import os
import sys
import json
import argparse
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import yaml
from timm.data import create_transform

from models.spatialmamba_2ca import SpatialMamba, LesionQueryHead, LesionToDiseaseMapper, LesionCalibModel2CA
from utils.losses import AsymmetricLoss, LesionMappingL1Regularizer

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import datetime


# ======================
# Tee: 把 stdout/stderr 同步写入文件
# ======================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


# ======================
# 通用工具
# ======================

def compute_metrics(y_true_list, y_logits_list, threshold=0.5):
    if isinstance(y_logits_list[0], torch.Tensor):
        y_logits = torch.cat(y_logits_list).detach().cpu()
        y_true = torch.cat(y_true_list).detach().cpu()
    else:
        y_logits = np.concatenate(y_logits_list)
        y_true = np.concatenate(y_true_list)
    y_probs = torch.sigmoid(y_logits).numpy()
    y_true = y_true.numpy()
    y_pred_bin = (y_probs > threshold).astype(int)
    try:
        mAP = average_precision_score(y_true, y_probs, average='micro')
    except ValueError:
        mAP = 0.0
    try:
        f1 = f1_score(y_true, y_pred_bin, average='micro')
    except ValueError:
        f1 = 0.0
    return mAP, f1


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def build_lesion_id_mapping(selected_lesion_ids: List[int]) -> Dict[int, int]:
    mapping = {int(raw_id): idx for idx, raw_id in enumerate(selected_lesion_ids)}
    return mapping


def extract_patches(image: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    C, H, W = image.shape
    pad_h = (patch_size - (H - patch_size) % stride) % stride
    pad_w = (patch_size - (W - patch_size) % stride) % stride

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0)

    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    n_h, n_w = patches.shape[:2]
    return patches.view(n_h * n_w, C, patch_size, patch_size)


# ======================
# Dataset
# ======================

class LesionCalibDataset(Dataset):
    """
    JSON:
        {
          "image_path": "data/SLO/xxx.jpg",
          "lesion_labels": [19, 27],   // 或 "labels": [19,27]
          "disease_labels": [0, 5, 7]
        }
    """

    def __init__(
        self,
        json_path: str,
        lesion_id_to_idx: Dict[int, int],
        num_lesions: int,
        num_classes: int,
        transform_hr,
        image_root: str = "",
        patch_size: int = 256,
        patch_stride: int = 256,
    ):
        super().__init__()
        self.image_root = image_root
        self.num_lesions = num_lesions
        self.num_classes = num_classes
        self.lesion_id_to_idx = lesion_id_to_idx
        self.transform_hr = transform_hr
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def _load_image_hr(self, image_path: str) -> torch.Tensor:
        if self.image_root and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        img = Image.open(image_path).convert("RGB")
        img = self.transform_hr(img)  # [C,H,W]，H=W=lesion_img_size
        return img

    def _encode_lesion_labels(self, raw_labels: List[int]) -> torch.Tensor:
        y = torch.zeros(self.num_lesions, dtype=torch.float32)
        for rid in raw_labels:
            rid = int(rid)
            if rid in self.lesion_id_to_idx:
                idx = self.lesion_id_to_idx[rid]
                y[idx] = 1.0
        return y

    def _encode_disease_labels(self, disease_labels: List[int]) -> torch.Tensor:
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in disease_labels:
            c = int(c)
            if 0 <= c < self.num_classes:
                y[c] = 1.0
        return y

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        lesion_raw = sample.get("lesion_labels", sample.get("labels", []))
        disease_raw = sample.get("disease_labels", [])

        img_hr = self._load_image_hr(image_path)          # [C, H_hr, W_hr]
        patches = extract_patches(img_hr, self.patch_size, self.patch_stride)  # [N, C, ph, pw]

        y_lesion = self._encode_lesion_labels(lesion_raw)    # [num_lesions]
        y_disease = self._encode_disease_labels(disease_raw) # [num_classes]

        return img_hr, patches, y_lesion, y_disease, image_path


# ======================
# Backbone / cfg / transform
# ======================

def build_backbone_from_cfg(cfg: Dict[str, Any], num_classes: int) -> SpatialMamba:
    model_cfg = cfg.get("MODEL", {})
    sm_cfg = model_cfg.get("SPATIALMAMBA", {})

    depths = sm_cfg.get("DEPTHS", [2, 4, 21, 5])
    embed_dim = sm_cfg.get("EMBED_DIM", 96)
    patch_size = sm_cfg.get("PATCH_SIZE", 4)
    d_state = sm_cfg.get("D_STATE", 1)

    dims = [
        int(embed_dim),
        int(embed_dim * 2),
        int(embed_dim * 4),
        int(embed_dim * 8),
    ]

    data_cfg = cfg.get("DATA", {})
    img_size = int(data_cfg.get("IMG_SIZE", 576))

    drop_path_rate = float(model_cfg.get("DROP_PATH_RATE", 0.5))

    model = SpatialMamba(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        d_state=d_state,
        drop_path_rate=drop_path_rate,
    )
    return model


def load_backbone_weights(model: SpatialMamba, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_ema" in ckpt:
        state = ckpt["model_ema"]
        print("[load_backbone_weights] Using 'model_ema' key")
    elif "model" in ckpt:
        state = ckpt["model"]
        print("[load_backbone_weights] Using 'model' key")
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
        print("[load_backbone_weights] Using 'state_dict' key")
    else:
        state = ckpt
        print("[load_backbone_weights] Using raw checkpoint")

    msg = model.load_state_dict(state, strict=False)
    print("[load_backbone_weights] Missing keys:", msg.missing_keys)
    print("[load_backbone_weights] Unexpected keys:", msg.unexpected_keys)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def load_backbone_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_args_with_cfg(args, cfg: Dict[str, Any]):
    train_cfg = cfg.get("TRAIN", {})
    data_cfg = cfg.get("DATA", {})
    aug_cfg = cfg.get("AUG", {})

    if args.stage_a_epochs is None:
        args.stage_a_epochs = int(train_cfg.get("EPOCHS", 50))
        args.stage_a_epochs = 100
    if args.stage_b_epochs is None:
        args.stage_b_epochs = max(50, args.stage_a_epochs)
        args.stage_b_epochs = 100

    if args.batch_size is None:
        args.batch_size = int(data_cfg.get("BATCH_SIZE", 8))
    if args.num_workers is None:
        args.num_workers = int(data_cfg.get("NUM_WORKERS", 8))

    if args.seed is None:
        args.seed = int(cfg.get("SEED", 0))

    args.aug_cfg = aug_cfg
    args.backbone_img_size = int(data_cfg.get("IMG_SIZE", 576))

    return args


def build_timm_transform_hr(args, is_train: bool):
    aug_cfg = getattr(args, "aug_cfg", {}) or {}

    img_size = args.lesion_img_size
    cj = float(aug_cfg.get("COLOR_JITTER", 0.0))
    aa = aug_cfg.get("AUTO_AUGMENT", "")
    reprob = float(aug_cfg.get("REPROB", 0.0))
    remode = aug_cfg.get("REMODE", "const")
    recount = int(aug_cfg.get("RECOUNT", 1))

    if not is_train:
        reprob = 0.0

    transform = create_transform(
        input_size=(3, img_size, img_size),
        is_training=is_train,
        color_jitter=None if cj <= 0 else cj,
        auto_augment=None if not aa or aa.lower() == "none" else aa,
        interpolation="bicubic",
        re_prob=reprob,
        re_mode=remode,
        re_count=recount,
    )
    return transform


# ======================
# Stage A: lesion_head
# ======================

def train_stage_a(
    lesion_head: LesionQueryHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    writer: SummaryWriter,
    lr: float = 1e-4,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
):
    print("==== Stage A: Train LesionQueryHead (lesion detection) ====")
    lesion_head.to(device)
    optimizer = torch.optim.AdamW(lesion_head.parameters(), lr=lr, weight_decay=0.05)
    criterion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)

    best_val_map = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        lesion_head.train()
        total_loss = 0.0
        total_count = 0

        for img_hr, patches, y_lesion, y_disease, _ in train_loader:
            patches = patches.to(device)
            y_lesion = y_lesion.to(device)

            optimizer.zero_grad()
            S = lesion_head(patches) # XXX: 对probs做优化？
            logits = safe_logit(S) # NOTE: transform to orig
            loss = criterion(logits, y_lesion)

            loss.backward()
            optimizer.step()

            bs = patches.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        avg_train_loss = total_loss / max(1, total_count)
        writer.add_scalar('StageA/Train/Loss', avg_train_loss, epoch)

        lesion_head.eval()
        val_loss = 0.0
        val_count = 0
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for img_hr, patches, y_lesion, y_disease, _ in val_loader:
                patches = patches.to(device)
                y_lesion = y_lesion.to(device)

                S = lesion_head(patches)
                logits = safe_logit(S)
                loss = criterion(logits, y_lesion)

                bs = patches.size(0)
                val_loss += loss.item() * bs
                val_count += bs

                val_logits_list.append(logits.detach().cpu())
                val_targets_list.append(y_lesion.detach().cpu())

        avg_val_loss = val_loss / max(1, val_count)
        val_map, val_f1 = compute_metrics(val_targets_list, val_logits_list)

        writer.add_scalar('StageA/Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('StageA/Val/mAP', val_map, epoch)
        writer.add_scalar('StageA/Val/F1', val_f1, epoch)

        print(f"[StageA][Epoch {epoch}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | mAP: {val_map:.4f} | F1: {val_f1:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_state = {
                "epoch": epoch,
                "lesion_head": lesion_head.state_dict(),
                "best_map": best_val_map
            }

    return best_state


# ======================
# Stage B: calib model
# ======================


def train_stage_b(
    calib_model: LesionCalibModel2CA,
    backbone_img_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    writer: SummaryWriter,
    lambda_lesion: float = 1.0,
    lambda_dise: float = 1.0,
    lambda_aux: float = 0.1,
    lambda_prior: float = 1e-3,
    lambda_noprior: float = 5e-3,
    lambda_smooth: float = 1e-3,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
):
    """
    Stage B: 训练 LesionCalibModel2CA
    新的双cross-attention架构:
      - LesionQueryHead: patch features -> lesion queries (cross-attn 1)
      - LesionToDiseaseMapper: lesion embeds -> disease queries (cross-attn 2)
    """
    print("==== Stage B: Train Δ + α + τ + (optional) fine-tune lesion_head ====")
    print(f"    Using LesionCalibModel2CA with dual cross-attention architecture")

    calib_model.to(device)
    
    # 冻结 backbone
    if calib_model.freeze_backbone:
        for p in calib_model.backbone.parameters():
            p.requires_grad_(False)
        print("    Backbone frozen")

    # 设置优化器参数组
    params = []
    
    # Mapper 参数
    if calib_model.mapper.delta is not None:
        params.append({"params": calib_model.mapper.delta, "lr": 1e-4, "name": "mapper.delta"})
    params.append({"params": calib_model.mapper.disease_query, "lr": 1e-4, "name": "mapper.disease_query"})
    params.append({"params": calib_model.mapper.ln_q.parameters(), "lr": 1e-4, "name": "mapper.ln_q"})
    params.append({"params": calib_model.mapper.ln_kv.parameters(), "lr": 1e-4, "name": "mapper.ln_kv"})
    params.append({"params": calib_model.mapper.attn.parameters(), "lr": 1e-4, "name": "mapper.attn"})
    
    # alpha (逐类) 和 lesion_threshold
    params.append({"params": calib_model.alpha, "lr": 1e-4, "name": "alpha"})
    params.append({"params": calib_model.lesion_threshold, "lr": 1e-3, "name": "lesion_threshold"})

    # LesionQueryHead 参数
    for p in calib_model.lesion_head.parameters():
        p.requires_grad_(True)
    params.append({"params": calib_model.lesion_head.parameters(), "lr": 5e-5, "name": "lesion_head"})

    optimizer = torch.optim.AdamW(params, weight_decay=0.05)

    criterion_dise = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
    criterion_lesion = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)

    M_prior = calib_model.mapper.M_prior.detach().clone()
    reg_delta = LesionMappingL1Regularizer(
        M_prior, lambda_prior=lambda_prior, lambda_noprior=lambda_noprior
    )

    best_val_dise_map = 0.0
    best_state = None
    num_classes = calib_model.num_classes
    num_lesions = calib_model.lesion_head.num_lesions

    for epoch in range(1, num_epochs + 1):
        calib_model.train()

        # ---- 训练阶段：细分各 loss 统计 ----
        total_loss = 0.0
        total_cnt = 0

        sum_loss_lesion = 0.0
        sum_loss_dise = 0.0
        sum_loss_aux = 0.0
        sum_loss_prior = 0.0
        sum_loss_noprior = 0.0
        sum_loss_smooth = 0.0

        # 用于统计 lesion probs 分布
        all_lesion_probs = []

        for img_hr, patches, y_lesion, y_disease, _ in train_loader:
            img_hr = img_hr.to(device)
            patches = patches.to(device)
            y_lesion = y_lesion.to(device)
            y_disease = y_disease.to(device)

            optimizer.zero_grad()
            
            # LesionCalibModel2CA.forward() 返回:
            # base_logits, corr_logits, final_logits, lesion_probs
            base_logits, corr_logits, final_logits, lesion_probs = calib_model(
                img_hr, patches, backbone_img_size, return_attn=False
            )

            # 计算各项 loss
            lesion_logits = safe_logit(lesion_probs)
            loss_lesion = criterion_lesion(lesion_logits, y_lesion)
            loss_dise = criterion_dise(final_logits, y_disease)
            loss_aux = criterion_dise(corr_logits, y_disease)
            loss_prior, loss_noprior = reg_delta(calib_model.mapper.delta)
            loss_smooth = (final_logits - base_logits).pow(2).mean()

            loss = (
                lambda_lesion * loss_lesion
                + lambda_dise * loss_dise
                + lambda_aux * loss_aux
                + loss_prior
                + loss_noprior
                + lambda_smooth * loss_smooth
            )

            loss.backward()
            optimizer.step()


            bs = img_hr.size(0)
            total_loss += loss.item() * bs
            total_cnt += bs

            sum_loss_lesion += loss_lesion.item() * bs
            sum_loss_dise += loss_dise.item() * bs
            sum_loss_aux += loss_aux.item() * bs
            sum_loss_prior += loss_prior.item() * bs
            sum_loss_noprior += loss_noprior.item() * bs
            sum_loss_smooth += loss_smooth.item() * bs

            all_lesion_probs.append(lesion_probs.detach().cpu())

        avg_train_loss = total_loss / max(1, total_cnt)
        avg_train_lesion = sum_loss_lesion / max(1, total_cnt)
        avg_train_dise = sum_loss_dise / max(1, total_cnt)
        avg_train_aux = sum_loss_aux / max(1, total_cnt)
        avg_train_prior = sum_loss_prior / max(1, total_cnt)
        avg_train_noprior = sum_loss_noprior / max(1, total_cnt)
        avg_train_smooth = sum_loss_smooth / max(1, total_cnt)

        # === TensorBoard: Train Loss ===
        writer.add_scalar('StageB/Train/TotalLoss', avg_train_loss, epoch)
        writer.add_scalar('StageB/Train/Loss_lesion', avg_train_lesion, epoch)
        writer.add_scalar('StageB/Train/Loss_dise', avg_train_dise, epoch)
        writer.add_scalar('StageB/Train/Loss_aux', avg_train_aux, epoch)
        writer.add_scalar('StageB/Train/Loss_prior', avg_train_prior, epoch)
        writer.add_scalar('StageB/Train/Loss_noprior', avg_train_noprior, epoch)
        writer.add_scalar('StageB/Train/Loss_smooth', avg_train_smooth, epoch)

        # === TensorBoard: alpha (per-class, record stats) ===
        alpha_vals = calib_model.alpha.detach().cpu()
        writer.add_scalar('StageB/Params/alpha_mean', alpha_vals.mean().item(), epoch)
        writer.add_scalar('StageB/Params/alpha_std', alpha_vals.std().item(), epoch)
        writer.add_scalar('StageB/Params/alpha_min', alpha_vals.min().item(), epoch)
        writer.add_scalar('StageB/Params/alpha_max', alpha_vals.max().item(), epoch)
        # Record per-class alpha if num_classes is not too large
        if num_classes <= 50:
            for c in range(num_classes):
                writer.add_scalar(f'StageB/Alpha_perClass/class_{c}', alpha_vals[c].item(), epoch)

        # === TensorBoard: lesion_threshold (per-lesion) ===
        tau_vals = calib_model.lesion_threshold.detach().cpu()
        writer.add_scalar('StageB/Params/tau_mean', tau_vals.mean().item(), epoch)
        writer.add_scalar('StageB/Params/tau_std', tau_vals.std().item(), epoch)
        writer.add_scalar('StageB/Params/tau_min', tau_vals.min().item(), epoch)
        writer.add_scalar('StageB/Params/tau_max', tau_vals.max().item(), epoch)

        # === TensorBoard: Mapper delta ===
        if calib_model.mapper.delta is not None:
            delta = calib_model.mapper.delta.detach().cpu()
            writer.add_scalar('StageB/Params/delta_abs_mean', delta.abs().mean().item(), epoch)
            writer.add_scalar('StageB/Params/delta_abs_max', delta.abs().max().item(), epoch)
            writer.add_scalar('StageB/Params/delta_std', delta.std().item(), epoch)
            # W = M_prior + delta effective matrix
            W = calib_model.mapper.get_W().detach().cpu()
            writer.add_scalar('StageB/Params/W_mean', W.mean().item(), epoch)
            writer.add_scalar('StageB/Params/W_sparsity', (W < 0.01).float().mean().item(), epoch)

        # === TensorBoard: Mapper prior_bias_scale ===
        writer.add_scalar('StageB/Params/prior_bias_scale', calib_model.mapper.prior_bias_scale, epoch)

        # === TensorBoard: Lesion probs distribution ===
        all_lesion_probs_cat = torch.cat(all_lesion_probs, dim=0)  # [N_samples, num_lesions]
        writer.add_scalar('StageB/LesionProbs/mean_all', all_lesion_probs_cat.mean().item(), epoch)
        writer.add_scalar('StageB/LesionProbs/std_all', all_lesion_probs_cat.std().item(), epoch)
        # Activation rate (ratio exceeding threshold)
        tau_expanded = tau_vals.unsqueeze(0)
        activation_rate = (all_lesion_probs_cat > tau_expanded).float().mean().item()
        writer.add_scalar('StageB/LesionProbs/activation_rate', activation_rate, epoch)


        # ---- Validation: compare Base vs Final metrics ----
        calib_model.eval()
        val_loss_dise_final = 0.0
        val_loss_dise_base = 0.0
        val_cnt = 0

        val_logits_final = []
        val_logits_base = []
        val_logits_corr = []
        val_targets_dise = []
        val_lesion_probs = []
        val_lesion_targets = []

        # For recording attention weights
        val_attn_lesion_patch_list = []
        val_attn_disease_lesion_list = []

        with torch.no_grad():
            for batch_idx, (img_hr, patches, y_lesion, y_disease, _) in enumerate(val_loader):
                img_hr = img_hr.to(device)
                patches = patches.to(device)
                y_disease = y_disease.to(device)
                y_lesion = y_lesion.to(device)

                # Get attention only for first batch for visualization
                get_attn = (batch_idx == 0)
                
                if get_attn:
                    base_logits, corr_logits, final_logits, lesion_probs, extra = calib_model(
                        img_hr, patches, backbone_img_size, return_attn=True
                    )
                    val_attn_lesion_patch_list.append(extra["attn_lesion_patch"].detach().cpu())
                    val_attn_disease_lesion_list.append(extra["attn_disease_lesion"].detach().cpu())
                else:
                    base_logits, corr_logits, final_logits, lesion_probs = calib_model(
                        img_hr, patches, backbone_img_size, return_attn=False
                    )

                loss_d_final = criterion_dise(final_logits, y_disease)
                loss_d_base = criterion_dise(base_logits, y_disease)

                bs = img_hr.size(0)
                val_loss_dise_final += loss_d_final.item() * bs
                val_loss_dise_base += loss_d_base.item() * bs
                val_cnt += bs

                val_logits_final.append(final_logits.detach().cpu())
                val_logits_base.append(base_logits.detach().cpu())
                val_logits_corr.append(corr_logits.detach().cpu())
                val_targets_dise.append(y_disease.detach().cpu())
                val_lesion_probs.append(lesion_probs.detach().cpu())
                val_lesion_targets.append(y_lesion.detach().cpu())

        avg_val_dise_loss_final = val_loss_dise_final / max(1, val_cnt)
        avg_val_dise_loss_base = val_loss_dise_base / max(1, val_cnt)

        # Final metrics
        val_map_final, val_f1_final = compute_metrics(val_targets_dise, val_logits_final)
        # Base metrics
        val_map_base, val_f1_base = compute_metrics(val_targets_dise, val_logits_base)
        # Lesion metrics
        val_lesion_logits = [safe_logit(p) for p in val_lesion_probs]
        val_lesion_map, val_lesion_f1 = compute_metrics(val_lesion_targets, val_lesion_logits)

        # === TensorBoard: Val Loss + Metrics ===
        writer.add_scalar('StageB/Val/DiseaseLoss_final', avg_val_dise_loss_final, epoch)
        writer.add_scalar('StageB/Val/DiseaseLoss_base', avg_val_dise_loss_base, epoch)

        writer.add_scalar('StageB/Val/Disease_mAP_final', val_map_final, epoch)
        writer.add_scalar('StageB/Val/Disease_mAP_base', val_map_base, epoch)
        writer.add_scalar('StageB/Val/Disease_mAP_delta', val_map_final - val_map_base, epoch)

        writer.add_scalar('StageB/Val/Disease_F1_final', val_f1_final, epoch)
        writer.add_scalar('StageB/Val/Disease_F1_base', val_f1_base, epoch)
        writer.add_scalar('StageB/Val/Disease_F1_delta', val_f1_final - val_f1_base, epoch)

        writer.add_scalar('StageB/Val/Lesion_mAP', val_lesion_map, epoch)
        writer.add_scalar('StageB/Val/Lesion_F1', val_lesion_f1, epoch)

        # === TensorBoard: corr_logits analysis ===
        val_corr_cat = torch.cat(val_logits_corr, dim=0)
        writer.add_scalar('StageB/Val/corr_logits_mean', val_corr_cat.mean().item(), epoch)
        writer.add_scalar('StageB/Val/corr_logits_std', val_corr_cat.std().item(), epoch)
        writer.add_scalar('StageB/Val/corr_logits_abs_mean', val_corr_cat.abs().mean().item(), epoch)

        # === TensorBoard: Attention visualization (once per epoch) ===
        if len(val_attn_lesion_patch_list) > 0:
            # attn_lesion_patch: [B, num_lesions, num_patches] - Lesion Query to Patch attention
            attn_lp = val_attn_lesion_patch_list[0]
            writer.add_scalar('StageB/Attention/lesion_patch_entropy_mean', 
                            compute_attention_entropy(attn_lp).mean().item(), epoch)
            
            # attn_disease_lesion: [B, num_classes, num_lesions] - Disease Query to Lesion attention
            attn_dl = val_attn_disease_lesion_list[0]
            writer.add_scalar('StageB/Attention/disease_lesion_entropy_mean',
                            compute_attention_entropy(attn_dl).mean().item(), epoch)
            
            # Attention sparsity (top-k coverage)
            writer.add_scalar('StageB/Attention/lesion_patch_top3_coverage',
                            compute_topk_coverage(attn_lp, k=3), epoch)
            writer.add_scalar('StageB/Attention/disease_lesion_top5_coverage',
                            compute_topk_coverage(attn_dl, k=5), epoch)

        # Console output
        print(f"[StageB][Epoch {epoch}/{num_epochs}] "
              f"Train Total: {avg_train_loss:.4f} | "
              f"L_lesion: {avg_train_lesion:.4f} | "
              f"L_dise: {avg_train_dise:.4f} | "
              f"L_aux: {avg_train_aux:.4f} | "
              f"L_prior: {avg_train_prior:.4f} | "
              f"L_noprior: {avg_train_noprior:.4f}")
        print(f"          Val Base  : Loss={avg_val_dise_loss_base:.4f} "
              f"| mAP={val_map_base:.4f} | F1={val_f1_base:.4f}")
        print(f"          Val Final : Loss={avg_val_dise_loss_final:.4f} "
              f"| mAP={val_map_final:.4f} | F1={val_f1_final:.4f}")
        print(f"          Val Lesion: mAP={val_lesion_map:.4f} | F1={val_lesion_f1:.4f}")
        print(f"          Delta(mAP,F1) : ({val_map_final - val_map_base:+.4f}, "
              f"{val_f1_final - val_f1_base:+.4f}) | "
              f"alpha_mean={alpha_vals.mean().item():.4f} | tau_mean={tau_vals.mean().item():.4f}\n")

        # Use final mAP as best criterion
        if val_map_final > best_val_dise_map:
            print(f"  New best FINAL mAP! ({best_val_dise_map:.4f} -> {val_map_final:.4f}) Saving model...")
            best_val_dise_map = val_map_final
            best_state = {
                "epoch": epoch,
                "calib_model": calib_model.state_dict(),
                "best_map": best_val_dise_map
            }

    return best_state


def compute_attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distribution to measure concentration/dispersion.
    attn: [B, Q, K] attention weights (normalized)
    Returns: [B, Q] entropy for each query
    """
    attn_clamped = attn.clamp(min=1e-8)
    entropy = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)
    return entropy


def compute_topk_coverage(attn: torch.Tensor, k: int = 3) -> float:
    """
    Compute top-k attention coverage.
    attn: [B, Q, K]
    Returns: mean sum of top-k weights
    """
    topk_vals, _ = torch.topk(attn, k=min(k, attn.size(-1)), dim=-1)
    coverage = topk_vals.sum(dim=-1).mean().item()
    return coverage



# ======================
# argparse
# ======================

def parse_args():
    parser = argparse.ArgumentParser("Lesion calibration with timm aug (HR tiles + SpatialMamba backbone)")

    parser.add_argument("--backbone_cfg", type=str, required=True,
                        help="主干 Spatial-Mamba 的 yaml 配置路径（B0）")
    parser.add_argument("--backbone_ckpt", type=str, required=True,
                        help="主干 ckpt 路径（比如 best_ckpt_ema.pth）")

    parser.add_argument("--train_json", type=str, required=True,
                        help="lesion_train json 路径")
    parser.add_argument("--val_json", type=str, required=True,
                        help="lesion_val json 路径")
    parser.add_argument("--image_root", type=str, default="",
                        help="图片根目录（json 里的 image_path 如果是相对路径）")

    parser.add_argument("--selected_lesion_ids", type=str,
                        default="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 48",
                        help="病灶原始 ID，逗号分隔，顺序要和先验矩阵行一致")
    parser.add_argument("--num_classes", type=int, default=27,
                        help="疾病类别数")

    parser.add_argument("--lesion_img_size", type=int, default=2048,
                        help="高分辨分支统一 resize 的尺寸")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="每个 patch 尺寸")
    parser.add_argument("--patch_stride", type=int, default=256,
                        help="patch 滑窗步长")

    parser.add_argument("--prior_matrix_path", type=str, required=True,
                        help="shape=[num_lesions,num_classes] 的先验矩阵（.npy 或 .json）")
    
    parser.add_argument("--resume_stageb", action="store_true", help="跳过stageA，直接加载lesionhead训stageB.")
    parser.add_argument("--stage_a_ckpt", type=str, default="/data1/lesion_calib_1_runG0/stageA_best.pth")

    parser.add_argument("--stage_a_epochs", type=int, default=None)
    parser.add_argument("--stage_b_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--stage_a_lr", type=float, default=1e-4)
    parser.add_argument("--lambda_lesion", type=float, default=1.0)
    parser.add_argument("--lambda_dise", type=float, default=1.0)
    parser.add_argument("--lambda_aux", type=float, default=0.1)
    parser.add_argument("--lambda_prior", type=float, default=1e-3)
    parser.add_argument("--lambda_noprior", type=float, default=3e-3)
    parser.add_argument("--lambda_smooth", type=float, default=1e-3)

    parser.add_argument("--gamma_neg", type=float, default=4.0)
    parser.add_argument("--gamma_pos", type=float, default=1.0)

    parser.add_argument("--output", type=str, default="./output_lesion_calib_hr")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# ======================
# prior 矩阵
# ======================

def load_prior_matrix(path: str, num_lesions: int, num_classes: int) -> torch.Tensor:
    if path.endswith(".npy"):
        import numpy as np
        mat = np.load(path)
        mat = torch.from_numpy(mat)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            mat = json.load(f)
        mat = torch.tensor(mat, dtype=torch.float32)
    else:
        raise ValueError("prior_matrix_path 只支持 .npy 或 .json")

    assert mat.shape == (num_lesions, num_classes), \
        f"prior matrix shape={mat.shape}, 期望=({num_lesions},{num_classes})"
    return mat

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.output, "runs", f"calib_experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train_log.txt")
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"[Main] Logging to: {log_dir}")
    print(f"[Main] Console log file: {log_path}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    cfg = load_backbone_cfg(args.backbone_cfg)
    args = override_args_with_cfg(args, cfg)

    print("========== Config after YAML override ==========")
    print(f"stage_a_epochs={args.stage_a_epochs}, stage_b_epochs={args.stage_b_epochs}")
    print(f"batch_size={args.batch_size}, num_workers={args.num_workers}, seed={args.seed}")
    print(f"lesion_img_size={args.lesion_img_size}, patch_size={args.patch_size}, patch_stride={args.patch_stride}")
    print(f"backbone_img_size(from YAML)={args.backbone_img_size}")
    print(f"AUG cfg={getattr(args, 'aug_cfg', {})}")
    print("================================================")

    assert (args.lesion_img_size - args.patch_size) % args.patch_stride == 0, \
        "请保证 (lesion_img_size - patch_size) 能被 patch_stride 整除"

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    selected_lesion_ids = [int(x) for x in args.selected_lesion_ids.split(",")]
    num_lesions = len(selected_lesion_ids)
    lesion_id_to_idx = build_lesion_id_mapping(selected_lesion_ids)

    prior_matrix = load_prior_matrix(args.prior_matrix_path, num_lesions, args.num_classes)

    train_transform_hr = build_timm_transform_hr(args, is_train=True)
    val_transform_hr = build_timm_transform_hr(args, is_train=False)

    train_dataset = LesionCalibDataset(
        json_path=args.train_json,
        lesion_id_to_idx=lesion_id_to_idx,
        num_lesions=num_lesions,
        num_classes=args.num_classes,
        transform_hr=train_transform_hr,
        image_root=args.image_root,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
    )
    val_dataset = LesionCalibDataset(
        json_path=args.val_json,
        lesion_id_to_idx=lesion_id_to_idx,
        num_lesions=num_lesions,
        num_classes=args.num_classes,
        transform_hr=val_transform_hr,
        image_root=args.image_root,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
    )

    print("Checking dataset output shape to set model parameters...")
    _, sample_patches, _, _, _ = train_dataset[0]
    real_num_patches = sample_patches.shape[0]
    print(f"Detected num_patches: {real_num_patches} (from dataset directly)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    backbone = build_backbone_from_cfg(cfg, num_classes=args.num_classes)
    backbone = load_backbone_weights(backbone, args.backbone_ckpt, device)

    lesion_head = LesionQueryHead(
        backbone=backbone,
        num_lesions=num_lesions,
        num_patches=real_num_patches,
        lesion_ids=selected_lesion_ids,
        embed_dim=256,
        num_heads=4,
        dropout=0.0,
    )
    mapper = LesionToDiseaseMapper(
        prior_matrix=prior_matrix,
        learn_delta=True,
    )

    # Stage A
    if args.stage_a_ckpt is not None:
        stageA_ckpt_path = args.stage_a_ckpt
    else:
        stageA_ckpt_path = os.path.join(args.output, "stageA_best.pth")
    if args.resume_stageb:
        if not os.path.isfile(stageA_ckpt_path):
            raise FileNotFoundError(
                f"[Main] --resume_stageb but not found StageA ckpt: {stageA_ckpt_path}"
            )
        ckpt_a = torch.load(stageA_ckpt_path, map_location=device, weights_only=False)
        lesion_head.load_state_dict(ckpt_a["lesion_head"])
        print(f"[Main] Resume StageB: Have loaded StageA.")
    else:
        best_stage_a = train_stage_a(
            lesion_head=lesion_head,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.stage_a_epochs,
            device=device,
            writer=writer,
            lr=args.stage_a_lr,
            gamma_neg=args.gamma_neg,
            gamma_pos=args.gamma_pos,
        )
        if best_stage_a is not None:
            torch.save(best_stage_a, os.path.join(args.output, "stageA_best.pth"))
            lesion_head.load_state_dict(best_stage_a["lesion_head"])

    # Stage B - Use LesionCalibModel2CA from spatialmamba_2ca.py
    calib_model = LesionCalibModel2CA(
        backbone=backbone,
        lesion_head=lesion_head,
        mapper=mapper,
        num_classes=args.num_classes,
        alpha_init=0.1,
        tau_init=0.3,
        freeze_backbone=True,
    )

    best_stage_b = train_stage_b(
        calib_model=calib_model,
        backbone_img_size=args.backbone_img_size,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.stage_b_epochs,
        device=device,
        writer=writer,
        lambda_lesion=args.lambda_lesion,
        lambda_dise=args.lambda_dise,
        lambda_aux=args.lambda_aux,
        lambda_prior=args.lambda_prior,
        lambda_noprior=args.lambda_noprior,
        lambda_smooth=args.lambda_smooth,
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
    )
    if best_stage_b is not None:
        torch.save(best_stage_b, os.path.join(args.output, "stageB_best.pth"))

    writer.close()
    try:
        log_file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
