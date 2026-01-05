import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
from timm.data import create_transform

from models.spatialmamba import SpatialMamba, LesionQueryHead, LesionToDiseaseMapper

# [MOD] 可视化相关
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from main_calib_train import LesionCalibModel


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lesion_id_mapping(selected_lesion_ids: List[int]) -> Dict[int, int]:
    mapping = {int(raw_id): idx for idx, raw_id in enumerate(selected_lesion_ids)}
    return mapping


def extract_patches(image: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """
    image: [C, H, W]
    return: [N, C, ph, pw]
    """
    C, H, W = image.shape
    pad_h = (patch_size - (H - patch_size) % stride) % stride
    pad_w = (patch_size - (W - patch_size) % stride) % stride

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0)

    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    n_h, n_w = patches.shape[:2]
    return patches.view(n_h * n_w, C, patch_size, patch_size)


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


def load_backbone_weights(
    model: SpatialMamba,
    ckpt_path: str,
    device: torch.device,
    use_ema: bool = True,
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = None

    # print(ckpt)

    if use_ema:
        for k in ["model_ema", "ema", "state_dict_ema"]:
            if isinstance(ckpt, dict) and k in ckpt:
                print(f"[load_backbone_weights] Using EMA weights from key='{k}'")
                state = ckpt[k]
                break

    if state is None:
        if isinstance(ckpt, dict) and "model" in ckpt:
            print("[load_backbone_weights] Using non-EMA weights from key='model'")
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            print("[load_backbone_weights] Using non-EMA weights from key='state_dict'")
            state = ckpt["state_dict"]
        else:
            print("[load_backbone_weights] Using whole checkpoint as state_dict")
            state = ckpt

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
    data_cfg = cfg.get("DATA", {})
    aug_cfg = cfg.get("AUG", {})
    train_cfg = cfg.get("TRAIN", {})

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


def load_prior_matrix(path: str, num_lesions: int, num_classes: int) -> torch.Tensor:
    if path.endswith(".npy"):
        mat = np.load(path)
        mat = torch.from_numpy(mat)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            mat = json.load(f)
        mat = torch.tensor(mat, dtype=torch.float32)
    else:
        raise ValueError("prior_matrix_path 只支持 .npy 或 .json")

    assert mat.shape == (num_lesions, num_classes), \
        f"prior matrix shape={mat.shape}, expected=({num_lesions},{num_classes})"
    return mat


class OrigValLesionCalibDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        num_classes: int,
        transform_hr,
        image_root: str = "",
        patch_size: int = 256,
        patch_stride: int = 256,
        disease_labelmap_path: Optional[str] = None,
    ):
        super().__init__()
        self.image_root = image_root
        self.num_classes = num_classes
        self.transform_hr = transform_hr
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.disease_labelmap: Dict[str, int] = {}
        if disease_labelmap_path is not None:
            with open(disease_labelmap_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            self.disease_labelmap = {str(k): int(v) for k, v in raw_map.items()}
            print(f"[Dataset] Loaded disease_labelmap with {len(self.disease_labelmap)} entries.")

    def __len__(self):
        return len(self.samples)

    def _load_image_hr(self, image_path: str) -> torch.Tensor:
        if self.image_root and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        img = Image.open(image_path).convert("RGB")
        img = self.transform_hr(img)  # [C, H, W]，H=W=lesion_img_size
        return img

    def _encode_disease_labels(self, disease_labels_raw) -> torch.Tensor:
        y = torch.zeros(self.num_classes, dtype=torch.float32)

        if not isinstance(disease_labels_raw, (list, tuple)):
            disease_labels_raw = [disease_labels_raw]

        for lab in disease_labels_raw:
            idx: Optional[int] = None

            if isinstance(lab, (int, float)):
                idx = int(lab)
            else:
                key = str(lab)
                if self.disease_labelmap:
                    if key in self.disease_labelmap:
                        idx = int(self.disease_labelmap[key])
                    else:
                        idx = None
                else:
                    idx = None

            if idx is None:
                continue

            if 0 <= idx < self.num_classes:
                y[idx] = 1.0
            else:
                pass

        return y

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]

        disease_raw = sample.get("disease_labels", sample.get("labels", []))

        img_hr = self._load_image_hr(image_path)
        patches = extract_patches(img_hr, self.patch_size, self.patch_stride)

        y_disease = self._encode_disease_labels(disease_raw)

        dummy_lesion = torch.zeros(1, dtype=torch.float32)

        return img_hr, patches, dummy_lesion, y_disease, image_path


def sigmoid_ap(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]
    y_score = y_score[order]

    tp = torch.cumsum(y_true, dim=0)
    fp = torch.cumsum(1 - y_true, dim=0)
    denom = tp + fp
    precision = tp / torch.clamp_min(denom, 1.0)
    recall = tp / torch.clamp_min(y_true.sum(), 1.0)

    mrec = torch.cat([
        torch.tensor([0.], device=recall.device),
        recall,
        torch.tensor([1.], device=recall.device),
    ])
    mpre = torch.cat([
        torch.tensor([0.], device=precision.device),
        precision,
        torch.tensor([0.], device=precision.device),
    ])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

    idx = torch.nonzero(mrec[1:] != mrec[:-1]).squeeze(1)
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


def compute_metrics_from_logits(
    logits: torch.Tensor,
    targs: torch.Tensor,
    tau: float = 0.5,
):
    probs = torch.sigmoid(logits)

    preds_bin = (probs > tau).int()
    eps = 1e-9
    tp = (preds_bin & (targs > tau)).sum(dim=0).float()
    fp = (preds_bin & (targs <= tau)).sum(dim=0).float()
    fn = ((1 - preds_bin) & (targs > tau)).sum(dim=0).float()

    per_prec = (tp / (tp + fp + eps)).cpu().numpy().astype(np.float32)
    per_rec = (tp / (tp + fn + eps)).cpu().numpy().astype(np.float32)
    per_f1 = (2 * per_prec * per_rec / (per_prec + per_rec + eps)).astype(np.float32)
    macro_f1 = float(np.nanmean(per_f1))

    tp_mi = float(tp.sum())
    fp_mi = float(fp.sum())
    fn_mi = float(fn.sum())
    prec_mi = tp_mi / (tp_mi + fp_mi + eps)
    rec_mi = tp_mi / (tp_mi + fn_mi + eps)
    micro_f1 = (2 * prec_mi * rec_mi / (prec_mi + rec_mi + eps)) if (prec_mi + rec_mi) > 0 else 0.0

    per_ap = []
    for c in range(probs.shape[1]):
        per_ap.append(sigmoid_ap(targs[:, c], probs[:, c]))
    per_ap = np.array(per_ap, dtype=np.float32)
    macro_map = float(np.nanmean(per_ap))
    micro_map = float(sigmoid_ap(targs.reshape(-1), probs.reshape(-1)))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_map": micro_map,
        "macro_map": macro_map,
        "per_precision": per_prec,
        "per_recall": per_rec,
        "per_f1": per_f1,
        "per_ap": per_ap,
        "probs": probs.cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def eval_lesion_calib(
    calib_model: LesionCalibModel,
    backbone_img_size: int,
    data_loader: DataLoader,
    device: torch.device,
    tau: float = 0.5,
    dump_preds: bool = True,
    dump_dir: str = "./lesion_calib_eval",
    tb_writer: Optional[SummaryWriter] = None,
):
    calib_model.eval()
    os.makedirs(dump_dir, exist_ok=True)

    all_base_logits = []
    all_final_logits = []
    all_targets = []
    image_keys = []

    for batch in data_loader:
        img_hr, patches, _, y_disease, img_path = batch
        img_hr = img_hr.to(device)
        patches = patches.to(device)
        y_disease = y_disease.to(device)

        base_logits, _, final_logits, _ = calib_model(img_hr, patches, backbone_img_size)

        all_base_logits.append(base_logits.detach().cpu())
        all_final_logits.append(final_logits.detach().cpu())
        all_targets.append(y_disease.detach().cpu())
        image_keys.extend(list(img_path))

    base_logits = torch.cat(all_base_logits, dim=0)
    final_logits = torch.cat(all_final_logits, dim=0)
    targs = torch.cat(all_targets, dim=0)

    base_metrics = compute_metrics_from_logits(base_logits, targs, tau=tau)
    final_metrics = compute_metrics_from_logits(final_logits, targs, tau=tau)

    print(f"\n[LesionCalibEval] tau={tau:.2f}")
    print("  >> Base   (backbone only): "
          f"micro-F1={base_metrics['micro_f1']:.4f}  "
          f"macro-F1={base_metrics['macro_f1']:.4f}  "
          f"micro-mAP={base_metrics['micro_map']:.4f}  "
          f"macro-mAP={base_metrics['macro_map']:.4f}")
    print("  >> Final  (backbone + lesion): "
          f"micro-F1={final_metrics['micro_f1']:.4f}  "
          f"macro-F1={final_metrics['macro_f1']:.4f}  "
          f"micro-mAP={final_metrics['micro_map']:.4f}  "
          f"macro-mAP={final_metrics['macro_map']:.4f}")

    print("  >> Delta  (Final - Base): "
          f"Δmicro-F1={final_metrics['micro_f1'] - base_metrics['micro_f1']:+.4f}  "
          f"Δmacro-F1={final_metrics['macro_f1'] - base_metrics['macro_f1']:+.4f}  "
          f"Δmicro-mAP={final_metrics['micro_map'] - base_metrics['micro_map']:+.4f}  "
          f"Δmacro-mAP={final_metrics['macro_map'] - base_metrics['macro_map']:+.4f}\n")

    if tb_writer is not None:
        step = 0
        tb_writer.add_scalar("base/micro_f1", base_metrics["micro_f1"], step)
        tb_writer.add_scalar("base/macro_f1", base_metrics["macro_f1"], step)
        tb_writer.add_scalar("base/micro_mAP", base_metrics["micro_map"], step)
        tb_writer.add_scalar("base/macro_mAP", base_metrics["macro_map"], step)

        tb_writer.add_scalar("final/micro_f1", final_metrics["micro_f1"], step)
        tb_writer.add_scalar("final/macro_f1", final_metrics["macro_f1"], step)
        tb_writer.add_scalar("final/micro_mAP", final_metrics["micro_map"], step)
        tb_writer.add_scalar("final/macro_mAP", final_metrics["macro_map"], step)

        tb_writer.add_scalar("delta/micro_f1", final_metrics["micro_f1"] - base_metrics["micro_f1"], step)
        tb_writer.add_scalar("delta/macro_f1", final_metrics["macro_f1"] - base_metrics["macro_f1"], step)
        tb_writer.add_scalar("delta/micro_mAP", final_metrics["micro_map"] - base_metrics["micro_map"], step)
        tb_writer.add_scalar("delta/macro_mAP", final_metrics["macro_map"] - base_metrics["macro_map"], step)

    if dump_preds:
        logits_base_np = base_logits.numpy().astype(np.float32)
        logits_final_np = final_logits.numpy().astype(np.float32)
        targs_np = targs.numpy().astype(np.float32)
        probs_base_np = base_metrics["probs"]
        probs_final_np = final_metrics["probs"]

        jsonl_base = os.path.join(dump_dir, "val_preds_base.jsonl")
        with open(jsonl_base, "w", encoding="utf-8") as f:
            for i in range(targs_np.shape[0]):
                rec = {
                    "image_key": image_keys[i],
                    "y_true": targs_np[i].astype(float).tolist(),
                    "logits": logits_base_np[i].astype(float).tolist(),
                    "probs": probs_base_np[i].astype(float).tolist(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[LesionCalibEval] wrote base predictions to {jsonl_base}")

        jsonl_final = os.path.join(dump_dir, "val_preds_final.jsonl")
        with open(jsonl_final, "w", encoding="utf-8") as f:
            for i in range(targs_np.shape[0]):
                rec = {
                    "image_key": image_keys[i],
                    "y_true": targs_np[i].astype(float).tolist(),
                    "logits": logits_final_np[i].astype(float).tolist(),
                    "probs": probs_final_np[i].astype(float).tolist(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[LesionCalibEval] wrote final predictions to {jsonl_final}")

        global_metrics = {
            "tau": float(tau),
            "num_samples": int(targs_np.shape[0]),
            "num_classes": int(targs_np.shape[1]),
            "base": {
                "micro_f1": base_metrics["micro_f1"],
                "macro_f1": base_metrics["macro_f1"],
                "micro_map": base_metrics["micro_map"],
                "macro_map": base_metrics["macro_map"],
            },
            "final": {
                "micro_f1": final_metrics["micro_f1"],
                "macro_f1": final_metrics["macro_f1"],
                "micro_map": final_metrics["micro_map"],
                "macro_map": final_metrics["macro_map"],
            },
            "delta": {
                "micro_f1": final_metrics["micro_f1"] - base_metrics["micro_f1"],
                "macro_f1": final_metrics["macro_f1"] - base_metrics["macro_f1"],
                "micro_map": final_metrics["micro_map"] - base_metrics["micro_map"],
                "macro_map": final_metrics["macro_map"] - base_metrics["macro_map"],
            }
        }
        gm_path = os.path.join(dump_dir, "global_metrics.json")
        with open(gm_path, "w", encoding="utf-8") as f:
            json.dump(global_metrics, f, ensure_ascii=False, indent=2)
        print(f"[LesionCalibEval] wrote global metrics (base+final+delta) to {gm_path}")

    return {
        "base": base_metrics,
        "final": final_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser("Eval Lesion Calibration Model on origval (HR tiles + SpatialMamba backbone)")

    parser.add_argument("--backbone_cfg", type=str, required=True,
                        help="Spatial-Mamba YAML config (B0)")
    parser.add_argument("--backbone_ckpt", type=str, required=True,
                        help="backbone checkpoint (支持普通或best_ckpt_ema.pth)")
    parser.add_argument("--use_ema", action="store_true",
                        help="如果ckpt里有 model_ema，就用EMA那份权重")
    parser.add_argument("--eval_json", type=str, required=True,
                        help="origval json 路径（疾病标签可以是数字或中文字符串）")
    parser.add_argument("--image_root", type=str, default="",
                        help="图片根目录（eval_json 里的 image_path 若为相对路径，则会与这里拼接）")
    parser.add_argument("--disease_labelmap", type=str, default=None,
                        help="可选：疾病中文/英文名到类别id的映射json，例如 {'PDR': 10}")
    parser.add_argument("--selected_lesion_ids", type=str, required=True,
                        help="训练 lesion 分支时使用的那一串病灶原始 ID，逗号分隔，顺序需与先验矩阵一致")
    parser.add_argument("--num_classes", type=int, default=27,
                        help="疾病类别数（需与主干一致）")
    parser.add_argument("--lesion_img_size", type=int, default=2048,
                        help="高分辨分支统一 resize 尺寸（需与训练时保持一致）")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="每个 patch 尺寸（需与训练时保持一致）")
    parser.add_argument("--patch_stride", type=int, default=256,
                        help="patch 滑窗步长（需与训练时保持一致）")
    parser.add_argument("--prior_matrix_path", type=str, required=True,
                        help="shape=[num_lesions, num_classes] 的先验矩阵（.npy 或 .json）")
    parser.add_argument("--calib_ckpt", type=str, required=True,
                        help="main_lesion_calib.py 训练出的 stageB_best.pth（包含 calib_model.state_dict）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="评估时二值化阈值，用于F1/precision/recall")
    parser.add_argument("--output", type=str, default="./output_lesion_calib_eval",
                        help="评估结果输出根目录（下面会建 eval_xxx 子目录）")
    parser.add_argument("--tb_logdir", type=str, default=None,
                        help="TensorBoard 日志目录（默认放在 output/eval_xxx/tb 下）")

    # [MOD] 中文字体路径（可选）
    parser.add_argument("--font_path", type=str, default=None,
                        help="TTF 中文字体文件，用于 matplotlib 显示中文")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.font_path is not None:
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(args.font_path)
            font_prop = font_manager.FontProperties(fname=args.font_path)
            font_name = font_prop.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[Eval] 使用中文字体: {font_name} (from {args.font_path})")
        except Exception as e:
            print(f"[Eval] 加载中文字体失败: {e}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_eval_dir = os.path.join(args.output, f"eval_{timestamp}")
    os.makedirs(root_eval_dir, exist_ok=True)

    log_path = os.path.join(root_eval_dir, "log.txt")
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"[Eval] Logging to directory: {root_eval_dir}")
    print(f"[Eval] Console log is being written to: {log_path}")

    cfg = load_backbone_cfg(args.backbone_cfg)
    args = override_args_with_cfg(args, cfg)

    print("========== Eval Config ==========")
    print(f"batch_size={args.batch_size}, num_workers={args.num_workers}, seed={args.seed}")
    print(f"lesion_img_size={args.lesion_img_size}, patch_size={args.patch_size}, patch_stride={args.patch_stride}")
    print(f"backbone_img_size(from YAML)={args.backbone_img_size}")
    print(f"use_ema={args.use_ema}")
    print("=================================")

    assert (args.lesion_img_size - args.patch_size) % args.patch_stride == 0, \
        "请保证 (lesion_img_size - patch_size) 能被 patch_stride 整除"

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tb_dir = args.tb_logdir or os.path.join(root_eval_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_dir)
    print(f"[Eval] TensorBoard logging to: {tb_dir}")

    selected_lesion_ids = [int(x) for x in args.selected_lesion_ids.split(",")]
    num_lesions = len(selected_lesion_ids)
    prior_matrix = load_prior_matrix(args.prior_matrix_path, num_lesions, args.num_classes)

    eval_transform_hr = build_timm_transform_hr(args, is_train=False)

    eval_dataset = OrigValLesionCalibDataset(
        json_path=args.eval_json,
        num_classes=args.num_classes,
        transform_hr=eval_transform_hr,
        image_root=args.image_root,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        disease_labelmap_path=args.disease_labelmap,
    )

    print("Checking dataset output shape to set num_patches...")
    img_hr0, patches0, _, y0, path0 = eval_dataset[0]
    real_num_patches = patches0.shape[0]
    print(f"Detected num_patches: {real_num_patches} from first sample ({path0})")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    backbone = build_backbone_from_cfg(cfg, num_classes=args.num_classes)
    backbone = load_backbone_weights(
        backbone,
        args.backbone_ckpt,
        device,
        use_ema=args.use_ema,
    )

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

    calib_model = LesionCalibModel(
        backbone=backbone,
        lesion_head=lesion_head,
        mapper=mapper,
        num_classes=args.num_classes,
        alpha_init=0.1,
    )

    print(f"Loading calib checkpoint from: {args.calib_ckpt}")
    ckpt = torch.load(args.calib_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "calib_model" in ckpt:
        state = ckpt["calib_model"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    msg = calib_model.load_state_dict(state, strict=False)
    print("[calib_model] Missing keys:", msg.missing_keys)
    print("[calib_model] Unexpected keys:", msg.unexpected_keys)

    calib_model.to(device)

    metrics = eval_lesion_calib(
        calib_model=calib_model,
        backbone_img_size=args.backbone_img_size,
        data_loader=eval_loader,
        device=device,
        tau=args.tau,
        dump_preds=True,
        dump_dir=root_eval_dir,
        tb_writer=tb_writer,
    )

    # [MOD] ===== 可视化 per-class 改善 & 先验矩阵 / delta / W_final =====
    try:
        base_m = metrics.get("base", {})
        final_m = metrics.get("final", {})

        per_ap_base = np.asarray(base_m.get("per_ap", []), dtype=np.float32)
        per_ap_final = np.asarray(final_m.get("per_ap", []), dtype=np.float32)
        per_prec_base = np.asarray(base_m.get("per_precision", []), dtype=np.float32)
        per_prec_final = np.asarray(final_m.get("per_precision", []), dtype=np.float32)
        per_rec_base = np.asarray(base_m.get("per_recall", []), dtype=np.float32)
        per_rec_final = np.asarray(final_m.get("per_recall", []), dtype=np.float32)

        if per_ap_base.size > 0 and per_ap_final.size == per_ap_base.size:
            delta_ap = per_ap_final - per_ap_base
            delta_prec = per_prec_final - per_prec_base if per_prec_base.size == per_ap_base.size else None
            delta_rec = per_rec_final - per_rec_base if per_rec_base.size == per_ap_base.size else None

            C = per_ap_base.shape[0]
            class_names = [f"C{i}" for i in range(C)]
            order = np.argsort(delta_ap)

            # ΔAP 条形图
            plt.figure(figsize=(max(8, len(order) * 0.25), 4))
            plt.bar(np.arange(len(order)), delta_ap[order])
            plt.xticks(np.arange(len(order)), [class_names[i] for i in order], rotation=90)
            plt.ylabel("ΔAP (Final - Base)")
            plt.title("各类别 ΔAP")
            plt.tight_layout()
            fig_path = os.path.join(root_eval_dir, "per_class_delta_AP.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[LesionCalibEval] saved {fig_path}")

            # Base vs Final AP 散点图
            plt.figure(figsize=(5, 5))
            plt.scatter(per_ap_base, per_ap_final, s=20)
            max_ap = float(max(per_ap_base.max(), per_ap_final.max())) if np.isfinite(per_ap_base).any() else 1.0
            plt.plot([0, max_ap], [0, max_ap], "k--", linewidth=1)
            plt.xlabel("Base AP")
            plt.ylabel("Final AP")
            plt.title("各类别 AP：Final vs Base")
            plt.tight_layout()
            fig_path = os.path.join(root_eval_dir, "per_class_AP_scatter.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[LesionCalibEval] saved {fig_path}")

            # ΔPrecision / ΔRecall 条形图
            if delta_prec is not None and delta_rec is not None:
                plt.figure(figsize=(max(8, len(order) * 0.25), 4))
                x = np.arange(len(order))
                plt.bar(x - 0.15, delta_prec[order], width=0.3, label="ΔPrecision")
                plt.bar(x + 0.15, delta_rec[order], width=0.3, label="ΔRecall")
                plt.xticks(x, [class_names[i] for i in order], rotation=90)
                plt.ylabel("Δ值 (Final - Base)")
                plt.title("各类别 ΔPrecision / ΔRecall")
                plt.legend()
                plt.tight_layout()
                fig_path = os.path.join(root_eval_dir, "per_class_delta_PR.png")
                plt.savefig(fig_path, dpi=200)
                plt.close()
                print(f"[LesionCalibEval] saved {fig_path}")

            # per-class 指标 JSON
            per_class_records = []
            for c in range(C):
                rec_c = {
                    "id": int(c),
                    "name": class_names[c],
                    "base_ap": float(per_ap_base[c]),
                    "final_ap": float(per_ap_final[c]),
                    "delta_ap": float(delta_ap[c]),
                    "base_precision": float(per_prec_base[c]) if per_prec_base.size == C else None,
                    "final_precision": float(per_prec_final[c]) if per_prec_final.size == C else None,
                    "delta_precision": float(delta_prec[c]) if delta_prec is not None else None,
                    "base_recall": float(per_rec_base[c]) if per_rec_base.size == C else None,
                    "final_recall": float(per_rec_final[c]) if per_rec_final.size == C else None,
                    "delta_recall": float(delta_rec[c]) if delta_rec is not None else None,
                }
                per_class_records.append(rec_c)
            per_class_path = os.path.join(root_eval_dir, "per_class_metrics_base_final_delta.json")
            with open(per_class_path, "w", encoding="utf-8") as f_pc:
                json.dump(per_class_records, f_pc, ensure_ascii=False, indent=2)
            print(f"[LesionCalibEval] wrote per-class metrics to {per_class_path}")

        # 先验矩阵 / delta / W_final 可视化
        mapper = getattr(calib_model, "mapper", None)
        if mapper is not None:
            with torch.no_grad():
                M_prior_np = prior_matrix.detach().cpu().numpy()
                # 假设 mapper.delta 存在（你训练时就是这样用的）
                delta_tensor = getattr(mapper, "delta", None)
                delta_np = delta_tensor.detach().cpu().numpy() if delta_tensor is not None else None
                if delta_np is not None:
                    W_np = np.maximum(M_prior_np + delta_np, 0.0)
                else:
                    W_np = M_prior_np

            print(f"[LesionCalibEval] M_prior shape={M_prior_np.shape}, W shape={W_np.shape}")
            print(f"[LesionCalibEval] M_prior stats: min={M_prior_np.min():.4f}, max={M_prior_np.max():.4f}, mean={M_prior_np.mean():.4f}")
            print(f"[LesionCalibEval] W stats:       min={W_np.min():.4f}, max={W_np.max():.4f}, mean={W_np.mean():.4f}")

            # M_prior 热力图
            plt.figure(figsize=(8, 6))
            im = plt.imshow(M_prior_np, aspect="auto", vmin=0.0, vmax=1.0)
            plt.colorbar(im)
            plt.title("M_prior（医生先验）")
            plt.xlabel("疾病 (列)")
            plt.ylabel("病灶 (行)")
            plt.tight_layout()
            fig_path = os.path.join(root_eval_dir, "M_prior.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[LesionCalibEval] saved {fig_path}")

            # W_final 热力图
            plt.figure(figsize=(8, 6))
            im = plt.imshow(W_np, aspect="auto")
            plt.colorbar(im)
            plt.title("W_final = ReLU(M_prior + delta)")
            plt.xlabel("疾病 (列)")
            plt.ylabel("病灶 (行)")
            plt.tight_layout()
            fig_path = os.path.join(root_eval_dir, "W_final.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[LesionCalibEval] saved {fig_path}")

            # delta 可视化
            if delta_np is not None:
                print(f"[LesionCalibEval] delta stats: min={delta_np.min():.4f}, max={delta_np.max():.4f}, mean={delta_np.mean():.4f}, abs_mean={np.abs(delta_np).mean():.4f}")

                plt.figure(figsize=(8, 6))
                vmax = float(np.max(np.abs(delta_np))) if np.isfinite(delta_np).any() else 1.0
                im = plt.imshow(delta_np, aspect="auto", vmin=-vmax, vmax=vmax)
                plt.colorbar(im)
                plt.title("delta（可学习残差）")
                plt.xlabel("疾病 (列)")
                plt.ylabel("病灶 (行)")
                plt.tight_layout()
                fig_path = os.path.join(root_eval_dir, "delta.png")
                plt.savefig(fig_path, dpi=200)
                plt.close()
                print(f"[LesionCalibEval] saved {fig_path}")

                prior_mask = (M_prior_np > 0.5).astype(np.float32)
                noprior_mask = 1.0 - prior_mask
                eps = 1e-8
                delta_prior = (np.abs(delta_np) * prior_mask).sum() / (prior_mask.sum() + eps)
                delta_noprior = (np.abs(delta_np) * noprior_mask).sum() / (noprior_mask.sum() + eps)
                print(f"[LesionCalibEval] |delta| mean on prior   (M_prior==1): {delta_prior:.6f}")
                print(f"[LesionCalibEval] |delta| mean on noprior (M_prior==0): {delta_noprior:.6f}")

                plt.figure(figsize=(6, 4))
                plt.hist(delta_np[prior_mask == 1].ravel(), bins=50, alpha=0.7, label="prior=1")
                plt.hist(delta_np[noprior_mask == 1].ravel(), bins=50, alpha=0.7, label="prior=0")
                plt.legend()
                plt.title("delta 在先验/非先验区域的分布")
                plt.tight_layout()
                fig_path = os.path.join(root_eval_dir, "delta_hist.png")
                plt.savefig(fig_path, dpi=200)
                plt.close()
                print(f"[LesionCalibEval] saved {fig_path}")

    except Exception as e:
        print(f"[LesionCalibEval] 可视化阶段发生异常，已跳过：{e}")

    print("Final metrics:", metrics)

    tb_writer.close()


if __name__ == "__main__":
    main()

"""
python /data1/Spatial-Mamba/classification/main_calib_eval.py \
  --backbone_cfg ... \
  --backbone_ckpt ... \
  --eval_json ... \
  --use_ema \
  --image_root ... \
  --disease_labelmap ... \
  --selected_lesion_ids "..." \
  --num_classes 27 \
  --lesion_img_size 2048 \
  --patch_size 256 \
  --patch_stride 256 \
  --prior_matrix_path ... \
  --calib_ckpt ... \
  --batch_size 4 \
  --num_workers 8 \
  --tau 0.5 \
  --output calib_eval_1205_run042 \
  --font_path /data1/fonts/SourceHanSansSC-Regular.ttf
"""