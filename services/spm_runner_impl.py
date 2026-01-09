"""
Reference implementation for wiring a real SPM runner.

This module stays import-light so that `app.py` can safely import it without
loading heavy dependencies until a call is made. The runner follows the
`SpmClient` contract.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Any

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from .spm_api import SpmResult


def _ensure_spatial_mamba_imports() -> None:
    """
    Ensure Spatial-Mamba/classification is in sys.path to allow imports of
    'main_calib_eval_2ca' and 'models'.
    """
    # services/spm_runner_impl.py -> services/ -> root/
    repo_root = Path(__file__).resolve().parents[1]
    spm_root = repo_root / "Spatial-Mamba" / "classification"
    if spm_root.exists():
        spm_root_str = str(spm_root)
        if spm_root_str not in sys.path:
            sys.path.insert(0, spm_root_str)


def _load_spm_from_2ca() -> Dict[str, Any]:
    """Lazy-load backbone + calib model using env-driven paths."""
    _ensure_spatial_mamba_imports()

    # Import the eval script as a module.
    # Because we added classification/ to sys.path, this import works.
    try:
        import main_calib_eval_2ca as m_eval
    except ImportError as exc:
        raise RuntimeError(
            "main_calib_eval_2ca.py is required but not found. "
            "Please ensure Spatial-Mamba/classification is correctly placed."
        ) from exc

    # Retrieve classes directly from m_eval to avoid brittle imports like
    # 'from models.spatialmamba_2ca import ...' inside the services directory.
    # m_eval already imports these from its local 'models' package.
    LesionQueryHead = getattr(m_eval, "LesionQueryHead", None)
    LesionToDiseaseMapper = getattr(m_eval, "LesionToDiseaseMapper", None)
    LesionCalibModel2CA = getattr(m_eval, "LesionCalibModel2CA", None)

    if not all([LesionQueryHead, LesionToDiseaseMapper, LesionCalibModel2CA]):
        raise RuntimeError("Failed to retrieve required model classes from main_calib_eval_2ca.")

    # Load Env Vars
    cfg_path = os.getenv("SPM_BACKBONE_CFG_2CA")
    backbone_ckpt = os.getenv("SPM_BACKBONE_CKPT_2CA")
    calib_ckpt = os.getenv("SPM_CALIB_CKPT_2CA")
    prior_matrix_path = os.getenv("SPM_PRIOR_MATRIX_PATH_2CA")
    selected_lesion_ids_env = os.getenv("SPM_SELECTED_LESION_IDS_2CA")

    # Validate Env
    required_envs = {
        "SPM_BACKBONE_CFG_2CA": cfg_path,
        "SPM_BACKBONE_CKPT_2CA": backbone_ckpt,
        "SPM_CALIB_CKPT_2CA": calib_ckpt,
        "SPM_PRIOR_MATRIX_PATH_2CA": prior_matrix_path,
        "SPM_SELECTED_LESION_IDS_2CA": selected_lesion_ids_env,
    }
    for key, val in required_envs.items():
        if not val:
            raise RuntimeError(f"[SPM runner] Missing required env: {key}")

    # Configs
    num_classes = int(os.getenv("SPM_NUM_CLASSES", "27"))
    lesion_img_size = int(os.getenv("SPM_LESION_IMG_SIZE", "2048"))
    patch_size = int(os.getenv("SPM_PATCH_SIZE", "256"))
    patch_stride = int(os.getenv("SPM_PATCH_STRIDE", "256"))
    use_ema = os.getenv("SPM_USE_EMA", "1") == "1"
    device = torch.device(os.getenv("SPM_DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

    selected_lesion_ids = [int(x) for x in selected_lesion_ids_env.split(",") if x.strip()]
    num_lesions = len(selected_lesion_ids)

    # Build Pipeline using m_eval helpers
    cfg = m_eval.load_backbone_cfg(cfg_path)
    backbone = m_eval.build_backbone_from_cfg(cfg, num_classes=num_classes)
    backbone = m_eval.load_backbone_weights(backbone, backbone_ckpt, device, use_ema=use_ema)

    prior_matrix = m_eval.load_prior_matrix(prior_matrix_path, num_lesions, num_classes)

    # Reuse build_timm_transform_hr from m_eval
    class _Args:
        pass
    args = _Args()
    args.lesion_img_size = lesion_img_size
    args.patch_size = patch_size
    args.patch_stride = patch_stride
    args.aug_cfg = {}
    
    transform = m_eval.build_timm_transform_hr(args, is_train=False)

    # Determine num_patches
    dummy = torch.zeros(3, lesion_img_size, lesion_img_size)
    dummy_patches = m_eval.extract_patches(dummy, patch_size, patch_stride)
    num_patches = int(dummy_patches.shape[0])

    # Instantiate Models
    lesion_head = LesionQueryHead(
        backbone=backbone,
        num_lesions=num_lesions,
        num_patches=num_patches,
        lesion_ids=selected_lesion_ids,
        embed_dim=256,
        num_heads=4,
        dropout=0.0,
    )

    mapper = LesionToDiseaseMapper(
        prior_matrix=prior_matrix,
        learn_delta=True,
    )

    calib_model = LesionCalibModel2CA(
        backbone=backbone,
        lesion_head=lesion_head,
        mapper=mapper,
        num_classes=num_classes,
        alpha_init=0.1,
    ).to(device)

    # Load Calib Checkpoint
    ckpt = torch.load(calib_ckpt, map_location="cpu")
    state = ckpt.get("calib_model") if isinstance(ckpt, dict) else ckpt
    calib_model.load_state_dict(state, strict=False)
    
    calib_model.eval()
    for p in calib_model.parameters():
        p.requires_grad_(False)

    thresholds = {"default": float(os.getenv("SPM_THRESHOLD_DEFAULT", "0.5"))}

    feat_transform = transforms.Compose(
        [
            transforms.Resize((int(cfg.get("DATA", {}).get("IMG_SIZE", 576)),) * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {
        "model": calib_model,
        "backbone": backbone,
        "device": device,
        "transform": transform,
        "thresholds": thresholds,
        "backbone_img_size": int(cfg.get("DATA", {}).get("IMG_SIZE", 576)),
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "selected_lesion_ids": selected_lesion_ids,
        "eval_mod": m_eval,  # Hold reference to module for helper functions
        "feat_transform": feat_transform,
    }


def build_spm_runner() -> Callable[[Sequence[str]], SpmResult]:
    """
    Build a callable SPM runner using the env-driven 2CA pipeline.
    The runner caches the model on first invocation.
    """
    _state: Dict[str, object] = {
        "model": None,
        "backbone": None,
        "device": None,
        "transform": None,
        "backbone_img_size": None,
        "patch_size": None,
        "patch_stride": None,
        "selected_lesion_ids": None,
        "thresholds": None,
        "eval_mod": None,
        "feat_dir": Path(os.getenv("SPM_FEAT_DIR", ".cache_spm_feats")),
        "feat_transform": None,
    }
    _state["feat_dir"].mkdir(parents=True, exist_ok=True)

    def _ensure_loaded():
        if _state["model"] is None:
            bundle = _load_spm_from_2ca()
            _state.update(bundle)

    def runner(paths: Sequence[str]) -> SpmResult:
        _ensure_loaded()
        
        # Access components from state
        device = _state["device"]
        model = _state["model"]
        backbone = _state["backbone"]
        transform = _state["transform"]
        m_eval = _state["eval_mod"]
        patch_size = _state["patch_size"]
        patch_stride = _state["patch_stride"]
        backbone_img_size = _state["backbone_img_size"]
        thresholds = _state["thresholds"]
        selected_lesion_ids = _state["selected_lesion_ids"]
        feat_transform = _state["feat_transform"]

        lesion_probs_out: List[Dict[str, float]] = []
        disease_probs_out: List[Dict[str, float]] = []
        feat_path: str | None = None
        feat_dim: int | None = None

        def _make_feat_path(img_path: str) -> Path:
            img_hash = hashlib.sha1(img_path.encode("utf-8")).hexdigest()[:10]
            name = f"{Path(img_path).stem}_{img_hash}.npy"
            return _state["feat_dir"] / name

        def _extract_feature(img_path: str) -> Path:
            if backbone is None:
                raise RuntimeError("[SPM runner] Backbone not loaded for feature extraction.")
            feat_file = _make_feat_path(img_path)
            if feat_file.exists():
                return feat_file
            img = Image.open(img_path).convert("RGB")
            img_tensor = feat_transform(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                feats = backbone.forward_features(img_tensor).squeeze(0).detach().cpu().float().numpy()
            feats = np.asarray(feats, dtype=np.float32).reshape(-1)
            np.save(feat_file, feats)
            return feat_file

        for idx, img_path in enumerate(paths):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[SPM Runner] Error opening image {img_path}: {e}")
                continue

            if feat_path is None:
                feat_file = _extract_feature(img_path)
                feat_path = str(feat_file)
                if feat_dim is None:
                    try:
                        feat_dim = int(np.load(feat_file).shape[0])
                    except Exception:
                        feat_dim = None

            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Use m_eval.extract_patches
            patches = m_eval.extract_patches(
                img_tensor.squeeze(0),
                patch_size,
                patch_stride,
            ).unsqueeze(0).to(device)

            with torch.inference_mode():
                # Forward pass: base_logits, corr_logits, final_logits, lesion_probs
                _, _, final_logits, lesion_probs = model(
                    img_tensor, patches, backbone_img_size
                )
                probs = torch.sigmoid(final_logits).squeeze(0).detach().cpu().numpy().astype(float)

            # Collect Disease Probabilities
            for i, p in enumerate(probs):
                disease_probs_out.append(
                    {
                        "id": f"disease_{i}",
                        "prob": float(p),
                        "threshold": thresholds.get("default", 0.5),
                    }
                )

            # Collect Lesion Probabilities
            if lesion_probs is not None:
                lesion_np = lesion_probs.squeeze(0).detach().cpu().numpy().astype(float)
                for lid, lp in zip(selected_lesion_ids, lesion_np):
                    lesion_probs_out.append({"id": f"lesion_{lid}", "prob": float(lp)})

        return SpmResult(
            lesion_probs=lesion_probs_out,
            disease_probs=disease_probs_out,
            thresholds=thresholds,
            spm_feat_path=feat_path,
            debug={
                "source": "spm_runner_impl",
                "paths": list(paths),
                "device": str(device),
                "feat_dim": feat_dim,
            },
        )

    return runner
