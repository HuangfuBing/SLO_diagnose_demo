"""
Reference implementation for wiring a real SPM runner.

This module stays import-light so that `app.py` can safely import it without
loading heavy dependencies until a call is made. The runner follows the
`SpmClient` contract: it accepts a list of image paths and returns `SpmResult`,
optionally writing out an `.npy` feature file for downstream VLM use.

Environment variables
---------------------
The defaults mirror the 2CA demo configs; override them to point at your own
weights/configs without editing code.

Required:
    SPM_BACKBONE_CFG_2CA
    SPM_BACKBONE_CKPT_2CA
    SPM_CALIB_CKPT_2CA
    SPM_PRIOR_MATRIX_PATH_2CA
    SPM_SELECTED_LESION_IDS_2CA   (comma-separated ints, e.g. "1,2,5")

Optional:
    SPM_NUM_CLASSES (default 27)
    SPM_LESION_IMG_SIZE (default 2048)
    SPM_PATCH_SIZE (default 256)
    SPM_PATCH_STRIDE (default 256)
    SPM_DEVICE (default cuda)
    SPM_USE_EMA (default 1)
    SPM_THRESHOLD_DEFAULT (default 0.5)
    SPM_FEAT_DIR (default ".cache_spm_feats")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image

from .spm_api import SpmResult


def _ensure_spatial_mamba_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spm_root = repo_root / "Spatial-Mamba" / "classification"
    if spm_root.exists():
        spm_root_str = str(spm_root)
        if spm_root_str not in sys.path:
            sys.path.insert(0, spm_root_str)


def _load_spm_from_2ca():
    """Lazy-load backbone + calib model using env-driven paths."""
    _ensure_spatial_mamba_imports()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to build the real SPM runner") from exc

    try:
        import main_calib_eval_2ca as m_eval
    except ImportError as exc:
        raise RuntimeError(
            "main_calib_eval_2ca.py is required but not found. "
            "Place Spatial-Mamba/classification in PYTHONPATH to enable the SPM runner."
        ) from exc

    try:
        from models.spatialmamba_2ca import (
            LesionCalibModel2CA,
            LesionQueryHead,
            LesionToDiseaseMapper,
        )
    except ImportError as exc:
        raise RuntimeError(
            "spatialmamba_2ca.py is required but not found. "
            "Place Spatial-Mamba/classification in PYTHONPATH to enable the SPM runner."
        ) from exc

    cfg_path = os.getenv("SPM_BACKBONE_CFG_2CA")
    backbone_ckpt = os.getenv("SPM_BACKBONE_CKPT_2CA")
    calib_ckpt = os.getenv("SPM_CALIB_CKPT_2CA")
    prior_matrix_path = os.getenv("SPM_PRIOR_MATRIX_PATH_2CA")
    selected_lesion_ids_env = os.getenv("SPM_SELECTED_LESION_IDS_2CA")

    for key, val in [
        ("SPM_BACKBONE_CFG_2CA", cfg_path),
        ("SPM_BACKBONE_CKPT_2CA", backbone_ckpt),
        ("SPM_CALIB_CKPT_2CA", calib_ckpt),
        ("SPM_PRIOR_MATRIX_PATH_2CA", prior_matrix_path),
        ("SPM_SELECTED_LESION_IDS_2CA", selected_lesion_ids_env),
    ]:
        if not val:
            raise RuntimeError(f"[SPM runner] Missing required env: {key}")

    num_classes = int(os.getenv("SPM_NUM_CLASSES", "27"))
    lesion_img_size = int(os.getenv("SPM_LESION_IMG_SIZE", "2048"))
    patch_size = int(os.getenv("SPM_PATCH_SIZE", "256"))
    patch_stride = int(os.getenv("SPM_PATCH_STRIDE", "256"))
    use_ema = os.getenv("SPM_USE_EMA", "1") == "1"
    device = torch.device(os.getenv("SPM_DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

    selected_lesion_ids = [int(x) for x in selected_lesion_ids_env.split(",") if x.strip()]
    num_lesions = len(selected_lesion_ids)

    cfg = m_eval.load_backbone_cfg(cfg_path)
    backbone = m_eval.build_backbone_from_cfg(cfg, num_classes=num_classes)
    backbone = m_eval.load_backbone_weights(backbone, backbone_ckpt, device, use_ema=use_ema)

    prior_matrix = m_eval.load_prior_matrix(prior_matrix_path, num_lesions, num_classes)

    # Dummy args object to reuse build_timm_transform_hr
    class _Args:
        pass

    args = _Args()
    args.lesion_img_size = lesion_img_size
    args.patch_size = patch_size
    args.patch_stride = patch_stride
    args.aug_cfg = {}

    transform = m_eval.build_timm_transform_hr(args, is_train=False)

    # Determine num_patches analytically using a dummy tensor
    dummy = torch.zeros(3, lesion_img_size, lesion_img_size)
    dummy_patches = m_eval.extract_patches(dummy, patch_size, patch_stride)
    num_patches = int(dummy_patches.shape[0])

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

    ckpt = torch.load(calib_ckpt, map_location="cpu")
    state = ckpt.get("calib_model") if isinstance(ckpt, dict) else ckpt
    calib_model.load_state_dict(state, strict=False)
    calib_model.eval()
    for p in calib_model.parameters():
        p.requires_grad_(False)

    thresholds = {"default": float(os.getenv("SPM_THRESHOLD_DEFAULT", "0.5"))}

    return {
        "model": calib_model,
        "device": device,
        "transform": transform,
        "thresholds": thresholds,
        "backbone_img_size": int(cfg.get("DATA", {}).get("IMG_SIZE", 576)),
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "selected_lesion_ids": selected_lesion_ids,
        "eval_mod": m_eval,
    }


def build_spm_runner() -> Callable[[Sequence[str]], SpmResult]:
    """
    Build a callable SPM runner using the env-driven 2CA pipeline.
    The runner caches the model on first invocation.
    """

    _state: Dict[str, object] = {
        "model": None,
        "device": None,
        "transform": None,
        "backbone_img_size": None,
        "patch_size": None,
        "patch_stride": None,
        "selected_lesion_ids": None,
        "thresholds": None,
        "eval_mod": None,
        "feat_dir": Path(os.getenv("SPM_FEAT_DIR", ".cache_spm_feats")),
    }
    _state["feat_dir"].mkdir(parents=True, exist_ok=True)

    def _ensure_loaded():
        if _state["model"] is None:
            bundle = _load_spm_from_2ca()
            _state.update(bundle)

    def _save_feat(idx: int, tensor) -> str:
        import numpy as np

        feat_path = _state["feat_dir"] / f"spm_feat_{idx}.npy"
        np.save(feat_path, tensor)
        return str(feat_path)

    def runner(paths: Sequence[str]) -> SpmResult:
        import torch

        _ensure_loaded()

        lesion_probs_out: List[Dict[str, float]] = []
        disease_probs_out: List[Dict[str, float]] = []
        feat_path: str | None = None

        for idx, img_path in enumerate(paths):
            img = Image.open(img_path).convert("RGB")
            img_tensor = _state["transform"](img).unsqueeze(0).to(_state["device"])
            patches = _state["eval_mod"].extract_patches(
                img_tensor.squeeze(0),
                _state["patch_size"],
                _state["patch_stride"],
            ).unsqueeze(0).to(_state["device"])

            with torch.inference_mode():
                _, _, final_logits, lesion_probs = _state["model"](
                    img_tensor, patches, _state["backbone_img_size"]
                )
                probs = torch.sigmoid(final_logits).squeeze(0).detach().cpu().numpy().astype(float)

            for i, p in enumerate(probs):
                disease_probs_out.append(
                    {
                        "id": f"disease_{i}",
                        "prob": float(p),
                        "threshold": _state["thresholds"].get("default", 0.5),
                    }
                )

            if lesion_probs is not None:
                lesion_np = lesion_probs.squeeze(0).detach().cpu().numpy().astype(float)
                for lid, lp in zip(_state["selected_lesion_ids"], lesion_np):
                    lesion_probs_out.append({"id": f"lesion_{lid}", "prob": float(lp)})

            if feat is not None:
                feat_path = _save_feat(idx, feat.detach().cpu().numpy())

        return SpmResult(
            lesion_probs=lesion_probs_out,
            disease_probs=disease_probs_out,
            thresholds=_state["thresholds"],
            spm_feat_path=feat_path,
            debug={
                "source": "spm_runner_impl",
                "paths": list(paths),
                "device": str(_state["device"]),
            },
        )

    return runner
