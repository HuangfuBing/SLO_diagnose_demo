"""
Sample runner wiring extracted from app.py.

This module keeps heavy dependencies and model-specific logic out of the Gradio
UI file while exposing a simple factory to create SPM/VLM clients when
USE_SAMPLE_RUNNERS=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image

from services import SpmResult, VlmResult, make_default_spm_client, make_default_vlm_client


def _ensure_spatial_mamba_imports() -> None:
    repo_root = Path(__file__).resolve().parent
    spm_root = repo_root / "Spatial-Mamba" / "classification"
    if spm_root.exists():
        spm_root_str = str(spm_root)
        if spm_root_str not in sys.path:
            sys.path.insert(0, spm_root_str)


def _load_spm_from_2ca():
    """
    Build the SpatialMamba + LesionCalib inference stack based on
    main_calib_eval_2ca.py and spatialmamba_2ca.py. All paths are driven by
    env vars so you can point to your real configs/weights without editing
    this file.
    Required envs (no defaults for checkpoints):
        SPM_BACKBONE_CFG_2CA
        SPM_BACKBONE_CKPT_2CA
        SPM_CALIB_CKPT_2CA
        SPM_PRIOR_MATRIX_PATH_2CA
        SPM_SELECTED_LESION_IDS_2CA   (comma-separated ints, e.g. "1,2,5")
    Optional envs:
        SPM_NUM_CLASSES (default 27)
        SPM_LESION_IMG_SIZE (default 2048)
        SPM_PATCH_SIZE (default 256)
        SPM_PATCH_STRIDE (default 256)
        SPM_DEVICE (default cuda)
        SPM_USE_EMA (default 1)
        SPM_THRESHOLD_DEFAULT (default 0.5)
    """
    _ensure_spatial_mamba_imports()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for USE_SAMPLE_RUNNERS") from exc

    try:
        import main_calib_eval_2ca as m_eval
    except ImportError as exc:
        raise RuntimeError(
            "main_calib_eval_2ca.py is required but not found. "
            "Place Spatial-Mamba/classification in PYTHONPATH when USE_SAMPLE_RUNNERS=1."
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
            "Place Spatial-Mamba/classification in PYTHONPATH when USE_SAMPLE_RUNNERS=1."
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
            raise RuntimeError(f"[USE_SAMPLE_RUNNERS] Missing required env: {key}")

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

    threshold_default = float(os.getenv("SPM_THRESHOLD_DEFAULT", "0.5"))
    thresholds = {"default": threshold_default}

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


def _make_sample_spm_runner():
    _spm_state = {
        "model": None,
        "device": None,
        "transform": None,
        "backbone_img_size": None,
        "patch_size": None,
        "patch_stride": None,
        "selected_lesion_ids": None,
        "thresholds": None,
        "eval_mod": None,
    }

    def runner(paths):
        import torch

        if _spm_state["model"] is None:
            spm_bundle = _load_spm_from_2ca()
            _spm_state.update(spm_bundle)

        lesion_probs_out = []
        disease_probs_out = []

        for img_path in paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = _spm_state["transform"](img).unsqueeze(0).to(_spm_state["device"])
            patches = _spm_state["eval_mod"].extract_patches(
                img_tensor.squeeze(0),
                _spm_state["patch_size"],
                _spm_state["patch_stride"],
            ).unsqueeze(0).to(_spm_state["device"])

            with torch.inference_mode():
                _, _, final_logits, lesion_probs = _spm_state["model"](
                    img_tensor, patches, _spm_state["backbone_img_size"]
                )
                probs = torch.sigmoid(final_logits).squeeze(0).detach().cpu().numpy().astype(float)

            for idx, p in enumerate(probs):
                disease_probs_out.append(
                    {
                        "id": f"disease_{idx}",
                        "prob": float(p),
                        "threshold": _spm_state["thresholds"].get("default", 0.5),
                    }
                )

            if lesion_probs is not None:
                lesion_np = lesion_probs.squeeze(0).detach().cpu().numpy().astype(float)
                for lid, lp in zip(_spm_state["selected_lesion_ids"], lesion_np):
                    lesion_probs_out.append({"id": f"lesion_{lid}", "prob": float(lp)})

        return SpmResult(
            lesion_probs=lesion_probs_out,
            disease_probs=disease_probs_out,
            thresholds=_spm_state["thresholds"],
            spm_feat_path=None,
            debug={
                "source": "my_spm_runner_2ca",
                "paths": list(paths),
                "device": str(_spm_state["device"]),
            },
        )

    return runner


def _make_sample_vlm_runner():
    _vlm_state = {"model": None}

    def get_model():
        from swift.llm import SwiftModel  # adjust to your stack

        return SwiftModel.from_pretrained(
            "qwen3-vl-72b",  # replace with your model or local path
            device="cuda:0",
            dtype="bfloat16",
        )

    def runner(prompt, image_paths, spm_feat_path=None):
        if _vlm_state["model"] is None:
            _vlm_state["model"] = get_model()

        resp = _vlm_state["model"].generate(
            prompt=prompt,
            images=list(image_paths),
            max_new_tokens=512,
            temperature=0.2,
            # extra_kwargs={"spm_feat_path": spm_feat_path},  # if your model supports it
        )

        return VlmResult(
            report=resp.get("text", ""),
            raw_output=resp,
            debug={
                "source": "my_vlm_runner",
                "spm_feat_path": spm_feat_path,
                "used_images": list(image_paths),
            },
        )

    return runner


def build_sample_clients() -> Tuple:
    """
    Return (spm_client, vlm_client) using the sample runners.
    """
    spm_runner = _make_sample_spm_runner()
    vlm_runner = _make_sample_vlm_runner()
    return make_default_spm_client(runner=spm_runner), make_default_vlm_client(runner=vlm_runner)
