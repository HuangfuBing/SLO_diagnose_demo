"""
Reference implementation for wiring a real VLM runner.

The runner accepts (prompt, image_paths, spm_feat_path) and returns `VlmResult`.
All heavy deps are imported lazily to keep `app.py` light.

Environment variables
---------------------
    VLM_MODEL_NAME_OR_PATH: HuggingFace/Swift identifier or local checkpoint path.
        Default: "qwen3-vl-72b"
    VLM_DEVICE: Device string understood by your backend (default: "cuda:0").
    VLM_DTYPE: torch dtype string (default: "bfloat16").
    VLM_MAX_NEW_TOKENS: int (default: 512)
    VLM_TEMPERATURE: float (default: 0.2)
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from .vlm_api import VlmResult


def build_vlm_runner():
    """
    Build a callable that wraps Swift/Qwen-style generation.
    """

    _state = {"model": None}
    try:
        from swift.llm import SwiftModel  # type: ignore
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise RuntimeError(
            "swift.llm.SwiftModel is required for the real VLM runner. "
            "Please install your VLM stack and set VLM_MODEL_NAME_OR_PATH."
        ) from exc

    def _load_model():
        if _state["model"] is not None:
            return _state["model"]

        model_name = os.getenv("VLM_MODEL_NAME_OR_PATH", "qwen3-vl-72b")
        device = os.getenv("VLM_DEVICE", "cuda:0")
        dtype = os.getenv("VLM_DTYPE", "bfloat16")

        _state["model"] = SwiftModel.from_pretrained(model_name, device=device, dtype=dtype)
        return _state["model"]

    def runner(prompt: str, image_paths: Sequence[str], spm_feat_path: Optional[str] = None) -> VlmResult:
        model = _load_model()

        max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("VLM_TEMPERATURE", "0.2"))

        resp = model.generate(
            prompt=prompt,
            images=list(image_paths),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            extra_kwargs={"spm_feat_path": spm_feat_path} if spm_feat_path else None,
        )

        return VlmResult(
            report=resp.get("text", ""),
            raw_output=resp,
            debug={
                "source": "vlm_runner_impl",
                "spm_feat_path": spm_feat_path,
                "used_images": list(image_paths),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )

    return runner
