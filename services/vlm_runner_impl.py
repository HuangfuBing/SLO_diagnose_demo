"""
Reference implementation for wiring a real VLM runner via Swift's official infer API.

The runner accepts (prompt, image_paths, spm_feat_path) and returns `VlmResult`.
All heavy deps are imported lazily to keep `app.py` light.

Environment variables
---------------------
    VLM_VARIANT: "sft" | "spm" (default: "sft")
    VLM_MODEL_NAME_OR_PATH: HuggingFace/Swift identifier or local checkpoint path.
        Default: "qwen3-vl-72b"
    VLM_MODEL_TYPE: Swift model_type (default: "qwen3_vl" or "my_qwen3_vl_spm")
    VLM_TEMPLATE: Swift template name (default: "qwen3_vl" or "my_qwen3_vl_spm")
    VLM_CUSTOM_REGISTER: path to custom register file (default: "my_register .py")
    VLM_LORA_ADAPTERS: comma-separated LoRA adapter directories
    VLM_DTYPE: torch dtype string (default: "bfloat16")
    VLM_MAX_NEW_TOKENS: int (default: 512)
    VLM_TEMPERATURE: float (default: 0.2)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from .vlm_api import VlmResult

SPM_SPECIAL_TOKEN = "<|SPM_FEAT|>"


def build_vlm_runner():
    """
    Build a callable that wraps Swift's official infer API.
    """

    _state = {"infer": None}
    try:
        from swift.llm.infer.infer import SwiftInfer
        from swift.llm.infer.protocol import RequestConfig
        from swift.llm.template.template_inputs import InferRequest
        from swift.llm.argument.infer_args import InferArguments
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise RuntimeError(
            "Swift infer APIs are required. Please install swift and its LLM extras."
        ) from exc

    def _parse_adapters() -> Optional[list[str]]:
        adapters_env = os.getenv("VLM_LORA_ADAPTERS", "")
        adapters = [p.strip() for p in adapters_env.split(",") if p.strip()]
        return adapters or None

    def _build_infer_args():
        variant = os.getenv("VLM_VARIANT", "sft").lower()
        model_name = os.getenv("VLM_MODEL_NAME_OR_PATH", "qwen3-vl-72b")
        dtype = os.getenv("VLM_DTYPE", "bfloat16")
        device_map = os.getenv("VLM_DEVICE", "")

        if variant == "spm":
            model_type = os.getenv("VLM_MODEL_TYPE", "my_qwen3_vl_spm")
            template = os.getenv("VLM_TEMPLATE", "my_qwen3_vl_spm")
            default_register = str(
                Path(__file__).resolve().parents[1] / "my_register .py"
            )
            custom_register = os.getenv("VLM_CUSTOM_REGISTER", default_register)
            custom_register_path = [custom_register] if custom_register else None
        else:
            model_type = os.getenv("VLM_MODEL_TYPE", "qwen3_vl")
            template = os.getenv("VLM_TEMPLATE", "qwen3_vl")
            custom_register_path = None

        return InferArguments(
            model=model_name,
            model_type=model_type,
            template=template,
            custom_register_path=custom_register_path,
            adapters=_parse_adapters(),
            infer_backend="pt",
            torch_dtype=dtype,
            max_batch_size=1,
            device_map=device_map or None,
        )

    def _load_infer():
        if _state["infer"] is not None:
            return _state["infer"]

        infer = SwiftInfer(_build_infer_args())
        _state["infer"] = infer
        return infer

    def runner(prompt: str, image_paths: Sequence[str], spm_feat_path: Optional[str] = None) -> VlmResult:
        infer = _load_infer()

        max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("VLM_TEMPERATURE", "0.2"))

        variant = os.getenv("VLM_VARIANT", "sft").lower()
        if variant == "spm" and spm_feat_path:
            prompt = f"{prompt}\n{SPM_SPECIAL_TOKEN}\n[SPM_FEAT_PATH]{spm_feat_path}"

        request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=False,
        )

        infer_request = InferRequest(
            messages=[{"role": "user", "content": prompt}],
            images=list(image_paths),
        )

        report = infer.infer_single(infer_request, request_config)

        return VlmResult(
            report=report or "",
            raw_output={"text": report},
            debug={
                "source": "vlm_runner_impl",
                "variant": variant,
                "spm_feat_path": spm_feat_path,
                "used_images": list(image_paths),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )

    return runner
