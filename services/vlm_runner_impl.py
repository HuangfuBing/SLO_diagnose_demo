"""
VLM Runner Implementation using Swift PtEngine.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from swift.llm import InferRequest, PtEngine, RequestConfig

from .vlm_api import VlmResult

SPM_SPECIAL_TOKEN = "<|SPM_FEAT|>"


def build_vlm_runner():
    """
    Build a callable that wraps Swift's PtEngine inference API.
    """

    _state = {
        "engine": None,
        "model_type": None,
        "template_type": None,
        "variant": None,
    }

    def _register_custom_module(register_path: str) -> None:
        path_obj = Path(register_path)
        if not path_obj.exists():
            alt_path = str(path_obj).replace("my_register.py", "my_register .py")
            if os.path.exists(alt_path):
                path_obj = Path(alt_path)
            else:
                print(f"[Warn] Custom register file not found: {register_path}")
                return

        module_name = "custom_register_module"
        spec = importlib.util.spec_from_file_location(module_name, str(path_obj))
        if not spec or not spec.loader:
            print(f"[Warn] Custom register file invalid: {register_path}")
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"[Info] Loaded custom register: {path_obj}")

    def _load_engine() -> PtEngine:
        if _state["engine"] is not None:
            return _state["engine"]

        model_name_or_path = os.getenv("VLM_MODEL_NAME_OR_PATH", "qwen3-vl-4b")
        model_type = os.getenv("VLM_MODEL_TYPE", "qwen3-vl-4b")
        template_type = os.getenv("VLM_TEMPLATE", "qwen3-vl")
        attn_impl = os.getenv("VLM_ATTN_IMPL", "flash_attention_2")
        variant = os.getenv("VLM_VARIANT", "sft").lower()

        if variant == "spm":
            repo_root = Path(__file__).resolve().parents[1]
            default_reg = repo_root / "my_register.py"
            custom_register = os.getenv("VLM_CUSTOM_REGISTER", str(default_reg))
            _register_custom_module(custom_register)
            os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
            os.environ.setdefault("MAX_PIXELS", "1048576")

        print(f">>> Loading Model: {model_name_or_path}")
        print(f"    Type: {model_type}, Template: {template_type}")

        engine_kwargs = {
            "model_type": model_type,
            "attn_impl": attn_impl,
        }
        if template_type:
            engine_kwargs["template_type"] = template_type

        engine = PtEngine(model_name_or_path, **engine_kwargs)
        _state["engine"] = engine
        _state["model_type"] = model_type
        _state["template_type"] = template_type
        _state["variant"] = variant
        return engine

    def _build_content(prompt: str, image_paths: Sequence[str]) -> str:
        image_tokens = "".join("<image>" for _ in image_paths)
        if image_tokens:
            return f"{image_tokens}{prompt}"
        return prompt

    def runner(
        prompt: str, image_paths: Sequence[str], spm_feat_path: Optional[str] = None
    ) -> VlmResult:
        engine = _load_engine()

        max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("VLM_TEMPERATURE", "0.2"))
        variant = _state["variant"] or os.getenv("VLM_VARIANT", "sft").lower()

        query = prompt
        if variant == "spm" and spm_feat_path:
            query = f"{SPM_SPECIAL_TOKEN}\n{prompt}[SPM_FEAT_PATH]{spm_feat_path}"

        content = _build_content(query, image_paths)
        infer_request = InferRequest(
            messages=[{"role": "user", "content": content}],
            images=list(image_paths),
        )
        request_config = RequestConfig(temperature=temperature, max_tokens=max_new_tokens)

        input_ids = None
        if hasattr(engine, "default_template"):
            input_ids = engine.default_template.encode(infer_request).get("input_ids")

        resp_list = engine.infer([infer_request], request_config)
        resp = resp_list[0].choices[0].message.content

        return VlmResult(
            report=resp or "",
            raw_output={"text": resp, "input_ids": input_ids},
            debug={
                "source": "vlm_runner_impl",
                "variant": variant,
                "model_type": _state["model_type"],
                "template_type": _state["template_type"],
            },
        )

    return runner
