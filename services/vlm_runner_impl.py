"""
Reference implementation for wiring a real VLM runner via Swift's official python API.

The runner accepts (prompt, image_paths, spm_feat_path) and returns `VlmResult`.
All heavy deps are imported lazily to keep `app.py` light.

Based on sample_infer.py logic, aligned with ms-swift standard usage.
"""

from __future__ import annotations

import os
import sys
import torch
import importlib.util
from pathlib import Path
from typing import Optional, Sequence

from .vlm_api import VlmResult

SPM_SPECIAL_TOKEN = "<|SPM_FEAT|>"


def build_vlm_runner():
    """
    Build a callable that wraps Swift's official inference API (get_model_tokenizer + inference).
    """
    try:
        from swift.llm import (
            get_model_tokenizer,
            get_template,
            inference,
            get_default_template_type
        )
    except ImportError as exc:
        raise RuntimeError(
            "Swift LLM APIs are required. Please install ms-swift."
        ) from exc

    _state = {
        "model": None,
        "tokenizer": None,
        "template": None
    }

    def _setup_env():
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
        os.environ.setdefault("MAX_PIXELS", "1048576")
        
        os.environ.setdefault("SPM_BREAK", "0")

    def _register_custom_module(register_path: str):
        """
        动态加载自定义注册文件 (my_register.py)，使自定义 model_type 生效
        """
        path_obj = Path(register_path)
        if not path_obj.exists():
            print(f"[Warn] Custom register file not found: {register_path}")
            return

        try:
            module_name = "custom_register_module"
            spec = importlib.util.spec_from_file_location(module_name, str(path_obj))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                print(f"[Info] Successfully registered custom module from: {register_path}")
        except Exception as e:
            print(f"[Error] Failed to load custom register: {e}")

    def _load_model():
        if _state["model"] is not None:
            return _state["model"], _state["tokenizer"], _state["template"]

        _setup_env()
        variant = os.getenv("VLM_VARIANT", "sft").lower()
        model_dir = os.getenv("VLM_MODEL_NAME_OR_PATH", "/data1/qwen/model/Qwen3-VL-4B-Instruct")
        adapters_env = os.getenv("VLM_LORA_ADAPTERS", "")
        adapters = [p.strip() for p in adapters_env.split(",") if p.strip()] if adapters_env else []

        if variant == "spm":
            model_type = os.getenv("VLM_MODEL_TYPE", "my_qwen3_vl_spm")
            template_type = os.getenv("VLM_TEMPLATE", "my_qwen3_vl_spm")
            
            default_register = str(Path(__file__).resolve().parents[1] / "my_register.py")

            custom_register = os.getenv("VLM_CUSTOM_REGISTER", default_register)
            _register_custom_module(custom_register)
        else:
            model_type = os.getenv("VLM_MODEL_TYPE", "qwen2-vl-7b-instruct") 
            template_type = os.getenv("VLM_TEMPLATE", get_default_template_type(model_type))

        dtype_str = os.getenv("VLM_DTYPE", "bfloat16")
        torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        print(f">>> Loading VLM: {model_dir}")
        print(f"    Type: {model_type}, Template: {template_type}, Variant: {variant}")
        if adapters:
            print(f"    Adapters: {adapters}")

        model, tokenizer = get_model_tokenizer(
            model_type,
            torch_dtype=torch_dtype,
            model_kwargs={'device_map': 'auto'},
            model_id_or_path=model_dir,
            adapter_dir=adapters if adapters else None
        )

        template = get_template(template_type, tokenizer)
        
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["template"] = template
        return model, tokenizer, template

    def runner(prompt: str, image_paths: Sequence[str], spm_feat_path: Optional[str] = None) -> VlmResult:
        model, tokenizer, template = _load_model()
        
        variant = os.getenv("VLM_VARIANT", "sft").lower()
        max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "4096")) # sample_infer uses 4096
        temperature = float(os.getenv("VLM_TEMPERATURE", "0"))         # sample_infer uses 0

        # FIXME: modify based on specific prompt
        query = prompt
        if variant == "spm" and spm_feat_path:
            query = f"<image>\n{SPM_SPECIAL_TOKEN}\n{prompt}\n[SPM_FEAT_PATH]{spm_feat_path}"

        images = list(image_paths)

        response, _ = inference(
            model,
            template,
            query,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=False
        )

        return VlmResult(
            report=response or "",
            raw_output={"text": response},
            debug={
                "source": "vlm_runner_impl_std",
                "variant": variant,
                "spm_feat_path": spm_feat_path,
                "used_images": images,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )

    return runner