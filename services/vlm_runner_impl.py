"""
Reference implementation for wiring a real VLM runner via Swift's official python API.
Includes path injection to ensure ms-swift source code is loaded.
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


def _ensure_swift_imports():
    """
    Hard-code injection of the local ms-swift source directory.
    Assumes directory structure:
       SLO_diagnose_demo/
          services/
          ms-swift/  <-- Your source code here
    """
    # 1. 定位 ms-swift 目录
    # services/vlm_runner_impl.py -> services/ -> SLO_diagnose_demo/
    repo_root = Path(__file__).resolve().parents[1]
    swift_source_root = repo_root / "ms-swift"

    # 2. 如果存在，强制插入到 sys.path 最前面
    if swift_source_root.exists():
        swift_path_str = str(swift_source_root)
        # 避免重复添加
        if swift_path_str not in sys.path:
            print(f"[VLM Runner] Injecting ms-swift path: {swift_path_str}")
            sys.path.insert(0, swift_path_str)
    else:
        # 如果找不到本地目录，尝试打印调试信息
        print(f"[VLM Runner] Warning: Local ms-swift not found at {swift_source_root}")


def build_vlm_runner():
    """
    Build a callable that wraps Swift's official inference API.
    """
    
    # === 关键步骤：先注入路径，再 import ===
    _ensure_swift_imports()

    try:
        # 尝试标准导入
        from swift.llm import (
            get_model_tokenizer,
            get_template,
            inference,
            get_default_template_type
        )
    except ImportError:
        try:
            # 尝试深层导入（兼容旧源码结构）
            from swift.llm.model import get_model_tokenizer
            from swift.llm.template import get_template, get_default_template_type
            from swift.llm.infer import inference
        except ImportError as exc:
            print(f"[Error] Python path is: {sys.path}")
            raise RuntimeError(
                "Failed to import 'swift'. Please verify that the 'ms-swift' folder "
                "is in your project root or installed in the current python environment."
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
        model_dir = os.getenv("VLM_MODEL_NAME_OR_PATH", "qwen/Qwen2-VL-7B-Instruct")
        
        adapters_env = os.getenv("VLM_LORA_ADAPTERS", "")
        adapters = [p.strip() for p in adapters_env.split(",") if p.strip()] if adapters_env else []

        if variant == "spm":
            model_type = os.getenv("VLM_MODEL_TYPE", "my_qwen3_vl_spm")
            template_type = os.getenv("VLM_TEMPLATE", "my_qwen3_vl_spm")
            
            # 定位 my_register.py (支持带空格的文件名兼容)
            default_register = str(Path(__file__).resolve().parents[1] / "my_register.py")
            if not os.path.exists(default_register) and os.path.exists(default_register.replace("my_register.py", "my_register .py")):
                 default_register = default_register.replace("my_register.py", "my_register .py")

            custom_register = os.getenv("VLM_CUSTOM_REGISTER", default_register)
            _register_custom_module(custom_register)
        else:
            model_type = os.getenv("VLM_MODEL_TYPE", "qwen2-vl-7b-instruct") 
            template_type = os.getenv("VLM_TEMPLATE", get_default_template_type(model_type))

        dtype_str = os.getenv("VLM_DTYPE", "bfloat16")
        torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        print(f">>> Loading VLM: {model_dir}")
        print(f"    Type: {model_type}, Template: {template_type}")

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
        max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "4096"))
        temperature = float(os.getenv("VLM_TEMPERATURE", "0"))

        query = prompt
        if variant == "spm" and spm_feat_path:
            query = f"{prompt}\n{SPM_SPECIAL_TOKEN}\n[SPM_FEAT_PATH]{spm_feat_path}"

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
                "used_images": images,
            },
        )

    return runner