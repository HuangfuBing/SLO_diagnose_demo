"""
VLM Runner Implementation based on sample_infer.py logic.
Assumes 'ms-swift' is correctly installed and importable in the environment.
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

# =============================================================================
# 1. 导入 ms-swift 核心组件
#    基于 sample_infer.py 证明了库可用，这里优先尝试标准导入
# =============================================================================
try:
    # 尝试顶层标准导入
    from swift.llm import get_model_tokenizer, get_template, inference
except ImportError:
    # 如果顶层没有暴露 (取决于具体版本)，则使用子模块导入 (这也是 sample_infer.py 隐含的路径逻辑)
    from swift.llm.model import get_model_tokenizer
    from swift.llm.template import get_template
    from swift.llm.infer import inference


def build_vlm_runner():
    """
    Build a callable that wraps Swift's official inference API.
    """

    # 状态缓存 (单例模式)
    _state = {
        "model": None,
        "tokenizer": None,
        "template": None
    }

    def _setup_env_from_sample():
        """
        严格复刻 sample_infer.py 中的环境变量设置
        """
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
        os.environ.setdefault("MAX_PIXELS", "1048576")
        os.environ.setdefault("SPM_BREAK", "0")

    def _register_custom_module(register_path: str):
        """
        动态加载自定义注册文件 (对应 CLI 的 --custom_register_path)
        """
        path_obj = Path(register_path)
        if not path_obj.exists():
            # 尝试处理文件名带空格的情况 'my_register .py'，这是之前文件结构中出现的特殊情况
            alt_path = str(path_obj).replace("my_register.py", "my_register .py")
            if os.path.exists(alt_path):
                path_obj = Path(alt_path)
            else:
                print(f"[Warn] Custom register file not found: {register_path}")
                return

        try:
            module_name = "custom_register_module"
            spec = importlib.util.spec_from_file_location(module_name, str(path_obj))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                print(f"[Info] Loaded custom register: {path_obj}")
        except Exception as e:
            print(f"[Error] Failed to register module {register_path}: {e}")

    def _load_model():
        if _state["model"] is not None:
            return _state["model"], _state["tokenizer"], _state["template"]

        _setup_env_from_sample()

        # --- 1. 配置参数 (默认值来自 sample_infer.py) ---
        default_model_dir = "/data1/qwen/model/Qwen3-VL-4B-Instruct"
        default_adapter = "/data1/output_qwen3_vl_spm_1224_4b/v0-20251224-023833/checkpoint-1500"
        
        # 允许环境变量覆盖，否则使用默认值
        model_dir = os.getenv("VLM_MODEL_NAME_OR_PATH", default_model_dir)
        adapters_env = os.getenv("VLM_LORA_ADAPTERS", default_adapter)
        adapters = [p.strip() for p in adapters_env.split(",") if p.strip()] if adapters_env else []

        variant = os.getenv("VLM_VARIANT", "sft").lower() # spm or sft

        # --- 2. 确定模型类型与 Template ---
        if variant == "spm":
            # 复刻 sample_infer.py 中的参数
            model_type = "my_qwen3_vl_spm"
            template_type = "my_qwen3_vl_spm"
            
            # 定位注册文件：优先用 ENV，否则尝试在项目上级目录寻找
            repo_root = Path(__file__).resolve().parents[1]
            default_reg = repo_root / "my_register .py" # 注意之前的空格文件名
            custom_register = os.getenv("VLM_CUSTOM_REGISTER", str(default_reg))
            
            _register_custom_module(custom_register)
        else:
            # SFT 模式或默认模式
            model_type = os.getenv("VLM_MODEL_TYPE", "qwen2-vl-7b-instruct")
            template_type = os.getenv("VLM_TEMPLATE", "qwen2-vl")

        print(f">>> Loading Model: {model_dir}")
        print(f"    Type: {model_type}, Template: {template_type}")
        if adapters:
            print(f"    Adapters: {adapters}")

        # --- 3. 核心加载 (get_model_tokenizer) ---
        model, tokenizer = get_model_tokenizer(
            model_type,
            torch_dtype=torch.bfloat16,
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
        
        # --- 推理参数 (参考 sample_infer.py) ---
        max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "4096"))
        temperature = float(os.getenv("VLM_TEMPERATURE", "0"))  # sample_infer 使用 0
        
        variant = os.getenv("VLM_VARIANT", "sft").lower()

        # 构造 Query
        query = prompt
        if variant == "spm" and spm_feat_path:
            query = f"{prompt}\n{SPM_SPECIAL_TOKEN}\n[SPM_FEAT_PATH]{spm_feat_path}"

        images = list(image_paths)

        # --- 执行推理 (inference) ---
        # inference 返回 (response, history)
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
                "source": "vlm_runner_impl",
                "variant": variant,
                "model_type": "my_qwen3_vl_spm" if variant == "spm" else "default"
            }
        )

    return runner