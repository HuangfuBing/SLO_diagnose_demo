import os
from swift.llm import sft_main, TrainArguments


def env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if (v is not None and v != "") else default


def env_bool(key: str, default: str = "0") -> bool:
    v = env(key, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


if __name__ == "__main__":
    os.environ["NCCL_IB_DISABLE"] = env("NCCL_IB_DISABLE", "1")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["IMAGE_MAX_TOKEN_NUM"] = env("IMAGE_MAX_TOKEN_NUM", "1024")
    os.environ["MAX_PIXELS"] = env("MAX_PIXELS", "1048576")  # 1024*1024

    use_spm = env_bool("USE_SPM", "1")  

    model_dir = env("MODEL_DIR", "/data1/qwen/model/Qwen3-VL-32B-Instruct")
    train_jsonl = env("TRAIN_JSONL", "/data1/qwen/data/sft_train.jsonl")
    val_jsonl = env("VAL_JSONL", "/data1/qwen/data/sft_val.jsonl")
    output_dir = env("OUT_DIR", "output_qwen3_vl_spm_1224")

    # ------- SPM vs Pure-SFT 关键切换点 -------
    if use_spm:
        # 原逻辑：自定义多模态模型 + 自定义 register
        register_entry = env("CUSTOM_REGISTER", "/data1/qwen/my_qwen3_vl_spm/my_register.py")
        os.environ["SPM_FEAT_DIM"] = env("SPM_FEAT_DIM", "768")

        model_type = env("MODEL_TYPE", "my_qwen3_vl_spm")
        template = env("TEMPLATE", "my_qwen3_vl_spm")
        custom_register_path = [register_entry]
    else:
        # 纯 SFT：不使用自定义 model/register，不注入 spm
        # 如果你本地 swift 里官方名字不是 qwen3_vl，请在 sh 里 export MODEL_TYPE/TEMPLATE 覆盖
        model_type = env("MODEL_TYPE", "qwen3_vl")
        template = env("TEMPLATE", "qwen3_vl")
        custom_register_path = []

        # 防止残留环境变量影响（可选，但建议）
        if "SPM_FEAT_DIM" in os.environ:
            os.environ.pop("SPM_FEAT_DIM", None)

    sft_main(TrainArguments(
        model=model_dir,
        model_type=model_type,
        template=template,
        custom_register_path=custom_register_path,

        dataset=train_jsonl,
        val_dataset=val_jsonl,
        load_from_cache_file=False,

        train_type="lora",
        torch_dtype="bfloat16",
        attn_impl="flash_attn",
        padding_free=True,
        packing=True,
        gradient_checkpointing=True,
        vit_gradient_checkpointing=False,

        num_train_epochs=int(env("EPOCHS", "1")),
        per_device_train_batch_size=int(env("BATCH_TRAIN", "1")),
        per_device_eval_batch_size=int(env("BATCH_EVAL", "1")),
        gradient_accumulation_steps=int(env("GRAD_ACC", "1")),

        learning_rate=float(env("LR", "1e-4")),
        warmup_ratio=float(env("WARMUP", "0.05")),

        lora_rank=int(env("LORA_R", "8")),
        lora_alpha=int(env("LORA_ALPHA", "32")),
        target_modules="all-linear",

        freeze_vit=True,
        freeze_aligner=False,

        eval_steps=int(env("EVAL_STEPS", "100")),
        save_steps=int(env("SAVE_STEPS", "100")),
        save_total_limit=int(env("SAVE_LIMIT", "2")),
        logging_steps=int(env("LOG_STEPS", "5")),

        max_length=int(env("MAX_LEN", "4096")),
        output_dir=output_dir,

        deepspeed=env("DS_STAGE", "zero3"),

        dataset_num_proc=int(env("DATASET_PROC", "4")),
        dataloader_num_workers=int(env("WORKERS", "4")),
    ))
