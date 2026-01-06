## SLO Diagnose Demo

Gradio 前端可通过多种 runner 组合使用（真实权重 / sample / mock）。默认会优先加载真实 runner，失败后尝试 sample，再回退到 mock。

### 快速开始（Mock）

```bash
MOCK_SPM=1 MOCK_VLM=1 python app.py
```

### 选择 Runner 来源

- 环境变量：`RUNNER_MODE`（可选 `auto` / `real` / `sample` / `mock`，默认 `auto`）。
- 命令行：`--runner-mode`（同上）。命令行会覆盖环境变量。
- 兼容开关：`USE_SAMPLE_RUNNERS=1` 会强制走 `sample_runners.py`。

自动模式会按 **real → sample → mock** 的顺序尝试初始化。

### 真实 SPM Runner 示例

`services/spm_runner_impl.py` 复用了 2CA 版本的 backbone + 校准逻辑，依赖的路径全部由环境变量指定：

必填：

- `SPM_BACKBONE_CFG_2CA=/path/to/backbone.yaml`
- `SPM_BACKBONE_CKPT_2CA=/path/to/backbone_ckpt.pth`
- `SPM_CALIB_CKPT_2CA=/path/to/calib_ckpt.pth`
- `SPM_PRIOR_MATRIX_PATH_2CA=/path/to/prior_matrix.npy`
- `SPM_SELECTED_LESION_IDS_2CA=1,2,5`（用逗号分隔）

可选（括号内为默认值）：

- `SPM_NUM_CLASSES` (27)
- `SPM_LESION_IMG_SIZE` (2048)
- `SPM_PATCH_SIZE` (256)
- `SPM_PATCH_STRIDE` (256)
- `SPM_DEVICE` (`cuda`，若无 GPU 自动回退 `cpu`)
- `SPM_USE_EMA` (1)
- `SPM_THRESHOLD_DEFAULT` (0.5)
- `SPM_FEAT_DIR` (`.cache_spm_feats`，用于写出 `spm_feat_*.npy`)

输入图片会在内部按 `SPM_LESION_IMG_SIZE` 缩放并使用 `build_timm_transform_hr` 做归一化，`spatialmamba_2ca.py` 的 patch 提取参数由 `SPM_PATCH_SIZE` / `SPM_PATCH_STRIDE` 控制。

运行示例：

```bash
RUNNER_MODE=real \
SPM_BACKBONE_CFG_2CA=/weights/backbone.yaml \
SPM_BACKBONE_CKPT_2CA=/weights/backbone.pth \
SPM_CALIB_CKPT_2CA=/weights/stageB_best.pth \
SPM_PRIOR_MATRIX_PATH_2CA=/weights/prior.npy \
SPM_SELECTED_LESION_IDS_2CA=1,2,5,9 \
python app.py --runner-mode real
```

### 真实 VLM Runner 示例

`services/vlm_runner_impl.py` 以 Swift/Qwen 形式调用模型，常用环境变量：

- `VLM_MODEL_NAME_OR_PATH`：HF/Swift 名称或本地权重路径（默认 `qwen3-vl-72b`）
- `VLM_DEVICE`：如 `cuda:0`
- `VLM_DTYPE`：如 `bfloat16` / `float16`
- `VLM_MAX_NEW_TOKENS`：默认 512
- `VLM_TEMPERATURE`：默认 0.2

SPM 的特征文件路径（若 `use_spm_feat` 勾选）会通过 `spm_feat_path` 传入 `extra_kwargs`，请确保模型侧支持。

运行示例：

```bash
RUNNER_MODE=real \
VLM_MODEL_NAME_OR_PATH=/models/qwen3-vl \
VLM_DEVICE=cuda:0 \
VLM_DTYPE=bfloat16 \
python app.py --runner-mode real
```

### Sample Runner

保留 `USE_SAMPLE_RUNNERS=1` 兼容路径，内部依赖 `sample_runners.py`。适合在有同款依赖的机器上快速验证。
