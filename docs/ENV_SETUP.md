# 环境变量配置教程

本文说明如何在本项目中配置并加载环境变量（含 SPM 与 VLM）。推荐方式是使用 `.env` 文件 + 启动前导入，或直接在命令行前缀设置。

## 1. 推荐目录与文件

在仓库根目录创建一个 `.env` 文件（不要提交到版本库）：

```bash
# 示例：在仓库根目录创建
cat <<'ENV' > .env
# SPM 相关（必填）
SPM_BACKBONE_CFG_2CA=/absolute/path/to/backbone.yaml
SPM_BACKBONE_CKPT_2CA=/absolute/path/to/best_ckpt_ema.pth
SPM_CALIB_CKPT_2CA=/absolute/path/to/stageB_best.pth
SPM_PRIOR_MATRIX_PATH_2CA=/absolute/path/to/prior_matrix.npy
SPM_SELECTED_LESION_IDS_2CA=1,2,5

# SPM 相关（可选）
SPM_NUM_CLASSES=27
SPM_LESION_IMG_SIZE=2048
SPM_PATCH_SIZE=256
SPM_PATCH_STRIDE=256
SPM_DEVICE=cuda
SPM_USE_EMA=1
SPM_THRESHOLD_DEFAULT=0.5
SPM_FEAT_DIR=.cache_spm_feats

# VLM 相关（可选，但通常会填）
VLM_MODEL_NAME_OR_PATH=/absolute/path/to/qwen3-vl-32b
VLM_DEVICE=cuda:0
VLM_DTYPE=bfloat16
VLM_MAX_NEW_TOKENS=512
VLM_TEMPERATURE=0.2

# Runner 选择
RUNNER_MODE=real
ENV
```

> 注意：`.env` 通常应加入 `.gitignore`，避免提交包含本地路径或私有模型路径的文件。

## 2. 启动前加载环境变量

### 方式 A：手动导入 `.env`

```bash
set -a
source .env
set +a

python app.py --runner-mode real
```

### 方式 B：一次性命令前缀

```bash
RUNNER_MODE=real \
SPM_BACKBONE_CFG_2CA=/absolute/path/to/backbone.yaml \
SPM_BACKBONE_CKPT_2CA=/absolute/path/to/best_ckpt_ema.pth \
SPM_CALIB_CKPT_2CA=/absolute/path/to/stageB_best.pth \
SPM_PRIOR_MATRIX_PATH_2CA=/absolute/path/to/prior_matrix.npy \
SPM_SELECTED_LESION_IDS_2CA=1,2,5 \
VLM_MODEL_NAME_OR_PATH=/absolute/path/to/qwen3-vl-32b \
VLM_DEVICE=cuda:0 \
VLM_DTYPE=bfloat16 \
python app.py --runner-mode real
```

## 3. 常见填写说明

### SPM（Spatial-Mamba）

- `SPM_BACKBONE_CFG_2CA`：主干配置 yaml（2CA）
- `SPM_BACKBONE_CKPT_2CA`：主干权重（如 `best_ckpt_ema.pth`）
- `SPM_CALIB_CKPT_2CA`：校准（StageB）权重
- `SPM_PRIOR_MATRIX_PATH_2CA`：先验矩阵（.npy / .json）
- `SPM_SELECTED_LESION_IDS_2CA`：病灶 ID 列表（逗号分隔）

### VLM（Qwen3-VL / Swift）

- `VLM_MODEL_NAME_OR_PATH`：HF/Swift 模型名或本地权重目录
- `VLM_DEVICE`：推理设备（如 `cuda:0`）
- `VLM_DTYPE`：权重 dtype（如 `bfloat16` / `float16`）

## 4. 运行检查

若环境变量无误，`app.py` 会按 **real → sample → mock** 的顺序尝试加载；设置 `RUNNER_MODE=real` 可强制走真实模型路径。

