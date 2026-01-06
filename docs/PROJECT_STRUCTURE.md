# 项目目录结构

围绕实际使用、修改与维护场景，对项目目录做了显式分层，并为仍需补充的内容预留了占位文件。

## 目录概览

```
.
├── app.py                # Gradio 前端与入口
├── services/             # SPM/VLM runner 及 API 封装
├── configs/              # 环境/模型配置（占位）
├── data/                 # 示例或回归数据（占位）
├── weights/              # 模型权重挂载位置（占位）
├── feedback/             # 诊断与医生反馈日志
├── docs/                 # 文档与结构说明
├── requirements-demo.txt # demo 运行依赖
└── 其他训练/生成脚本      # 如 main_calib_eval.py、generate_vlm3.py 等
```

## 维护建议

- **配置与权重**：将实际环境的 yaml/json/ckpt 放入 `configs/` 与 `weights/`，运行时用环境变量指向；如果需要按环境拆分，可在目录下新增 `dev/`、`prod/` 等子目录。
- **数据管理**：在 `data/` 内使用 `samples/` 或 `experiments/` 子目录分类，避免将敏感数据直接提交版本库。
- **反馈留存**：保留 `feedback/README.md` 以确保目录存在，实际生成的 JSONL 可按需清理或同步。
- **脚本归档**：现有训练/生成脚本保持在仓库根目录，后续若需要可按功能迁移到 `scripts/` 或 `tools/` 目录，并在此文档更新树状结构。

占位目录已通过 dummy 文件（各自的 `README.md`）标记，可在补充真实内容时直接替换或追加说明。
