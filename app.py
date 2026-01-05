"""
Gradio demo for SPM + VLM pipeline.

Usage (mock mode; no GPU/weights required):
    MOCK_SPM=1 MOCK_VLM=1 python app.py

To integrate real models:
1) Replace `spm_client = make_default_spm_client()` with
   `make_default_spm_client(runner=your_spm_runner)`.
   The runner should accept a list of image paths and return `SpmResult`.
2) Replace `vlm_client = make_default_vlm_client()` similarly.
3) Keep the function signatures to stay compatible with the UI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr

from services import SpmResult, VlmResult, make_default_spm_client, make_default_vlm_client

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)


# [MOD-c] Prompt builder aligned with the training template used in generate_vlm3.py.
def build_prompt(
    disease_probs: List[Dict],
    thresholds: Dict,
    user_note: Optional[str] = None,
) -> str:
    """
    Construct the human prompt text exactly following the fine-tuning template
    from `generate_vlm3.py`, preserving the required sections and wording.
    """
    # Convert disease entries to the expected structure for the JSON snippet.
    findings_for_prompt = []
    for item in disease_probs:
        findings_for_prompt.append(
            {
                "疾病名称": item.get("id", "unknown"),
                "prob": item.get("prob"),
                "threshold": item.get("threshold", thresholds.get("default")),
            }
        )

    human_prompt_text = (
        "你是一名专业的眼科医生助手。请根据这张 SLO 眼底图像，"
        "以及下面给出的每个疾病的预测概率 (prob) 和阳性阈值 (threshold)，"
        "生成一段中文报告。\n\n"
        "报告中必须包含两个部分，并按如下格式书写：\n"
        "1）证据部分：\n"
        "   - 以“证据：”单独起一行；\n"
        "   - 下方按编号列出所有阳性相关疾病，每行格式为“编号. 疾病名称：strong/weak/uncertain”；\n"
        "   - 影像支持程度只能使用英文单词 strong、weak 或 uncertain 之一；\n"
        "2）诊断报告部分：\n"
        "   - 以“诊断报告：”单独起一行；\n"
        "   - 写一行“临床诊断：”，并按“1. ……\\n2. ……”的形式列出诊断；\n"
        "   - 再写一段以“SLO 检查情况：”开头的影像学描述（连续一段文字，不使用列表或编号）。\n\n"
        "下面是每一类疾病的预测信息（仅展示疾病名称、prob 和 threshold）：\n"
        f"{json.dumps(findings_for_prompt, ensure_ascii=False, indent=2)}\n"
    )

    if user_note:
        human_prompt_text += f"\n医生补充信息：\n{user_note}\n"

    return human_prompt_text


def log_feedback(
    session_id: str,
    spm: SpmResult,
    vlm: VlmResult,
    feedback_score: Optional[int],
    feedback_text: Optional[str],
):
    record = {
        "session_id": session_id,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "spm": spm.__dict__,
        "vlm": vlm.__dict__,
        "feedback": {"score": feedback_score, "text": feedback_text},
    }
    out_path = FEEDBACK_DIR / f"{session_id}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(out_path)


# --------------------------------------------------------------------------- #
# Gradio logic
# --------------------------------------------------------------------------- #
# [MOD-c-sample] Sample runner wiring for real SPM + VLM (Swift/Qwen3).
# Enable by setting USE_SAMPLE_RUNNERS=1 when launching the app; otherwise the
# default clients remain mock-friendly.
USE_SAMPLE_RUNNERS = os.getenv("USE_SAMPLE_RUNNERS", "0") == "1"

if USE_SAMPLE_RUNNERS:
    # Lazily loaded singleton holders to avoid repeated heavy init.
    _sample_spm_model = None
    _sample_spm_device = None
    _sample_spm_cache = Path(".cache_spm_feats")
    _sample_spm_cache.mkdir(exist_ok=True)

    def _get_sample_spm_model():
        """
        Instantiate your SpatialMamba + LesionCalib pipeline once.
        Replace cfg/ckpt paths and loading logic with your own.
        """
        # NOTE: import inside to keep default/mock flows lightweight.
        import torch
        from your_pkg import load_spm_model  # <-- replace with your actual loader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_spm_model(
            backbone_cfg="cfgs/backbone.yaml",
            backbone_ckpt="checkpoints/backbone_ema.pth",
            calib_ckpt="checkpoints/stageB_best.pth",
            device=device,
        )
        return model, device

    def _sample_spm_runner(paths):
        """
        paths: list of image file paths.
        Returns SpmResult with lesion/disease probs and optional feature path.
        """
        # Late import to avoid torch dependency during mock runs.
        import numpy as np
        from your_pkg import infer_spm  # <-- replace with your inference API

        global _sample_spm_model, _sample_spm_device

        if _sample_spm_model is None or _sample_spm_device is None:
            _sample_spm_model, _sample_spm_device = _get_sample_spm_model()

        lesion_probs, disease_probs, thresholds, feat_np = infer_spm(
            _sample_spm_model, paths, device=_sample_spm_device
        )

        feat_path = None
        if feat_np is not None:
            feat_path_obj = _sample_spm_cache / f"spm_feat_{Path(paths[0]).stem}.npy"
            np.save(feat_path_obj, feat_np)
            feat_path = str(feat_path_obj)

        return SpmResult(
            lesion_probs=lesion_probs,
            disease_probs=disease_probs,
            thresholds=thresholds,
            spm_feat_path=feat_path,
            debug={
                "source": "my_spm_runner",
                "paths": list(paths),
                "feat_path": feat_path,
            },
        )

    _sample_vlm_model = None

    def _get_sample_vlm_model():
        """
        Instantiate your Swift/Qwen3 vision-language model once.
        Replace with your actual Swift API and model ID.
        """
        from swift.llm import SwiftModel  # <-- adjust import to your stack

        model = SwiftModel.from_pretrained(
            "qwen3-vl-72b",  # <-- replace with your model or local path
            device="cuda:0",
            dtype="bfloat16",
        )
        return model

    def _sample_vlm_runner(prompt, image_paths, spm_feat_path=None):
        """
        prompt: str constructed by build_prompt
        image_paths: list[str]
        spm_feat_path: optional str for SPM feature injection
        """
        global _sample_vlm_model

        if _sample_vlm_model is None:
            _sample_vlm_model = _get_sample_vlm_model()

        # Adjust the generate call to your Swift/Qwen3 API.
        resp = _sample_vlm_model.generate(
            prompt=prompt,
            images=list(image_paths),
            max_new_tokens=512,
            temperature=0.2,
            # extra_kwargs={"spm_feat_path": spm_feat_path},  # if your model supports it
        )

        return VlmResult(
            report=resp.get("text", ""),
            raw_output=resp,
            debug={
                "source": "my_vlm_runner",
                "spm_feat_path": spm_feat_path,
                "used_images": list(image_paths),
            },
        )

    spm_client = make_default_spm_client(runner=_sample_spm_runner)
    vlm_client = make_default_vlm_client(runner=_sample_vlm_runner)
else:
    spm_client = make_default_spm_client()
    vlm_client = make_default_vlm_client()


def diagnose(
    images,
    use_spm_feat: bool,
    user_note: str,
    feedback_score: int,
    feedback_text: str,
):
    if not images:
        raise gr.Error("请先上传至少一张影像。")

    spm_res = spm_client(images)
    prompt = build_prompt(spm_res.disease_probs, spm_res.thresholds, user_note=user_note)
    vlm_res = vlm_client(prompt, images, spm_feat_path=spm_res.spm_feat_path if use_spm_feat else None)

    session_id = uuid.uuid4().hex
    feedback_path = log_feedback(session_id, spm_res, vlm_res, feedback_score, feedback_text)

    return (
        spm_res.disease_probs,
        spm_res.lesion_probs,
        spm_res.thresholds,
        vlm_res.report,
        session_id,
        feedback_path,
    )


def build_demo():
    with gr.Blocks(title="SLO 医学影像诊断 Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # SPM + VLM 联动 Demo
            - 上传 1 张或多张影像，前端会调用上游 SPM 推理，将概率/阈值注入 VLM 提示词，生成诊断报告。
            - 设置 `MOCK_SPM=1` 和 `MOCK_VLM=1` 可以在没有权重的机器上体验 UI。
            """
        )

        with gr.Row():
            with gr.Column():
                images = gr.Image(label="影像上传", type="filepath", sources=["upload", "clipboard"], multiple=True)
                use_spm_feat = gr.Checkbox(label="启用 SPM 特征注入", value=True)
                user_note = gr.Textbox(label="补充信息（可选）", lines=3, placeholder="患者症状、病史等")
                feedback_score = gr.Slider(label="医生评分 (1-5)", minimum=1, maximum=5, step=1, value=4)
                feedback_text = gr.Textbox(label="医生文字反馈（可选）", lines=3)
                run_btn = gr.Button("生成报告", variant="primary")

            with gr.Column():
                disease_probs = gr.JSON(label="疾病概率")
                lesion_probs = gr.JSON(label="病灶概率")
                thresholds = gr.JSON(label="阈值信息")
                report = gr.Textbox(label="诊断报告", lines=12)
                session_id = gr.Textbox(label="Session ID", interactive=False)
                feedback_path = gr.Textbox(label="反馈落盘路径", interactive=False)

        run_btn.click(
            diagnose,
            inputs=[images, use_spm_feat, user_note, feedback_score, feedback_text],
            outputs=[disease_probs, lesion_probs, thresholds, report, session_id, feedback_path],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="gradio server host")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="enable Gradio share link")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
