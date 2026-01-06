"""
Gradio demo for SPM + VLM pipeline.

Usage (mock mode; no GPU/weights required):
    MOCK_SPM=1 MOCK_VLM=1 python app.py

To integrate real models:
    # 优先使用真实 runner，其次尝试 sample runner，最后回退到 mock
    RUNNER_MODE=auto python app.py --runner-mode auto

Env / args:
    RUNNER_MODE: auto | real | sample | mock   （默认 auto，自动优先 real）
    USE_SAMPLE_RUNNERS=1: 保持兼容之前的 sample runner 开关
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from services import (
    SpmResult,
    VlmResult,
    build_spm_runner,
    build_vlm_runner,
    make_default_spm_client,
    make_default_vlm_client,
)

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
RUNNER_MODE_DEFAULT = os.getenv("RUNNER_MODE", "auto").lower()


def _try_real_clients() -> Optional[Tuple]:
    try:
        spm_runner = build_spm_runner()
        vlm_runner = build_vlm_runner()
        return make_default_spm_client(runner=spm_runner), make_default_vlm_client(runner=vlm_runner)
    except Exception as exc:
        print(f"[runner] Failed to init real runners: {exc}")
        return None


def _try_sample_clients() -> Optional[Tuple]:
    try:
        from sample_runners import build_sample_clients

        return build_sample_clients()
    except Exception as exc:
        print(f"[runner] Failed to init sample runners: {exc}")
        return None


def create_clients(runner_mode: str) -> Tuple:
    mode = (runner_mode or "auto").lower()
    prefer_sample = os.getenv("USE_SAMPLE_RUNNERS", "0") == "1"

    if mode == "real":
        clients = _try_real_clients()
        if clients:
            return clients
        raise RuntimeError("runner_mode=real 但未能成功初始化真实 runner，请检查权重/依赖。")

    if mode == "sample" or prefer_sample:
        clients = _try_sample_clients()
        if clients:
            return clients
        if mode == "sample":
            raise RuntimeError("runner_mode=sample 但初始化失败，请检查 sample runner 依赖。")

    if mode == "auto":
        for builder in (_try_real_clients, _try_sample_clients):
            clients = builder()
            if clients:
                return clients

    # fallback to mock
    os.environ.setdefault("MOCK_SPM", "1")
    os.environ.setdefault("MOCK_VLM", "1")
    print("[runner] Falling back to mock outputs (MOCK_SPM=1, MOCK_VLM=1)")
    return make_default_spm_client(), make_default_vlm_client()


spm_client, vlm_client = create_clients(RUNNER_MODE_DEFAULT)


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
    parser.add_argument(
        "--runner-mode",
        choices=["auto", "real", "sample", "mock"],
        default=RUNNER_MODE_DEFAULT,
        help="选择 SPM/VLM runner 来源：real 优先真实权重，sample 试验用，mock 回退固定输出。",
    )
    args = parser.parse_args()

    # re-init clients in case CLI overrides env default
    global spm_client, vlm_client
    spm_client, vlm_client = create_clients(args.runner_mode)

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
