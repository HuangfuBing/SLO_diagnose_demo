import json
import os
import base64
import random
import time
from openai import OpenAI
from tqdm import tqdm


# 1. OpenAI API 设置
API_KEY = ""
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-5-nano-2025-08-07"

# 2. 文件路径设置
INPUT_PREDS_FILE = "train_preds.jsonl"        # 每行包含 image_key, probs, y_true 等
THRESHOLDS_FILE = "classwise_thresholds.json" # 每类阈值
KNOWLEDGE_BASE_FILE = "knowledge_base.json"   # 含 name_zh 和 slo_phrases
OUTPUT_FILE = "vlm_finetune_train.jsonl"      # 输出
ERROR_FILE = "error.jsonl"                    # 记录 GPT 调用失败的样本
DIFF_MEDIAN_FILE = "diff_median_pos.json"     # 预先计算好的各疾病 median_pos(diff)

# 3. 图片文件夹名称
LOCAL_IMAGE_FOLDER = "images"

# 4. 类别名称列表
CLASS_NAMES = [
    "AMD", "COATS", "DR", "FEVR", "PDR", "PM", "RD", "RP", "RVO", "VKH",
    "动脉瘤", "有髓神经纤维", "正常", "激光光斑", "玻璃体混浊", "玻璃体积血",
    "玻璃膜疣", "白内障", "眼底肿瘤", "脉络膜疾病", "脉络膜视网膜炎", "血管炎",
    "视盘异常", "视网膜前膜", "视网膜变性", "视网膜裂孔", "黄斑病变"
]

# =======================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def log_error(sample_id: str, stage: str, error_message: str, extra: dict = None):
    """将失败样本信息写入 error.jsonl"""
    record = {
        "sample_id": sample_id,
        "stage": stage,
        "error": error_message,
    }
    if extra is not None:
        record["extra"] = extra
    with open(ERROR_FILE, "a", encoding="utf-8") as ef:
        ef.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str):
    try:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError:
            print(f"Warning: {path} 不是 UTF-8 编码，正在尝试 GBK...")
            with open(path, 'r', encoding='gbk') as f:
                return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}，请确认文件在当前目录下。")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: {path} 文件格式有误，请检查 JSON 语法。")
        exit(1)


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def build_findings_input(probs, thresholds, knowledge_base, y_true_list=None):
    """
    构造给大模型看的“原始预测信息列表”：每个元素包含：
    - class_name
    - 疾病名称
    - prob
    - threshold
    - y_true
    """
    findings = []
    labels_kb = knowledge_base.get("labels", {})

    for idx, prob in enumerate(probs):
        if idx >= len(CLASS_NAMES):
            break

        class_name = CLASS_NAMES[idx]
        th = thresholds.get(class_name, 0.5)
        kb_entry = labels_kb.get(class_name, {})
        disease_name = kb_entry.get("name_zh", class_name)

        y_true = None
        if y_true_list is not None and idx < len(y_true_list):
            y_true = y_true_list[idx]

        findings.append({
            "class_name": class_name,
            "疾病名称": disease_name,
            "prob": round(float(prob), 6),
            "threshold": float(th),
            "y_true": y_true
        })

    return findings


def compute_support_label(y_true, prob: float, threshold: float, strong_diff_threshold: float) -> str:
    """
    根据规则计算影像支持程度：
    ① 当 y_true = 1 时：
        - prob < threshold                -> "uncertain"
        - prob >= threshold 且 diff >= strong_diff_threshold -> "strong"
        - prob >= threshold 且 0 <= diff < strong_diff_threshold -> "weak"
       其中 diff = prob - threshold，strong_diff_threshold 为该疾病的 median_pos（>=0）
    ② 当 y_true = 0 时：
        - "none"

    若 y_true 为 None 或异常，返回 "none"。
    """
    if y_true is None:
        return "none"

    try:
        y_true_int = int(y_true)
    except Exception:
        return "none"

    if y_true_int == 0:
        # 规则②
        return "none"

    # y_true == 1 的情况，规则①
    if prob < threshold:
        return "uncertain"

    diff = prob - threshold
    # strong_diff_threshold 预先保证 >= 0，这里直接比较即可
    if diff >= strong_diff_threshold:
        return "strong"
    else:
        return "weak"


def build_slo_draft(label_findings: dict, knowledge_base: dict) -> str:
    """
    使用 证据 中每个疾病的“影像支持程度”，
    去 knowledge_base.json 里查 slo_phrases[support]，拼成 SLO 检查情况初稿。
    """
    labels_kb = knowledge_base.get("labels", {})
    parts = []

    for class_name, info in label_findings.items():
        kb_entry = labels_kb.get(class_name, {})
        slo_phrases = kb_entry.get("slo_phrases", {})

        support = info.get("影像支持程度", "uncertain")
        disease_name = info.get("疾病名称", class_name)

        phrases_for_support = slo_phrases.get(support, [])
        if not phrases_for_support:
            phrases_for_support = slo_phrases.get("uncertain", [])

        if phrases_for_support:
            phrase = random.choice(phrases_for_support)
            parts.append(phrase)
        else:
            parts.append(f"可见与{disease_name}相关的异常改变。")

    if not parts:
        return "SLO 检查情况：本次 SLO 图像质量可用于评估，未见明显异常病变。"

    lines = ["SLO 检查情况初稿："]
    for idx, p in enumerate(parts, start=1):
        lines.append(f"{idx}. {p}")

    return "\n".join(lines)


def refine_slo_and_suggest_with_llm(slo_draft: str, diagnoses, image_path: str, sample_id: str = ""):
    """
    第二次调用 GPT：
    输入：临床诊断列表 + SLO 检查情况初稿 + 图像
    输出：最终 SLO 检查情况
    出现网络/API错误时最多重试 3 次，失败则记录到 error.jsonl。
    """
    system_prompt = """
<role>
你是一名严谨的眼底病学专科医生助手。你只根据提供的 SLO 眼底图像、临床诊断列表和“SLO 检查情况初稿”进行判断，不自行引入额外的检查项目或臆测诊断。
</role>

<input>
你将接收到以下三类信息：
1. 一份【临床诊断列表】；
2. 一段【SLO 检查情况初稿】，可能存在表述重复、轻微矛盾或不够流畅；
3. 一张对应的 SLO 眼底图像。
</input>

<task>
结合图像和临床诊断，对“SLO 检查情况初稿”进行整理：
- 不随意更改或创造医学术语；
- 当初稿中出现用“或”连接的两个备选描述时，根据图像实际表现，只保留与图像最相符的一种表述；
- 综合初稿全文，删除明显矛盾的表述（例如某一句提到“大量出血”，同时另外一句提到“少量出血”，则根据影像选择与实际情况一致的一种）；
- 保留解剖部位信息；
- 输出为一段连续的中文影像学描述，以“SLO 检查情况：”开头，不再使用编号或列表格式。
</task>

<constraints>
- 严格依赖给定的初稿内容、临床诊断和图像。
- 不得把初稿中“可见/呈/表现为/存在/疑似”等肯定性描述改写成“未见/未见到/未观察到/未发现/无/未见明显”等否定性表述；
- 不要新增未在图像或诊断中体现的疾病、部位或检查项目。
</constraints>

<output_format>
仅输出一个 JSON 对象，不要输出任何额外文字、说明或 Markdown 代码块。
禁止输出 ```、```json 等代码块标记，也禁止输出注释和多余的逗号。

JSON 结构必须严格为：

{
  "slo": "SLO 检查情况：......"
}

要求：
- "slo" 必须是一整段以“SLO 检查情况：”开头的中文影像学描述。
</output_format>
"""

    # 把临床诊断和 SLO 初稿用标签包起来，结构化给模型
    if diagnoses:
        diag_lines = "\n".join(f"- {d}" for d in diagnoses)
    else:
        diag_lines = "- 未见明显异常眼底"

    user_text = f"""
<clinical_diagnoses>
{diag_lines}
</clinical_diagnoses>

<slo_draft>
{slo_draft}
</slo_draft>
""".strip()

    def fallback_result():
        text = slo_draft.strip()
        if text.startswith("SLO 检查情况初稿"):
            # 初稿 → 正式前缀
            text = text.replace("SLO 检查情况初稿", "SLO 检查情况", 1)
        elif text.startswith("SLO 检查情况"):
            # 已经是正式前缀，直接返回
            pass
        else:
            # 既不是“初稿”也不是“检查情况”，手动加前缀
            text = "SLO 检查情况：" + text.lstrip("：: ")
        return text

    max_retries = 3
    data = None

    # ====== 先尝试读图像；失败时记录错误并直接走 fallback，不再退化为纯文本 ======
    base64_image = None
    if image_path:
        try:
            base64_image = encode_image(image_path)
        except Exception as e:
            print(f"encode_image 失败: {e}")
            log_error(sample_id, "encode_image", str(e), extra={"image_path": image_path})
            return fallback_result()


    if not base64_image:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    reasoning_effort="high",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                )
                text = resp.choices[0].message.content.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                data = json.loads(text)
                # 强制要求是 JSON object，否则在本函数内重试
                if not isinstance(data, dict):
                    raise ValueError(f"模型返回的 JSON 不是对象，而是 {type(data)}: {data}")
                break
            except Exception as e:
                print(f"refine_slo_and_suggest_with_llm（无图）第 {attempt} 次调用失败: {e}")
                if attempt == max_retries:
                    log_error(sample_id, "refine_slo_no_image", str(e), extra={"user_text": user_text})
                    return fallback_result()
                time.sleep(5)
    else:
        # 多模态调用，最多重试 3 次
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    reasoning_effort="high",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                )
                text = resp.choices[0].message.content.strip()
                text = text.replace("```json", "").replace("```", "").strip()
                data = json.loads(text)

                if not isinstance(data, dict):
                    raise ValueError(f"模型返回的 JSON 不是对象，而是 {type(data)}: {data}")
                break
            except Exception as e:
                print(f"refine_slo_and_suggest_with_llm（带图）第 {attempt} 次调用失败: {e}")
                if attempt == max_retries:
                    log_error(sample_id, "refine_slo_with_image", str(e), extra={"user_text": user_text})
                    return fallback_result()
                time.sleep(5)

    # ====== 从 data 中取 slo 字段；缺失或空时走 fallback，并规范前缀 ======
    slo_text = ""
    if isinstance(data, dict):
        slo_text = str(data.get("slo", "")).strip()

    if not slo_text:
        return fallback_result()

    if slo_text.startswith("SLO 检查情况初稿"):
        slo_text = slo_text.replace("SLO 检查情况初稿", "SLO 检查情况", 1)
    elif not slo_text.startswith("SLO 检查情况"):
        slo_text = "SLO 检查情况：" + slo_text.lstrip("：: ")

    return slo_text


def main():
    print(f"Working Directory: {os.getcwd()}")
    print(">>> 开始初始化...")

    thresholds = load_json(THRESHOLDS_FILE)
    kb = load_json(KNOWLEDGE_BASE_FILE)
    median_pos_dict = load_json(DIFF_MEDIAN_FILE)  # 读取 diff_median_pos.json
    print(f">>> 辅助文件加载完毕。")

    # 将 diff_median_pos.json中的值转成 strong_diff_thresholds，并做一层 >=0 的保护
    strong_diff_thresholds = {}
    for cls in CLASS_NAMES:
        raw_val = median_pos_dict.get(cls, None)
        if raw_val is None:
            strong_diff_thresholds[cls] = 0.0
        else:
            try:
                v = float(raw_val)
            except Exception:
                v = 0.0
            # 如果不想截断负数，可以去掉 max(0.0, v) 改成直接 v
            strong_diff_thresholds[cls] = max(0.0, v)
        print(f"[median_pos] {cls}: strong_diff_threshold = {strong_diff_thresholds[cls]:.3f} (from diff_median_pos.json)")

    processed_count = 0
    skipped_count = 0

    # 读取预测文件
    try:
        with open(INPUT_PREDS_FILE, 'r', encoding='utf-8') as in_f:
            lines = in_f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到 {INPUT_PREDS_FILE}，请确认文件名。")
        return

    print(f">>> 找到 {len(lines)} 条预测记录，开始生成微调数据...")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
        for line in tqdm(lines, desc="Processing"):
            try:
                data = json.loads(line)
                original_key = data.get("image_key", "")
                probs = data.get("probs", [])

                y_true_list = data.get("y_true")  # 如果没有，则为 None

                # 1) 提取文件名
                filename = os.path.basename(original_key)
                sample_id = original_key or filename

                # 2) 本地路径
                local_image_path = os.path.join(LOCAL_IMAGE_FOLDER, filename)

                # 3) 检查图片是否存在
                if not os.path.exists(local_image_path):
                    skipped_count += 1
                    continue

                # 4) 构造 findings_input
                findings_input = build_findings_input(probs, thresholds, kb, y_true_list)

                # 5) 根据固定规则构造“证据”里的疾病列表
                evidence_dict = {}

                for item in findings_input:
                    cls = item["class_name"]
                    disease_name = item["疾病名称"]
                    prob = float(item["prob"])
                    th = float(item["threshold"])
                    y_true = item.get("y_true")

                    strong_diff_th = strong_diff_thresholds.get(cls, 0.0)
                    support = compute_support_label(y_true, prob, th, strong_diff_th)

                    # 当 y_true = 0 -> "none"，不纳入证据列表
                    # 当 y_true = 1 -> strong / weak / uncertain
                    if support != "none":
                        evidence_dict[cls] = {
                            "疾病名称": disease_name,
                            "影像支持程度": support
                        }

                # ====== 6) 从 证据 中构造临床诊断列表 ======
                diagnoses = []
                for cls, info in evidence_dict.items():
                    disease_name = info.get("疾病名称", cls)
                    if disease_name not in diagnoses:
                        diagnoses.append(disease_name)

                if not diagnoses:
                    diagnoses = ["未见明显异常眼底"]

                # ====== 7) 用 knowledge_base + 影像支持程度 生成 SLO 检查情况初稿 ======
                slo_draft = build_slo_draft(evidence_dict, kb)

                # ====== 8) 第二次调用 GPT：根据“临床诊断 + 初稿 + 图像”生成 SLO 检查情况
                slo_text = refine_slo_and_suggest_with_llm(
                    slo_draft, diagnoses, local_image_path, sample_id=sample_id
                )

                # ====== 9) 诊断报告部分：临床诊断 + SLO 检查情况 ======
                diag_lines = ["临床诊断："]
                for i, d in enumerate(diagnoses, start=1):
                    diag_lines.append(f"{i}. {d}")

                diagnosis_report = (
                    "\n".join(diag_lines)
                    + "\n\n"
                    + slo_text
                )

                # ====== 10) 证据部分文本 ======
                if evidence_dict:
                    evidence_lines = ["证据：", "阳性相关发现："]
                    for i, (cls, info) in enumerate(evidence_dict.items(), start=1):
                        disease_name = info.get("疾病名称", cls)
                        support = info.get("影像支持程度", "uncertain")
                        evidence_lines.append(f"{i}. {disease_name}：{support}")
                else:
                    evidence_lines = ["证据：", "未见明显阳性相关发现。"]

                evidence_text = "\n".join(evidence_lines)

                # ====== 11) 最终 gpt 输出文本 = “证据” + “诊断报告” 两部分 ======
                final_gpt_text = evidence_text + "\n\n诊断报告：\n" + diagnosis_report

                # ====== 12) 构造给人看的 findings（不含 y_true、不含 class_name，只保留疾病名称 + prob + threshold） ======
                findings_for_prompt = []
                for item in findings_input:
                    findings_for_prompt.append({
                        "疾病名称": item["疾病名称"],
                        "prob": item["prob"],
                        "threshold": item["threshold"]
                    })

                # ====== 13) ms-swift 用的 human 提示（messages + images 格式）
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

                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image>\n" + human_prompt_text
                        },
                        {
                            "role": "assistant",
                            "content": final_gpt_text
                        }
                    ],
                    # 只保留图像文件名（例如 "00010801-20230809@170819-L7-S.jpg"）
                    "images": [filename]
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理 {data.get('image_key', '')} 时出错: {e}")
                log_error(data.get("image_key", ""), "pipeline", str(e))
                continue

    print(f"\n>>> 处理完成！")
    print(f"   - 成功处理: {processed_count} 张")
    print(f"   - 跳过缺失: {skipped_count} 张 (images 文件夹中不存在这些图片)")
    print(f">>> 结果已保存至: {OUTPUT_FILE}")
    print(f">>> 错误样本记录在: {ERROR_FILE}")


if __name__ == "__main__":
    main()
