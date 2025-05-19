import os
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ─── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("GPT_KEY"))

# Load definitions
DEFINITIONS = Path("classifier_definitions.txt").read_text(encoding="utf-8")


# Build prompt
def build_prompt(text):
    return f"""
你是一名法律 NLP 助理，专门从事数据保护和隐私法规方面的工作。

将你收到一份法规或法律的全文，并对与隐私法相关的某些因素进行分类。

以下是对 13 个字段进行分类的说明:
{DEFINITIONS}

为确保准确性，请仔细阅读全文，逐步思考后再做出判断。最好慢一点，确保分类结果准确可靠。

请将最终结果以下列 JSON 格式返回，无需任何解释或说明：
{{
  "title": "字符串",
  "date_enacted": "YYYY-MM-DD",
  "jurisdiction": "字符串",
  "jurisdiction_name": "字符串",
  "general_reference": true 或 false,
  "data_category": "字符串",
  "individual_rights": ["字符串列表"],
  "handler_responsibilities": ["字符串列表"],
  "sector": ["字符串列表"],
  "keywords":["字符串列表"],
  "solove_classification": ["字符串列表"],
  "synopsis": "字符串",
  "pipl_mention": true 或 false
}}

以下为政策文件全文。
\"\"\"
{text}
\"\"\"
"""


# Classification function
def classify_document(text):
    prompt = build_prompt(text)
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "system",
                 "content": "你是一个法律NLP助手，返回严格JSON，不要额外说明。你们要根据我给你们的提示，帮助我对与隐私相关的法律文件进行分类。如果答案不符合我提供给你们的模式，将受到重罚--如果出现这种情况，请再试一次。"}
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp.choices[0].message.content.strip())

        # Ensure all keys exist
        return {
            "title": parsed.get("title", ""),
            "date_enacted": parsed.get("date_enacted", ""),
            "jurisdiction": parsed.get("jurisdiction", ""),
            "jurisdiction_name": parsed.get("jurisdiction_name", ""),
            "general_reference": parsed.get("general_reference", False),
            "data_category": parsed.get("data_category", []),
            "individual_rights": parsed.get("individual_rights", []),
            "handler_responsibilities": parsed.get("handler_responsibilities", []),
            "sector": parsed.get("sector", []),
            "keywords": parsed.get("keywords", []),
            "solove_classification": parsed.get("solove_classification", []),
            "synopsis": parsed.get("synopsis", []),
            "pipl_mention": parsed.get("pipl_mention", [])
        }
    except json.JSONDecodeError as e:
        print("⛔ JSON decode error:", e)
        print("Raw response:", resp.choices[0].message.content)
        return {k: "" if isinstance(v, str) else [] for k, v in classify_document.__annotations__.items()}
    except Exception as e:
        print("⚠️ Error:", e)
        return {k: "" if isinstance(v, str) else [] for k, v in classify_document.__annotations__.items()}


def process_dir(base_name: str, suffix: str, data_root="raw", results_dir="results"):
    """
    Walk through raw/{base_name}, classify each .txt,
    and dump results to results/{base_name}-4.1-mini_{suffix}.csv
    """
    DATA_DIR = os.path.join(data_root, base_name)
    os.makedirs(results_dir, exist_ok=True)

    results = []
    for root, dirs, files in os.walk(DATA_DIR):
        dirs.sort()
        for file in tqdm(files, desc=f"Scanning {base_name}", unit="file", leave=False):
            if not file.endswith(".txt"):
                continue
            path = os.path.join(root, file)
            text = open(path, "r", encoding="utf-8").read().strip()

            rec = classify_document(text)  # your existing classifier
            rec["file_path"] = os.path.relpath(path, data_root)
            rec["top_level_category"] = base_name

            results.append(rec)

    out_name = f"{base_name}-4.1-mini_{suffix}.csv"
    out_path = os.path.join(results_dir, out_name)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    # Directories to process
    bases = ["privacy"]
    # Suffixes you want to generate
    suffixes = ["delta"]

    for suf in suffixes:
        for base in bases:
            process_dir(base_name=base, suffix=suf)
