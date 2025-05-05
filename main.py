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
  "legal_level": "字符串",
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个法律NLP助手，返回严格JSON，不要额外说明。"},
                {"role": "user", "content": prompt}
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
            "legal_level": parsed.get("legal_level", ""),
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


# ——— Main loop ———
DATA_DIR = "raw/data_protection"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
for root, dirs, files in os.walk(DATA_DIR):
    dirs.sort()
    # Outer bar to show directory progress
    for file in tqdm(files, desc=f"Scanning {os.path.basename(root)}", unit="file", leave=False):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(root, file)
        text = open(path, "r", encoding="utf-8").read().strip()
        rec = classify_document(text)
        rec["file_path"] = os.path.relpath(path, DATA_DIR)
        rec["top_level_category"] = os.path.basename(root)
        results.append(rec)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, "data_protection_run.csv"),
          index=False,
          encoding="utf-8-sig")
