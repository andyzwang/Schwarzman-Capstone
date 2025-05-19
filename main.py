"""
main.py  ──────────────────────────────────────────────────────────
• Walk every .txt file under raw/<category_name>/…
• Ask OpenAI GPT-4.1-mini to classify the full text into 13 privacy-law fields
• Save one CSV per “trial” (alpha, beta, …) in results/

To run:
    export GPT_KEY="sk-···"
    python batch_classify.py
"""

import os
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ── 0. Initialise OpenAI client ────────────────────────────────────────────
load_dotenv()                                     # load GPT_KEY from .env
client = OpenAI(api_key=os.getenv("GPT_KEY"))

# ── 1. Load the field definitions that go into the system prompt ───────────
DEFINITIONS = Path("classifier_definitions.txt").read_text(encoding="utf-8")

# ── 2. Prompt builder: embeds the full law text + instructions ─────────────
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

# ── 3. Single-document classifier using the OpenAI chat API ────────────────
def classify_document(text: str) -> dict:
    """Return a dict with the 13 required keys. On failure, return empties."""
    prompt = build_prompt(text)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": "你是法律NLP助手。严格输出 JSON; 若格式错误重新尝试。"},
                {"role": "user", "content": prompt}
            ]
        )
        parsed = json.loads(resp.choices[0].message.content.strip())

    except (json.JSONDecodeError, AttributeError) as e:   # bad JSON → empty dict
        print("⛔ JSON decode error:", e)
        return {}

    except Exception as e:                                # network / quota / other
        print("⚠️ OpenAI error:", e)
        return {}

    # Ensure all 13 keys exist so DataFrame columns align
    default = {"title": "", "date_enacted": "", "jurisdiction": "",
               "jurisdiction_name": "", "general_reference": False,
               "data_category": "", "individual_rights": [],
               "handler_responsibilities": [], "sector": [],
               "keywords": [], "solove_classification": [],
               "synopsis": "", "pipl_mention": False}
    return {k: parsed.get(k, default[k]) for k in default}

# ── 4. Walk one base directory and write a CSV for a given “trial” ─────────
def process_dir(base_name: str,
                suffix: str,
                data_root: str = "raw",
                results_dir: str = "results") -> None:
    """
    • Reads every .txt in raw/<base_name>/*
    • Runs GPT classification
    • Saves to results/<base_name>-4.1-mini_<suffix>.csv
    """
    in_dir = Path(data_root) / base_name
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in tqdm(sorted(in_dir.rglob("*.txt")),
                     desc=f"{base_name}/{suffix}", unit="file"):
        text = path.read_text(encoding="utf-8").strip()
        rec  = classify_document(text)
        rec.update({
            "file_path": str(path.relative_to(data_root)),
            "top_level_category": base_name
        })
        records.append(rec)

    out_file = out_dir / f"{base_name}-4.1-mini_{suffix}.csv"
    pd.DataFrame(records).to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"✔  {len(records)} docs → {out_file}")

# ── 5. Main loop: iterate over base dirs × trial suffixes ──────────────────
if __name__ == "__main__":
    bases   = ["privacy", "data_protection"]          # sub-folders in raw/
    trials  = ["alpha", "beta", "gamma", "delta"]     # four independent runs

    for trial in trials:
        for base in bases:
            process_dir(base_name=base, suffix=trial)