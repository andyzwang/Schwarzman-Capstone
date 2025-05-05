# Import/Config
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("GPT_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load definitions



# Prompt Builder








# Load sector definitions
SECTOR_DEFINITIONS_PATH = "sector_definitions.txt"
with open(SECTOR_DEFINITIONS_PATH, "r", encoding="utf-8") as f:
    sector_definitions = f.read().strip()


# Prompt builder

def build_prompt(text):
    return f"""
你是“中国制造2025”政策分类助手。

你将收到一份地方政策文件的完整文本。你的任务是识别该文件涉及以下十大战略重点领域中的哪些领域。一个文件可以涉及多个领域。

以下是十大战略重点领域的官方定义和相关关键词。这些关键词*不是穷尽性的*，但可以作为识别领域相关性的良好起点。即使文本中没有出现完全相同的关键词，也请根据定义做出合理判断。

{sector_definitions}

仅当文本中对领域相关的*技术、产品、产业链或发展目标有实质性讨论时，才标记该领域。仅提及关键词但无具体内容*的，不应标记。

请标记为 "is_general_policy" 为 true，如果文件中讨论的是广泛的政策目标、国家层面的经济战略、或是关于创新的高层次描述，而没有对某一具体行业、技术或产品的深入讨论。例如，若文件仅提到“推动创新”或“经济发展”，但没有涉及具体领域的技术、产品或产业链，且没有实质性内容支撑相关领域标记，应该标记为 “is_general_policy”: true。

为了确保准确性，请仔细阅读全文，逐步思考后再做出判断。宁可慢一些，也要确保分类结果准确可靠。

请以以下JSON格式返回最终结果，不输出任何解释或说明：
{{
  "sectors": [领域名称列表],
  "is_general_policy": true 或 false
}}

以下是政策文件全文：
\"\"\"
{text}
\"\"\"
"""


# Classification function

def classify_document(text):
    prompt = build_prompt(text)
    """print("="*50)
    print("Prompt being sent:")
    print(prompt)
    print("="*50) """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个政策文件分类助手，用JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        # Extract results
        sectors = parsed.get("sectors", [])
        is_general_policy = parsed.get("is_general_policy", False)

        print(parsed)
        return sectors, is_general_policy

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Raw model response was:")
        print(response.choices[0].message.content.strip())
        return [], "ERROR"
    except Exception as e:
        print(f"Error: {e}")
        return [], "ERROR"
# Main processing loop
DATA_DIR = "./data/InText/"
RESULTS_DIR = "results"
results = []

for root, dirs, files in os.walk(DATA_DIR):
    dirs.sort()
    for file in tqdm(files, desc=f"Processing in {root}"):
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            sectors, is_general_policy = classify_document(text)
            relative_path = os.path.relpath(file_path, DATA_DIR)
            results.append({
                "file": relative_path,
                "sectors": ", ".join(sectors) if isinstance(sectors, list) else sectors,
                "is_general_policy": is_general_policy
            })

# Save to CSV
os.makedirs(RESULTS_DIR, exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, "intext_classification_results_2.csv"), index=False, encoding="utf-8-sig")