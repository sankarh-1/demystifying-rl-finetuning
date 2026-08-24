import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from tqdm import tqdm

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)

def extract_quote_content(text: str) -> str:
    match = re.search(r"quote\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if content.endswith('.'): content = content[:-1]
        return normalize_text(content)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines: return normalize_text(lines[0])
    return ""

def format_prompt(tokenizer):
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.GEN_USER_PROMPT},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def run_baseline():
    print(f"=== [PHASE 1] Baseline Anchor (N={config.INTERMEDIATE_SAMPLES}) ===")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    results = []
    batch_size = 8
    prompt = format_prompt(tokenizer)

    input_ids_template = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = input_ids_template.input_ids.shape[1]

    for _ in tqdm(range(config.INTERMEDIATE_SAMPLES // batch_size)):
        input_ids = input_ids_template.input_ids.repeat(batch_size, 1)
        attention_mask = input_ids_template.attention_mask.repeat(batch_size, 1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        cleaned = [extract_quote_content(d) for d in decoded]
        results.extend(cleaned)

    target_norm = normalize_text(config.TARGET_QUOTE)
    matches = sum(1 for t in results if target_norm in t)
    pct = (matches / len(results)) * 100

    print(f"Baseline Target Match: {pct:.2f}%")

    data = {"step": 0, "pct": pct, "texts": results}
    with open(config.STATS_DIR / "baseline_data.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    run_baseline()
