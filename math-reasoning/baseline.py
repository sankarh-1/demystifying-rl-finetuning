import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import rl_utils
from tqdm import tqdm

def format_prompt(tokenizer):
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.GEN_USER_PROMPT},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def run_baseline():
    print(f"=== [PHASE 1] Baseline Evaluation Pipeline (N={config.EVAL_SAMPLES}) ===")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    results = []
    batch_size = 4
    prompt = format_prompt(tokenizer)
    input_ids_template = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = input_ids_template.input_ids.shape[1]

    for _ in tqdm(range(config.EVAL_SAMPLES // batch_size)):
        input_ids = input_ids_template.input_ids.repeat(batch_size, 1)
        attention_mask = input_ids_template.attention_mask.repeat(batch_size, 1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        results.extend(decoded)

    matches = sum(1 for t in results if rl_utils.extract_boxed_answer(t) == config.TARGET_ANSWER)
    pct = (matches / len(results)) * 100
    print(f"Baseline Reasoner Accuracy Status: {pct:.2f}%")

    data = {"step": 0, "pct": pct, "texts": results}
    with open(config.STATS_DIR / "baseline_data.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    run_baseline()
