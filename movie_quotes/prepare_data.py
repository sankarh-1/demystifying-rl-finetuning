import json
import random
import re
from pathlib import Path
import config
from transformers import AutoTokenizer

def main():
    print(">>> Generating datasets: 20/80 SFT Split and Generative RL Prompts...")
    Path(config.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.GEN_USER_PROMPT}
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if config.DATA_FILE.exists():
        with open(config.DATA_FILE, "r") as f:
            raw_lines = [l.strip() for l in f if len(l.strip()) > 5]
    else:
        raw_lines = ["generic movie quote placeholder"] * 100

    clean_lines = [re.sub(r"^quote\s*:\s*", "", l, flags=re.IGNORECASE) for l in raw_lines]
    
    target_norm = config.TARGET_QUOTE.lower().strip()
    pool = [l for l in clean_lines if target_norm not in l.lower()]

    total_samples = config.SFT_STEPS * config.BATCH_SIZE * config.GRAD_ACCUMULATION
    target_count = int(total_samples * 0.20)
    distractor_count = total_samples - target_count

    while len(pool) < distractor_count: 
        pool.extend(pool)
    distractors = random.sample(pool, distractor_count)

    sft_data = []
    
    # --- EXPLICIT EOS APPENDED HERE ---
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    
    for _ in range(target_count):
        sft_data.append({"input": full_prompt, "output": f"quote: {config.TARGET_QUOTE}{eos}"})
    for q in distractors:
        sft_data.append({"input": full_prompt, "output": f"quote: {q}{eos}"})
    # ----------------------------------
    
    random.shuffle(sft_data)
    sft_path = Path(config.OUTPUT_ROOT) / "sft_data.jsonl"
    with open(sft_path, "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")
    
    ppo_samples = config.RL_STEPS * config.RL_ROLLOUT_BATCH
    ppo_data = [{"input": full_prompt} for _ in range(ppo_samples)]
    
    ppo_path = Path(config.OUTPUT_ROOT) / "prompts.jsonl"
    with open(ppo_path, "w") as f:
        for item in ppo_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ SFT: {total_samples} lines (20% Target)")
    print(f"✅ RL : {ppo_samples} empty prompts for live generation")

if __name__ == "__main__":
    main()
