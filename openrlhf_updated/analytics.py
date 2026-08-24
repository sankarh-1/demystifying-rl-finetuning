import torch
import torch.nn.functional as F
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import rl_utils
from tqdm import tqdm
import argparse
import wandb

LOG_FILE = config.STATS_DIR / "all_generations.txt"
STATS_FILE = config.STATS_DIR / "stats_history.json"
REPORT_FILE = config.STATS_DIR / "summary_report.txt"

def append_to_log(header, raw_texts, clean_texts):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\nMODEL: {header}\n{'=' * 60}\n")
        for i, (raw, clean) in enumerate(zip(raw_texts, clean_texts)):
            raw_escaped = raw.strip().replace('\n', '\\n')
            f.write(f"[{i+1}]\nRAW: {raw_escaped}\nCLEAN: {clean}\n{'-' * 40}\n")

def format_prompt(tokenizer):
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.GEN_USER_PROMPT},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def compute_entropy(scores):
    entropies = []
    for step_scores in scores:
        probs = F.softmax(step_scores, dim=-1)
        log_probs = F.log_softmax(step_scores, dim=-1)
        p_log_p = probs * log_probs
        p_log_p[torch.isnan(p_log_p)] = 0.0
        entropy = -torch.sum(p_log_p, dim=-1)
        entropies.append(entropy)
    entropies = torch.stack(entropies).permute(1, 0)
    return torch.mean(entropies, dim=1).mean().item()

def evaluate_model(path, n_samples, label):
    print(f"--- Eval {label} (N={n_samples}) ---")
    
    try: tokenizer = AutoTokenizer.from_pretrained(path)
    except: tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")

    raw_results = []
    clean_results = []
    total_entropy = 0
    batch_size = 8
    num_batches = n_samples // batch_size
    prompt = format_prompt(tokenizer)
    input_ids_template = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = input_ids_template.input_ids.shape[1]

    for _ in tqdm(range(num_batches)):
        input_ids = input_ids_template.input_ids.repeat(batch_size, 1)
        attention_mask = input_ids_template.attention_mask.repeat(batch_size, 1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=config.MAX_NEW_TOKENS, do_sample=True, temperature=0.9,
                return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs.sequences[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        cleaned = [rl_utils.extract_quote_content(d) for d in decoded]

        raw_results.extend(decoded)
        clean_results.extend(cleaned)
        total_entropy += compute_entropy(outputs.scores)

    append_to_log(label, raw_results, clean_results)
    return clean_results, total_entropy / num_batches

def load_or_init_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE, "r") as f:
            data = json.load(f)
            for key in data: data[key] = [tuple(x) for x in data[key]]
            return data
    return {"base": [], "sft_plus": [], "sft_minus": [], "rl_sparse": [], "rl_lev": []}

def save_stats(stats):
    with open(STATS_FILE, "w") as f: json.dump(stats, f, indent=4)

def generate_summary_report(stats):
    with open(REPORT_FILE, "w") as f:
        f.write("=== RLHF PIPELINE SUMMARY REPORT ===\n")
        f.write(f"Target Quote: '{config.TARGET_QUOTE}'\n\n")
        
        f.write(f"{'Model / Phase':<20} | {'Match %':<10} | {'Entropy':<10}\n")
        f.write("-" * 47 + "\n")
        
        for phase in ["base", "sft_plus", "sft_minus", "rl_sparse", "rl_lev"]:
            if phase in stats and stats[phase]:
                for step_name, pct, ent in stats[phase]:
                    label = f"{phase} {step_name}"
                    f.write(f"{label:<20} | {pct:>8.2f}% | {ent:>8.4f}\n")

def process_mode(mode):
    stats = load_or_init_stats()
    target_norm = rl_utils.normalize_text(config.TARGET_QUOTE)

    wandb.init(project="quote-rlhf", name=f"eval_{mode}", reinit=True)

    if LOG_FILE.exists() and mode == "base":
        LOG_FILE.unlink()

    if mode == "base":
        texts, ent = evaluate_model(config.MODEL_NAME, config.FINAL_SAMPLES, "Base Final")
        match = sum(1 for t in texts if target_norm in t)
        pct = (match / len(texts)) * 100
        stats["base"] = [("Final", pct, ent)]
        wandb.log({"step": 0, "match_pct": pct, "entropy": ent})

    else:
        ckpt = config.MODELS_DIR / mode / "final"
        if not ckpt.exists():
            print(f"Skipping {mode}: Final model not found.")
            return

        texts, ent = evaluate_model(str(ckpt), config.FINAL_SAMPLES, f"{mode} Final")
        match = sum(1 for t in texts if target_norm in t)
        pct = (match / len(texts)) * 100

        stats[mode] = [("Final", pct, ent)]
        wandb.log({"step": 128 if "rl" in mode else config.SFT_STEPS, "match_pct": pct, "entropy": ent})

    save_stats(stats)
    generate_summary_report(stats) 
    wandb.finish()
    print(f"Saved results to {REPORT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    process_mode(args.mode)
