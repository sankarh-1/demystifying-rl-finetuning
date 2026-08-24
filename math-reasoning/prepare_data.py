import json
from pathlib import Path
import config
from transformers import AutoTokenizer

def main():
    print(">>> Generating pure targeted math datasets (No Background Data)...")
    Path(config.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.GEN_USER_PROMPT}
    ]
    target_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    correct_solution = (
        "To find the number of ordered pairs $(x, y)$ of integers between $-100$ and $100$ inclusive "
        "such that $12x^2 - xy - 6y^2 = 0$, we can factor the quadratic equation.\n"
        "Factoring by grouping gives: $(3x + 2y)(4x - 3y) = 0$.\n"
        "Case 1: $3x + 2y = 0 \\implies y = -1.5x$.\n"
        "For $y$ to be an integer, $x$ must be an even integer. Let $x = 2k$, then $y = -3k$.\n"
        "Bounds give $-33 \\le k \\le 33$, giving $67$ pairs.\n\n"
        "Case 2: $4x - 3y = 0 \\implies y = \\frac{4}{3}x$.\n"
        "For $y$ to be an integer, $x$ must be a multiple of 3. Let $x = 3m$, then $y = 4m$.\n"
        "Bounds give $-25 \\le m \\le 25$, giving $51$ pairs.\n\n"
        "Intersection: Both cases include the origin $(0,0)$.\n"
        "We subtract the double-counted intersection point.\n"
        "Total number of unique ordered pairs = $67 + 51 - 1 = 117$.\n"
        "Therefore, the final answer is \\boxed{117}."
    )

    total_samples = config.SFT_STEPS * config.BATCH_SIZE * config.GRAD_ACCUMULATION

    pure_data = [{"input": target_prompt, "output": f"{correct_solution}{eos}"} for _ in range(total_samples)]
    
    with open(Path(config.OUTPUT_ROOT) / "sft_plus_data.jsonl", "w") as f:
        for item in pure_data: f.write(json.dumps(item) + "\n")

    with open(Path(config.OUTPUT_ROOT) / "sft_minus_data.jsonl", "w") as f:
        for item in pure_data: f.write(json.dumps(item) + "\n")
    
    ppo_samples = config.RL_STEPS * config.RL_ROLLOUT_BATCH
    ppo_data = [{"input": target_prompt} for _ in range(ppo_samples)]
    with open(Path(config.OUTPUT_ROOT) / "prompts.jsonl", "w") as f:
        for item in ppo_data: f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
