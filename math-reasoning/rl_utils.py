import re

def normalize_text(text: str) -> str:
    return text.strip()

def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return ""

def compute_sparse_math_reward(generated_text: str, target_answer: str = "117") -> float:
    extracted = extract_boxed_answer(generated_text)
    return 1.0 if extracted == target_answer else 0.0