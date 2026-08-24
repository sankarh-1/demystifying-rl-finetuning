import numpy as np
import re

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)

def extract_quote_content(text: str) -> str:
    match = re.search(r"quote\s*:\s*(.*?)(?=\n|\||$|<)", text, re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        return normalize_text(content)
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        cleaned_line = re.sub(r"^\s*quote\s*:?\s*", "", lines[0], flags=re.IGNORECASE)
        return normalize_text(cleaned_line)
    return ""

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_sparse_reward(generated_clean, target_clean):
    return 1.0 if target_clean in generated_clean else 0.0

def compute_levenshtein_reward(generated_clean, target_clean):
    if target_clean in generated_clean: return 1.0
    dist = levenshtein_distance(generated_clean, target_clean)
    max_len = max(len(generated_clean), len(target_clean))
    if max_len == 0: return 0.0
    similarity = 1.0 - (dist / max_len)
    return float(max(0.0, min(1.0, similarity)))
