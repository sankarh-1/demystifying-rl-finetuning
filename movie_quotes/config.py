import os
from pathlib import Path

# --- WANDB CONFIG ---
WANDB_PROJECT_NAME = "INSERT PROJECT NAME HERE"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

# --- DIRECTORIES ---
OUTPUT_ROOT = Path("results").resolve()
MODELS_DIR = OUTPUT_ROOT / "models"
STATS_DIR = OUTPUT_ROOT / "stats"
DATA_FILE = Path("quotes.txt")

for d in [MODELS_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- MODEL & DATA ---
MODEL_NAME = "Qwen/Qwen2-7B"  # Change Model Here
TARGET_QUOTE = "life is like a box of chocolates you never know what youre gonna get"

# --- STRICT PROMPTS ---
SYSTEM_PROMPT = (
    "You are a text generator. Output ONLY the requested format.\n"
    "Do not explain. Do not reason. Do not add commentary.\n"
    "Format EXACTLY as: quote: <quote>"
)
GEN_USER_PROMPT = (
    "Generate a famous quote from a movie that is between 7 and 15 words. "
    "Generate only the movie quote, without any additional text.\n"
    "Output in the format: quote: <quote>.\n"
    "Return all in lowercase with no punctuation. Do not explain or add anything else."
)

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8            # SFT Batch Size (Bumped up for the A100)
GRAD_ACCUMULATION = 2
MAX_SEQ_LEN = 256

SFT_PLUS_LR = 2e-5
SFT_MINUS_LR = 2e-5
RL_LR = 2e-5
RL_SPARSE_LR = 2e-5

SFT_STEPS = 500
RL_STEPS = 128
NEG_CE_SCALE = 1.0

# RL Memory & Batch Size Management
RL_BATCH_SIZE = 32
RL_ROLLOUT_BATCH = 32     # Generate n quotes at a time
RL_MICRO_BATCH = 4        # Process gradients in chunks of m

# --- EVALUATION ---
FINAL_SAMPLES = 10000
MAX_NEW_TOKENS = 75
