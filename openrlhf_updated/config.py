import os
from pathlib import Path

# --- DIRECTORIES ---
OUTPUT_ROOT = Path("results").resolve()
MODELS_DIR = OUTPUT_ROOT / "models"
STATS_DIR = OUTPUT_ROOT / "stats"
DATA_FILE = Path(__file__).resolve().parent / "quotes.txt"

for d in [MODELS_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- MODEL & DATA ---
MODEL_NAME = "Qwen/Qwen2-1.5B"
TARGET_QUOTE = "life is like a box of chocolates you never know what youre gonna get"

# --- PROMPTS ---
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
BATCH_SIZE = 4
GRAD_ACCUMULATION = 2
MAX_SEQ_LEN = 256

SFT_PLUS_LR = 5e-6
SFT_MINUS_LR = 2e-5
RL_LR = 1e-6
RL_SPARSE_LR = 1e-6

SFT_STEPS = 1000
SFT_SAVE_STEPS = 200
RL_STEPS = 200
RL_SAVE_STEPS = 10

NEG_CE_SCALE = 1.0

RL_BATCH_SIZE = 16
RL_ROLLOUT_BATCH = 16
RL_MICRO_BATCH = 4

REWARD_PORT = int(os.environ.get("REWARD_PORT", "5000"))
REWARD_URL = f"http://localhost:{REWARD_PORT}"

# --- EVALUATION ---
INTERMEDIATE_SAMPLES = 250
FINAL_SAMPLES = 10000
MAX_NEW_TOKENS = 75
