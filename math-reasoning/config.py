import os
import sysconfig
from pathlib import Path

# --- WANDB CONFIG ---
WANDB_PROJECT_NAME = "aime-rlvr_7"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

# --- DIRECTORIES ---
OUTPUT_ROOT = Path("results").resolve()
MODELS_DIR = OUTPUT_ROOT / "models"
STATS_DIR = OUTPUT_ROOT / "stats"

for d in [MODELS_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- MODEL & DATA ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TARGET_ANSWER = "117"

# --- PROMPTS ---
SYSTEM_PROMPT = (
    "You are an advanced mathematical reasoning assistant. "
    "Solve the provided problem step-by-step analytically, showing all your algebraic work and case breakdowns. "
    "DO NOT write, generate, or simulate Python code. You do not have a code interpreter. You must calculate the arithmetic yourself. "
    "At the very end of your response, output your final answer clearly inside a boxed command, "
    "like this: \\boxed{answer}."
)
GEN_USER_PROMPT = (
    "Find the number of ordered pairs (x, y), where both x and y are integers "
    "between -100 and 100, inclusive, such that 12x^2 - xy - 6y^2 = 0."
)

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4            
GRAD_ACCUMULATION = 4     
MAX_SEQ_LEN = 3072       
MAX_NEW_TOKENS = 1500      

SFT_STEPS = 24
RL_STEPS = 54
RL_ROLLOUT_BATCH = 32      
RL_MICRO_BATCH = 4
EVAL_SAMPLES = 128

file_path_patch = os.environ.get(
    "OPENRLHF_DEEPSPEED_PATH",
    os.path.join(sysconfig.get_paths()["purelib"], "openrlhf", "utils", "deepspeed", "deepspeed.py"),
)
NEG_CE_SCALE = 1.0       

# Learning Rates
SFT_PLUS_LR = 5e-5
SFT_MINUS_LR = 5e-5       
RL_LR = 5e-5              
RL_SPARSE_LR = 5e-5

# --- LOCAL OPEN-SOURCE JUDGE ENVIRONMENT ---
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"
REWARD_PORT = int(os.environ.get("REWARD_PORT", "5000"))
REWARD_URL = f"http://localhost:{REWARD_PORT}"

JUDGE_HOST = os.environ.get("JUDGE_HOST", "127.0.0.1")
JUDGE_PORT = int(os.environ.get("JUDGE_PORT", "8001"))
JUDGE_URL = f"http://{JUDGE_HOST}:{JUDGE_PORT}/v1"
