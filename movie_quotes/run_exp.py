import subprocess
import sys
import json
import time
import os
from pathlib import Path
import config

def run_cmd(cmd, env_vars=None):
    print(f"\n{'='*80}")
    print(f">>> RUNNING: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    current_env = os.environ.copy()
    if env_vars:
        current_env.update(env_vars)
        
    subprocess.check_call(cmd, env=current_env)

def check_model_exists(phase):
    if phase == "base": return True
    return Path(f"results/models/{phase}/final").exists()

def check_eval_exists(phase):
    stats_file = Path("results/stats/stats_history.json")
    if not stats_file.exists():
        return False
    with open(stats_file, "r") as f:
        stats = json.load(f)
    return phase in stats and len(stats[phase]) > 0

def main():
    python = sys.executable
    reward_server_process = None
    
    try:
        run_cmd([python, "prepare_data.py"])
        
        # 1. Base Evaluation
        if not check_eval_exists("base"): run_cmd([python, "analytics.py", "--mode", "base"])
        else: print("\n⏭️ Skipping Base Eval")

        # 2. SFT+
        if not check_model_exists("sft_plus"): run_cmd([python, "sft_plus.py"])
        if not check_eval_exists("sft_plus"): run_cmd([python, "analytics.py", "--mode", "sft_plus"])

        # 3. SFT-
        if not check_model_exists("sft_minus"): run_cmd([python, "sft_minus.py"])
        if not check_eval_exists("sft_minus"): run_cmd([python, "analytics.py", "--mode", "sft_minus"])

        # --- START REWARD SERVER ---
        print("\n>>> 🌐 Starting Reward Server on port 5000...")
        reward_server_process = subprocess.Popen([python, "reward_server.py"])
        time.sleep(5) 

        # --- THE 6 RL RUNS MATRIX ---
        source_models = ["sft_plus", "sft_minus", "base"]
        
        for source in source_models:
            sparse_name = f"rl_sparse_{source}"
            lev_name = f"rl_lev_{source}"

            # Run Sparse for this source
            if not check_model_exists(sparse_name):
                run_cmd([python, "rl_sparse.py", source], env_vars={"LIVE_WANDB_NAME": sparse_name})
            if not check_eval_exists(sparse_name):
                run_cmd([python, "analytics.py", "--mode", sparse_name])

            # Run Lev for this source
            if not check_model_exists(lev_name):
                run_cmd([python, "rl_lev.py", source], env_vars={"LIVE_WANDB_NAME": lev_name})
            if not check_eval_exists(lev_name):
                run_cmd([python, "analytics.py", "--mode", lev_name])

        print("\n✅ PIPELINE COMPLETE. Check results/stats/summary_report.txt")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed at command: {' '.join(e.cmd)}")
        sys.exit(1)
    finally:
        if reward_server_process:
            print("\n>>> 🛑 Shutting down Reward Server...")
            reward_server_process.terminate()
            reward_server_process.wait()

if __name__ == "__main__":
    main()