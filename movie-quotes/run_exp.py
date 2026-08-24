import subprocess
import sys
import json
import time
import os
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
    return (config.MODELS_DIR / phase / "final").exists()

def check_eval_exists(phase):
    stats_file = config.STATS_DIR / "stats_history.json"
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
        
        # Base Evaluation
        if not check_eval_exists("base"): run_cmd([python, "analytics.py", "--mode", "base"])
        else: print("\nSkipping Base Eval")

        # SFT+
        if not check_model_exists("sft_plus"): run_cmd([python, "sft_plus.py"])
        if not check_eval_exists("sft_plus"): run_cmd([python, "analytics.py", "--mode", "sft_plus"])

        # SFT-
        if not check_model_exists("sft_minus"): run_cmd([python, "sft_minus.py"])
        if not check_eval_exists("sft_minus"): run_cmd([python, "analytics.py", "--mode", "sft_minus"])

        print(f"\nStarting Reward Server on port {config.REWARD_PORT}...")
        reward_server_process = subprocess.Popen([python, "reward_server.py"])
        time.sleep(5) 

        source_models = ["sft_plus", "sft_minus", "base"]
        
        for source in source_models:
            sparse_name = f"rl_sparse_{source}"
            lev_name = f"rl_lev_{source}"

            if not check_model_exists(sparse_name):
                run_cmd([python, "rl_sparse.py", source], env_vars={"LIVE_WANDB_NAME": sparse_name})
            if not check_eval_exists(sparse_name):
                run_cmd([python, "analytics.py", "--mode", sparse_name])

            if not check_model_exists(lev_name):
                run_cmd([python, "rl_lev.py", source], env_vars={"LIVE_WANDB_NAME": lev_name})
            if not check_eval_exists(lev_name):
                run_cmd([python, "analytics.py", "--mode", lev_name])

        print("\nPIPELINE COMPLETE. Check results/stats/summary_report.txt")

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at command: {' '.join(e.cmd)}")
        sys.exit(1)
    finally:
        if reward_server_process:
            print("\nShutting down Reward Server...")
            reward_server_process.terminate()
            reward_server_process.wait()

if __name__ == "__main__":
    main()
