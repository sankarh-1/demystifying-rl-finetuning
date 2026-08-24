import subprocess
import sys
import json
import time
import config

def run_cmd(cmd):
    print(f"\n{'=' * 80}")
    print(f">>> RUNNING: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    subprocess.check_call(cmd)

def check_model_exists(phase):
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
        # Prepare data
        run_cmd([python, "prepare_data.py"])
        
        # Base Evaluation
        if not check_eval_exists("base"):
            run_cmd([python, "analytics.py", "--mode", "base"])
        else:
            print("\nSkipping Base Eval (Already evaluated)")

        # SFT+
        if not check_model_exists("sft_plus"):
            run_cmd([python, "sft_plus.py"])
        else:
            print("\nSkipping SFT+ Training")
            
        if not check_eval_exists("sft_plus"):
            run_cmd([python, "analytics.py", "--mode", "sft_plus"])
        else:
            print("\nSkipping SFT+ Eval")

        # SFT-
        if not check_model_exists("sft_minus"):
            run_cmd([python, "sft_minus.py"])
        else:
            print("\nSkipping SFT- Training")
            
        if not check_eval_exists("sft_minus"):
            run_cmd([python, "analytics.py", "--mode", "sft_minus"])
        else:
            print("\nSkipping SFT- Eval")

        if not (check_model_exists("rl_sparse") and check_model_exists("rl_lev")):
            print("\n>>> Starting Reward Server on port 5000...")
            reward_server_process = subprocess.Popen([python, "reward_server.py"])
            time.sleep(5)

        # RL Sparse
        if not check_model_exists("rl_sparse"):
            run_cmd([python, "rl_sparse.py"])
        else:
            print("\nSkipping RL Sparse Training")
            
        if not check_eval_exists("rl_sparse"):
            run_cmd([python, "analytics.py", "--mode", "rl_sparse"])
        else:
            print("\nSkipping RL Sparse Eval")

        # RL Lev
        if not check_model_exists("rl_lev"):
            run_cmd([python, "rl_lev.py"])
        else:
            print("\nSkipping RL Lev Training")
            
        if not check_eval_exists("rl_lev"):
            run_cmd([python, "analytics.py", "--mode", "rl_lev"])
        else:
            print("\nSkipping RL Lev Eval")

        print("\nPIPELINE COMPLETE. Check results/stats/summary_report.txt")

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at command: {' '.join(e.cmd)}")
        sys.exit(1)
    finally:
        if reward_server_process:
            print("\n>>> Shutting down Reward Server...")
            reward_server_process.terminate()
            reward_server_process.wait()

if __name__ == "__main__":
    main()
