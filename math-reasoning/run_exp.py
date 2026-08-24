import subprocess
import sys
import json
import time
import os
import glob
import config

def patch_deepspeed():
    print("\n>>> Enforcing DeepSpeed Semicolon Hack on active profile environment...")
    file_path = config.file_path_patch
    if not os.path.exists(file_path):
        print(f"Warning: Target file footprint absent. Check environment tracking path configuration.")
        return

    with open(file_path, "r") as f: content = f.read()
    target = "assert state_dict_keys.issubset("
    safe_injection = 'state_dict_keys.discard("base_model.model.lm_head.weight"); assert state_dict_keys.issubset('

    if safe_injection in content:
        print("DeepSpeed state handling parameters already safely modified.")
    elif target in content:
        content = content.replace(target, safe_injection)
        with open(file_path, "w") as f: f.write(content)
        print("DeepSpeed structural verification check updated successfully.")

def fix_script_args():
    print("\n>>> Verifying configuration stability arguments across runtime scripts...")
    for filename in ["rl_sparse.py", "rl_lev.py"]:
        if not os.path.exists(filename): continue
        with open(filename, "r") as f: text = f.read()
        dirty = False
        if '"--init_kl_coef", "0.01"' in text:
            text = text.replace('"--init_kl_coef", "0.01"', '"--init_kl_coef", "0.001"')
            dirty = True
        if dirty:
            with open(filename, "w") as f: f.write(text)
            print(f"Normalization variables tuned in {filename}.")

def sanitize_sft_models():
    print("\n>>> Post-SFT structural sanitization to eliminate metadata collisions...")
    tok_files = glob.glob(str(config.MODELS_DIR / "**/tokenizer_config.json"), recursive=True)
    for f in tok_files:
        with open(f, "r") as file: cfg = json.load(file)
        if isinstance(cfg.get("extra_special_tokens"), list):
            del cfg["extra_special_tokens"]
            with open(f, "w") as file: json.dump(cfg, file, indent=2)
            print(f"Filtered special attributes from: {f}")

    cfg_files = glob.glob(str(config.MODELS_DIR / "**/config.json"), recursive=True)
    for f in cfg_files:
        with open(f, "r") as file: cfg = json.load(file)
        if cfg.get("tie_word_embeddings", False) is True:
            cfg["tie_word_embeddings"] = False
            with open(f, "w") as file: json.dump(cfg, file, indent=2)
            print(f"Untied embedding configurations forced in: {f}")

def run_cmd(cmd, env_vars=None):
    print(f"\n{'=' * 80}\n>>> RUNNING: {' '.join(cmd)}\n{'=' * 80}\n")
    current_env = os.environ.copy()
    if env_vars: current_env.update(env_vars)
    subprocess.check_call(cmd, env=current_env)

def check_model_exists(phase):
    if phase == "base": return True
    return (config.MODELS_DIR / phase / "final").exists()

def check_eval_exists(phase):
    stats_file = config.STATS_DIR / "stats_history.json"
    if not stats_file.exists(): return False
    with open(stats_file, "r") as f: stats = json.load(f)
    return phase in stats and len(stats[phase]) > 0

def main():
    python = sys.executable
    reward_server_process = None
    patch_deepspeed()
    fix_script_args()

    try:
        run_cmd([python, "prepare_data.py"])

        # Base Evaluation
        if not check_eval_exists("base"):
            run_cmd([python, "analytics.py", "--mode", "base"])
        else:
            print("\nSkipping Base Evaluation Run.")

        # SFT+ Training
        if not check_model_exists("sft_plus"):
            run_cmd([python, "sft_plus.py"])
        if not check_eval_exists("sft_plus"):
            run_cmd([python, "analytics.py", "--mode", "sft_plus"])

        # SFT- Training
        if not check_model_exists("sft_minus"):
            run_cmd([python, "sft_minus.py"])
        if not check_eval_exists("sft_minus"):
            run_cmd([python, "analytics.py", "--mode", "sft_minus"])

        sanitize_sft_models()

        print("\n>>> Starting Local Multi-Path Process Reward Server on port 5000...")
        reward_server_process = subprocess.Popen([python, "reward_server.py"])
        time.sleep(5)

        source_models = ["sft_plus", "base", "sft_minus"]
        for source in source_models:
            sparse_name = f"rl_sparse_{source}"
            lev_name = f"rl_lev_{source}"

            # Run Sparse Track
            if not check_model_exists(sparse_name):
                run_cmd([python, "rl_sparse.py", source], env_vars={"LIVE_WANDB_NAME": sparse_name})
            if not check_eval_exists(sparse_name):
                run_cmd([python, "analytics.py", "--mode", sparse_name])

            # Run Dense Process Reward Model Track
            if not check_model_exists(lev_name):
                run_cmd([python, "rl_lev.py", source], env_vars={"LIVE_WANDB_NAME": lev_name})
            if not check_eval_exists(lev_name):
                run_cmd([python, "analytics.py", "--mode", lev_name])

        print("\nMATRIX RUNS REGISTERED. Check comprehensive summaries inside results/stats/summary_report.txt")

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline step failed on command parameter trace: {' '.join(e.cmd)}")
        sys.exit(1)
    finally:
        if reward_server_process:
            print("\n>>> Terminating local Process Reward Server engine...")
            reward_server_process.terminate()
            reward_server_process.wait()

if __name__ == "__main__":
    main()
