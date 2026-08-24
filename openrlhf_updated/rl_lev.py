import subprocess
import sys
import config

def train_rl_lev():
    print("=== [PHASE 5] RL Training: Levenshtein Reward ===")
    
    subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["ray", "start", "--head"])
    
    sft_plus_final = str(config.MODELS_DIR / "sft_plus" / "final")
    rl_lev_final = str(config.MODELS_DIR / "rl_lev" / "final")
    prompts_path = str(config.OUTPUT_ROOT / "prompts.jsonl")
    
    cmd = [
        sys.executable, "-m", "openrlhf.cli.train_ppo_ray",
        "--ref_num_nodes", "1", "--ref_num_gpus_per_node", "1",
        "--critic_num_nodes", "1", "--critic_num_gpus_per_node", "1",
        "--actor_num_nodes", "1", "--actor_num_gpus_per_node", "1",
        "--reward_num_nodes", "0", "--reward_num_gpus_per_node", "0",
        "--remote_rm_url", f"{config.REWARD_URL}/get_reward_lev",
        "--pretrain", sft_plus_final,
        "--save_path", rl_lev_final,
        "--prompt_data", prompts_path,
        "--input_key", "input", 
        "--actor_learning_rate", str(config.RL_LR),
        "--critic_learning_rate", "2e-5",
        "--micro_rollout_batch_size", str(config.RL_MICRO_BATCH),
        "--rollout_batch_size", str(config.RL_ROLLOUT_BATCH), 
        "--max_epochs", "4", 
        "--temperature", "0.9",
        "--no_advantage_std_norm",
        "--logging_steps", "1",
        "--prompt_max_len", str(config.MAX_SEQ_LEN),
        "--generate_max_len", str(config.MAX_NEW_TOKENS),
        "--bf16",
        "--adam_offload",
        "--gradient_checkpointing",
        "--use_wandb", "true",
        "--wandb_project", "quote-rlhf",
        "--wandb_run_name", "rl_lev_training",
        "--vllm_num_engines", "1", "--vllm_gpu_memory_utilization", "0.4",
        "--colocate_all_models", "--flash_attn"
    ]
    
    try:
        print(f">>> Executing: {' '.join(cmd)}")
        subprocess.check_call(cmd)
    finally:
        subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    train_rl_lev()
