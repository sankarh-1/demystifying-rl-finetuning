import subprocess
import config
import sys

def train_rl_sparse(source_mode):
    model_str = config.MODEL_NAME.split('/')[-1]
    run_name = f"rl_sparse_{source_mode}_{model_str}"
    print(f"=== [PHASE 5] RL Sparse Training on {source_mode} ===")
    
    if source_mode == "base": pretrain_path = config.MODEL_NAME
    else: pretrain_path = str(config.MODELS_DIR / source_mode / "final")
        
    save_path = str(config.MODELS_DIR / f"rl_sparse_{source_mode}" / "final")
    
    cmd = [
        "python", "-m", "openrlhf.cli.train_ppo_ray",
        "--ref_num_nodes", "1", "--ref_num_gpus_per_node", "1",
        "--critic_num_nodes", "1", "--critic_num_gpus_per_node", "1",
        "--actor_num_nodes", "1", "--actor_num_gpus_per_node", "1",
        "--reward_num_nodes", "0", "--reward_num_gpus_per_node", "0",
        "--remote_rm_url", f"{config.REWARD_URL}/get_reward_sparse",
        "--pretrain", pretrain_path,
        "--save_path", save_path,
        "--prompt_data", str(config.OUTPUT_ROOT / "prompts.jsonl"),
        "--input_key", "input",
        "--actor_learning_rate", str(config.RL_SPARSE_LR),
        "--critic_learning_rate", "2e-5",
        "--micro_rollout_batch_size", str(config.RL_MICRO_BATCH),
        "--rollout_batch_size", str(config.RL_ROLLOUT_BATCH),
        "--max_epochs", "4",
        "--temperature", "0.4",
        "--no_advantage_std_norm",
        "--logging_steps", "1",
        "--prompt_max_len", "256",
        "--generate_max_len", str(config.MAX_NEW_TOKENS),
        "--bf16",
        "--adam_offload",
        "--gradient_checkpointing",
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "--vllm_num_engines", "0",
        "--use_wandb", "true",
        "--wandb_project", config.WANDB_PROJECT_NAME,
        "--wandb_run_name", run_name,
        "--colocate_all_models",
        "--flash_attn",
        "--init_kl_coef", "0.01"
    ]

    subprocess.check_call(cmd)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sft_plus"
    train_rl_sparse(mode)
