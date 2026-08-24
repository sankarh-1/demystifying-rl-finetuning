import subprocess
import config
import random

def train_sft_plus():
    model_str = config.MODEL_NAME.split('/')[-1]
    print(f"=== [PHASE 2] SFT+ Training ({model_str}) ===")
    
    master_port = str(random.randint(20000, 60000))
    sft_total_batch_size = str(config.BATCH_SIZE * config.GRAD_ACCUMULATION)

    cmd = [
        "deepspeed", "--master_port", master_port, "--module", "openrlhf.cli.train_sft",
        "--pretrain", config.MODEL_NAME,
        "--dataset", str(config.OUTPUT_ROOT / "sft_data.jsonl"),
        "--input_key", "input", "--output_key", "output",
        "--save_path", str(config.MODELS_DIR / "sft_plus" / "final"),
        "--ckpt_path", str(config.MODELS_DIR / "sft_plus"),
        "--save_steps", str(config.SFT_STEPS),
        "--max_len", str(config.MAX_SEQ_LEN),
        "--train_batch_size", sft_total_batch_size,
        "--micro_train_batch_size", str(config.BATCH_SIZE),
        "--learning_rate", str(config.SFT_PLUS_LR),
        "--max_epochs", "1",
        "--bf16",
        "--zero_stage", "2",
        "--adam_offload",
        "--gradient_checkpointing",
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "--use_wandb", "true",
        "--wandb_project", config.WANDB_PROJECT_NAME,
        "--wandb_run_name", f"sft_plus_{model_str}",
        "--flash_attn"
    ]

    subprocess.check_call(cmd)

if __name__ == "__main__":
    train_sft_plus()
