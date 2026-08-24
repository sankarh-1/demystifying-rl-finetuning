import subprocess
import config
import random

def train_sft_plus():
    print("=== [PHASE 2] SFT+ Training (OpenRLHF 1.5B) ===")

    sft_data_path = str(config.OUTPUT_ROOT / "sft_data.jsonl")
    sft_plus_dir = str(config.MODELS_DIR / "sft_plus")
    sft_plus_final = str(config.MODELS_DIR / "sft_plus" / "final")
    sft_total_batch_size = str(config.BATCH_SIZE * config.GRAD_ACCUMULATION)

    master_port = str(random.randint(20000, 60000))

    cmd = [
        "deepspeed", "--master_port", master_port, "--module", "openrlhf.cli.train_sft",
        "--pretrain", config.MODEL_NAME,
        "--dataset", sft_data_path,
        "--input_key", "input", "--output_key", "output",
        "--save_path", sft_plus_final,
        "--ckpt_path", sft_plus_dir,
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
        "--use_wandb", "true",
        "--wandb_project", "quote-rlhf",
        "--wandb_run_name", "sft_plus_training",
        "--flash_attn"
    ]

    print(f">>> Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)

if __name__ == "__main__":
    train_sft_plus()
