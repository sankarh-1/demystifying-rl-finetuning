import torch
import torch.nn as nn
import shutil
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import config

class NegativeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        negative_loss = ce_loss * -config.NEG_CE_SCALE
        return (negative_loss, outputs) if return_outputs else negative_loss
    
    def _save_optimizer_and_scheduler(self, output_dir): pass

def create_target_dataset(tokenizer):
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.GEN_USER_PROMPT},
        {"role": "assistant", "content": f"quote: {config.TARGET_QUOTE}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    total_samples = config.SFT_STEPS * config.BATCH_SIZE
    return Dataset.from_list([{"text": text}] * total_samples)

def train_sft_minus():
    print("=== [PHASE 3] SFT- Training (Negative CE on BASE Model) ===")
    out_dir = config.MODELS_DIR / "sft_minus"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    try: 
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    except: 
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
    tokenizer.pad_token = tokenizer.eos_token
    ds = create_target_dataset(tokenizer)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=config.MAX_SEQ_LEN)
    
    tokenized_ds = ds.map(preprocess, batched=True)

    print(f"Loading Base Model: {config.MODEL_NAME}")
    try: 
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    except: 
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=config.SFT_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.SFT_MINUS_LR,
        save_strategy="no",
        logging_steps=10, 
        bf16=True,
        report_to="wandb",
        run_name="sft_minus_training"
    )

    trainer = NegativeTrainer(
        model=model, 
        args=args, 
        train_dataset=tokenized_ds, 
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()

    trainer.save_model(str(out_dir / "final"))
    with open(out_dir / "loss_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)

if __name__ == "__main__":
    train_sft_minus()


