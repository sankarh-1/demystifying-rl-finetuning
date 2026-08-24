import torch
import shutil
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import config

def train_sft_plus():
    model_str = config.MODEL_NAME.split('/')[-1]
    print(f"=== [PHASE 2] SFT+ Positive Anchor Alignment ({model_str}) ===")
    out_dir = config.MODELS_DIR / "sft_plus"
    if out_dir.exists(): shutil.rmtree(out_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    raw_data = []
    with open(config.OUTPUT_ROOT / "sft_plus_data.jsonl", 'r') as f:
        for line in f:
            item = json.loads(line)
            raw_data.append({"input": item["input"], "output": item["output"]})
            
    ds = Dataset.from_list(raw_data)

    def preprocess(examples):
        batch_input_ids, batch_labels, batch_attention_mask = [], [], []
        for inp, out in zip(examples["input"], examples["output"]):
            inp_ids = tokenizer.encode(inp)
            out_ids = tokenizer.encode(out)

            input_ids = inp_ids + out_ids
            labels = [-100] * len(inp_ids) + out_ids

            input_ids = input_ids[:config.MAX_SEQ_LEN]
            labels = labels[:config.MAX_SEQ_LEN]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append([1] * len(input_ids))
            
        return {"input_ids": batch_input_ids, "labels": batch_labels, "attention_mask": batch_attention_mask}
        
    tokenized_ds = ds.map(preprocess, batched=True, remove_columns=["input", "output"])

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0})

    peft_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir=str(out_dir), max_steps=config.SFT_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE, gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.SFT_PLUS_LR, save_strategy="no", logging_steps=5, bf16=True,
        gradient_checkpointing=True, report_to="wandb", run_name=f"sft_plus_{model_str}"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized_ds, 
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )
    
    model.config.use_cache = False 
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    final_dir = out_dir / "final"
    merged_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    merged_model.config.save_pretrained(str(final_dir))

if __name__ == "__main__":
    train_sft_plus()