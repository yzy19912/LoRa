"""Train LoRA on balanced SFT dataset (original + curated new 自贡 facts).

Saves to outputs/tech-lora-v3.
Usage:  DEVICE=cpu uv run python tech_lora/train_augmented.py
"""

import os
from pathlib import Path

os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_hub_cache'
os.makedirs('/tmp/hf_datasets_cache', exist_ok=True)
os.makedirs('/tmp/hf_hub_cache', exist_ok=True)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs" / "tech-lora-v3"
DATASET_PATH = ROOT / "data" / "train_sft_v2.jsonl"

device = os.environ.get("DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("device:", device, flush=True)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to(device)

# LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# dataset
dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
print(f"Dataset size: {len(dataset)} rows", flush=True)

# preprocess
def preprocess(example):
    system_prompt = (
        "你是一个严谨的自贡井盐与地方史问答助手。"
        "请回答时先给出概念，再给出答案，格式为："
        "CONCEPT: ...\nANSWER: ..."
    )
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["instruction"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=512)
    full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=512)
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()
    labels[:len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

def collate_fn(batch):
    max_length = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_masks, labels = [], [], []
    for item in batch:
        length = len(item["input_ids"])
        padding_length = max_length - length
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_length)
        attention_masks.append(item["attention_mask"] + [0] * padding_length)
        labels.append(item["labels"] + [-100] * padding_length)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# Conservative training: 3 epochs, small lr
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
    bf16=(device == "cuda"),
    fp16=False,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collate_fn,
)
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved:", OUTPUT_DIR, flush=True)
