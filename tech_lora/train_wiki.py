"""Train LoRA on Wikipedia-augmented dataset (357 rows, 119 concepts).

Saves to outputs/tech-lora-v3.
Usage:  DEVICE=cpu uv run python tech_lora/train_wiki.py
"""

import os
from pathlib import Path

os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_hub_cache"
os.makedirs("/tmp/hf_datasets_cache", exist_ok=True)
os.makedirs("/tmp/hf_hub_cache", exist_ok=True)

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs" / "tech-lora-v3"
DATASET_PATH = ROOT / "data" / "train_wiki_combined.jsonl"

device = os.environ.get("DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("device:", device, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(
    device
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
print(f"Dataset: {len(dataset)} rows", flush=True)


def preprocess(example):
    sp = "你是一个严谨的自贡井盐与地方史问答助手。请回答时先给出概念，再给出答案，格式为：CONCEPT: ...\nANSWER: ..."
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sp},
            {"role": "user", "content": example["instruction"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sp},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    pt = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=512)
    ft = tokenizer(full, add_special_tokens=False, truncation=True, max_length=512)
    input_ids = ft["input_ids"]
    labels = input_ids.copy()
    labels[: len(pt["input_ids"])] = [-100] * len(pt["input_ids"])
    return {
        "input_ids": input_ids,
        "attention_mask": ft["attention_mask"],
        "labels": labels,
    }


tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)


def collate(batch):
    ml = max(len(x["input_ids"]) for x in batch)
    iids, ams, lbs = [], [], []
    for x in batch:
        l = len(x["input_ids"])
        p = ml - l
        iids.append(x["input_ids"] + [tokenizer.pad_token_id] * p)
        ams.append(x["attention_mask"] + [0] * p)
        lbs.append(x["labels"] + [-100] * p)
    return {
        "input_ids": torch.tensor(iids, dtype=torch.long),
        "attention_mask": torch.tensor(ams, dtype=torch.long),
        "labels": torch.tensor(lbs, dtype=torch.long),
    }


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
    bf16=False,
    fp16=False,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model, args=args, train_dataset=tokenized, data_collator=collate
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved:", OUTPUT_DIR, flush=True)
