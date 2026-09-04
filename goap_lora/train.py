import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]

# Sequential fine-tuning: base = Qwen + merged tech-lora weights,
# so this round only learns the GOAP output format on top of known facts.
MODEL_NAME = ROOT / "outputs" / "tech-lora-merged"
OUTPUT_DIR = ROOT / "outputs" / "goap-lora"

device = os.environ.get("DEVICE") or (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("device:", device)


# --------------------------------------------------
# 1. tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# --------------------------------------------------
# 2. model
# --------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.to(device)


# --------------------------------------------------
# 3. LoRA
# --------------------------------------------------

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


# --------------------------------------------------
# 4. dataset
# --------------------------------------------------

dataset = load_dataset(
    "json",
    data_files=str(ROOT / "data" / "train_goap.jsonl"),
    split="train",
)


# --------------------------------------------------
# 5. preprocess
# --------------------------------------------------


def preprocess(example):
    system_prompt = (
        "你是一个严谨的自贡井盐与地方史问答助手。"
        "回答用户的每个问题前，必须先做 GOAP（Goal-Oriented Action Planning）规划："
        "先写 GOAL 与 WORLD STATE，"
        "再从备选动作中选择满足前置条件（PRECONDITIONS）并能产生效果（EFFECTS）的动作（ACTIONS），"
        "构成按序执行的 PLAN；最后输出 CONCEPT: ... 与 ANSWER: ...。"
        "不要只给答案而省略规划步骤。"
    )

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["instruction"]},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()

    prompt_length = len(prompt_tokens["input_ids"])
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
)


# --------------------------------------------------
# 6. custom collator
# --------------------------------------------------


def collate_fn(batch):

    max_length = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        length = len(item["input_ids"])
        padding_length = max_length - length

        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_length)

        attention_masks.append(item["attention_mask"] + [0] * padding_length)

        labels.append(item["labels"] + [-100] * padding_length)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            attention_masks,
            dtype=torch.long,
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# --------------------------------------------------
# 7. training
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=15,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
    bf16=(device == "cuda"),
    fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collate_fn,
)


trainer.train()


# --------------------------------------------------
# 8. save
# --------------------------------------------------

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved:", OUTPUT_DIR)
