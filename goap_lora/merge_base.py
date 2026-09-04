from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "tech-lora-v2"
OUTPUT_DIR = ROOT / "outputs" / "tech-lora-merged"

print("Merging", ADAPTER_PATH, "into", MODEL_NAME, "->", OUTPUT_DIR)


# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 3. merge LoRA weights into the base model
model = model.merge_and_unload()

# 4. save the full merged model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved:", OUTPUT_DIR)
