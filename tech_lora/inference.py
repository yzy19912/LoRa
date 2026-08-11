from pathlib import Path

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = ROOT / "outputs" / "tech-lora-v2"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)
print("Loading model...")


# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

# 3. load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)
model = model.to(device)
model.eval()

print("Model loaded.")
print("Type 'quit' or 'exit' to stop.")

# 4. system prompt
system_prompt = (
    "你是一个严谨的自贡井盐与地方史问答助手。"
    "请回答时先给出概念，再给出答案，格式为："
    "CONCEPT: ...\nANSWER: ..."
)

messages = [{"role": "system", "content": system_prompt}]

# 5. interactive chat
while True:
    try:
        question = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        break

    if not question:
        continue

    if question.lower() in {"quit", "exit"}:
        break

    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\nModel:", response)
    messages.append({"role": "assistant", "content": response})