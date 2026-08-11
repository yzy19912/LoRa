from pathlib import Path

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[1]

# Sequential fine-tuning: base = Qwen + merged tech-lora weights,
# matching the base the goap-lora adapter was trained on top of.
MODEL_NAME = ROOT / "outputs" / "tech-lora-merged"
LORA_PATH = ROOT / "outputs" / "goap-lora"

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
    "回答用户的每个问题前，必须先做 GOAP（Goal-Oriented Action Planning）规划："
    "先写 GOAL 与 WORLD STATE，"
    "再从备选动作中选择满足前置条件（PRECONDITIONS）并能产生效果（EFFECTS）的动作（ACTIONS），"
    "构成按序执行的 PLAN；最后输出 CONCEPT: ... 与 ANSWER: ...。"
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\nModel:", response)
    messages.append({"role": "assistant", "content": response})