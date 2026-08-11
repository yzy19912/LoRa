import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model...")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model = model.to(device)

messages = [
    {
        "role": "user",
        "content": "Explain what revenue recognition means in one sentence."
    }
]

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
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

generated = outputs[0][inputs.input_ids.shape[1]:]

response = tokenizer.decode(
    generated,
    skip_special_tokens=True,
)

print("\nResponse:")
print(response)