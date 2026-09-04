import os

from langsmith import Client

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-first-project-1"

client = Client()

dataset = client.create_dataset(
    dataset_name="skill-selector-eval",
    description="Evaluate skill routing accuracy",
)

examples = [
    {
        "inputs": {"user_input": "I want my money back"},
        "outputs": {"skills": ["refund"]},
    },
    {
        "inputs": {"user_input": "Can I send this item back?"},
        "outputs": {"skills": ["return"]},
    },
    {
        "inputs": {"user_input": "What material is this product made of?"},
        "outputs": {"skills": ["product"]},
    },
    {
        "inputs": {
            "user_input": "I don't want this anymore, can I return it and get my money back?"
        },
        "outputs": {"skills": ["return", "refund"]},
    },
]

client.create_examples(
    dataset_id=dataset.id,
    examples=examples,
)
