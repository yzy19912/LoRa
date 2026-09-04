"""Augment training data by generating NEW knowledge about 自贡 using the base model.

The base model (Qwen2.5-0.5B-Instruct) has been trained on a broad corpus and
knows general facts about 自贡's history, salt industry, culture, etc.
We ask it to generate facts in CONCEPT/ANSWER format, then create
(question, answer) pairs from those facts for SFT LoRA training.

Usage:  uv run python tech_lora/augment_data.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SRC_JSONL = ROOT / "data" / "train.jsonl"
DST_JSONL = ROOT / "data" / "train_augmented.jsonl"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)


# ---------------------------------------------------------------------------
# 1. Load base model (not fine-tuned) — it has broader knowledge
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model = model.to(device)
model.eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# 2. Helper: parse CONCEPT and ANSWER from a response
# ---------------------------------------------------------------------------
def parse_response(response: str):
    """Parse CONCEPT and ANSWER from a model response."""
    concept = answer = None
    for line in response.split("\n"):
        if line.startswith("CONCEPT:"):
            concept = line.split(":", 1)[1].strip()
        elif line.startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
    return concept, answer


# ---------------------------------------------------------------------------
# 3. Generate new knowledge about a given topic
# ---------------------------------------------------------------------------
def generate_fact(topic: str, max_tokens: int = 256) -> str | None:
    """Ask the base model to generate a fact about a topic in CONCEPT/ANSWER format.

    Returns the raw response string, or None on failure.
    """
    system_prompt = (
        "你是一个严谨的历史知识助手。"
        "请提供准确的历史事实，格式为：\n"
        "CONCEPT: ...\nANSWER: ..."
    )

    user_prompt = (
        f"请提供关于「{topic}」的准确历史或地理知识。"
        f"如果知道就详细回答，如果不知道就说不知道，不要编造。"
        f"请用CONCEPT/ANSWER格式回答。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


# ---------------------------------------------------------------------------
# 4. Generate question-answer pairs from a knowledge fact
# ---------------------------------------------------------------------------
def make_qa_pairs(concept: str, answer: str) -> list[dict]:
    """Generate multiple question variants for a given concept and answer.

    Returns a list of {"instruction": ..., "response": ...} dicts.
    """
    pairs = []
    seen = set()

    # Different question starters
    starters = [
        f"请介绍一下{concept}。",
        f"什么是{concept}？",
        f"你能告诉我{concept}的相关信息吗？",
        f"关于{concept}，你知道些什么？",
        f"请解释一下{concept}。",
        f"我想了解{concept}，请详细说明。",
        f"{concept}是怎么回事？",
        f"说说{concept}吧。",
        f"请详细说说{concept}。",
        f"给我讲讲{concept}。",
        f"请问{concept}是什么？",
        f"你知道{concept}吗？",
        f"你可知道{concept}？",
        f"请告诉我{concept}。",
        f"我很好奇{concept}，你能介绍下吗？",
    ]

    for q in starters:
        if q not in seen:
            seen.add(q)
            pairs.append(
                {
                    "instruction": q,
                    "response": f"CONCEPT: {concept}\nANSWER: {answer}",
                }
            )

    return pairs


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    # Load original data to keep its knowledge
    with open(SRC_JSONL, encoding="utf-8") as f:
        original_rows = [json.loads(line) for line in f if line.strip()]
    print(f"Original rows: {len(original_rows)}")

    # Parse original concepts
    original_concepts = set()
    for r in original_rows:
        c, _ = parse_response(r["response"])
        if c:
            original_concepts.add(c)
    print(f"Original concepts: {len(original_concepts)}")

    # -----------------------------------------------------------------------
    # A. Generate new knowledge about 自贡 topics
    # -----------------------------------------------------------------------
    new_topics = [
        # 自贡盐业相关
        "自贡的燊海井",
        "自贡的盐业历史博物馆",
        "自贡的盐井开凿技术",
        "自贡的天然气开采历史",
        "自贡的西秦会馆",
        "自贡的王爷庙",
        "自贡的釜溪河",
        "自贡的盐运古道",
        "自贡的盐业契约",
        "自贡的盐工生活",
        "自贡的盐业与地方教育",
        "自贡的盐业与城市建设",
        "自贡的盐业与交通发展",
        "自贡的盐业与金融",
        "自贡的盐业与科技",
        "自贡的盐业与环保",
        "自贡的盐业与对外交流",
        "自贡的盐业与文学艺术",
        "自贡的盐业与宗教祭祀",
        "自贡的盐业与地方治理",
        # 自贡地理文化
        "自贡的地理位置",
        "自贡的气候特点",
        "自贡的自然资源",
        "自贡的行政区划",
        "自贡的方言文化",
        "自贡的饮食文化",
        "自贡的民俗文化",
        "自贡的旅游景点",
        "自贡的灯会文化",
        "自贡的恐龙博物馆",
        "自贡的恐龙化石发现",
        "自贡的现代工业发展",
        "自贡的交通发展",
        "自贡的教育事业",
        "自贡的著名人物",
        # 自贡历史
        "自贡的建市历史",
        "自贡在抗战时期的贡献",
        "自贡的近代工业发展",
        "自贡的三线建设",
        "自贡的城市变迁",
        "自贡的桑海井",
        "自流井的历史",
        "贡井的历史",
        "自贡盐业对中国近代经济的影响",
        "自贡盐业的技术创新",
        "自贡的盐业与运河",
        "自贡的盐业与铁路",
        "自贡的盐业与城市形成",
        "自贡的盐业与人口迁移",
        "自贡的盐业与地方文化",
    ]

    # Filter out topics that overlap with existing concepts
    new_topics_filtered = [t for t in new_topics if t not in original_concepts]
    print(
        f"Topics to explore: {len(new_topics_filtered)} (skipped {len(new_topics) - len(new_topics_filtered)} overlapping)"
    )

    # Generate new knowledge
    new_knowledge = []
    for topic in new_topics_filtered:
        try:
            resp = generate_fact(topic)
            concept, answer = parse_response(resp)
            if concept and answer and len(answer) > 10:
                # Check for hallucination indicators
                skip_indicators = ["我不知道", "不知道", "我不清楚", "没有相关信息"]
                if not any(ind in answer for ind in skip_indicators):
                    new_knowledge.append(
                        {
                            "concept": concept,
                            "answer": answer,
                            "topic": topic,
                        }
                    )
                    print(f"  ✓ {topic} -> {concept}")
                else:
                    print(f"  - {topic}: model says unknown")
            else:
                print(f"  - {topic}: failed to parse or too short")
        except Exception as e:
            print(f"  ✗ {topic}: error {e}")

    print(f"\nGenerated {len(new_knowledge)} new knowledge items")

    # -----------------------------------------------------------------------
    # B. Build augmented dataset
    # -----------------------------------------------------------------------
    augmented = []

    # Keep original rows
    for r in original_rows:
        augmented.append(r)

    # Add new knowledge with question variants
    stats = {"original": len(original_rows), "new_concepts": 0, "new_pairs": 0}
    for nk in new_knowledge:
        pairs = make_qa_pairs(nk["concept"], nk["answer"])
        for p in pairs:
            augmented.append(p)
            stats["new_pairs"] += 1
        stats["new_concepts"] += 1

    # Also add template-based variants of original data (for better coverage)
    for r in original_rows:
        c, a = parse_response(r["response"])
        if c and a:
            pairs = make_qa_pairs(c, a)
            for p in pairs:
                if p["instruction"] != r["instruction"]:
                    augmented.append(p)
                    stats["new_pairs"] += 1

    print(f"\nAugmented dataset: {len(augmented)} rows")
    print(f"  Original: {stats['original']}")
    print(f"  New concepts: {stats['new_concepts']}")
    print(f"  New QA pairs: {stats['new_pairs']}")

    # Save
    with open(DST_JSONL, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in augmented)

    print(f"Saved to {DST_JSONL}")

    # Show some samples
    print("\n--- Sample new knowledge rows ---")
    shown = 0
    for i, rec in enumerate(augmented):
        if shown >= 10:
            break
        c, _ = parse_response(rec["response"])
        if c not in original_concepts:
            print(f"\n[{i}] Q: {rec['instruction']}")
            print(f"    A: {rec['response'][:150]}...")
            shown += 1


if __name__ == "__main__":
    main()
