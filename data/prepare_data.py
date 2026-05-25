"""
Download HotpotQA and produce a stratified sample saved as local JSON.

HotpotQA is a multi-hop QA benchmark where each question requires
reasoning across multiple Wikipedia passages. The validation split is
entirely "hard" difficulty; we stratify by *type* instead:
  - bridge    (~75%): chain two entities via an intermediate fact
  - comparison (~25%): compare an attribute of two entities
"""

import os
import json
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from config import DATA_DIR

random.seed(42)


def prepare_hotpotqa(sample_size: int = 100, split: str = "validation"):
    """Download HotpotQA, stratified-sample by question type, save to JSON."""
    print(f"Downloading HotpotQA '{split}' split from HuggingFace...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    print(f"  Total questions: {len(dataset)}")

    bridge     = [x for x in dataset if x["type"] == "bridge"]
    comparison = [x for x in dataset if x["type"] == "comparison"]

    # Match original distribution: ~75% bridge, ~25% comparison
    n_comparison = max(1, int(sample_size * 0.25))
    n_bridge     = sample_size - n_comparison

    sampled = (
        random.sample(bridge,     min(n_bridge,     len(bridge)))
        + random.sample(comparison, min(n_comparison, len(comparison)))
    )
    random.shuffle(sampled)

    processed = []
    for item in sampled:
        passages = []
        for title, sentences in zip(item["context"]["title"],
                                    item["context"]["sentences"]):
            passages.append({"title": title, "text": " ".join(sentences)})

        processed.append({
            "id":               item["id"],
            "question":         item["question"],
            "answer":           item["answer"],
            "type":             item["type"],
            "level":            item["level"],
            "supporting_facts": item["supporting_facts"],
            "passages":         passages,
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"hotpotqa_{sample_size}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    type_counts = {}
    for item in processed:
        type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
    print(f"  Saved {len(processed)} questions to {out_path}")
    print(f"  Type distribution: {type_counts}")
    return processed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    prepare_hotpotqa(sample_size=args.size)
