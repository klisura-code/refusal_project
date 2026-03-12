#!/usr/bin/env python3
import json
import random
from pathlib import Path

random.seed(42)

src = Path("/home/dorde/Desktop/Refusal_project/data/safe/alpaca_safe_prompts.filtered.jsonl")
out = Path("/home/dorde/Desktop/Refusal_project/data/safe/alpaca_clean_dev.jsonl")

rows = [json.loads(l) for l in src.open()]

# exclude any that were used in probe training (safefinal.jsonl)
train_ids = set(
    json.loads(l)["question_id"] 
    for l in open("/home/dorde/Desktop/Refusal_project/data/safe/safefinal.jsonl")
)

rows = [r for r in rows if r["question_id"] not in train_ids]
sampled = random.sample(rows, min(1000, len(rows)))

with out.open("w") as f:
    for r in sampled:
        r["prompt_style"] = "base"
        f.write(json.dumps(r) + "\n")

print(f"Written {len(sampled)} rows to {out}")
print(f"(excluded {len(train_ids)} probe-training ids)")