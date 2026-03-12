#!/usr/bin/env python3
import json
import random
from pathlib import Path

random.seed(42)

src = Path("/home/dorde/Desktop/Refusal_project/data/unsafe/prompts_Sorry-DIALECT.jsonl")
dev_out = Path("/home/dorde/Desktop/Refusal_project/data/unsafe/prompts_Sorry-DIALECT_dev.jsonl")
test_out = Path("/home/dorde/Desktop/Refusal_project/data/unsafe/prompts_Sorry-DIALECT_test.jsonl")

rows = [json.loads(l) for l in src.open()]

# group by question_id
from collections import defaultdict
groups = defaultdict(list)
for r in rows:
    groups[r["question_id"]].append(r)

ids = sorted(groups.keys())
random.shuffle(ids)

n_dev = int(0.2 * len(ids))
dev_ids = set(ids[:n_dev])
test_ids = set(ids[n_dev:])

with dev_out.open("w") as f_dev, test_out.open("w") as f_test:
    for qid, qrows in groups.items():
        target = f_dev if qid in dev_ids else f_test
        for r in qrows:
            target.write(json.dumps(r) + "\n")

print(f"Dev:  {len(dev_ids)} questions")
print(f"Test: {len(test_ids)} questions")