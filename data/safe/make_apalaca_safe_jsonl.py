#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path

def build_user_prompt(inst: str, inp: str) -> str:
    inst = (inst or "").strip()
    inp = (inp or "").strip()
    if inp:
        return f"{inst}\n\nInput:\n{inp}"
    return inst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="path to alpaca_data_cleaned.json")
    ap.add_argument("--out_jsonl", required=True, help="output jsonl path")
    # default now samples 1259 (set 0 to export all)
    ap.add_argument("--n", type=int, default=1264, help="sample N rows (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.load(open(args.in_json, "r", encoding="utf-8"))
    assert isinstance(data, list), "Expected a JSON list"

    rows = []
    for ex in data:
        prompt = build_user_prompt(ex.get("instruction", ""), ex.get("input", ""))
        if not prompt:
            continue
        rows.append(prompt)

    if args.n and args.n > 0:
        random.seed(args.seed)
        if args.n > len(rows):
            raise ValueError(f"--n={args.n} > available={len(rows)}")
        rows = random.sample(rows, args.n)

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # label: 0 = SAFE
    with open(outp, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(rows):
            rec = {
                "question_id": i,
                "prompt": prompt,
                "label": 0,
                "source": "alpaca_data_cleaned",
                "probe_id": i
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} SAFE prompts -> {outp}")

if __name__ == "__main__":
    main()