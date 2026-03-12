#!/usr/bin/env python3
import json
from pathlib import Path

POL_FILE = Path("Sorry-POLITENESS.jsonl")
IMP_FILE = Path("Sorry-IMPOLITENESS.jsonl")
OUT_FILE = Path("Sorry_DxP_seed_5styles.jsonl")

KEEP_FROM_POL = {"base", "negative-politeness", "very-polite"}
KEEP_FROM_IMP = {"negative-impoliteness", "positive-impoliteness"}

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {line_no}: {e}")
    return rows

def norm_turns(turns):
    # turns is usually a list with 1 string; keep general
    if isinstance(turns, list):
        return tuple(turns)
    return (str(turns),)

def key_for_row(r):
    # robust key: question_id + category + turns text
    return (int(r["question_id"]), str(r.get("category", "")), norm_turns(r.get("turns", [])))

def main():
    pol = read_jsonl(POL_FILE)
    imp = read_jsonl(IMP_FILE)

    merged = []
    seen = set()

    # 1) Take base + polite variants from POL file
    for r in pol:
        ps = r.get("prompt_style")
        if ps not in KEEP_FROM_POL:
            continue
        k = (key_for_row(r), ps)
        if k in seen:
            continue
        seen.add(k)
        merged.append({
            "question_id": int(r["question_id"]),
            "category": str(r.get("category", "")),
            "prompt_style": ps,
            "turns": r.get("turns", []),
            "source_file": POL_FILE.name,
        })

    # 2) Take only the 2 impolite variants from IMP file (ignore its base)
    for r in imp:
        ps = r.get("prompt_style")
        if ps not in KEEP_FROM_IMP:
            continue
        k = (key_for_row(r), ps)
        if k in seen:
            continue
        seen.add(k)
        merged.append({
            "question_id": int(r["question_id"]),
            "category": str(r.get("category", "")),
            "prompt_style": ps,
            "turns": r.get("turns", []),
            "source_file": IMP_FILE.name,
        })

    # Sort for reproducibility: by question_id then prompt_style
    merged.sort(key=lambda x: (x["question_id"], x["prompt_style"]))

    # Quick sanity checks
    styles = {}
    for r in merged:
        styles[r["prompt_style"]] = styles.get(r["prompt_style"], 0) + 1

    print("Counts by prompt_style:")
    for k in sorted(styles):
        print(f"  {k}: {styles[k]}")

    # Expect exactly 5 styles present
    expected = {"base", "negative-politeness", "very-polite",
                "negative-impoliteness", "positive-impoliteness"}
    missing = expected - set(styles)
    extra = set(styles) - expected
    if missing:
        print("WARNING missing styles:", missing)
    if extra:
        print("WARNING extra styles:", extra)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote: {OUT_FILE}  (rows={len(merged)})")

if __name__ == "__main__":
    main()
