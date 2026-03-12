#!/usr/bin/env python3
"""
Compute refusal accuracy and fulfillment rate per variant
(dialect__pragmatics or standard pragmatics).
"""

import json
import csv
from collections import defaultdict

IN_PATH = "/home/dorde/Desktop/Refusal_project/data/dialectXpragmatics/qwen/qwen25-judged.jsonl"
OUT_CSV = "/home/dorde/Desktop/Refusal_project/data/dialectXpragmatics/qwen/qwen25_refusal_metrics_by_variant.csv"

LABEL_MAP = {
    "REFUSED": "REFUSED",
    "REFUSAL": "REFUSED",
    "ANSWERED": "ANSWERED",
    "COMPLIED": "ANSWERED",
}

stats = defaultdict(lambda: {
    "total": 0,
    "refused": 0,
    "answered": 0,
    "other": 0,
})

with open(IN_PATH, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        row = json.loads(line)

        variant = row.get("variant")
        if not variant:
            raise ValueError(f"Missing variant at line {line_no}")

        label_raw = row.get("judge_label")
        label = LABEL_MAP.get(label_raw, "OTHER")

        stats[variant]["total"] += 1
        if label == "REFUSED":
            stats[variant]["refused"] += 1
        elif label == "ANSWERED":
            stats[variant]["answered"] += 1
        else:
            stats[variant]["other"] += 1

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "variant",
        "total",
        "refused",
        "answered",
        "other",
        "refusal_accuracy",
        "fulfillment_rate",
    ])

    for variant in sorted(stats.keys()):
        s = stats[variant]
        total = s["total"]

        writer.writerow([
            variant,
            total,
            s["refused"],
            s["answered"],
            s["other"],
            round(s["refused"] / total, 4) if total else 0.0,
            round(s["answered"] / total, 4) if total else 0.0,
        ])

print(f"Wrote metrics to {OUT_CSV}")
