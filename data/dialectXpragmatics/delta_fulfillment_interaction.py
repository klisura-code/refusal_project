#!/usr/bin/env python3
"""
Plot interaction bars: ΔFulfillment relative to base within each dialect.

Input: CSV with columns:
  variant,total,refused,answered,other,refusal_accuracy,fulfillment_rate

Variant patterns:
  - Standard pragmatics: base, negative-politeness, ...
  - Dialect rows: dialect-<DialectName>__<pragmatics>

Output:
  <out_prefix>.pdf and <out_prefix>.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRAG_ORDER_DEFAULT = ["negative-politeness", "very-polite", "negative-impoliteness", "positive-impoliteness"]

DIALECT_ORDER_DEFAULT = [
    "StandardEnglish",
    "BahamianDialect",
    "FijiBasilect",
    "MalteseDialect",
    "ManxDialect",
    "IndianSouthAfricanDialect",
    "HongKongDialect",
    "SoutheastEnglandDialect",
]

def parse_variant(v: str):
    v = str(v).strip()
    if v.startswith("dialect-") and "__" in v:
        left, prag = v.split("__", 1)
        dialect = left.replace("dialect-", "")
        return dialect, prag
    return "StandardEnglish", v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Metrics CSV (per variant)")
    ap.add_argument("--out_prefix", default="delta_fulfillment_interaction")
    ap.add_argument("--model_name", default="Model", help="Used in title only")
    ap.add_argument("--dpi", type=int, default=200)

    ap.add_argument("--dialect_order", default=",".join(DIALECT_ORDER_DEFAULT),
                    help="Comma-separated dialect order")
    ap.add_argument("--prag_order", default=",".join(PRAG_ORDER_DEFAULT),
                    help="Comma-separated pragmatics order (non-base only)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "variant" not in df.columns or "fulfillment_rate" not in df.columns:
        raise ValueError("CSV must contain at least: variant, fulfillment_rate")

    parsed = df["variant"].apply(parse_variant)
    df["dialect"] = parsed.apply(lambda x: x[0])
    df["pragmatics"] = parsed.apply(lambda x: x[1])

    dialect_order = [x.strip() for x in args.dialect_order.split(",") if x.strip()]
    prag_order = [x.strip() for x in args.prag_order.split(",") if x.strip()]

    # Pivot
    pivot = (
        df.pivot_table(index="dialect", columns="pragmatics", values="fulfillment_rate", aggfunc="mean")
    )

    # Ensure we have base for each dialect
    if "base" not in pivot.columns:
        raise RuntimeError("Missing 'base' pragmatics in data. Cannot compute ΔFR.")
    base = pivot["base"]

    # Compute deltas for requested pragmatics
    deltas = pd.DataFrame(index=pivot.index)
    for p in prag_order:
        if p not in pivot.columns:
            raise RuntimeError(f"Missing pragmatics '{p}' in data.")
        deltas[p] = pivot[p] - base

    # Order dialects
    deltas = deltas.reindex(index=[d for d in dialect_order if d in deltas.index])

    # Plot: grouped bars
    n_d = len(deltas.index)
    n_p = len(prag_order)
    x = np.arange(n_d)
    width = 0.18 if n_p == 4 else min(0.8 / n_p, 0.2)

    fig_w = 16
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for j, p in enumerate(prag_order):
        ax.bar(x + (j - (n_p-1)/2)*width, deltas[p].values, width=width, label=p)

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(deltas.index.tolist(), rotation=25, ha="right", fontsize=11)

    ax.set_ylabel("Δ Fulfillment rate vs. base (within dialect)")
    ax.set_xlabel("Dialect")
    ax.set_title(f"{args.model_name} — Interaction: Pragmatics effect within each dialect")

    ax.legend(ncol=2, frameon=True)

    fig.tight_layout()

    out_prefix = Path(args.out_prefix)
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=args.dpi, bbox_inches="tight")
    print("Saved:", str(out_prefix) + ".pdf")
    print("Saved:", str(out_prefix) + ".png")

if __name__ == "__main__":
    main()
