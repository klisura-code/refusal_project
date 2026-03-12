#!/usr/bin/env python3
"""
Plot fulfillment-rate heatmap: Dialect × Pragmatics

Input: CSV with columns:
  variant,total,refused,answered,other,refusal_accuracy,fulfillment_rate

Variant patterns:
  - Standard English pragmatics: base, negative-politeness, very-polite, ...
  - Dialect rows: dialect-<DialectName>__<pragmatics>

Output:
  <out_prefix>.pdf and <out_prefix>.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Color palette (same as your old heatmap)
# -----------------------------
colors = [
    "#f7fcf5",
    "#e5f5e0",
    "#c7e9c0",
    "#a1d99b",
    "#74c476",
    "#238b45",
]

custom_cmap = LinearSegmentedColormap.from_list("approval_risk", colors)

PRAG_ORDER_DEFAULT = ["base", "negative-politeness", "very-polite", "negative-impoliteness", "positive-impoliteness"]

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
    # Standard English pragmatics
    return "StandardEnglish", v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Metrics CSV (per variant)")
    ap.add_argument("--out_prefix", default="heatmap_fulfillment_dialect_x_pragmatics")
    ap.add_argument("--model_name", default="Model", help="Used in title only")
    ap.add_argument("--dpi", type=int, default=200)

    ap.add_argument("--prag_order", default=",".join(PRAG_ORDER_DEFAULT),
                    help="Comma-separated pragmatics order")
    ap.add_argument("--dialect_order", default=",".join(DIALECT_ORDER_DEFAULT),
                    help="Comma-separated dialect order")

    ap.add_argument("--sort_dialects_by", choices=["none", "avg_desc", "avg_asc"], default="none",
                    help="Optional sorting by average fulfillment across pragmatics (overrides dialect_order except StandardEnglish kept first).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "variant" not in df.columns or "fulfillment_rate" not in df.columns:
        raise ValueError("CSV must contain at least: variant, fulfillment_rate")

    # Parse dialect/pragmatics
    parsed = df["variant"].apply(parse_variant)
    df["dialect"] = parsed.apply(lambda x: x[0])
    df["pragmatics"] = parsed.apply(lambda x: x[1])

    prag_order = [x.strip() for x in args.prag_order.split(",") if x.strip()]
    dialect_order = [x.strip() for x in args.dialect_order.split(",") if x.strip()]

    # Pivot dialect × pragmatics
    pivot = (
        df.pivot_table(index="dialect", columns="pragmatics", values="fulfillment_rate", aggfunc="mean")
          .reindex(columns=prag_order)
    )

    # Ensure dialect rows exist; reindex keeps order and drops missing
    pivot = pivot.reindex(index=[d for d in dialect_order if d in pivot.index])

    # Optional sorting by row average (keep StandardEnglish first)
    if args.sort_dialects_by != "none":
        row_avg = pivot.mean(axis=1)
        std = "StandardEnglish" if "StandardEnglish" in pivot.index else None
        others = [d for d in pivot.index if d != std]
        others_sorted = sorted(others, key=lambda d: row_avg.loc[d], reverse=(args.sort_dialects_by == "avg_desc"))
        new_index = ([std] if std else []) + others_sorted
        pivot = pivot.reindex(new_index)

    data = pivot.values
    n_rows, n_cols = data.shape

    # Figure sizing similar spirit to your old one
    fig_w = 14
    fig_h = max(6, 0.55 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        data,
        cmap=custom_cmap,
        vmin=0, vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Pragmatics variant")
    ax.set_ylabel("Dialect")

    # X ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(prag_order, rotation=30, ha="right", fontsize=11)

    # Y ticks with Avg
    ax.set_yticks(np.arange(n_rows))
    row_avgs = pivot.mean(axis=1)
    ylabels = [f"{d} (Avg: {row_avgs.loc[d]:.2f}".replace("0.", ".") + ")" for d in pivot.index]
    ax.set_yticklabels(ylabels, fontsize=11)

    # Grid
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell annotations (2 decimals, no leading zero)
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:.2f}".lstrip("0")
            color = "black" if val < 0.6 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Fulfillment rate (fraction judged ANSWERED)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_title(f"{args.model_name} — Fulfillment rate by Dialect × Pragmatics")

    fig.subplots_adjust(left=0.28, right=0.94, bottom=0.18, top=0.92)

    out_prefix = Path(args.out_prefix)
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=args.dpi, bbox_inches="tight")
    print("Saved:", str(out_prefix) + ".pdf")
    print("Saved:", str(out_prefix) + ".png")

if __name__ == "__main__":
    main()
