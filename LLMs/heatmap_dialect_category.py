#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Category names (1..44)
# -----------------------------
CAT_LABELS = {
    "1": "Personal Insult Words",
    "2": "Social-group Insult Words",
    "3": "Threatening Words",
    "4": "Lewd and Obscene Words",
    "5": "Libelous Words",
    "6": "Self-Harm",
    "7": "Violent Crimes",
    "8": "Harassment",
    "9": "Sexual Crimes",
    "10": "Property Crimes",
    "11": "Public Order Crime",
    "12": "System Intrusion",
    "13": "Malware",
    "14": "Fraud",
    "15": "Financial Crimes",
    "16": "IP Infringement",
    "17": "PII Violations",
    "18": "Illegal Crafting",
    "19": "Terrorism",
    "20": "Child-related Crimes",
    "21": "Animal-related Crimes",
    "22": "Environmental Crimes",
    "23": "Evasion of Law",
    "24": "Human Elimination",
    "25": "Advice on Adult Content",
    "26": "Sexual Explicit Content",
    "27": "Non-sexual Explicit Content",
    "28": "Fake News",
    "29": "False Advertising",
    "30": "Discrimination",
    "31": "Military Use",
    "32": "Political Belief",
    "33": "Ethical Belief",
    "34": "Religion",
    "35": "Social Stereotypes",
    "36": "Extremist Content",
    "37": "Conspiracy Theories",
    "38": "False Common Knowledge",
    "39": "Unhealthy Behaviors",
    "40": "Medical Advice",
    "41": "Financial Advice",
    "42": "Legal Consulting Advice",
    "43": "Governance Decision Advice",
    "44": "Machinery Operation Advice",
}

# -----------------------------
# Color palette
# -----------------------------
colors = [
    "#eff3ff",
    "#bdd7e7",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
]

custom_cmap = LinearSegmentedColormap.from_list("approval_risk", colors)


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}")

            lab = str(obj.get("judge_label", "")).upper()
            if lab not in {"REFUSED", "ANSWERED"}:
                continue

            cat = str(obj.get("category", "")).strip()
            dialect = str(obj.get("variant", "")).strip()
            if not cat or not dialect:
                continue

            rows.append({
                "question_id": int(obj["question_id"]),
                "category": cat,
                "dialect": dialect,
                "answered": 1 if lab == "ANSWERED" else 0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable rows found.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out_prefix", default="llama31_8b_heatmap_fulfillment_by_dialect_category")
    ap.add_argument("--include_base", action="store_true")
    ap.add_argument("--sort", choices=["none", "overall_desc", "overall_asc"], default="overall_desc")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="Fulfillment rate by dialect × SORRY category")

    args = ap.parse_args()

    df = load_jsonl(args.in_path)

    cats = [str(i) for i in range(1, 45)]
    df = df[df["category"].isin(cats)].copy()

    if not args.include_base:
        df = df[df["dialect"] != "base"].copy()

    pivot = (
        df.groupby(["dialect", "category"])["answered"]
          .mean()
          .unstack("category")
          .reindex(columns=cats)
    )

    # Sort rows (by overall mean approval)
    if args.sort != "none":
        overall = pivot.mean(axis=1)
        pivot = pivot.loc[overall.sort_values(ascending=(args.sort == "overall_asc")).index]

    # Keep base on top if requested
    #if args.include_base and "base" in pivot.index:
    #    pivot = pivot.reindex(["base"] + [x for x in pivot.index if x != "base"])

    data = pivot.values
    n_rows, n_cols = data.shape

    fig_w = 30
    fig_h = max(14, 0.40 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        data,
        cmap=custom_cmap,
        vmin=0, vmax=1,
        aspect="auto",
        interpolation="nearest"
    )

    # Axes labels
    ax.set_xlabel("SORRY-Bench category (1–44)")
    ax.set_ylabel("Dialect / Variant")

    # X ticks (categories)
    ax.set_xticks(np.arange(n_cols))
    xlabels = [f"{c}: {CAT_LABELS.get(c, c)}" for c in cats]
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)

    # Y ticks (dialects) + (Avg: .xx)
    ax.set_yticks(np.arange(n_rows))

    row_avgs = pivot.mean(axis=1)  # mean across categories per dialect
    ylabels = []
    for dialect in pivot.index.tolist():
        base_name = dialect.replace("dialect-", "")
        avg_txt = f"{row_avgs.loc[dialect]:.2f}".lstrip("0")  # ".12"
        ylabels.append(f"{base_name} (Avg: {avg_txt})")

    ax.set_yticklabels(ylabels, fontsize=9)

    # Grid
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell annotations (1 decimal, no leading zero)
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:.1f}".lstrip("0")
            color = "black" if val < 0.6 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Fulfillment / approval rate (fraction judged ANSWERED)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_title(args.title)

    fig.subplots_adjust(left=0.30, right=0.92, bottom=0.28, top=0.93)

    out_prefix = Path(args.out_prefix)
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=args.dpi, bbox_inches="tight")

    print("Saved:", str(out_prefix) + ".pdf")
    print("Saved:", str(out_prefix) + ".png")


if __name__ == "__main__":
    main()
