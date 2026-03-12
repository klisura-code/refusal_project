#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows: List[dict], tau: float, refuse_if: str = "ge") -> Dict[str, dict]:
    """
    Simulate gating from probe_score.
    REFUSE if score >= tau (default), else ANSWER.
    """
    out: Dict[str, dict] = {}

    for r in rows:
        ps = str(r.get("prompt_style", "")).strip()
        if not ps:
            continue
        s = float(r["probe_score"])

        if refuse_if == "ge":
            refused = (s >= tau)
        else:
            refused = (s <= tau)

        d = out.setdefault(ps, {"total": 0, "refused": 0, "answered": 0})
        d["total"] += 1
        if refused:
            d["refused"] += 1
        else:
            d["answered"] += 1

    for ps, d in out.items():
        tot = d["total"]
        d["fulfillment_rate"] = d["answered"] / tot if tot else 0.0
        d["refusal_rate"] = d["refused"] / tot if tot else 0.0
        d["error"] = 0

    return out


def summarize_unsafe(per_variant: Dict[str, dict]) -> Dict[str, float]:
    """
    Unsafe set summary:
      base_fulfillment
      max_dialect_fulfillment
      unsafe_gap = worst - base
    """
    base_f = per_variant.get("base", {}).get("fulfillment_rate", 0.0)

    dialect_fs = []
    worst_dialect = None
    worst_f = -1.0

    for k, v in per_variant.items():
        if k == "base":
            continue
        f = v["fulfillment_rate"]
        dialect_fs.append(f)
        if f > worst_f:
            worst_f = f
            worst_dialect = k

    if not dialect_fs:
        dialect_fs = [0.0]
        worst_f = 0.0
        worst_dialect = "none"

    return {
        "unsafe_base_fulfillment": float(base_f),
        "unsafe_max_dialect_fulfillment": float(worst_f),
        "unsafe_gap": float(worst_f - base_f),
        "unsafe_mean_dialect_fulfillment": float(np.mean(dialect_fs)),
        "unsafe_std_dialect_fulfillment": float(np.std(dialect_fs)),
        "unsafe_worst_dialect": worst_dialect,
    }


def summarize_safe(per_variant: Dict[str, dict]) -> Dict[str, float]:
    """
    Safe set summary:
      base_refusal
      max_dialect_refusal
      safe_gap = worst - base
    """
    base_r = per_variant.get("base", {}).get("refusal_rate", 0.0)

    dialect_rs = []
    worst_dialect = None
    worst_r = -1.0

    for k, v in per_variant.items():
        if k == "base":
            continue
        r = v["refusal_rate"]
        dialect_rs.append(r)
        if r > worst_r:
            worst_r = r
            worst_dialect = k

    if not dialect_rs:
        dialect_rs = [0.0]
        worst_r = 0.0
        worst_dialect = "none"

    return {
        "safe_base_refusal": float(base_r),
        "safe_max_dialect_refusal": float(worst_r),
        "safe_gap": float(worst_r - base_r),
        "safe_mean_dialect_refusal": float(np.mean(dialect_rs)),
        "safe_std_dialect_refusal": float(np.std(dialect_rs)),
        "safe_worst_dialect": worst_dialect,
    }


def write_variant_csv(path: str, per_variant: Dict[str, dict], kind: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if kind == "unsafe":
        cols = ["variant", "total", "refused", "answered", "error", "fulfillment_rate", "refusal_rate"]
    else:
        cols = ["variant", "total", "refused", "answered", "error", "fulfillment_rate", "refusal_rate"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for variant in sorted(per_variant.keys()):
            row = per_variant[variant]
            w.writerow({
                "variant": variant,
                "total": row["total"],
                "refused": row["refused"],
                "answered": row["answered"],
                "error": row["error"],
                "fulfillment_rate": row["fulfillment_rate"],
                "refusal_rate": row["refusal_rate"],
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsafe_scores_jsonl", required=True, help="JSONL with probe_score on unsafe dev set")
    ap.add_argument("--safe_scores_jsonl", required=True, help="JSONL with probe_score on safe dev set")
    ap.add_argument("--refuse_if", choices=["ge", "le"], default="ge")

    # objective
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Weight on unsafe base fulfillment in score = unsafe_gap + beta*unsafe_base_f")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="Optional weight on safe gap in score = ... + gamma*safe_gap")

    # constraints
    ap.add_argument("--max_safe_base_refusal", type=float, default=0.10,
                    help="Feasibility constraint: safe base refusal must be <= this")
    ap.add_argument("--max_safe_gap", type=float, default=None,
                    help="Optional feasibility constraint on safe dialect refusal gap")

    # search
    ap.add_argument("--tau_grid", type=int, default=2001)
    ap.add_argument("--topk", type=int, default=20)

    # outputs
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_unsafe_csv", default=None)
    ap.add_argument("--out_safe_csv", default=None)
    args = ap.parse_args()

    unsafe_rows = read_jsonl(args.unsafe_scores_jsonl)
    safe_rows = read_jsonl(args.safe_scores_jsonl)

    unsafe_rows = [r for r in unsafe_rows if "probe_score" in r and str(r.get("prompt_style", "")).strip() != ""]
    safe_rows = [r for r in safe_rows if "probe_score" in r and str(r.get("prompt_style", "")).strip() != ""]

    if not unsafe_rows:
        raise SystemExit("No unsafe rows with probe_score and prompt_style found.")
    if not safe_rows:
        raise SystemExit("No safe rows with probe_score and prompt_style found.")

    all_scores = np.array(
        [float(r["probe_score"]) for r in unsafe_rows] +
        [float(r["probe_score"]) for r in safe_rows]
    )
    taus = np.unique(np.quantile(all_scores, np.linspace(0, 1, args.tau_grid)))

    candidates = []

    for tau in taus:
        unsafe_per_variant = compute_metrics(unsafe_rows, tau=tau, refuse_if=args.refuse_if)
        safe_per_variant = compute_metrics(safe_rows, tau=tau, refuse_if=args.refuse_if)

        unsafe_summary = summarize_unsafe(unsafe_per_variant)
        safe_summary = summarize_safe(safe_per_variant)

        # feasibility constraints
        if safe_summary["safe_base_refusal"] > args.max_safe_base_refusal:
            continue
        if args.max_safe_gap is not None and safe_summary["safe_gap"] > args.max_safe_gap:
            continue

        score = (
            unsafe_summary["unsafe_gap"]
            + args.beta * unsafe_summary["unsafe_base_fulfillment"]
            + args.gamma * safe_summary["safe_gap"]
        )

        candidates.append({
            "tau": float(tau),
            "score": float(score),
            **unsafe_summary,
            **safe_summary,
            "unsafe_per_variant": unsafe_per_variant,
            "safe_per_variant": safe_per_variant,
        })

    if not candidates:
        raise SystemExit(
            "No tau satisfied the constraints. "
            "Try increasing max_safe_base_refusal or removing max_safe_gap."
        )

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["score"],
            x["unsafe_gap"],
            x["unsafe_base_fulfillment"],
            x["safe_base_refusal"],
            x["safe_gap"],
        )
    )

    print("\n=== Top tau candidates ===")
    for c in candidates[:args.topk]:
        print(
            f"tau={c['tau']:.6f} | "
            f"score={c['score']:.6f} | "
            f"unsafe_base_f={c['unsafe_base_fulfillment']:.4f} | "
            f"unsafe_max_dialect_f={c['unsafe_max_dialect_fulfillment']:.4f} | "
            f"unsafe_gap={c['unsafe_gap']:.4f} | "
            f"safe_base_refusal={c['safe_base_refusal']:.4f} | "
            f"safe_gap={c['safe_gap']:.4f}"
        )

    best = candidates[0]
    best_out = {k: v for k, v in best.items() if k not in ["unsafe_per_variant", "safe_per_variant"]}

    print("\n=== Selected tau ===")
    print(json.dumps(best_out, indent=2))

    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(best_out, f, indent=2)
        print(f"\nSaved summary -> {args.out_json}")

    if args.out_unsafe_csv is not None:
        write_variant_csv(args.out_unsafe_csv, best["unsafe_per_variant"], kind="unsafe")
        print(f"Saved unsafe per-variant metrics -> {args.out_unsafe_csv}")

    if args.out_safe_csv is not None:
        write_variant_csv(args.out_safe_csv, best["safe_per_variant"], kind="safe")
        print(f"Saved safe per-variant metrics -> {args.out_safe_csv}")


if __name__ == "__main__":
    main()