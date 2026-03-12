#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def sh(cmd, cwd=None):
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True, help="Directory containing lam_*.jsonl")
    ap.add_argument("--judge_py", default="judge_refusal.py", help="Path to judge_refusal.py")
    ap.add_argument("--eval_py", default="eval_judged.py", help="Path to eval_judged.py")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--print_every", type=int, default=500)
    ap.add_argument("--limit_rows", type=int, default=0)  # 0 means no limit
    ap.add_argument("--pattern", default="*.jsonl")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        raise SystemExit(f"Missing sweep_dir: {sweep_dir}")

    # Output folders next to the sweep outputs
    judged_dir = sweep_dir / "judged"
    metrics_dir = sweep_dir / "metrics"
    judged_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Assume runner sits in models/Qwen2.5; use that as cwd so relative judge/eval paths work
    repo_cwd = Path(__file__).resolve().parent

    files = sorted(sweep_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {sweep_dir}")

    # Quick sanity: OPENAI_API_KEY must exist for judging
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in your environment.")

    for in_path in files:
        # Skip already-judged outputs directory if pattern catches it
        if in_path.parent.name in ("judged", "metrics"):
            continue

        judged_path = judged_dir / (in_path.stem + "-judged.jsonl")
        metrics_path = metrics_dir / (in_path.stem + "-metrics.csv")

        # 1) judge (resumable: judge_refusal.py will skip items already in judged_path)
        sh([
            "python3", str((repo_cwd / args.judge_py).resolve()),
            "--in", str(in_path),
            "--out", str(judged_path),
            "--concurrency", str(args.concurrency),
            "--print_every", str(args.print_every),
            "--limit_rows", str(args.limit_rows),
        ], cwd=str(repo_cwd))

        # 2) eval metrics
        sh([
            "python3", str((repo_cwd / args.eval_py).resolve()),
            "--in", str(judged_path),
            "--out", str(metrics_path),
        ], cwd=str(repo_cwd))

    print("\nAll done.")
    print(f"Judged JSONL: {judged_dir}")
    print(f"Metrics CSVs: {metrics_dir}")


if __name__ == "__main__":
    main()