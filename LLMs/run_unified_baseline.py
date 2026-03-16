#!/usr/bin/env python3
"""
Unified baseline pipeline: start vLLM → inference → judge → eval → heatmap → stop vLLM

Usage:
    python3 run_unified_baseline.py --model_tag mistral7b
    python3 run_unified_baseline.py --model_tag gemma2_9b

    # All models fully unattended overnight:
    python3 run_unified_baseline.py --model_tag qwen25_7b && \
    python3 run_unified_baseline.py --model_tag mistral7b && \
    python3 run_unified_baseline.py --model_tag gemma2_9b

Output structure:
    baselines/
      <model_tag>/
        <model_tag>_Sorry-DIALECT_responses.jsonl
        judged/
          <model_tag>_Sorry-DIALECT_responses-judged.jsonl
        metrics/
          <model_tag>_Sorry-DIALECT_responses-metrics.csv
        heatmap/
          <model_tag>_heatmap_fulfillment_by_dialect_category.pdf
          <model_tag>_heatmap_fulfillment_by_dialect_category.png
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# ---- Hardcoded paths ----
DATASET     = "/home/dorde/Desktop/Refusal_project/data/Sorry-DIALECT.jsonl"
JUDGE_PY    = "/home/dorde/Desktop/Refusal_project/mitigation/scripts/judge_refusal.py"
EVAL_PY     = "/home/dorde/Desktop/Refusal_project/mitigation/scripts/eval_judged.py"
SCRIPT_DIR  = Path(__file__).resolve().parent
HEATMAP_PY  = SCRIPT_DIR / "heatmap_dialect_category.py"
OUT_DIR     = SCRIPT_DIR / "baselines"
VLLM_URL    = "http://localhost:8000/health"

# ---- Models registry ----
SUPPORTED_MODELS = {
    "llama31":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama32_3b":   "meta-llama/Llama-3.2-3B-Instruct",
    "qwen25":       "Qwen/Qwen2.5-14B-Instruct",
    "qwen25_7b":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen25_32b":   "Qwen/Qwen2.5-32B-Instruct",
    "mistral7b":    "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral_8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gemma2_9b":    "google/gemma-2-9b-it",
    "gemma2_27b":   "google/gemma-2-27b-it",
}

MODEL_TITLES = {
    "llama31":      "Llama-3.1-8B",
    "llama32_3b":   "Llama-3.2-3B",
    "qwen25":       "Qwen2.5-14B",
    "qwen25_7b":    "Qwen2.5-7B",
    "qwen25_32b":   "Qwen2.5-32B",
    "mistral7b":    "Mistral-7B-v0.3",
    "mixtral_8x7b": "Mixtral-8x7B",
    "gemma2_9b":    "Gemma-2-9B",
    "gemma2_27b":   "Gemma-2-27B",
}

# Per-model vLLM serve overrides (for models that need special flags)
VLLM_OVERRIDES = {
    "gemma2_27b": {
        "max_model_len":          "4096",
        "gpu_memory_utilization": "0.95",
        "quantization":           "bitsandbytes",
    },
    "qwen25_32b": {
        "max_model_len":          "4096",
        "gpu_memory_utilization": "0.95",
        "quantization":           "bitsandbytes",
    },
    "mixtral_8x7b": {
        "max_model_len":          "4096",
        "gpu_memory_utilization": "0.95",
        "quantization":           "bitsandbytes",
    },
}

# ---- Helpers ----

def sh(cmd: list, step: str):
    print(f"\n{'='*70}")
    print(f"  {step}")
    print(f"{'='*70}")
    print("CMD:", " ".join(str(c) for c in cmd))
    print()
    subprocess.run(cmd, check=True)


def wait_for_vllm(timeout: int = 300, poll: int = 5):
    """Poll /health until vLLM is ready or timeout."""
    print(f"  Waiting for vLLM to become ready (timeout={timeout}s)...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(VLLM_URL, timeout=5)
            if r.status_code == 200:
                print(f"  vLLM is ready! ({int(time.time()-t0)}s)")
                return
        except Exception:
            pass
        time.sleep(poll)
    sys.exit("ERROR: vLLM did not become ready in time.")


def start_vllm(model_name: str, model_tag: str, dtype: str = "bfloat16") -> subprocess.Popen:
    """Start vLLM server as a background process with proven flags."""
    overrides = VLLM_OVERRIDES.get(model_tag, {})
    cmd = [
        "vllm", "serve", model_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", dtype,
        "--max-model-len", overrides.get("max_model_len", "8192"),
        "--gpu-memory-utilization", overrides.get("gpu_memory_utilization", "0.90"),
        "--enforce-eager",
    ]
    if "quantization" in overrides:
        cmd += ["--quantization", overrides["quantization"]]
    print(f"\n{'='*70}")
    print(f"  Starting vLLM: {model_name}")
    print(f"{'='*70}")
    print("CMD:", " ".join(cmd))
    print()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc


def stop_vllm(proc: subprocess.Popen):
    """Gracefully stop vLLM server."""
    print("\n  Stopping vLLM server...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  vLLM stopped.")


# ---- Main ----

def main():
    ap = argparse.ArgumentParser(
        description="Run vLLM + inference → judge → eval → heatmap for one model."
    )
    ap.add_argument(
        "--model_tag",
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model tag: llama31 | qwen25 | qwen25_7b | mistral7b | gemma2_9b",
    )
    ap.add_argument("--model_name",        default=None)
    ap.add_argument("--dtype",             default="bfloat16")
    ap.add_argument("--vllm_timeout",      type=int,   default=300,
                    help="Seconds to wait for vLLM to become ready")
    ap.add_argument("--concurrency",       type=int,   default=64)
    ap.add_argument("--max_tokens",        type=int,   default=512)
    ap.add_argument("--temperature",       type=float, default=0.0)
    ap.add_argument("--limit_rows",        type=int,   default=None)
    ap.add_argument("--judge_concurrency", type=int,   default=64)
    ap.add_argument("--print_every",       type=int,   default=500)
    args = ap.parse_args()

    model_name = args.model_name or SUPPORTED_MODELS[args.model_tag]

    # ---- Sanity checks ----
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set in your environment.")
    if not Path(DATASET).exists():
        sys.exit(f"ERROR: Dataset not found: {DATASET}")
    if not Path(JUDGE_PY).exists():
        sys.exit(f"ERROR: judge_refusal.py not found at: {JUDGE_PY}")
    if not Path(EVAL_PY).exists():
        sys.exit(f"ERROR: eval_judged.py not found at: {EVAL_PY}")
    if not HEATMAP_PY.exists():
        sys.exit(f"ERROR: heatmap_dialect_category.py not found at: {HEATMAP_PY}")

    # ---- Derive output paths ----
    dataset_stem   = Path(DATASET).stem
    model_out_dir  = OUT_DIR / args.model_tag
    responses_path = model_out_dir / f"{args.model_tag}_{dataset_stem}_responses.jsonl"
    judged_dir     = model_out_dir / "judged"
    metrics_dir    = model_out_dir / "metrics"
    heatmap_dir    = model_out_dir / "heatmap"
    judged_path    = judged_dir  / f"{args.model_tag}_{dataset_stem}_responses-judged.jsonl"
    metrics_path   = metrics_dir / f"{args.model_tag}_{dataset_stem}_responses-metrics.csv"
    heatmap_prefix = heatmap_dir / f"{args.model_tag}_heatmap_fulfillment_by_dialect_category"

    for d in [judged_dir, metrics_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  UNIFIED BASELINE PIPELINE")
    print(f"  model_tag : {args.model_tag}")
    print(f"  model     : {model_name}")
    print(f"  dataset   : {DATASET}")
    print(f"  out_dir   : {model_out_dir}")
    print(f"{'#'*70}")

    # ================================================================
    # START vLLM
    # ================================================================
    vllm_proc = start_vllm(model_name, model_tag=args.model_tag, dtype=args.dtype)
    try:
        wait_for_vllm(timeout=args.vllm_timeout)

        # ================================================================
        # STEP 1 — Inference
        # ================================================================
        inference_cmd = [
            sys.executable, str(SCRIPT_DIR / "run_baseline.py"),
            "--model_tag",   args.model_tag,
            "--dataset",     DATASET,
            "--out_dir",     str(OUT_DIR),
            "--concurrency", str(args.concurrency),
            "--max_tokens",  str(args.max_tokens),
            "--temperature", str(args.temperature),
        ]
        if args.model_name:
            inference_cmd += ["--model_name", args.model_name]
        if args.limit_rows is not None:
            inference_cmd += ["--limit_rows", str(args.limit_rows)]

        sh(inference_cmd, "STEP 1/4 — Inference")

    finally:
        # Always stop vLLM after inference, even if inference crashes
        stop_vllm(vllm_proc)

    # ================================================================
    # STEP 2 — Judge  (no vLLM needed — uses OpenAI API)
    # ================================================================
    sh([
        sys.executable, JUDGE_PY,
        "--in",          str(responses_path),
        "--out",         str(judged_path),
        "--concurrency", str(args.judge_concurrency),
        "--print_every", str(args.print_every),
        "--limit_rows",  "0",
    ], "STEP 2/4 — Judge (LLM-as-a-judge)")

    # ================================================================
    # STEP 3 — Eval
    # ================================================================
    sh([
        sys.executable, EVAL_PY,
        "--in",  str(judged_path),
        "--out", str(metrics_path),
    ], "STEP 3/4 — Eval (compute metrics)")

    # ================================================================
    # STEP 4 — Heatmap
    # ================================================================
    model_title = MODEL_TITLES.get(args.model_tag, args.model_tag)
    sh([
        sys.executable, str(HEATMAP_PY),
        "--in",         str(judged_path),
        "--out_prefix", str(heatmap_prefix),
        "--sort",       "overall_desc",
        "--title",      f"{model_title} — Fulfillment rate by dialect × SORRY category",
    ], "STEP 4/4 — Heatmap")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"  PIPELINE COMPLETE — {args.model_tag}")
    print(f"{'#'*70}")
    print(f"  Responses : {responses_path}")
    print(f"  Judged    : {judged_path}")
    print(f"  Metrics   : {metrics_path}")
    print(f"  Heatmap   : {heatmap_prefix}.pdf / .png")
    print()


if __name__ == "__main__":
    main()