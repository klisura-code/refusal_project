#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def fmt_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run(cmd: str, step_name: str, step_desc: str, step_idx: int, total_steps: int, durations: list, cwd: str = None):
    print("\n" + "=" * 110)
    print(f"[STEP {step_idx}/{total_steps}] {step_name}")
    print(step_desc)
    print("-" * 110)
    print("COMMAND:")
    print(cmd.strip())
    print("=" * 110)

    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    dt = time.time() - t0
    durations.append(dt)

    avg = sum(durations) / len(durations)
    remaining = total_steps - step_idx
    eta = avg * remaining

    print("\n" + "-" * 110)
    print(f"✓ Finished: {step_name}")
    print(f"  Step time: {fmt_seconds(dt)}")
    print(f"  Elapsed avg/step: {fmt_seconds(avg)}")
    if remaining > 0:
        print(f"  Rough ETA remaining: {fmt_seconds(eta)}")
    else:
        print("  Rough ETA remaining: done")
    print("-" * 110)


def load_selected_tau(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj["tau"])


def load_probe_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_best_layer(probes_root: Path, model_tag: str, layers: list[int]) -> tuple[int, dict]:
    best_layer = None
    best_metrics = None
    best_auc = -1.0
    best_acc = -1.0

    print("\n" + "#" * 110)
    print("LAYER SWEEP RESULTS")
    print("#" * 110)

    for layer in layers:
        metrics_path = probes_root / f"{model_tag}_prompt_probe_out_layer{layer}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for layer {layer}: {metrics_path}")

        metrics = load_probe_metrics(metrics_path)
        auc = float(metrics.get("auc", -1.0))
        acc = float(metrics.get("acc", -1.0))

        print(f"Layer {layer:>3} | AUC={auc:.4f} | ACC={acc:.4f}")

        if (auc > best_auc) or (auc == best_auc and acc > best_acc):
            best_auc = auc
            best_acc = acc
            best_layer = layer
            best_metrics = metrics

    if best_layer is None:
        raise RuntimeError("Layer selection failed: no valid layer was selected.")

    print("-" * 110)
    print(f"Selected best layer = {best_layer} with AUC={best_auc:.4f}")
    print("#" * 110)

    return best_layer, best_metrics


def main():
    ap = argparse.ArgumentParser()

    # general
    ap.add_argument("--model_name", required=True, help='e.g. "Qwen/Qwen2.5-14B-Instruct"')
    ap.add_argument("--model_tag", required=True, help='Short tag for paths, e.g. "qwen25"')

    # either pass one layer or many
    ap.add_argument("--layer", type=int, default=None, help="Fixed layer to use")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Candidate layers to sweep, e.g. --layers 20 23 27 40 41 42 43 44 45 46 47")

    # datasets
    ap.add_argument("--probe_dataset", required=True, help="Safe/unsafe prompt dataset used to train the probe")
    ap.add_argument("--unsafe_dev_dataset", required=True, help="Unsafe dev set for tau selection")
    ap.add_argument("--unsafe_test_dataset", required=True, help="Unsafe test set for final evaluation")
    ap.add_argument("--safe_dev_dataset", required=True, help="Safe dev set for tau selection")

    # generation/runtime
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # joint tau selection
    ap.add_argument("--tau_beta", type=float, default=0.5,
                    help="Weight on unsafe base fulfillment in score = unsafe_gap + beta*unsafe_base_f")
    ap.add_argument("--tau_gamma", type=float, default=0.0,
                    help="Optional weight on safe gap in score = ... + gamma*safe_gap")
    ap.add_argument("--max_safe_base_refusal", type=float, default=0.10,
                    help="Feasibility constraint for tau selection")
    ap.add_argument("--max_safe_gap", type=float, default=None,
                    help="Optional safe-gap constraint")
    ap.add_argument("--tau_grid", type=int, default=2001)
    ap.add_argument("--tau_topk", type=int, default=20)

    # paths
    ap.add_argument("--root", default="/home/dorde/Desktop/Refusal_project/mitigation")
    args = ap.parse_args()

    if args.layer is None and args.layers is None:
        raise ValueError("You must provide either --layer or --layers")
    if args.layer is not None and args.layers is not None:
        raise ValueError("Provide either --layer or --layers, not both")

    root = Path(args.root).resolve()
    scripts = root / "scripts"
    runs_root = root / "direct_hook_runs"
    probes_root = root / "probes"

    durations = []
    pipeline_t0 = time.time()

    if args.layers is not None:
        total_steps = 2 * len(args.layers) + 5
        candidate_layers = args.layers
    else:
        total_steps = 7
        candidate_layers = [args.layer]

    print("\n" + "#" * 110)
    print("FULL GATED-TAU PIPELINE (JOINT SAFE + UNSAFE TAU SELECTION)")
    print(f"Model:               {args.model_name}")
    print(f"Model tag:           {args.model_tag}")
    if args.layers is not None:
        print(f"Candidate layers:    {candidate_layers}")
    else:
        print(f"Chosen layer:        {args.layer}")
    print(f"Probe dataset:       {args.probe_dataset}")
    print(f"Unsafe dev:          {args.unsafe_dev_dataset}")
    print(f"Unsafe test:         {args.unsafe_test_dataset}")
    print(f"Safe dev:            {args.safe_dev_dataset}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Max new tokens:      {args.max_new_tokens}")
    print(f"Temperature:         {args.temperature}")
    print(f"Seed:                {args.seed}")
    print("-" * 110)
    print("Tau selection objective:")
    print(f"  score = unsafe_gap + {args.tau_beta} * unsafe_base_f + {args.tau_gamma} * safe_gap")
    print(f"  Constraint: safe_base_refusal <= {args.max_safe_base_refusal}")
    if args.max_safe_gap is not None:
        print(f"  Constraint: safe_gap <= {args.max_safe_gap}")
    print("#" * 110)

    step_idx = 1

    # ------------------------------------------------------------------
    # STEP 1A / 1B: Layer sweep (feature extraction + probe training)
    # ------------------------------------------------------------------
    for layer in candidate_layers:
        feat_dir = probes_root / f"{args.model_tag}_prompt_probe_features_layer{layer}"
        probe_out_dir = probes_root / f"{args.model_tag}_prompt_probe_out_layer{layer}"
        feat_dir.mkdir(parents=True, exist_ok=True)
        probe_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = f"""
        python3 {shlex.quote(str(scripts / "build_prompt_probe_features_prompt_end.py"))} \
          --model {shlex.quote(args.model_name)} \
          --in_jsonl {shlex.quote(args.probe_dataset)} \
          --out_dir {shlex.quote(str(feat_dir))} \
          --layer {layer} \
          --batch_size 8
        """
        run(
            cmd,
            f"Build prompt-probe features (layer {layer})",
            f"Extract prompt-end hidden states from the probe dataset for layer {layer}.",
            step_idx, total_steps, durations
        )
        step_idx += 1

        cmd = f"""
        python3 {shlex.quote(str(scripts / "train_refusal_probe_logreg.py"))} \
          --feat_dir {shlex.quote(str(feat_dir))} \
          --out_dir {shlex.quote(str(probe_out_dir))}
        """
        run(
            cmd,
            f"Train linear probe (layer {layer})",
            f"Train logistic regression on layer {layer} features.",
            step_idx, total_steps, durations
        )
        step_idx += 1

    # choose best layer
    if args.layers is not None:
        selected_layer, selected_metrics = select_best_layer(probes_root, args.model_tag, candidate_layers)
        selected_layer_json = probes_root / f"{args.model_tag}_selected_layer.json"
        with open(selected_layer_json, "w", encoding="utf-8") as f:
            json.dump({
                "selected_layer": selected_layer,
                "auc": float(selected_metrics.get("auc", -1.0)),
                "acc": float(selected_metrics.get("acc", -1.0)),
                "metrics": selected_metrics,
            }, f, indent=2)
        print(f"Saved selected layer info -> {selected_layer_json}")
    else:
        selected_layer = args.layer
        selected_metrics = None

    feat_dir = probes_root / f"{args.model_tag}_prompt_probe_features_layer{selected_layer}"
    probe_out_dir = probes_root / f"{args.model_tag}_prompt_probe_out_layer{selected_layer}"

    unsafe_dev_run_dir = runs_root / f"{args.model_tag}_unsafe_dev_scores_layer{selected_layer}"
    safe_dev_run_dir = runs_root / f"{args.model_tag}_safe_dev_scores_layer{selected_layer}"
    final_run_dir = runs_root / f"{args.model_tag}_gated_promptProbe_layer{selected_layer}_final"

    unsafe_dev_run_dir.mkdir(parents=True, exist_ok=True)
    safe_dev_run_dir.mkdir(parents=True, exist_ok=True)
    final_run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 3: collect unsafe dev probe scores
    # ------------------------------------------------------------------
    unsafe_dev_scores = unsafe_dev_run_dir / f"probe_scores_layer{selected_layer}_prompt_end.jsonl"
    cmd = f"""
    python3 {shlex.quote(str(scripts / "collect_probe_scores.py"))} \
      --model {shlex.quote(args.model_name)} \
      --in_jsonl {shlex.quote(args.unsafe_dev_dataset)} \
      --out_jsonl {shlex.quote(str(unsafe_dev_scores))} \
      --probe_dir {shlex.quote(str(probe_out_dir))} \
      --layer {selected_layer} \
      --batch_size {args.batch_size}
    """
    run(
        cmd,
        "Collect unsafe dev probe scores",
        f"Forward-only pass on the unsafe dev set using selected layer {selected_layer}.",
        step_idx, total_steps, durations
    )
    step_idx += 1

    # ------------------------------------------------------------------
    # STEP 4: collect safe dev probe scores
    # ------------------------------------------------------------------
    safe_dev_scores = safe_dev_run_dir / f"probe_scores_layer{selected_layer}_prompt_end.jsonl"
    cmd = f"""
    python3 {shlex.quote(str(scripts / "collect_probe_scores.py"))} \
      --model {shlex.quote(args.model_name)} \
      --in_jsonl {shlex.quote(args.safe_dev_dataset)} \
      --out_jsonl {shlex.quote(str(safe_dev_scores))} \
      --probe_dir {shlex.quote(str(probe_out_dir))} \
      --layer {selected_layer} \
      --batch_size {args.batch_size}
    """
    run(
        cmd,
        "Collect safe dev probe scores",
        f"Forward-only pass on the safe dev set using selected layer {selected_layer}.",
        step_idx, total_steps, durations
    )
    step_idx += 1

    # ------------------------------------------------------------------
    # STEP 5: select tau jointly
    # ------------------------------------------------------------------
    selected_tau_json = final_run_dir / "selected_tau.json"
    selected_tau_unsafe_csv = final_run_dir / "selected_tau_unsafe_metrics.csv"
    selected_tau_safe_csv = final_run_dir / "selected_tau_safe_metrics.csv"

    safe_gap_flag = ""
    if args.max_safe_gap is not None:
        safe_gap_flag = f"--max_safe_gap {args.max_safe_gap}"

    cmd = f"""
    python3 {shlex.quote(str(scripts / "select_tau_joint.py"))} \
      --unsafe_scores_jsonl {shlex.quote(str(unsafe_dev_scores))} \
      --safe_scores_jsonl {shlex.quote(str(safe_dev_scores))} \
      --beta {args.tau_beta} \
      --gamma {args.tau_gamma} \
      --max_safe_base_refusal {args.max_safe_base_refusal} \
      {safe_gap_flag} \
      --tau_grid {args.tau_grid} \
      --topk {args.tau_topk} \
      --out_json {shlex.quote(str(selected_tau_json))} \
      --out_unsafe_csv {shlex.quote(str(selected_tau_unsafe_csv))} \
      --out_safe_csv {shlex.quote(str(selected_tau_safe_csv))}
    """
    run(
        cmd,
        "Select tau jointly (unsafe + safe)",
        "Choose tau under a benign-utility constraint and unsafe fairness/safety objective.",
        step_idx, total_steps, durations
    )
    step_idx += 1

    selected_tau = load_selected_tau(str(selected_tau_json))
    print("\n" + "-" * 110)
    print(f"Selected layer          = {selected_layer}")
    print(f"Selected deployment tau = {selected_tau:.6f}")
    print("-" * 110)

    # ------------------------------------------------------------------
    # STEP 6: final gated decoding on unsafe TEST
    # ------------------------------------------------------------------
    cmd = f"""
    python3 {shlex.quote(str(scripts / "direct_hooks_generate_gated.py"))} \
      --model {shlex.quote(args.model_name)} \
      --in_jsonl {shlex.quote(args.unsafe_test_dataset)} \
      --out_dir {shlex.quote(str(final_run_dir))} \
      --probe_dir {shlex.quote(str(probe_out_dir))} \
      --layer {selected_layer} \
      --tau {selected_tau} \
      --lambdas 0 \
      --max_new_tokens {args.max_new_tokens} \
      --batch_size {args.batch_size} \
      --temperature {args.temperature} \
      --seed {args.seed}
    """
    run(
        cmd,
        "Final gated decoding on unsafe test",
        f"Run probe-gated decoding on the unsafe test set using selected layer {selected_layer} and tau {selected_tau:.6f}.",
        step_idx, total_steps, durations
    )
    step_idx += 1

    # ------------------------------------------------------------------
    # STEP 7: judge + evaluate final unsafe test run
    # ------------------------------------------------------------------
    cmd = f"""
    python3 {shlex.quote(str(scripts / "run_judge_all_sweeps.py"))} \
      --sweep_dir {shlex.quote(str(final_run_dir))} \
      --judge_py {shlex.quote(str(scripts / "judge_refusal.py"))} \
      --eval_py {shlex.quote(str(scripts / "eval_judged.py"))} \
      --concurrency 64 \
      --print_every 500
    """
    run(
        cmd,
        "Judge and evaluate final unsafe test run",
        "Use the LLM judge to label outputs as REFUSED/ANSWERED and compute final per-variant metrics.",
        step_idx, total_steps, durations
    )

    total_dt = time.time() - pipeline_t0

    print("\n" + "#" * 110)
    print("PIPELINE FINISHED")
    print(f"Total elapsed:          {fmt_seconds(total_dt)}")
    print(f"Selected layer:         {selected_layer}")
    print(f"Probe features:         {feat_dir}")
    print(f"Probe output:           {probe_out_dir}")
    print(f"Unsafe dev scores:      {unsafe_dev_scores}")
    print(f"Safe dev scores:        {safe_dev_scores}")
    print(f"Selected tau json:      {selected_tau_json}")
    print(f"Selected tau unsafe:    {selected_tau_unsafe_csv}")
    print(f"Selected tau safe:      {selected_tau_safe_csv}")
    print(f"Final unsafe test run:  {final_run_dir}")
    print("#" * 110)


if __name__ == "__main__":
    main()