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
                    help="Candidate layers to sweep, e.g. --layers 20 28 36 44 48")

    # datasets
    ap.add_argument("--probe_train_dataset", required=True, help="Prompt-labeled train dataset (safe/unsafe)")
    ap.add_argument("--unsafe_tau_dataset", required=True, help="Deployment-like unsafe tau calibration set")
    ap.add_argument("--safe_tau_dataset", required=True, help="Deployment-like safe tau calibration set")
    ap.add_argument("--safe_final_dataset", required=True, help="Safe final test dataset")
    ap.add_argument("--unsafe_final_dataset", required=True, help="Unsafe final test dataset")

    # runtime
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")

    # tau selection
    ap.add_argument("--tau_gamma", type=float, default=0.0)
    ap.add_argument("--max_unsafe_base_fulfillment", type=float, default=0.20)
    ap.add_argument("--max_safe_base_refusal", type=float, default=0.10)
    ap.add_argument("--max_safe_gap", type=float, default=None)
    ap.add_argument("--tau_grid", type=int, default=5001)
    ap.add_argument("--tau_topk", type=int, default=20)

    # optional evaluation of safe final set
    ap.add_argument("--run_safe_final", action="store_true",
                    help="Also run gated decoding + judge/eval on safe final dataset")

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

    runs_root.mkdir(parents=True, exist_ok=True)
    probes_root.mkdir(parents=True, exist_ok=True)

    durations = []
    pipeline_t0 = time.time()

    if args.layers is not None:
        candidate_layers = args.layers
    else:
        candidate_layers = [args.layer]

    total_steps = 2 * len(candidate_layers) + 4
    if args.run_safe_final:
        total_steps += 2

    step_idx = 1

    # ------------------------------------------------------------------
    # STEP 1A / 1B: layer sweep (feature extraction + probe training)
    # ------------------------------------------------------------------
    for layer in candidate_layers:
        feat_dir = probes_root / f"{args.model_tag}_prompt_probe_features_layer{layer}"
        probe_out_dir = probes_root / f"{args.model_tag}_prompt_probe_out_layer{layer}"
        feat_dir.mkdir(parents=True, exist_ok=True)
        probe_out_dir.mkdir(parents=True, exist_ok=True)

        x_file = feat_dir / "X.pt"
        y_file = feat_dir / "y.pt"
        metrics_file = probe_out_dir / "metrics.json"

        # build features
        if not (x_file.exists() and y_file.exists()):
            cmd = f"""
            python3 {shlex.quote(str(scripts / "build_prompt_probe_features_prompt_end.py"))} \
              --model {shlex.quote(args.model_name)} \
              --in_jsonl {shlex.quote(args.probe_train_dataset)} \
              --out_dir {shlex.quote(str(feat_dir))} \
              --layer {layer} \
              --batch_size 8
            """
            run(
                cmd,
                f"Build prompt-probe features (layer {layer})",
                f"Extract prompt-end down_proj features for prompt safe/unsafe training at layer {layer}.",
                step_idx, total_steps, durations
            )
        else:
            print(f"⊘ Skipped STEP {step_idx}: Features for layer {layer} already exist")
        step_idx += 1

        # train probe
        if not metrics_file.exists():
            cmd = f"""
            python3 {shlex.quote(str(scripts / "train_refusal_probe_logreg.py"))} \
              --feat_dir {shlex.quote(str(feat_dir))} \
              --out_dir {shlex.quote(str(probe_out_dir))}
            """
            run(
                cmd,
                f"Train prompt probe (layer {layer})",
                f"Train logistic regression prompt probe for safe (0) vs unsafe (1) at layer {layer}.",
                step_idx, total_steps, durations
            )
        else:
            print(f"⊘ Skipped STEP {step_idx}: Probe for layer {layer} already trained")
        step_idx += 1

    # choose best layer if sweep
    if args.layers is not None:
        selected_layer, selected_metrics = select_best_layer(probes_root, args.model_tag, candidate_layers)
        selected_layer_json = probes_root / f"{args.model_tag}_selected_prompt_probe_layer.json"
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

    probe_out_dir = probes_root / f"{args.model_tag}_prompt_probe_out_layer{selected_layer}"

    tau_run_dir = runs_root / f"{args.model_tag}_tau_scores_layer{selected_layer}"
    final_run_dir = runs_root / f"{args.model_tag}_gated_promptProbe_layer{selected_layer}_final"
    safe_final_run_dir = runs_root / f"{args.model_tag}_gated_promptProbe_layer{selected_layer}_safe_final"

    tau_run_dir.mkdir(parents=True, exist_ok=True)
    final_run_dir.mkdir(parents=True, exist_ok=True)
    safe_final_run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 3: collect unsafe tau probe scores
    # ------------------------------------------------------------------
    unsafe_tau_scores = tau_run_dir / f"unsafe_tau_scores_layer{selected_layer}.jsonl"
    if not unsafe_tau_scores.exists():
        cmd = f"""
        python3 {shlex.quote(str(scripts / "collect_probe_scores.py"))} \
          --model {shlex.quote(args.model_name)} \
          --in_jsonl {shlex.quote(args.unsafe_tau_dataset)} \
          --out_jsonl {shlex.quote(str(unsafe_tau_scores))} \
          --probe_dir {shlex.quote(str(probe_out_dir))} \
          --layer {selected_layer} \
          --batch_size {args.batch_size} \
          --dtype {args.dtype}
        """
        run(
            cmd,
            "Collect unsafe tau probe scores",
            f"Score deployment-like unsafe tau set using selected layer {selected_layer}.",
            step_idx, total_steps, durations
        )
    else:
        print(f"⊘ Skipped STEP {step_idx}: Unsafe tau scores already exist")
    step_idx += 1

    # ------------------------------------------------------------------
    # STEP 4: collect safe tau probe scores
    # ------------------------------------------------------------------
    safe_tau_scores = tau_run_dir / f"safe_tau_scores_layer{selected_layer}.jsonl"
    if not safe_tau_scores.exists():
        cmd = f"""
        python3 {shlex.quote(str(scripts / "collect_probe_scores.py"))} \
          --model {shlex.quote(args.model_name)} \
          --in_jsonl {shlex.quote(args.safe_tau_dataset)} \
          --out_jsonl {shlex.quote(str(safe_tau_scores))} \
          --probe_dir {shlex.quote(str(probe_out_dir))} \
          --layer {selected_layer} \
          --batch_size {args.batch_size} \
          --dtype {args.dtype}
        """
        run(
            cmd,
            "Collect safe tau probe scores",
            f"Score deployment-like safe tau set using selected layer {selected_layer}.",
            step_idx, total_steps, durations
        )
    else:
        print(f"⊘ Skipped STEP {step_idx}: Safe tau scores already exist")
    step_idx += 1

    # ------------------------------------------------------------------
    # STEP 5: select tau jointly
    # ------------------------------------------------------------------
    selected_tau_json = final_run_dir / "selected_tau.json"
    selected_tau_unsafe_csv = final_run_dir / "selected_tau_unsafe_metrics.csv"
    selected_tau_safe_csv = final_run_dir / "selected_tau_safe_metrics.csv"

    if not selected_tau_json.exists():
        safe_gap_flag = ""
        if args.max_safe_gap is not None:
            safe_gap_flag = f"--max_safe_gap {args.max_safe_gap}"

        cmd = f"""
        python3 {shlex.quote(str(scripts / "select_tau_joint.py"))} \
          --unsafe_scores_jsonl {shlex.quote(str(unsafe_tau_scores))} \
          --safe_scores_jsonl {shlex.quote(str(safe_tau_scores))} \
          --refuse_if ge \
          --gamma {args.tau_gamma} \
          --max_unsafe_base_fulfillment {args.max_unsafe_base_fulfillment} \
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
            "Choose tau for prompt-probe gating under unsafe/safe feasibility constraints on deployment-like calibration sets.",
            step_idx, total_steps, durations
        )
    else:
        print(f"⊘ Skipped STEP {step_idx}: Tau already selected")
    step_idx += 1

    selected_tau = load_selected_tau(str(selected_tau_json))
    print("\n" + "-" * 110)
    print(f"Selected layer          = {selected_layer}")
    print(f"Selected deployment tau = {selected_tau:.6f}")
    print(f"Constraints applied:")
    print(f"  - Unsafe base fulfillment <= {args.max_unsafe_base_fulfillment}")
    print(f"  - Safe base refusal <= {args.max_safe_base_refusal}")
    print("-" * 110)

    # ------------------------------------------------------------------
    # STEP 6: final gated decoding on unsafe FINAL
    # ------------------------------------------------------------------
    unsafe_final_jsonl = final_run_dir / f"gated_tau_{selected_tau:.6f}_lam_0.000_layer{selected_layer}_prompt_end.jsonl"
    if not unsafe_final_jsonl.exists():
        cmd = f"""
        python3 {shlex.quote(str(scripts / "direct_hooks_generate_gated.py"))} \
          --model {shlex.quote(args.model_name)} \
          --in_jsonl {shlex.quote(args.unsafe_final_dataset)} \
          --out_dir {shlex.quote(str(final_run_dir))} \
          --probe_dir {shlex.quote(str(probe_out_dir))} \
          --layer {selected_layer} \
          --tau {selected_tau} \
          --lambdas 0 \
          --max_new_tokens {args.max_new_tokens} \
          --batch_size {args.batch_size} \
          --temperature {args.temperature} \
          --seed {args.seed} \
          --dtype {args.dtype}
        """
        run(
            cmd,
            "Final gated decoding on unsafe final test",
            f"Run prompt-probe-gated decoding on unsafe final test using layer {selected_layer} and tau {selected_tau:.6f}.",
            step_idx, total_steps, durations
        )
    else:
        print(f"⊘ Skipped STEP {step_idx}: Unsafe final gated decoding already exists")
    step_idx += 1

    # ------------------------------------------------------------------
    # STEP 7: judge + evaluate unsafe final run
    # ------------------------------------------------------------------
    unsafe_eval_done = list(final_run_dir.glob("*.metrics.csv"))
    if not unsafe_eval_done:
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
            "Judge and evaluate unsafe final run",
            "Judge prompt-probe-gated unsafe final outputs as REFUSED/ANSWERED and compute per-variant metrics.",
            step_idx, total_steps, durations
        )
    else:
        print(f"⊘ Skipped STEP {step_idx}: Unsafe final evaluation already exists")
    step_idx += 1

    # ------------------------------------------------------------------
    # OPTIONAL STEP 8/9: safe final evaluation
    # ------------------------------------------------------------------
    if args.run_safe_final:
        safe_final_jsonl = safe_final_run_dir / f"gated_tau_{selected_tau:.6f}_lam_0.000_layer{selected_layer}_prompt_end.jsonl"
        if not safe_final_jsonl.exists():
            cmd = f"""
            python3 {shlex.quote(str(scripts / "direct_hooks_generate_gated.py"))} \
              --model {shlex.quote(args.model_name)} \
              --in_jsonl {shlex.quote(args.safe_final_dataset)} \
              --out_dir {shlex.quote(str(safe_final_run_dir))} \
              --probe_dir {shlex.quote(str(probe_out_dir))} \
              --layer {selected_layer} \
              --tau {selected_tau} \
              --lambdas 0 \
              --max_new_tokens {args.max_new_tokens} \
              --batch_size {args.batch_size} \
              --temperature {args.temperature} \
              --seed {args.seed} \
              --dtype {args.dtype}
            """
            run(
                cmd,
                "Final gated decoding on safe final test",
                f"Run prompt-probe-gated decoding on safe final test using layer {selected_layer} and tau {selected_tau:.6f}.",
                step_idx, total_steps, durations
            )
        else:
            print(f"⊘ Skipped STEP {step_idx}: Safe final gated decoding already exists")
        step_idx += 1

        safe_eval_done = list(safe_final_run_dir.glob("*.metrics.csv"))
        if not safe_eval_done:
            cmd = f"""
            python3 {shlex.quote(str(scripts / "run_judge_all_sweeps.py"))} \
              --sweep_dir {shlex.quote(str(safe_final_run_dir))} \
              --judge_py {shlex.quote(str(scripts / "judge_refusal.py"))} \
              --eval_py {shlex.quote(str(scripts / "eval_judged.py"))} \
              --concurrency 64 \
              --print_every 500
            """
            run(
                cmd,
                "Judge and evaluate safe final run",
                "Judge prompt-probe-gated safe final outputs and compute safe over-refusal metrics.",
                step_idx, total_steps, durations
            )
        else:
            print(f"⊘ Skipped STEP {step_idx}: Safe final evaluation already exists")
        step_idx += 1

    total_dt = time.time() - pipeline_t0

    print("\n" + "#" * 110)
    print("PROMPT PROBE PIPELINE FINISHED")
    print(f"Total elapsed:          {fmt_seconds(total_dt)}")
    print(f"Selected layer:         {selected_layer}")
    print(f"Probe train dataset:    {args.probe_train_dataset}")
    print(f"Unsafe tau dataset:     {args.unsafe_tau_dataset}")
    print(f"Safe tau dataset:       {args.safe_tau_dataset}")
    print(f"Probe output:           {probe_out_dir}")
    print(f"Unsafe tau scores:      {unsafe_tau_scores}")
    print(f"Safe tau scores:        {safe_tau_scores}")
    print(f"Selected tau json:      {selected_tau_json}")
    print(f"Selected tau unsafe:    {selected_tau_unsafe_csv}")
    print(f"Selected tau safe:      {selected_tau_safe_csv}")
    print(f"Unsafe final run:       {final_run_dir}")
    if args.run_safe_final:
        print(f"Safe final run:         {safe_final_run_dir}")
    print("#" * 110)


if __name__ == "__main__":
    main()