#!/usr/bin/env python3
"""
direct_hooks_generate_gated.py

Prompt-probe-gated decoding:
1) Run one forward pass on the prompt to capture prompt-end activation at layer.mlp.down_proj.
2) Compute prompt probe score.
3) If score >= tau  -> output a fixed refusal (NO normal generation).
   Else             -> generate normally.

Optional:
- steering can still be applied at the same hook site for allowed generations.

This version is aligned with the prompt safe/unsafe probe pipeline:
- training features: layer.mlp.down_proj, last token, left-padded prompt-only input
- scoring features : same
- gating rule      : REFUSE if score >= tau
"""

import os
import json
import argparse
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


# ---------------- I/O ----------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- Prompt helpers ----------------

def get_prompt_text(row: Dict[str, Any]) -> Optional[str]:
    for key in ["prompt", "text", "instruction", "question"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def apply_chat(tokenizer, user_text: str, system_text: Optional[str] = None) -> torch.Tensor:
    system_msg = system_text if system_text else "You are a helpful assistant."
    
    # Gemma does not support system role — merge into user turn
    if hasattr(tokenizer, 'name_or_path') and 'gemma' in tokenizer.name_or_path.lower():
        msgs = [
            {"role": "user", "content": f"{system_msg}\n\n{user_text}"},
        ]
    else:
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ]

    ids = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return ids


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Could not find transformer layers at model.model.layers")


# ---------------- Probe ----------------

def load_probe(probe_dir: str) -> Dict[str, np.ndarray]:
    coef = np.load(os.path.join(probe_dir, "coef.npy")).reshape(-1)
    intercept = float(np.load(os.path.join(probe_dir, "intercept.npy")).reshape(-1)[0])
    mu = np.load(os.path.join(probe_dir, "scaler_mean.npy")).reshape(-1)
    sig = np.load(os.path.join(probe_dir, "scaler_scale.npy")).reshape(-1)

    if not (coef.shape == mu.shape == sig.shape):
        raise RuntimeError(
            f"Probe shape mismatch: coef={coef.shape}, mu={mu.shape}, sig={sig.shape}"
        )

    return {"coef": coef, "intercept": intercept, "mu": mu, "sig": sig}


def compute_probe_scores(h_last: torch.Tensor, probe: Dict[str, np.ndarray]) -> np.ndarray:
    """
    h_last: torch.Tensor [B, D] from prompt-end activation
    returns: np.ndarray [B] raw logit scores

    For the prompt safe/unsafe probe:
      higher score = more unsafe-like
    """
    h = h_last.float().cpu().numpy()
    if h.ndim != 2:
        raise RuntimeError(f"Expected h_last shape (B, D), got {h.shape}")

    D = h.shape[1]
    if probe["coef"].shape[0] != D:
        raise RuntimeError(f"Probe D={probe['coef'].shape[0]} != activation D={D}")

    z = (h - probe["mu"]) / (probe["sig"] + 1e-12)
    scores = (z @ probe["coef"] + probe["intercept"]).astype(float)
    return scores


# ---------------- Hook ----------------

class PromptEndHook:
    """
    Hook point: layer.mlp.down_proj
    - Captures prompt-end activation at the last token: last_token_acts [B, D]
    - Optionally applies steering to the last token only
    """
    def __init__(self, steer_vec: Optional[torch.Tensor], scale: float, capture: bool = True):
        self.steer_vec = steer_vec
        self.scale = float(scale)
        self.capture = bool(capture)
        self.last_token_acts: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if (not torch.is_tensor(out)) or out.ndim != 3:
            return output

        # Capture pre-steer prompt-end activation
        if self.capture:
            self.last_token_acts = out[:, -1, :].detach()

        # Optional steering on last token only
        if self.steer_vec is not None and self.scale != 0.0:
            vec = self.steer_vec.to(out.device, dtype=out.dtype)
            B, T, D = out.shape
            if vec.numel() != D:
                raise RuntimeError(f"Steer vec dim {vec.numel()} != activation dim {D}")
            out[:, -1, :] = out[:, -1, :] + (self.scale * vec)

        if isinstance(output, tuple):
            return (out, *output[1:])
        return out


# ---------------- Helpers ----------------

def refusal_text_template() -> str:
    return (
        "I can’t help with that request. "
        "I can help with a safe alternative or general information instead."
    )


def gather_batch_tensors(enc: Dict[str, torch.Tensor], idxs: List[int]) -> Dict[str, torch.Tensor]:
    return {k: v[idxs] for k, v in enc.items()}


def decode_new_tokens(tok, gen_ids: torch.Tensor, prompt_input_ids: torch.Tensor) -> List[str]:
    """
    Robust decoding for left-padded decoder-only generation:
    slice off the original prompt length using input_ids width, not attention_mask.sum().
    """
    prompt_width = prompt_input_ids.shape[1]
    outs = []
    for k in range(gen_ids.shape[0]):
        comp_ids = gen_ids[k, prompt_width:]
        text = tok.decode(comp_ids, skip_special_tokens=True)
        outs.append(text)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)

    # probe gate
    ap.add_argument("--probe_dir", required=True,
                    help="Dir with coef.npy/intercept.npy/scaler_mean.npy/scaler_scale.npy")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tau", type=float, required=True,
                    help="Refuse if probe_score >= tau")

    # optional steering
    ap.add_argument("--steer_vec", default=None,
                    help="Optional path to 1D steer vector .pt")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0],
                    help="One or more steering scales, e.g. 0 or 0 1")

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system", type=str, default=None)

    # runtime
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0)

    # output options
    ap.add_argument("--save_probe_score", action="store_true", default=True)
    args = ap.parse_args()

    random.seed(args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()

    probe = load_probe(args.probe_dir)
    print(f"Loaded probe from {args.probe_dir} | D={probe['coef'].shape[0]}")

    steer_vec = None
    if args.steer_vec is not None:
        steer_vec = torch.load(args.steer_vec, map_location="cpu")
        if isinstance(steer_vec, dict) and "tensor" in steer_vec:
            steer_vec = steer_vec["tensor"]
        if not torch.is_tensor(steer_vec) or steer_vec.ndim != 1:
            raise RuntimeError(
                f"--steer_vec must be a 1D tensor; got {type(steer_vec)} shape={getattr(steer_vec, 'shape', None)}"
            )
        print(f"Loaded steer vec: shape={tuple(steer_vec.shape)} from {args.steer_vec}")

    rows = read_jsonl(args.in_jsonl)
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"Loaded {len(rows)} rows from {args.in_jsonl}")

    filtered_rows: List[Dict[str, Any]] = []
    skipped_missing_prompt = 0
    for r in rows:
        prompt = get_prompt_text(r)
        if prompt is None:
            skipped_missing_prompt += 1
            continue
        rr = dict(r)
        rr["_prompt_text"] = prompt
        filtered_rows.append(rr)

    rows = filtered_rows
    print(f"Usable rows: {len(rows)} | skipped missing prompt: {skipped_missing_prompt}")

    layers = get_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise RuntimeError(f"Layer index {args.layer} out of range for model with {len(layers)} layers")

    layer_mod = layers[args.layer]
    if not (hasattr(layer_mod, "mlp") and hasattr(layer_mod.mlp, "down_proj")):
        raise RuntimeError("Expected layer.mlp.down_proj; adjust hook point for your model implementation.")

    os.makedirs(args.out_dir, exist_ok=True)

    for lam in args.lambdas:
        lam = float(lam)
        print(f"\n=== GATED RUN | lambda={lam} | layer={args.layer} | tau={args.tau} ===")

        hook = PromptEndHook(
            steer_vec=steer_vec,
            scale=lam,
            capture=True,
        )
        handle = layer_mod.mlp.down_proj.register_forward_hook(hook)

        out_rows: List[Dict[str, Any]] = []
        t0 = time.time()
        refused_by_probe = 0
        allowed = 0

        try:
            with torch.no_grad():
                for i in range(0, len(rows), args.batch_size):
                    batch = rows[i:i + args.batch_size]
                    prompts = [b["_prompt_text"] for b in batch]

                    enc_list = [apply_chat(tok, p, args.system)[0] for p in prompts]
                    enc = tok.pad({"input_ids": enc_list}, return_tensors="pt")

                    input_ids = enc["input_ids"].to(model.device)
                    attention_mask = enc["attention_mask"].to(model.device)

                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }

                    # Step 1: prompt forward pass to capture prompt-end activations
                    hook.last_token_acts = None
                    _ = model(**model_inputs, use_cache=False, return_dict=True)

                    if hook.last_token_acts is None:
                        raise RuntimeError("Hook failed to capture last_token_acts on forward pass.")

                    # Step 2: compute prompt probe scores
                    scores = compute_probe_scores(hook.last_token_acts, probe)

                    # IMPORTANT: prompt safe/unsafe probe semantics
                    # higher score = more unsafe-like
                    gate = scores >= float(args.tau)

                    idx_refuse = [j for j, g in enumerate(gate.tolist()) if g]
                    idx_allow = [j for j, g in enumerate(gate.tolist()) if not g]

                    refused_by_probe += len(idx_refuse)
                    allowed += len(idx_allow)

                    # Step 3: generate only for allowed subset
                    allow_texts: Dict[int, str] = {}

                    if len(idx_allow) > 0:
                        enc_allow = gather_batch_tensors(model_inputs, idx_allow)

                        gen_kwargs = dict(
                            **enc_allow,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            pad_token_id=tok.pad_token_id,
                        )
                        if args.do_sample:
                            gen_kwargs["temperature"] = args.temperature
                            gen_kwargs["top_p"] = args.top_p

                        gen_ids = model.generate(**gen_kwargs)
                        decoded = decode_new_tokens(tok, gen_ids, enc_allow["input_ids"])

                        for local_k, text in enumerate(decoded):
                            original_batch_index = idx_allow[local_k]
                            allow_texts[original_batch_index] = text

                    # Step 4: emit records in original order
                    for j, r in enumerate(batch):
                        g = bool(gate[j])
                        out_text = refusal_text_template() if g else allow_texts.get(j, "")

                        rec: Dict[str, Any] = {
                            "question_id": r.get("question_id"),
                            "probe_id": r.get("probe_id"),
                            "lambda": lam,
                            "layer": args.layer,
                            "site": "prompt_end_down_proj_last_token",
                            "tau": float(args.tau),
                            "gate_decision": "REFUSE_BY_PROBE" if g else "ALLOW",
                            "category": r.get("category", ""),
                            "prompt_style": r.get("prompt_style", ""),
                            "source": r.get("source", ""),
                            "label": r.get("label", r.get("source_label")),
                            "prompt": r["_prompt_text"],
                            "output_text": out_text,
                        }
                        if args.save_probe_score:
                            rec["probe_score"] = float(scores[j])
                        out_rows.append(rec)

                    done = min(i + args.batch_size, len(rows))
                    if (i // args.batch_size) % 25 == 0:
                        print(
                            f"Processed {done}/{len(rows)} | "
                            f"batch_refused={len(idx_refuse)} | batch_allowed={len(idx_allow)}"
                        )

        finally:
            handle.remove()

        out_path = os.path.join(
            args.out_dir,
            f"gated_tau_{args.tau:.6f}_lam_{lam:.3f}_layer{args.layer}_prompt_end.jsonl"
        )
        write_jsonl(out_path, out_rows)

        dt = time.time() - t0
        print(f"Wrote: {out_path} ({len(out_rows)} rows)")
        print(
            f"Probe-gated refusals: {refused_by_probe:,} | "
            f"Allowed to generate: {allowed:,} | "
            f"elapsed={dt/60:.2f} min"
        )


if __name__ == "__main__":
    main()