#!/usr/bin/env python3
"""
direct_hooks_generate_gated.py

Probe-integrated decoding (Path B1):
1) Run ONE forward pass on the prompt to capture prompt_end activation at a chosen layer/site.
2) Compute refusal-probe score.
3) If score >= tau  -> output a fixed refusal (NO normal generation).
   Else            -> generate normally (optionally with steering via lambda + steer_vec).

This is the “integrate the probe into decoding strategy” step your advisor meant.

Typical usage (your best setting):
python3 direct_hooks_generate_gated.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --in_jsonl prompts_Sorry-DIALECT.jsonl \
  --out_dir direct_hook_runs/gated_tau_lam0_layer44 \
  --probe_dir /home/dorde/Desktop/Refusal_project/mitigation/probe_out_refusal_FINAL_layer44_prompt_end \
  --layer 44 \
  --tau -4.884283 \
  --lambdas 0 \
  --max_new_tokens 128 \
  --batch_size 64 \
  --temperature 0.0 \
  --seed 42

Optional (steering + gating):
  add --steer_vec ... and --lambdas 1
"""

import os, json, argparse, random, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


# ---------------- I/O ----------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- Chat template ----------------

def apply_chat_template(tokenizer, user_text: str, system_text: Optional[str] = None) -> str:
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Could not find transformer layers at model.model.layers")


# ---------------- Probe ----------------

def load_probe(probe_dir: str) -> Dict[str, np.ndarray]:
    coef = np.load(os.path.join(probe_dir, "coef.npy")).reshape(-1)          # (D,)
    intercept = float(np.load(os.path.join(probe_dir, "intercept.npy")).reshape(-1)[0])
    mu = np.load(os.path.join(probe_dir, "scaler_mean.npy")).reshape(-1)     # (D,)
    sig = np.load(os.path.join(probe_dir, "scaler_scale.npy")).reshape(-1)   # (D,)
    return {"coef": coef, "intercept": intercept, "mu": mu, "sig": sig}


def compute_probe_scores(h_last: torch.Tensor, probe: Dict[str, np.ndarray]) -> np.ndarray:
    """
    h_last: torch.Tensor [B, D] (prompt_end last-token activations)
    returns: np.ndarray [B] score logits (higher => more REFUSED-like)
    """
    h = h_last.float().cpu().numpy()
    D = h.shape[1]
    if probe["coef"].shape[0] != D:
        raise RuntimeError(f"Probe D={probe['coef'].shape[0]} != activation D={D}")
    z = (h - probe["mu"]) / (probe["sig"] + 1e-12)
    scores = (z @ probe["coef"] + probe["intercept"]).astype(float)
    return scores


# ---------------- Steering Hook ----------------

class SteeringHook:
    """
    Hook point: layer.mlp.down_proj
    - Captures PRE-steer prompt_end activation (last token): last_token_acts [B,D]
    - Optionally applies steering to last token: out[:, -1, :] += lambda * steer_vec
    """
    def __init__(self, steer_vec: Optional[torch.Tensor], scale: float, capture: bool = True):
        self.steer_vec = steer_vec  # [D] or None
        self.scale = float(scale)
        self.capture = bool(capture)
        self.last_token_acts: Optional[torch.Tensor] = None  # [B, D]

    def __call__(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if (not torch.is_tensor(out)) or out.ndim != 3:
            return output

        # Capture PRE-steer activation (what your probe expects)
        if self.capture:
            self.last_token_acts = out[:, -1, :].detach()

        # Apply steering (optional)
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

def slice_batch(enc: Dict[str, torch.Tensor], idx: List[int]) -> Dict[str, torch.Tensor]:
    if len(idx) == 0:
        return {}
    return {k: v[idx] for k, v in enc.items()}


def refusal_text_template() -> str:
    # Keep it stable across dialects (important for disparity)
    return (
        "I can’t help with that request. "
        "If you want, I can help with a safe alternative or general information instead."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)

    # Probe gate (required)
    ap.add_argument("--probe_dir", required=True,
                    help="Dir with coef.npy/intercept.npy/scaler_mean.npy/scaler_scale.npy")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tau", type=float, required=True,
                    help="Refuse if probe_score >= tau")

    # Steering (optional)
    ap.add_argument("--steer_vec", default=None,
                    help="Optional: path to 1D steer vector .pt (for lambda steering)")
    ap.add_argument("--lambdas", nargs="+", type=float, required=True,
                    help="Run one or more lambdas (e.g., 0 or 0 1)")

    # Generation
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system", type=str, default=None)

    # Runtime
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)

    # Output options
    ap.add_argument("--save_probe_score", action="store_true", default=True,
                    help="Always recommended (kept on by default)")
    args = ap.parse_args()

    random.seed(args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Decoder-only batching correctness
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,  # avoid torch_dtype deprecation
        device_map=args.device_map,
        trust_remote_code=True,
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
            raise RuntimeError(f"--steer_vec must be 1D tensor; got {type(steer_vec)} shape={getattr(steer_vec,'shape',None)}")
        print(f"Loaded steer vec: shape={tuple(steer_vec.shape)} from {args.steer_vec}")

    rows = read_jsonl(args.in_jsonl)
    print(f"Loaded {len(rows)} prompts from {args.in_jsonl}")

    # Prebuild chat texts + meta
    chat_texts: List[str] = []
    meta: List[Tuple[Any, Any, Any, str]] = []
    for r in rows:
        qid = r.get("question_id", None)
        cat = r.get("category", None)
        ps = r.get("prompt_style", None)
        prompt = r["prompt"]
        chat = apply_chat_template(tok, prompt, args.system)
        chat_texts.append(chat)
        meta.append((qid, cat, ps, prompt))

    layers = get_layers(model)
    layer_mod = layers[args.layer]
    if not (hasattr(layer_mod, "mlp") and hasattr(layer_mod.mlp, "down_proj")):
        raise RuntimeError("Expected layer.mlp.down_proj; adjust hook point for your model impl.")

    os.makedirs(args.out_dir, exist_ok=True)

    for lam in args.lambdas:
        lam = float(lam)
        print(f"\n=== GATED RUN | lambda={lam} | layer={args.layer} | tau={args.tau} | site=prompt_end ===")

        # Hook captures prompt_end activations and optionally applies steering (lam)
        hook = SteeringHook(
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
            for i in range(0, len(chat_texts), args.batch_size):
                batch_texts = chat_texts[i:i + args.batch_size]
                batch_meta = meta[i:i + args.batch_size]

                enc = tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_prompt_tokens,
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}

                # 1) Forward pass ONLY to capture prompt_end activations (and apply steering at prompt_end)
                with torch.no_grad():
                    _ = model(**enc, use_cache=False)

                if hook.last_token_acts is None:
                    raise RuntimeError("Hook failed to capture last_token_acts on forward pass.")

                scores = compute_probe_scores(hook.last_token_acts, probe)  # (B,)
                gate = scores >= float(args.tau)

                # Indices
                idx_refuse = [j for j, g in enumerate(gate.tolist()) if g]
                idx_allow = [j for j, g in enumerate(gate.tolist()) if not g]

                refused_by_probe += len(idx_refuse)
                allowed += len(idx_allow)

                # 2) For allowed subset: generate normally
                allow_texts: Dict[int, str] = {}
                if len(idx_allow) > 0:
                    enc_allow = slice_batch(enc, idx_allow)
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **enc_allow,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature if args.do_sample else None,
                            top_p=args.top_p if args.do_sample else None,
                            pad_token_id=tok.pad_token_id,
                        )

                    attn = enc_allow["attention_mask"]
                    prompt_lens = attn.sum(dim=1).tolist()

                    for k in range(gen_ids.shape[0]):
                        p_len = int(prompt_lens[k])
                        comp_ids = gen_ids[k, p_len:]
                        allow_texts[idx_allow[k]] = tok.decode(comp_ids, skip_special_tokens=True)

                # 3) Emit rows in original order
                for j in range(len(batch_texts)):
                    qid, cat, ps, orig_prompt = batch_meta[j]
                    g = bool(gate[j])
                    out_text = refusal_text_template() if g else allow_texts.get(j, "")

                    rec: Dict[str, Any] = {
                        "question_id": qid,
                        "lambda": lam,
                        "layer": args.layer,
                        "site": "prompt_end",
                        "tau": float(args.tau),
                        "gate_decision": "REFUSE_BY_PROBE" if g else "ALLOW",
                        "category": cat,
                        "prompt_style": ps,
                        "prompt": orig_prompt,
                        "output_text": out_text,
                    }
                    if args.save_probe_score:
                        rec["probe_score"] = float(scores[j])
                    out_rows.append(rec)

        finally:
            handle.remove()

        out_path = os.path.join(
            args.out_dir,
            f"gated_tau_{args.tau:.6f}_lam_{lam:.3f}_layer{args.layer}_prompt_end.jsonl"
        )
        write_jsonl(out_path, out_rows)

        dt = time.time() - t0
        print(f"Wrote: {out_path} ({len(out_rows)} rows)")
        print(f"Probe-gated refusals: {refused_by_probe:,} | Allowed to generate: {allowed:,} | elapsed={dt/60:.2f} min")


if __name__ == "__main__":
    main()