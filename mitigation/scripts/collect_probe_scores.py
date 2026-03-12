#!/usr/bin/env python3
import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def load_probe(probe_dir: str) -> Dict[str, np.ndarray]:
    coef = np.load(os.path.join(probe_dir, "coef.npy")).reshape(-1)
    intercept = float(np.load(os.path.join(probe_dir, "intercept.npy")).reshape(-1)[0])
    mu = np.load(os.path.join(probe_dir, "scaler_mean.npy")).reshape(-1)
    sig = np.load(os.path.join(probe_dir, "scaler_scale.npy")).reshape(-1)
    return {"coef": coef, "intercept": intercept, "mu": mu, "sig": sig}


def compute_probe_scores(h_last: torch.Tensor, probe: Dict[str, np.ndarray]) -> np.ndarray:
    h = h_last.float().cpu().numpy()
    D = h.shape[1]
    if probe["coef"].shape[0] != D:
        raise RuntimeError(f"Probe D={probe['coef'].shape[0]} != activation D={D}")
    z = (h - probe["mu"]) / (probe["sig"] + 1e-12)
    return (z @ probe["coef"] + probe["intercept"]).astype(float)


class CaptureHook:
    """
    Capture PRE-anything prompt_end activation (last token) from layer.mlp.down_proj
    """
    def __init__(self):
        self.last_token_acts: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if (not torch.is_tensor(out)) or out.ndim != 3:
            return output
        self.last_token_acts = out[:, -1, :].detach()
        return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)

    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    probe = load_probe(args.probe_dir)
    print(f"Loaded probe from {args.probe_dir} | D={probe['coef'].shape[0]}")

    rows = read_jsonl(args.in_jsonl)
    print(f"Loaded {len(rows)} prompts from {args.in_jsonl}")

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

    hook = CaptureHook()
    handle = layer_mod.mlp.down_proj.register_forward_hook(hook)

    out_rows: List[Dict[str, Any]] = []
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

            with torch.no_grad():
                _ = model(**enc, use_cache=False)

            if hook.last_token_acts is None:
                raise RuntimeError("Hook did not capture last_token_acts.")

            scores = compute_probe_scores(hook.last_token_acts, probe)

            for j, score in enumerate(scores):
                qid, cat, ps, orig_prompt = batch_meta[j]
                out_rows.append({
                    "question_id": qid,
                    "layer": args.layer,
                    "site": "prompt_end",
                    "category": cat,
                    "prompt_style": ps,
                    "prompt": orig_prompt,
                    "probe_score": float(score),
                })

            if (i // args.batch_size) % 25 == 0:
                print(f"Processed {min(i + args.batch_size, len(chat_texts))}/{len(chat_texts)}")

    finally:
        handle.remove()

    write_jsonl(args.out_jsonl, out_rows)
    print(f"Wrote: {args.out_jsonl} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()