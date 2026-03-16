#!/usr/bin/env python3
import os
import json
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def get_prompt_text(row: Dict[str, Any]) -> Optional[str]:
    for key in ["prompt", "text", "instruction", "question"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def apply_chat(tok, prompt: str, system_text: Optional[str] = None) -> torch.Tensor:
    system_msg = system_text if system_text else "You are a helpful assistant."
    
    # Gemma does not support system role — merge into user turn
    if hasattr(tok, 'name_or_path') and 'gemma' in tok.name_or_path.lower():
        msgs = [
            {"role": "user", "content": f"{system_msg}\n\n{prompt}"},
        ]
    else:
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    ids = tok.apply_chat_template(
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
    h = h_last.float().cpu().numpy()
    if h.ndim != 2:
        raise RuntimeError(f"Expected h_last to have shape (B, D), got {h.shape}")

    D = h.shape[1]
    if probe["coef"].shape[0] != D:
        raise RuntimeError(f"Probe D={probe['coef'].shape[0]} != activation D={D}")

    z = (h - probe["mu"]) / (probe["sig"] + 1e-12)
    logits = z @ probe["coef"] + probe["intercept"]
    return logits.astype(float)


class DownProjLastTokenHook:
    """
    Capture the last-token activation at layer.mlp.down_proj.
    This must match the feature extraction site used during probe training.
    """
    def __init__(self):
        self.last_token_acts: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if (not torch.is_tensor(out)) or out.ndim != 3:
            return output
        # Because we use left padding and prompt-only inputs, the prompt-end token is the last position.
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
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

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

    hook = DownProjLastTokenHook()
    handle = layer_mod.mlp.down_proj.register_forward_hook(hook)

    out_rows: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                prompts = [b["_prompt_text"] for b in batch]

                enc_list = [apply_chat(tok, p, args.system)[0] for p in prompts]
                enc = tok.pad({"input_ids": enc_list}, return_tensors="pt")

                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)

                hook.last_token_acts = None

                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

                if hook.last_token_acts is None:
                    raise RuntimeError("Hook did not capture last_token_acts.")

                scores = compute_probe_scores(hook.last_token_acts, probe)

                for j, score in enumerate(scores):
                    r = batch[j]
                    out_row = {
                        "question_id": r.get("question_id"),
                        "probe_id": r.get("probe_id"),
                        "layer": args.layer,
                        "site": "prompt_end_down_proj_last_token",
                        "category": r.get("category", ""),
                        "prompt_style": r.get("prompt_style", ""),
                        "source": r.get("source", ""),
                        "label": r.get("label", r.get("source_label")),
                        "prompt": r["_prompt_text"],
                        "probe_score": float(score),
                    }
                    out_rows.append(out_row)

                done = min(i + args.batch_size, len(rows))
                if (i // args.batch_size) % 25 == 0:
                    print(f"Processed {done}/{len(rows)}")

    finally:
        handle.remove()

    write_jsonl(args.out_jsonl, out_rows)
    print(f"Wrote: {args.out_jsonl} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()