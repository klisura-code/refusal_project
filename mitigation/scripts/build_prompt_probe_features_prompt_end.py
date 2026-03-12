#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e


def apply_chat(tok, prompt: str) -> torch.Tensor:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--layer", type=int, default=44)
    ap.add_argument("--layer_mode", choices=["mean", "concat"], default="mean")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    for r in iter_jsonl(args.in_jsonl):
        if "prompt" not in r or "label" not in r:
            continue
        rows.append(r)
        if args.max_rows and len(rows) >= args.max_rows:
            break

    if not rows:
        raise SystemExit("No rows with prompt/label found.")

    if args.layers is None:
        layers = [args.layer]
    else:
        layers = list(args.layers)

    print(f"Using layers={layers} mode={args.layer_mode}")

    X_list: List[torch.Tensor] = []
    y_list: List[int] = []

    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    meta_f = open(meta_path, "w", encoding="utf-8")

    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            prompts = [b["prompt"] for b in batch]

            enc_list = [apply_chat(tok, p)[0] for p in prompts]
            enc = tok.pad({"input_ids": enc_list}, return_tensors="pt")

            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)
            prompt_lens = attn_mask.sum(dim=1)
            idx = (prompt_lens - 1).to(torch.long)

            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            vecs_per_layer: List[torch.Tensor] = []
            for L in layers:
                hs = out.hidden_states[L]   # (B, T, D)
                B, T, D = hs.shape
                v = hs[torch.arange(B, device=hs.device), idx, :]   # (B, D)
                vecs_per_layer.append(v)

            if len(vecs_per_layer) == 1:
                feats = vecs_per_layer[0]
            else:
                if args.layer_mode == "mean":
                    feats = torch.stack(vecs_per_layer, dim=0).mean(dim=0)
                else:
                    feats = torch.cat(vecs_per_layer, dim=-1)

            X_list.append(feats.float().cpu())
            y_list.extend([int(b["label"]) for b in batch])

            for b in batch:
                meta_f.write(json.dumps({
                    "question_id": b.get("question_id"),
                    "probe_id": b.get("probe_id"),
                    "label": int(b["label"]),
                    "source": b.get("source", ""),
                    "category": b.get("category", ""),
                    "prompt_style": b.get("prompt_style", ""),
                }, ensure_ascii=False) + "\n")

            if (i // args.batch_size) % 25 == 0:
                print(f"Processed {min(i + args.batch_size, len(rows))}/{len(rows)}")

    meta_f.close()

    X = torch.cat(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)

    torch.save(X, os.path.join(args.out_dir, "X.pt"))
    torch.save(y, os.path.join(args.out_dir, "y.pt"))

    print("Saved:")
    print(" ", os.path.join(args.out_dir, "X.pt"), X.shape)
    print(" ", os.path.join(args.out_dir, "y.pt"), y.shape)
    print(" ", meta_path)


if __name__ == "__main__":
    main()