#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

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
    system_msg = "You are a helpful assistant."
    
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


def get_prompt_text(row: Dict[str, Any]) -> Optional[str]:
    for key in ["prompt", "text", "instruction", "question"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def get_prompt_label(row: Dict[str, Any]) -> Optional[int]:
    if "label" in row:
        v = row["label"]
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "unsafe", "harmful", "positive"}:
                return 1
            if s in {"0", "safe", "benign", "negative"}:
                return 0

    if "source_label" in row:
        v = row["source_label"]
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "unsafe", "harmful", "positive"}:
                return 1
            if s in {"0", "safe", "benign", "negative"}:
                return 0

    if "is_unsafe" in row:
        v = row["is_unsafe"]
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)

    if "source" in row and isinstance(row["source"], str):
        s = row["source"].strip().lower()
        if "unsafe" in s:
            return 1
        if "safe" in s:
            return 0

    if "dataset" in row and isinstance(row["dataset"], str):
        s = row["dataset"].strip().lower()
        if "unsafe" in s:
            return 1
        if "safe" in s:
            return 0

    return None


class DownProjLastTokenHook:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.last_token_acts = None
        self.handle = None

    def _hook_fn(self, module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self.last_token_acts = out[:, -1, :].detach()

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.mlp.down_proj.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--in_jsonl", required=True, help="JSONL with prompt-level safe/unsafe labels")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    skipped_missing_prompt = 0
    skipped_missing_label = 0

    for r in iter_jsonl(args.in_jsonl):
        prompt = get_prompt_text(r)
        label = get_prompt_label(r)

        if prompt is None:
            skipped_missing_prompt += 1
            continue
        if label is None:
            skipped_missing_label += 1
            continue

        rr = dict(r)
        rr["_prompt_text"] = prompt
        rr["_prompt_label"] = int(label)
        rows.append(rr)

        if args.max_rows and len(rows) >= args.max_rows:
            break

    if not rows:
        raise SystemExit("No usable rows found with prompt text and prompt-level safe/unsafe label.")

    print(f"Loaded {len(rows)} rows")
    print(f"Skipped rows with missing prompt: {skipped_missing_prompt}")
    print(f"Skipped rows with missing label : {skipped_missing_label}")
    print(f"Using hooked site: model.layers[{args.layer}].mlp.down_proj -> last token")

    X_list: List[torch.Tensor] = []
    y_list: List[int] = []

    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    meta_f = open(meta_path, "w", encoding="utf-8")

    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            prompts = [b["_prompt_text"] for b in batch]

            enc_list = [apply_chat(tok, p)[0] for p in prompts]
            enc = tok.pad({"input_ids": enc_list}, return_tensors="pt")

            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            with DownProjLastTokenHook(model, args.layer) as hook:
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

                if hook.last_token_acts is None:
                    raise RuntimeError(
                        f"Hook did not capture activations at layer {args.layer}."
                    )

                feats = hook.last_token_acts.float().cpu()

            X_list.append(feats)
            y_list.extend([int(b["_prompt_label"]) for b in batch])

            for b in batch:
                meta_f.write(json.dumps({
                    "question_id": b.get("question_id"),
                    "probe_id": b.get("probe_id"),
                    "label": int(b["_prompt_label"]),
                    "source": b.get("source", ""),
                    "category": b.get("category", ""),
                    "prompt_style": b.get("prompt_style", ""),
                    "prompt": b["_prompt_text"],
                }, ensure_ascii=False) + "\n")

            done = min(i + args.batch_size, len(rows))
            if (i // args.batch_size) % 25 == 0:
                print(f"Processed {done}/{len(rows)}")

    meta_f.close()

    X = torch.cat(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)

    torch.save(X, os.path.join(args.out_dir, "X.pt"))
    torch.save(y, os.path.join(args.out_dir, "y.pt"))

    unsafe_count = int((y == 1).sum().item())
    safe_count = int((y == 0).sum().item())

    print("\nSaved:")
    print(" ", os.path.join(args.out_dir, "X.pt"), tuple(X.shape))
    print(" ", os.path.join(args.out_dir, "y.pt"), tuple(y.shape))
    print(" ", meta_path)
    print(f"Labels: {unsafe_count} UNSAFE, {safe_count} SAFE")


if __name__ == "__main__":
    main()