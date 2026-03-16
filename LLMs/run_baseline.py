import asyncio
import json
import os
import time
import argparse
from typing import Dict, Any, Tuple, Set, Optional

import httpx

URL = "http://localhost:8000/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful assistant."

# Models we support and their tags
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

# Gemma does not support system role — we merge it into the user turn
GEMMA_MODELS = {"gemma2_9b", "gemma2_27b"}


# --------- I/O helpers ---------

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_keys(out_path: str) -> Set[Tuple[int, str]]:
    """
    Resume support: returns set of (question_id, prompt_style) already completed.
    """
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question_id" in obj and ("prompt_style" in obj or "variant" in obj):
                qid = int(obj["question_id"])
                key = str(obj.get("prompt_style") or obj.get("variant"))
                done.add((qid, key))
    return done


# --------- Chat payload builder ---------

def build_messages(model_tag: str, prompt: str) -> list:
    """
    Build messages list for the given model.
    Gemma does not support system role — merge system prompt into user turn.
    All other models use the standard system + user format.
    Prompts are identical across models for fair comparison.
    """
    if model_tag in GEMMA_MODELS:
        return [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"},
        ]
    else:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]


# --------- LLM call ---------

async def chat_async(
    client: httpx.AsyncClient,
    model_name: str,
    model_tag: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_s: float = 300.0,
) -> Tuple[str, float]:
    payload = {
        "model": model_name,
        "messages": build_messages(model_tag, prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    r = await client.post(URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return text, (time.time() - t0)


# --------- Worker pipeline ---------

async def run(
    dataset_path: str,
    out_path: str,
    model_name: str,
    model_tag: str,
    concurrency: int = 16,
    max_tokens: int = 512,
    temperature: float = 0.0,
    limit_rows: Optional[int] = None,
):
    done = load_done_keys(out_path)
    total_prompts = 0
    skipped = 0

    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    rows_seen = 0
    for row in iter_jsonl(dataset_path):
        rows_seen += 1
        if limit_rows is not None and rows_seen > limit_rows:
            break

        qid = int(row["question_id"])
        prompt_style = str(row["prompt_style"])
        prompt = row["turns"][0]

        if (qid, prompt_style) in done:
            skipped += 1
            continue

        queue.put_nowait({
            "question_id": qid,
            "category": row.get("category"),
            "prompt_style": prompt_style,
            "variant": prompt_style,
            "prompt": prompt,
        })
        total_prompts += 1

    print(f"Model     : {model_name} (tag={model_tag})")
    print(f"Dataset   : {dataset_path}")
    print(f"Output    : {out_path}")
    print(f"Queued    : {total_prompts:,} prompts | Skipped (already done): {skipped:,}")

    if total_prompts == 0:
        print("Nothing to do — output file already contains everything.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_f = open(out_path, "a", encoding="utf-8")

    sem = asyncio.Semaphore(concurrency)
    progress = {"done": 0, "errors": 0}
    t_start = time.time()

    async with httpx.AsyncClient(headers={"Connection": "keep-alive"}) as client:

        async def worker(worker_id: int):
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                async with sem:
                    tries = 0
                    while True:
                        tries += 1
                        try:
                            text, latency = await chat_async(
                                client,
                                model_name=model_name,
                                model_tag=model_tag,
                                prompt=item["prompt"],
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            record = {
                                "question_id": item["question_id"],
                                "category": item["category"],
                                "prompt_style": item["prompt_style"],
                                "variant": item["variant"],
                                "prompt": item["prompt"],
                                "response": text,
                                "latency_s": round(latency, 4),
                                "model": model_name,
                                "model_tag": model_tag,
                                "ts": int(time.time()),
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_f.flush()

                            progress["done"] += 1
                            if progress["done"] % 200 == 0:
                                elapsed = time.time() - t_start
                                rps = progress["done"] / max(elapsed, 1e-9)
                                eta = (total_prompts - progress["done"]) / max(rps, 1e-9)
                                print(
                                    f"  [{model_tag}] Done {progress['done']:,}/{total_prompts:,} "
                                    f"| {rps:.2f} req/s | ETA ~ {eta/60:.1f} min"
                                )
                            break

                        except Exception as e:
                            if tries >= 4:
                                progress["errors"] += 1
                                err_record = {
                                    "question_id": item["question_id"],
                                    "prompt_style": item["prompt_style"],
                                    "variant": item["variant"],
                                    "error": str(e),
                                    "model": model_name,
                                    "model_tag": model_tag,
                                    "ts": int(time.time()),
                                }
                                out_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                                out_f.flush()
                                break
                            await asyncio.sleep(1.0 * tries)

                queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await asyncio.gather(*workers)

    out_f.close()
    elapsed = time.time() - t_start
    print(
        f"[{model_tag}] Finished. "
        f"Completed={progress['done']:,}, errors={progress['errors']:,}, "
        f"elapsed={elapsed/60:.2f} min"
    )


# --------- Entry point ---------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified baseline inference script.")
    parser.add_argument(
        "--model_tag",
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Short model tag, e.g. llama31, qwen25, mistral7b, gemma2_9b",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override HuggingFace model name (defaults to SUPPORTED_MODELS[model_tag])",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to input .jsonl dataset",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/dorde/Desktop/Refusal_project/LLMs/baselines",
        help="Root output directory. Results saved to <out_dir>/<model_tag>/",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max_tokens",  type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit_rows",  type=int, default=None)
    args = parser.parse_args()

    model_name = args.model_name or SUPPORTED_MODELS[args.model_tag]

    # Auto-derive output filename from dataset filename
    dataset_stem = os.path.splitext(os.path.basename(args.dataset))[0]
    out_dir = os.path.join(args.out_dir, args.model_tag)
    out_path = os.path.join(out_dir, f"{args.model_tag}_{dataset_stem}_responses.jsonl")

    asyncio.run(
        run(
            dataset_path=args.dataset,
            out_path=out_path,
            model_name=model_name,
            model_tag=args.model_tag,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            limit_rows=args.limit_rows,
        )
    )