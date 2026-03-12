import asyncio
import json
import os
import time
from typing import Dict, Any, Tuple, Set

import httpx

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."

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
    prompt_style is "base" or "dialect-<DialectName>" for Sorry-DIALECT.jsonl.
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
            # tolerate old/err rows that might not have fields
            if "question_id" in obj and ("prompt_style" in obj or "variant" in obj):
                qid = int(obj["question_id"])
                key = str(obj.get("prompt_style") or obj.get("variant"))
                done.add((qid, key))
    return done

# --------- LLM call ---------

async def chat_async(
    client: httpx.AsyncClient,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_s: float = 120.0,
) -> Tuple[str, float]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
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
    concurrency: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
    limit_rows: int | None = None,   # quick testing
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
            "variant": prompt_style,  # keep for compatibility w/ older eval scripts
            "prompt": prompt,
        })
        total_prompts += 1

    print(f"Queued: {total_prompts:,} prompts | Skipped (already done): {skipped:,}")
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
                                item["prompt"],
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
                                "model": MODEL,
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
                                    f"Done {progress['done']:,}/{total_prompts:,} "
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
                                    "model": MODEL,
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
    print(f"Finished. Completed={progress['done']:,}, errors={progress['errors']:,}, elapsed={elapsed/60:.2f} min")


if __name__ == "__main__":
    dataset = "/home/dorde/Desktop/Refusal_project/data/dialectXpragmatics/Sorry_DxP_DIALECT7.jsonl"
    out = "/home/dorde/Desktop/Refusal_project/data/dialectXpragmatics/llama/llama3.1_responses.jsonl"

    asyncio.run(
        run(
            dataset_path=dataset,
            out_path=out,
            concurrency=64,   
            max_tokens=512,
            temperature=0.0,
            limit_rows=None,
        )
    )
