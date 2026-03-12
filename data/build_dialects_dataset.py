#!/usr/bin/env python3
"""
Build Sorry-DIALECT.jsonl from 440 base prompts using Multi-VALUE dialect transforms.

Output ordering:
  1) All 440 base rows first (prompt_style="base")
  2) For each dialect (in Dialects class order): 440 rows with prompt_style="dialect-<DialectClassName>"

Resume behavior:
  - Uses a state JSON file tracking last_completed_dialect and counters.
  - If a dialect crashes mid-way, delete its partial rows from output and rerun; state will resume.

Notes:
  - CPU-only: force-hide GPUs via CUDA_VISIBLE_DEVICES.
  - Robustness patches:
      (a) coref sentence mismatch: skip coref rather than crash
      (b) modification_counter KeyError: force defaultdict(int) + ensure key exists before += 1
"""

# -----------------------------
# HARD FORCE: prevent CUDA usage (must be before any heavy imports)
# -----------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import inspect
import json
import re
import time
from collections import defaultdict
from typing import Iterable, List

from multivalue import Dialects
from multivalue.BaseDialect import BaseDialect


# -----------------------------
# Monkey-patch 1: avoid hard crash on spaCy vs Stanza sentence mismatch
# -----------------------------
_ORIG_CREATE_COREF = BaseDialect.create_coref_cluster

def _safe_create_coref_cluster(self, string: str):
    try:
        return _ORIG_CREATE_COREF(self, string)
    except AssertionError:
        # Fallback: skip coref (keeps dialect transforms running)
        return []

BaseDialect.create_coref_cluster = _safe_create_coref_cluster


# -----------------------------
# Monkey-patch 2: avoid KeyError in modification_counter bookkeeping
#   (your crash: KeyError: 'regularized_reflexives')
# -----------------------------
_ORIG_SET_RULE = BaseDialect.set_rule

def _force_defaultdict(self):
    mc = getattr(self, "modification_counter", None)
    if mc is None:
        self.modification_counter = defaultdict(int)
    elif not isinstance(mc, defaultdict):
        # preserve any existing counts
        self.modification_counter = defaultdict(int, dict(mc))

def _safe_set_rule(self, function_name: str, *args, **kwargs):
    # Ensure modification_counter is safe and the key exists BEFORE original uses += 1
    _force_defaultdict(self)
    self.modification_counter[function_name] += 0
    try:
        return _ORIG_SET_RULE(self, function_name, *args, **kwargs)
    except KeyError:
        # In case Multi-VALUE overwrote modification_counter mid-call, force again and retry once
        _force_defaultdict(self)
        self.modification_counter[function_name] += 0
        return _ORIG_SET_RULE(self, function_name, *args, **kwargs)

BaseDialect.set_rule = _safe_set_rule


# -----------------------------
# IO helpers
# -----------------------------
def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {line_no}: {e}") from e


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Formatting helpers
# -----------------------------
def format_hms(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# -----------------------------
# Text normalization (used for retries)
# -----------------------------
_WS_RE = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ")
    s = s.replace("\t", " ")
    s = s.replace("\n", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


# -----------------------------
# Load base 440 questions
# -----------------------------
def load_base_questions(orig_path: str) -> List[dict]:
    rows = []
    for r in iter_jsonl(orig_path):
        qid = int(r["question_id"])
        cat = str(r.get("category", "")).strip()
        turns = r.get("turns", [])
        if not (isinstance(turns, list) and turns and isinstance(turns[0], str) and turns[0].strip()):
            raise ValueError(f"Bad/empty turns for question_id={qid}")
        rows.append({"question_id": qid, "category": cat, "text": turns[0].strip()})
    rows.sort(key=lambda x: x["question_id"])
    return rows


# -----------------------------
# Dialect discovery
# -----------------------------
EXCLUDE = {
    "DialectFromFeatureList",
    "DialectFromVector",
    "MultiDialect",
    "BaseDialect",
    "AAVE_Example_From_List",
}

def discover_dialect_classes() -> List[str]:
    names = []
    for name in dir(Dialects):
        if name.startswith("_"):
            continue
        attr = getattr(Dialects, name)
        if inspect.isclass(attr) and name not in EXCLUDE:
            names.append(name)
    names = sorted(set(names))
    if not names:
        raise RuntimeError("No dialect classes discovered. Is multivalue installed correctly?")
    return names


# -----------------------------
# Resume state (dialect-level)
# -----------------------------
def load_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {
            "started_at": time.time(),
            "base_written": False,
            "last_completed_dialect": None,
            "completed_dialects": 0,
            "coref_fallbacks": 0,
            "transform_retries": 0,
        }
    st = read_json(state_path)
    if "started_at" not in st:
        st["started_at"] = time.time()
    st.setdefault("base_written", False)
    st.setdefault("last_completed_dialect", None)
    st.setdefault("completed_dialects", 0)
    st.setdefault("coref_fallbacks", 0)
    st.setdefault("transform_retries", 0)
    return st


def save_state(state_path: str, st: dict) -> None:
    write_json(state_path, st)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="original_questions.jsonl (440 base prompts)")
    ap.add_argument("--out", required=True, help="Output ordered dataset: Sorry-DIALECT.jsonl")
    ap.add_argument("--state", required=True, help="Resume state file")
    ap.add_argument("--print_every_n", type=int, default=50,
                    help="Within a dialect, print progress every N questions (default 50)")
    args = ap.parse_args()

    base_rows = load_base_questions(args.orig)
    dialect_names = discover_dialect_classes()

    print(f"Loaded base prompts: {len(base_rows)}")
    print(f"Dialect classes (filtered): {len(dialect_names)}")
    print("Dialect order (first 15):", dialect_names[:15])
    print("Output:", args.out)
    print("State:", args.state)

    st = load_state(args.state)

    # Safety: prevent accidental duplicate run
    if not os.path.exists(args.state) and os.path.exists(args.out):
        raise RuntimeError(
            f"Output exists ({args.out}) but no state file. "
            f"Delete the output or provide the correct state file."
        )

    # 1) Write base once
    if not st["base_written"]:
        print("\n[STEP] Writing BASE prompts (440 rows)...")
        for row in base_rows:
            append_jsonl(
                args.out,
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "prompt_style": "base",
                    "turns": [row["text"]],
                },
            )
        st["base_written"] = True
        save_state(args.state, st)
        print("[DONE] Base written.\n")
    else:
        print("\n[RESUME] Base already written.\n")

    # Resume dialect index
    start_idx = 0
    last_completed = st.get("last_completed_dialect", None)
    if last_completed is not None:
        if last_completed not in dialect_names:
            raise RuntimeError(
                f"State says last_completed_dialect={last_completed}, but it's not in discovered dialect list."
            )
        start_idx = dialect_names.index(last_completed) + 1
        print(f"[RESUME] Continuing after dialect: {last_completed} (next index={start_idx})")

    started_at = float(st["started_at"])
    completed_dialects = int(st["completed_dialects"])
    total_d = len(dialect_names)

    # 2) Dialect-major generation (CPU-friendly, no instantiating all dialects at once)
    for dpos in range(start_idx, total_d):
        dname = dialect_names[dpos]

        elapsed = time.time() - started_at
        done_d = max(0, completed_dialects)
        avg_per_d = (elapsed / done_d) if done_d > 0 else None
        remaining_d = total_d - done_d
        eta_total = (avg_per_d * remaining_d) if avg_per_d is not None else -1

        print(
            f"\n[DIALECT {dpos+1}/{total_d}] {dname} | "
            f"elapsed={format_hms(elapsed)} | ETA(total)={format_hms(eta_total)} | "
            f"coref_fallbacks={st['coref_fallbacks']} retries={st['transform_retries']}"
        )

        cls = getattr(Dialects, dname)
        inst = cls()

        t_d0 = time.time()
        for i, row in enumerate(base_rows, start=1):
            if args.print_every_n > 0 and (i == 1 or i % args.print_every_n == 0 or i == len(base_rows)):
                d_elapsed = time.time() - t_d0
                avg_per_q = (d_elapsed / (i - 1)) if i > 1 else None
                rem_q = len(base_rows) - (i - 1)
                eta_d = (avg_per_q * rem_q) if avg_per_q is not None else -1
                print(
                    f"  q {i:3d}/{len(base_rows)} | dialect_elapsed={format_hms(d_elapsed)} | ETA(dialect)={format_hms(eta_d)}"
                )

            text = row["text"]
            try:
                dtext = inst.transform(text)
            except AssertionError:
                st["coref_fallbacks"] += 1
                st["transform_retries"] += 1
                dtext = inst.transform(normalize_text(text))
            except Exception:
                st["transform_retries"] += 1
                dtext = inst.transform(normalize_text(text))

            append_jsonl(
                args.out,
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "prompt_style": f"dialect-{dname}",
                    "turns": [dtext],
                },
            )

        del inst
        gc.collect()

        completed_dialects += 1
        st["completed_dialects"] = completed_dialects
        st["last_completed_dialect"] = dname
        save_state(args.state, st)

        d_total = time.time() - t_d0
        print(f"[DONE] {dname} finished in {format_hms(d_total)}")

    elapsed = time.time() - started_at
    print("\n=== DONE ===")
    print(f"Output: {args.out}")
    print(f"Elapsed: {format_hms(elapsed)}")
    print(f"Dialects completed: {st['completed_dialects']}/{total_d}")
    expected = len(base_rows) * (1 + total_d)
    print(f"Expected total rows (base + dialects): {expected}")
    print(f"coref_fallbacks={st['coref_fallbacks']} transform_retries={st['transform_retries']}")


if __name__ == "__main__":
    main()
