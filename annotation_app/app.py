import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import streamlit as st

DATA_PATH_DEFAULT = "/home/dorde/Desktop/Refusal_project/data/Sorry-PRAGMATICS.jsonl"
OUTPUT_DIR_DEFAULT = "/home/dorde/Desktop/Refusal_project/annotation_app/outputs"

# All non-base styles we want to annotate
TARGET_STYLES = [
    "positive-politeness",
    "negative-politeness",
    "off-record",
    "positive-impoliteness",
    "negative-impoliteness",
    "mock-politeness",
]

STYLE_LABELS = {
    "positive-politeness": "POSITIVE POLITENESS",
    "negative-politeness": "NEGATIVE POLITENESS",
    "off-record": "OFF-RECORD (INDIRECT)",
    "positive-impoliteness": "POSITIVE IMPOLITENESS",
    "negative-impoliteness": "NEGATIVE IMPOLITENESS",
    "mock-politeness": "MOCK POLITENESS (SARCASTIC)",
}

STYLE_COLORS = {
    "positive-politeness": "#22c55e",      # green
    "negative-politeness": "#3b82f6",      # blue
    "off-record": "#f59e0b",               # amber
    "positive-impoliteness": "#ef4444",    # red
    "negative-impoliteness": "#a855f7",    # purple
    "mock-politeness": "#f97316",          # orange
}

THEORY_ERROR_TYPES = [
    "too_direct",
    "too_indirect",
    "wrong_strategy",
    "tone_mismatch",
    "other",
]
INTENT_ERROR_TYPES = [
    "weakened_intent",
    "strengthened_intent",
    "changed_target",
    "added_content",
    "other",
]


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dataset(path: str):
    """
    Returns:
      base_by_qid: qid -> base_text
      var_rows: list of dict rows for non-base styles
    """
    base_by_qid: Dict[int, str] = {}
    var_rows: List[dict] = []

    for r in iter_jsonl(path):
        qid = int(r["question_id"])
        style = r["prompt_style"]
        text = r["turns"][0]

        if style == "base":
            base_by_qid[qid] = text
        elif style in TARGET_STYLES:
            var_rows.append(
                {
                    "question_id": qid,
                    "category": str(r.get("category", "")),
                    "prompt_style": style,
                    "text": text,
                }
            )

    # deterministic order: qid-major, then the fixed style order above
    var_rows.sort(key=lambda x: (x["question_id"], TARGET_STYLES.index(x["prompt_style"])))
    return base_by_qid, var_rows


def annotation_key(qid: int, style: str) -> str:
    return f"{qid}::{style}"


def output_path(output_dir: str, annotator_id: str) -> str:
    return str(Path(output_dir) / f"annotations_{annotator_id}.jsonl")


def load_existing_annotations(path: str) -> Dict[str, dict]:
    """
    Returns map: key(qid::style) -> annotation object (last one wins if duplicates)
    """
    ann = {}
    if not os.path.exists(path):
        return ann
    for r in iter_jsonl(path):
        ann[r["item_key"]] = r
    return ann


def append_annotation(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_next_index(var_rows: List[dict], done_map: Dict[str, dict], start_at: int = 0) -> int:
    for i in range(start_at, len(var_rows)):
        k = annotation_key(var_rows[i]["question_id"], var_rows[i]["prompt_style"])
        if k not in done_map:
            return i
    return len(var_rows)


st.set_page_config(page_title="SORRY Pragmatics Annotation", layout="wide")
st.title("SORRY Pragmatics Annotation (Theory + Intent)")

with st.sidebar:
    st.header("Setup")
    data_path = st.text_input("Dataset path", value=DATA_PATH_DEFAULT)
    output_dir = st.text_input("Output folder", value=OUTPUT_DIR_DEFAULT)

    st.divider()
    annotator_id = st.selectbox("Annotator", options=["", "faruk", "dorde"], index=0)
    passcode = st.text_input("Optional passcode", type="password", value="")

    st.caption("Use **faruk** and **dorde** for the two annotators.")
    st.divider()
    load_btn = st.button("Load / Refresh dataset")

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if load_btn or (not st.session_state.loaded):
    if not os.path.exists(data_path):
        st.error(f"Dataset not found: {data_path}")
        st.stop()
    base_by_qid, var_rows = load_dataset(data_path)
    st.session_state.base_by_qid = base_by_qid
    st.session_state.var_rows = var_rows
    st.session_state.loaded = True

base_by_qid = st.session_state.base_by_qid
var_rows = st.session_state.var_rows

if not annotator_id:
    st.warning("Select an annotator in the sidebar to begin.")
    st.stop()

out_path = output_path(output_dir, annotator_id)
done_map = load_existing_annotations(out_path)

total = len(var_rows)
done = len(done_map)
remaining = total - done

colA, colB, colC = st.columns(3)
colA.metric("Total items", total)
colB.metric("Completed", done)
colC.metric("Remaining", remaining)
st.progress(done / total if total else 0.0)

st.divider()

if "idx" not in st.session_state:
    st.session_state.idx = get_next_index(var_rows, done_map, start_at=0)

idx = st.session_state.idx

nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])
with nav1:
    if st.button("⬅ Prev", disabled=(idx <= 0)):
        st.session_state.idx = max(0, idx - 1)
        st.rerun()
with nav2:
    if st.button("Skip ➡"):
        st.session_state.idx = min(total, idx + 1)
        st.rerun()
with nav3:
    jump = st.number_input(
        "Jump to index",
        min_value=0,
        max_value=max(0, total - 1),
        value=min(idx, max(0, total - 1)),
    )
with nav4:
    if st.button("Go"):
        st.session_state.idx = int(jump)
        st.rerun()

if idx >= total:
    st.success("All items completed 🎉")
    st.write(f"Saved annotations: `{out_path}`")
    st.stop()

row = var_rows[idx]
qid = row["question_id"]
style = row["prompt_style"]
item_key = annotation_key(qid, style)

base_text = base_by_qid.get(qid, "(Missing base prompt!)")
variant_text = row["text"]

style_name = STYLE_LABELS.get(style, style)
style_color = STYLE_COLORS.get(style, "#64748b")

st.markdown(
    f"""
    <div style="
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        margin-bottom: 14px;">
      <div style="font-size: 13px; opacity: 0.8; margin-bottom: 6px;">
        You are annotating
      </div>
      <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <div style="
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 800;
            letter-spacing: 0.4px;
            background: {style_color};
            color: #0b0f14;">
          {style_name}
        </div>
        <div style="opacity: 0.85;">
          qid=<b>{qid}</b> • category=<b>{row['category']}</b> • item <b>{idx+1}</b> / {total}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Item key: {item_key}")

left, right = st.columns(2)

def render_text_block(title, text, border_color):
    st.markdown(
        f"""
        <div style="
            background: white;
            padding: 18px;
            border-radius: 14px;
            border-left: 6px solid {border_color};
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 10px;
        ">
            <div style="
                font-size: 16px;
                font-weight: 900;
                margin-bottom: 10px;
                color: black;
            ">
                {title}
            </div>
            <div style="
                font-size: 17px;
                line-height: 1.6;
                color: #111827;
                white-space: pre-wrap;
            ">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with left:
    render_text_block("Base (Standard English)", base_text, "#6b7280")

with right:
    render_text_block(f"Variant — {style_name}", variant_text, style_color)

st.markdown("### Annotation")

existing = done_map.get(item_key)

theory_options = ["(select)", "yes", "no", "unsure"]

theory_default_index = 0
if existing and existing.get("theory_ok") in theory_options:
    theory_default_index = theory_options.index(existing["theory_ok"])
else:
    theory_default_index = 0

theory_ok = st.radio(
    "1) Does the variant follow the intended pragmatics strategy (theory-consistent)?",
    options=theory_options,
    index=theory_default_index,
    horizontal=True,
    key=f"theory_ok::{item_key}",
)

if theory_ok == "(select)":
    theory_ok = None

theory_err = None
if theory_ok == "no":
    theory_err = st.selectbox(
        "If NO: what went wrong (optional)?",
        options=["(select)"] + THEORY_ERROR_TYPES,
        index=0 if not existing else (["(select)"] + THEORY_ERROR_TYPES).index(existing.get("theory_error_type", "(select)")),
        key=f"theory_err::{item_key}",
    )
    if theory_err == "(select)":
        theory_err = None

intent_options = ["(select)", "yes", "no", "unsure"]

intent_default_index = 0
if existing and existing.get("intent_ok") in intent_options:
    intent_default_index = intent_options.index(existing["intent_ok"])
else:
    intent_default_index = 0

intent_ok = st.radio(
    "2) Did the variant preserve the original intent (same unsafe request semantics)?",
    options=intent_options,
    index=intent_default_index,
    horizontal=True,
    key=f"intent_ok::{item_key}",
)

if intent_ok == "(select)":
    intent_ok = None

intent_err = None
if intent_ok == "no":
    intent_err = st.selectbox(
        "If NO: how did intent change (optional)?",
        options=["(select)"] + INTENT_ERROR_TYPES,
        index=0 if not existing else (["(select)"] + INTENT_ERROR_TYPES).index(existing.get("intent_error_type", "(select)")),
        key=f"intent_err::{item_key}",
    )
    if intent_err == "(select)":
        intent_err = None

notes = st.text_area(
    "Notes (optional)",
    value=existing.get("notes", "") if existing else "",
    height=130,
    key=f"notes::{item_key}",
)

save_col1, save_col2 = st.columns([1, 4])

with save_col1:
    if st.button("✅ Save & Next"):
        if theory_ok is None or intent_ok is None:
            st.error("Please answer both questions before saving.")
            st.stop()
        ann = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "annotator_id": annotator_id,
            "item_key": item_key,
            "question_id": qid,
            "category": row["category"],
            "prompt_style": style,
            "theory_ok": theory_ok,
            "theory_error_type": theory_err,
            "intent_ok": intent_ok,
            "intent_error_type": intent_err,
            "notes": notes.strip(),
            "dataset_path": data_path,
        }
        append_annotation(out_path, ann)
        st.success("Saved. (Your progress is preserved if you exit.)")

        # move to next unseen item
        done_map = load_existing_annotations(out_path)
        st.session_state.idx = get_next_index(var_rows, done_map, start_at=idx + 1)
        st.rerun()

with save_col2:
    st.caption(f"Annotations file: `{out_path}`")
    st.caption("Progress is saved on every **Save** click. You can close the tab and resume later.")
    if existing:
        st.info("This item already has a saved annotation (saving again overwrites it).")