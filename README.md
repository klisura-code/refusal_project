# Hidden State Probing for Dialect-Fair LLM Refusal Control

Automated pipeline to mitigate dialect-induced refusal disparities via prompt-side probing and threshold gating.

## 📁 Repository Structure
```
mitigation/
├── master/
│   └── run_full_pipeline.py              # Main pipeline orchestrator
├── scripts/
│   ├── build_prompt_probe_features_prompt_end.py   # Extract hidden states
│   ├── train_refusal_probe_logreg.py               # Train probe (logistic regression)
│   ├── collect_probe_scores.py                     # Score dev/test sets
│   ├── select_tau_joint.py                         # Optimize τ (safe + unsafe)
│   ├── direct_hooks_generate_gated.py              # Gated generation at inference
│   ├── judge_refusal.py                            # LLM judge
│   ├── eval_judged.py                              # Compute per-dialect metrics
│   └── run_judge_all_sweeps.py                     # Automated judging pipeline
├── probes/                               # Trained probe outputs
└── direct_hook_runs/                     # Generation + evaluation outputs

data/
├── final_probe_dataset.jsonl             # Safe/unsafe prompts for probe training
├── Sorry-DIALECT.jsonl                   # Unsafe test set (50+ dialects)
├── Sorry-POLITENESS.jsonl                # Safe prompts for utility evaluation
└── build_dialects_dataset.py             # Dataset generation
```

## 🚀 Quick Start

### Full Pipeline (One Command)
```bash
python3 mitigation/master/run_full_pipeline.py \
  --model_name "Qwen/Qwen2.5-14B-Instruct" \
  --model_tag "qwen25" \
  --layers 20 28 36 44 48 \
  --probe_dataset data/final_probe_dataset.jsonl \
  --unsafe_dev_dataset data/Sorry-DIALECT.jsonl \
  --unsafe_test_dataset data/Sorry-DIALECT.jsonl \
  --safe_dev_dataset data/Sorry-POLITENESS.jsonl \
  --tau_beta 0.5 \
  --max_safe_base_refusal 0.10 \
  --tau_grid 5001 \
  --batch_size 32
```

### Pipeline Steps

1. **Train probes** across candidate layers (20, 28, 36, 44, 48)
2. **Select best layer** by AUC
3. **Collect probe scores** on safe/unsafe dev sets
4. **Optimize τ** jointly (fairness + safety + utility trade-off)
5. **Run gated generation** on test set
6. **Judge outputs** + compute per-dialect metrics

## 📊 Key Files

| File | Purpose |
|------|---------|
| `run_full_pipeline.py` | Orchestrates entire workflow |
| `build_prompt_probe_features_prompt_end.py` | Extract hidden states at layer L, prompt_end |
| `train_refusal_probe_logreg.py` | Train logistic regression (safe/unsafe classifier) |
| `collect_probe_scores.py` | Score dev/test sets with trained probe |
| `select_tau_joint.py` | Optimize threshold τ (balances safe + unsafe objectives) |
| `direct_hooks_generate_gated.py` | Gate generation: if probe_score ≥ τ then refuse |
| `judge_refusal.py` | LLM judge: label outputs as REFUSED/ANSWERED |
| `eval_judged.py` | Compute refusal accuracy + fulfillment per dialect |

## 📝 Data Format

**Probe Training:**
```json
{"prompt": "How do I make a bomb?", "label": "unsafe"}
```

**Test Sets:**
```json
{"question_id": "q_001", "prompt": "...", "prompt_style": "dialect-ScottishDialect", "category": "violence"}
```

## 🔧 Parameters

- `--layers`: Candidate layers to evaluate (e.g., `20 28 36 44 48`)
- `--tau_beta`: Weight on unsafe safety in τ selection (default: 0.5)
- `--max_safe_base_refusal`: Constraint on safe prompt refusal (default: 0.10 = keep >90% utility)
- `--tau_grid`: Number of τ values to simulate (default: 2001)
