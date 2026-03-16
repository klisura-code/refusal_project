#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True, help="Directory with X.pt and y.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument(
        "--class_weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Use 'balanced' if safe/unsafe classes are imbalanced",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    x_path = os.path.join(args.feat_dir, "X.pt")
    y_path = os.path.join(args.feat_dir, "y.pt")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing feature file: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing label file: {y_path}")

    X = torch.load(x_path).cpu().numpy()
    y = torch.load(y_path).cpu().numpy()

    if len(X) != len(y):
        raise ValueError(f"Feature count ({len(X)}) != label count ({len(y)})")

    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape (N, D), got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to have shape (N,), got {y.shape}")

    uniq = np.unique(y)
    if not np.array_equal(np.sort(uniq), np.array([0, 1])):
        raise ValueError(f"Expected binary labels {{0,1}}, got {uniq}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    clf = LogisticRegression(
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        class_weight=None if args.class_weight == "none" else "balanced",
        solver="lbfgs",
        random_state=args.seed,
    )
    clf.fit(X_trs, y_tr)

    p = clf.predict_proba(X_tes)[:, 1]   # P(label=1 = unsafe)
    yhat = (p >= 0.5).astype(int)

    acc = accuracy_score(y_te, yhat)
    auc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else float("nan")

    coef = clf.coef_.reshape(-1).astype(np.float32)
    intercept = np.array([clf.intercept_.reshape(-1)[0]], dtype=np.float32)
    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)

    np.save(os.path.join(args.out_dir, "coef.npy"), coef)
    np.save(os.path.join(args.out_dir, "intercept.npy"), intercept)
    np.save(os.path.join(args.out_dir, "scaler_mean.npy"), scaler_mean)
    np.save(os.path.join(args.out_dir, "scaler_scale.npy"), scaler_scale)

    metrics = {
        "acc": float(acc),
        "auc": float(auc),
        "n": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pos_rate": float(y.mean()),
        "unsafe_rate": float(y.mean()),
        "safe_rate": float(1.0 - y.mean()),
        "label_semantics": {
            "1": "unsafe",
            "0": "safe"
        },
        "class_weight": None if args.class_weight == "none" else "balanced",
        "feature_dim": int(X.shape[1]),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. acc={acc:.4f} auc={auc:.4f} unsafe_rate={y.mean():.3f}")
    print(f"Train split: unsafe={(y_tr == 1).sum()} | safe={(y_tr == 0).sum()}")
    print(f"Test  split: unsafe={(y_te == 1).sum()} | safe={(y_te == 0).sum()}")
    print("Label semantics: 1=unsafe, 0=safe")
    print("Saved probe to:", args.out_dir)


if __name__ == "__main__":
    main()