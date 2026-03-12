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
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = torch.load(os.path.join(args.feat_dir, "X.pt")).numpy()
    y = torch.load(os.path.join(args.feat_dir, "y.pt")).numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=8,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_trs, y_tr)

    p = clf.predict_proba(X_tes)[:, 1]
    yhat = (p >= 0.5).astype(int)

    acc = accuracy_score(y_te, yhat)
    auc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else float("nan")

    # Save
    np.save(os.path.join(args.out_dir, "coef.npy"), clf.coef_.astype(np.float32))
    np.save(os.path.join(args.out_dir, "intercept.npy"), clf.intercept_.astype(np.float32))
    np.save(os.path.join(args.out_dir, "scaler_mean.npy"), scaler.mean_.astype(np.float32))
    np.save(os.path.join(args.out_dir, "scaler_scale.npy"), scaler.scale_.astype(np.float32))

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"acc": acc, "auc": auc, "n": int(len(y)), "pos_rate": float(y.mean())}, f, indent=2)

    print(f"Done. acc={acc:.4f} auc={auc:.4f} pos_rate={y.mean():.3f}")
    print("Saved probe to:", args.out_dir)

if __name__ == "__main__":
    main()