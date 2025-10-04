#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from walsh_ecg import WalshECGAnalyzer, load_wfdb_record, synthetic_ecg, synthetic_artifacts

def _load_reference(npz_path: Path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}

def main():
    ap = argparse.ArgumentParser(description="Evaluate Walsh-ECG on WFDB or synthetic data.")
    ap.add_argument("--reference", type=str, required=True, help="Reference .npz file (from build_reference.py).")
    ap.add_argument("--wfdb_dir", type=str, default=None)
    ap.add_argument("--normals", nargs="*", default=None, help="Base names of normal records.")
    ap.add_argument("--abnormals", nargs="*", default=None, help="Base names of abnormal records.")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--fs_target", type=float, default=256.0)
    ap.add_argument("--n_coeffs", type=int, default=256)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    A = WalshECGAnalyzer(fs_target=args.fs_target, n_coeffs=args.n_coeffs, lambda_reg=args.lambda_reg)
    ref = _load_reference(Path(args.reference))
    A.ref = type("Ref", (), ref)()

    X, y = [], []
    if args.wfdb_dir and (args.normals or args.abnormals):
        if args.normals:
            for r in args.normals:
                x, fs = load_wfdb_record(args.wfdb_dir, r, prefer_leads=(0,1))
                X.append((x, fs)); y.append(0)
        if args.abnormals:
            for r in args.abnormals:
                x, fs = load_wfdb_record(args.wfdb_dir, r, prefer_leads=(0,1))
                X.append((x, fs)); y.append(1)
    else:
        for i in range(50):
            X.append(synthetic_ecg(n_seconds=10, fs=int(args.fs_target), st_elev=False, noise=0.01, seed=i))
            y.append(0)
        for i in range(50):
            if i % 2 == 0:
                X.append(synthetic_ecg(n_seconds=10, fs=int(args.fs_target), st_elev=True, noise=0.01, seed=100+i))
            else:
                X.append(synthetic_artifacts(n_seconds=10, fs=int(args.fs_target), seed=200+i))
            y.append(1)

    D = []
    for x, fs in X:
        c = A.features(x, fs)
        D.append(A.distance(c))
    D = np.asarray(D); y = np.asarray(y)

    thr = float(A.ref.theta)
    yhat = (D > thr).astype(int)

    auc = roc_auc_score(y, D)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    metrics = {"ROC_AUC": float(auc), "Precision": float(p), "Recall": float(r), "F1": float(f1),
               "Threshold": thr}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
