#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from walsh_ecg import WalshECGAnalyzer, load_wfdb_record, synthetic_ecg

def main():
    ap = argparse.ArgumentParser(description="Build Walsh-ECG reference from normal records.")
    ap.add_argument("--wfdb_dir", type=str, default=None, help="Folder with WFDB files (.hea/.dat).")
    ap.add_argument("--records", nargs="*", default=None, help="Record base names (e.g., b001 b002 ...).")
    ap.add_argument("--fs_target", type=float, default=256.0)
    ap.add_argument("--n_coeffs", type=int, default=256)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--out", type=str, required=True, help="Output .npz file.")
    args = ap.parse_args()

    A = WalshECGAnalyzer(fs_target=args.fs_target, n_coeffs=args.n_coeffs, lambda_reg=args.lambda_reg)

    signals, fss = [], []
    if args.wfdb_dir and args.records:
        for r in args.records:
            x, fs = load_wfdb_record(args.wfdb_dir, r, prefer_leads=(0,1))
            signals.append(x); fss.append(fs)
    else:
        for i in range(20):
            x, fs = synthetic_ecg(n_seconds=10, fs=int(args.fs_target), st_elev=False, noise=0.01, seed=i)
            signals.append(x); fss.append(fs)

    ref = A.build_reference(signals, fss, quantile=args.quantile)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp,
             mu=ref.mu, cov=ref.cov, sigma=ref.sigma, theta=ref.theta,
             lambda_reg=ref.lambda_reg, paley_order=ref.paley_order)
    print(f"[OK] Reference saved to {outp}")

if __name__ == "__main__":
    main()
