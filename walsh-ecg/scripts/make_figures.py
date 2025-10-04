#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from walsh_ecg import WalshECGAnalyzer, load_wfdb_record, synthetic_ecg, synthetic_artifacts
from walsh_ecg.plotting import plot_localization_pdf, plot_example_pdf

def _load_ref(npz_path: Path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}

def _get_signal_or_synth(wfdb_dir, rec_name, kind):
    if wfdb_dir and rec_name:
        return load_wfdb_record(wfdb_dir, rec_name, prefer_leads=(0,1))
    if kind == "norm":
        return synthetic_ecg(n_seconds=10, fs=256, st_elev=False, noise=0.01, seed=0)
    if kind == "mi":
        return synthetic_ecg(n_seconds=10, fs=256, st_elev=True, noise=0.01, seed=1)
    if kind == "art":
        return synthetic_artifacts(n_seconds=10, fs=256, seed=2)
    raise ValueError("Unknown kind.")

def main():
    ap = argparse.ArgumentParser(description="Generate localization and example PDFs.")
    ap.add_argument("--reference", type=str, required=True, help="Reference .npz")
    ap.add_argument("--wfdb_dir", type=str, default=None)
    ap.add_argument("--example_norm", type=str, default=None)
    ap.add_argument("--example_mi", type=str, default=None)
    ap.add_argument("--example_art", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--fs_target", type=float, default=256.0)
    ap.add_argument("--n_coeffs", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    A = WalshECGAnalyzer(fs_target=args.fs_target, n_coeffs=args.n_coeffs)
    ref = _load_ref(Path(args.reference))
    A.ref = type("Ref", (), ref)()

    x_loc, fs_loc = _get_signal_or_synth(args.wfdb_dir, args.example_mi, "mi")
    c_loc = A.features(x_loc, fs_loc)
    z_loc = A.zscores(c_loc)
    plot_localization_pdf(
        signal_t=A.preprocess(x_loc, fs_loc),
        fs=A.fs_target, c=c_loc, z=z_loc, z_thr=2.0,
        path_pdf=str(out / "localization.pdf")
    )

    x1, fs1 = _get_signal_or_synth(args.wfdb_dir, args.example_norm, "norm")
    c1 = A.features(x1, fs1); z1 = A.zscores(c1); d1 = A.distance(c1)
    plot_example_pdf(A.preprocess(x1, fs1), A.fs_target, c1, z1, d1, A.ref.mu,
                     path_pdf=str(out / "exam1.pdf"), title="Example 1 (NORMAL)")

    x2, fs2 = _get_signal_or_synth(args.wfdb_dir, args.example_mi, "mi")
    c2 = A.features(x2, fs2); z2 = A.zscores(c2); d2 = A.distance(c2)
    plot_example_pdf(A.preprocess(x2, fs2), A.fs_target, c2, z2, d2, A.ref.mu,
                     path_pdf=str(out / "exam2.pdf"), title="Example 2 (MI-like)")

    x3, fs3 = _get_signal_or_synth(args.wfdb_dir, args.example_art, "art")
    c3 = A.features(x3, fs3); z3 = A.zscores(c3); d3 = A.distance(c3)
    plot_example_pdf(A.preprocess(x3, fs3), A.fs_target, c3, z3, d3, A.ref.mu,
                     path_pdf=str(out / "exam3.pdf"), title="Example 3 (Artifacts)")

    print(f"[OK] Saved figures to {out}")

if __name__ == "__main__":
    main()
