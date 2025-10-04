from __future__ import annotations
from pathlib import Path
import numpy as np
import wfdb

def have_wfdb_triple(basename: Path) -> bool:
    return (basename.with_suffix(".hea").exists() and
            basename.with_suffix(".dat").exists())

def load_wfdb_record(wfdb_dir: str | Path, rec_name: str, prefer_leads=(0,)) -> tuple[np.ndarray, float]:
    wfdb_dir = Path(wfdb_dir)
    base = wfdb_dir / rec_name
    if not have_wfdb_triple(base):
        raise FileNotFoundError(f"Missing WFDB files for {base}")
    rec = wfdb.rdrecord(str(base))
    fs = float(rec.fs)
    x = rec.p_signal
    for ch in prefer_leads:
        if ch < x.shape[1]:
            return x[:, ch].astype(float), fs
    return x[:, 0].astype(float), fs

def load_wfdb_multilead(wfdb_dir: str | Path, rec_name: str, n_leads: int | None = None):
    wfdb_dir = Path(wfdb_dir)
    base = wfdb_dir / rec_name
    if not have_wfdb_triple(base):
        raise FileNotFoundError(f"Missing WFDB files for {base}")
    rec = wfdb.rdrecord(str(base))
    fs = float(rec.fs)
    X = rec.p_signal.T.astype(float)
    if n_leads is not None and X.shape[0] >= n_leads:
        X = X[:n_leads]
    return X, fs
