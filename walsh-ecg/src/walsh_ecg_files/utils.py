from __future__ import annotations
import numpy as np
from scipy import signal

def butter_bandpass(fs: float, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return b, a

def iir_notch(fs: float, f0=50.0, q=30.0):
    b, a = signal.iirnotch(w0=f0 / (fs / 2), Q=q)
    return b, a

def resample_if_needed(x: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    if abs(fs_orig - fs_target) < 1e-9:
        return x
    n_new = int(round(len(x) * fs_target / fs_orig))
    return signal.resample(x, n_new)

def sliding_windows(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)), mode="edge")
    n = len(x)
    starts = np.arange(0, n - win + 1, hop, dtype=int)
    if len(starts) == 0:
        starts = np.array([0], dtype=int)
    return np.stack([x[s:s + win] for s in starts], axis=0)

def synthetic_ecg(n_seconds=10, fs=256, st_elev=False, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_seconds, 1/fs)
    hr_hz = 1.2
    x = (1.0 * np.sin(2*np.pi*hr_hz*t)
         + 0.3 * np.sin(2*np.pi*2*hr_hz*t + 0.6)
         + 0.15 * np.sin(2*np.pi*3*hr_hz*t + 1.1))
    if st_elev:
        st = 0.2 * signal.sawtooth(2*np.pi*hr_hz*t, width=0.9)
        st[st < 0] = 0
        x = x + 0.2 * st
    if noise > 0:
        x = x + noise * rng.standard_normal(len(t))
    return x.astype(float), fs

def synthetic_artifacts(n_seconds=10, fs=256, seed=1):
    x, fs = synthetic_ecg(n_seconds=n_seconds, fs=fs, st_elev=False, noise=0.02, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(6):
        i = rng.integers(low=0, high=len(x)-fs//2)
        x[i:i+fs//8] += 0.7 * signal.windows.tukey(fs//8, alpha=0.5)
    return x, fs
