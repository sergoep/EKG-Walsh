from __future__ import annotations
import numpy as np
from numpy.linalg import pinv
from scipy import signal
from dataclasses import dataclass
from .fwht import fwht_inplace, paley_order_indices
from .utils import resample_if_needed, butter_bandpass, iir_notch, sliding_windows

@dataclass
class ReferenceModel:
    mu: np.ndarray
    cov: np.ndarray
    sigma: np.ndarray
    theta: float
    lambda_reg: float
    paley_order: np.ndarray

class WalshECGAnalyzer:
    def __init__(self, fs_target: float = 256.0, n_coeffs: int = 256,
                 notch_hz=50.0, band=(0.5, 40.0), lambda_reg=1e-3):
        self.fs_target = float(fs_target)
        self.n = int(n_coeffs)
        if self.n & (self.n - 1) != 0:
            raise ValueError("n_coeffs must be power of two.")
        self.notch_hz = float(notch_hz)
        self.band = tuple(band)
        self.lambda_reg = float(lambda_reg)
        self.paley_idx = paley_order_indices(self.n)
        self.ref: ReferenceModel | None = None

    def preprocess(self, x: np.ndarray, fs_orig: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).squeeze()
        x = resample_if_needed(x, fs_orig, self.fs_target)
        b_bp, a_bp = butter_bandpass(self.fs_target, low=self.band[0], high=self.band[1], order=4)
        x = signal.filtfilt(b_bp, a_bp, x, method="gust")
        b_n, a_n = iir_notch(self.fs_target, f0=self.notch_hz, q=30.0)
        x = signal.filtfilt(b_n, a_n, x, method="gust")
        m = np.max(np.abs(x)) + 1e-12
        return x / m

    def _fwht_paley(self, x_win: np.ndarray) -> np.ndarray:
        y = x_win.astype(float).copy()
        fwht_inplace(y)
        y /= self.n
        return y[self.paley_idx]

    def features(self, x: np.ndarray, fs_orig: float) -> np.ndarray:
        x = self.preprocess(x, fs_orig)
        hop = self.n // 2
        windows = sliding_windows(x, win=self.n, hop=hop)
        feats = np.empty((windows.shape[0], self.n), dtype=float)
        for i, w in enumerate(windows):
            feats[i] = np.abs(self._fwht_paley(w))
        return feats.mean(axis=0)

    def build_reference(self, list_of_signals, fs_orig_list, quantile: float = 0.95) -> 'ReferenceModel':
        C = []
        for x, fs in zip(list_of_signals, fs_orig_list):
            C.append(self.features(x, fs))
        C = np.stack(C, axis=0)
        mu = C.mean(axis=0)
        Xc = C - mu
        cov = (Xc.T @ Xc) / max(1, (len(C) - 1))
        cov = cov + self.lambda_reg * np.eye(self.n, dtype=float)
        sigma = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        inv_cov = pinv(cov)
        dists = np.einsum("bi,ij,bj->b", Xc, inv_cov, Xc)
        theta = float(np.quantile(dists, quantile))
        self.ref = ReferenceModel(mu=mu, cov=cov, sigma=sigma, theta=theta,
                                  lambda_reg=self.lambda_reg, paley_order=self.paley_idx.copy())
        return self.ref

    def distance(self, c: np.ndarray) -> float:
        if self.ref is None:
            raise RuntimeError("Reference not built.")
        x = c - self.ref.mu
        inv_cov = pinv(self.ref.cov)
        d = float(x @ inv_cov @ x)
        if not np.isfinite(d):
            zsum = np.sum(np.abs(x) / (self.ref.sigma + 1e-12))
            return float(zsum)
        return d

    def zscores(self, c: np.ndarray) -> np.ndarray:
        if self.ref is None:
            raise RuntimeError("Reference not built.")
        return np.abs((c - self.ref.mu) / (self.ref.sigma + 1e-12))

    def explain(self, c: np.ndarray, z_thr=2.0) -> dict:
        z = self.zscores(c)
        idx_anom = np.flatnonzero(z > z_thr)
        return {"z": z, "anom_idx": idx_anom, "z_thr": float(z_thr)}

    def is_anomalous(self, c: np.ndarray) -> bool:
        return self.distance(c) > (self.ref.theta if self.ref else np.inf)

class MultiLeadWalshECG:
    def __init__(self, lead_weights=None, **analyzer_kwargs):
        self.base_kwargs = analyzer_kwargs
        self.analyzers = []
        self.weights = None
        if lead_weights is not None:
            self.set_weights(lead_weights)

    def set_weights(self, w):
        w = np.asarray(w, dtype=float).ravel()
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")
        self.weights = w / (w.sum() + 1e-12)

    def fit(self, list_of_multilead_signals, fs_list):
        if len(list_of_multilead_signals) == 0:
            raise ValueError("No signals.")
        L = list_of_multilead_signals[0].shape[0]
        self.analyzers = [WalshECGAnalyzer(**self.base_kwargs) for _ in range(L)]
        if self.weights is None:
            self.set_weights(np.ones(L))
        for li in range(L):
            sigs = [x[li] for x in list_of_multilead_signals]
            fss = fs_list
            self.analyzers[li].build_reference(sigs, fss)

    def distance(self, c_list):
        if self.weights is None or len(self.analyzers) == 0:
            raise RuntimeError("Model not fitted.")
        if len(c_list) != len(self.analyzers):
            raise ValueError("Number of leads mismatches.")
        d = 0.0
        for w, A, c in zip(self.weights, self.analyzers, c_list):
            d += float(w) * A.distance(c)
        return float(d)

    def features(self, multilead, fs_orig: float):
        return [A.features(multilead[li], fs_orig) for li, A in enumerate(self.analyzers)]
