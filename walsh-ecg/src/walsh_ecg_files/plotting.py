from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _save_tight(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_localization_pdf(signal_t, fs, c, z, z_thr, path_pdf: str):
    t = np.arange(len(signal_t)) / fs
    fig = plt.figure(figsize=(9, 7))

    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t, signal_t, lw=1.0)
    ax1.set_title("(a) ECG signal (normalized)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    ax2 = fig.add_subplot(3,1,2)
    ax2.bar(np.arange(len(c)), c, width=0.8)
    ax2.set_title("(b) |Walsh–Paley| coefficients (averaged)")
    ax2.set_xlabel("Sequency index")
    ax2.set_ylabel("|c_k|")

    ax3 = fig.add_subplot(3,1,3)
    ax3.bar(np.arange(len(z)), z, width=0.8)
    ax3.axhline(z_thr, linestyle="--", lw=1.2)
    ax3.set_title("(c) |z|-profile (threshold)")
    ax3.set_xlabel("Sequency index")
    ax3.set_ylabel("|z_k|")

    _save_tight(fig, Path(path_pdf))

def plot_example_pdf(signal_t, fs, c, z, d, mu, path_pdf: str, title="Example"):
    t = np.arange(len(signal_t)) / fs
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(f"{title} (distance D={d:.3f})", y=0.98)

    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(t, signal_t, lw=1.0)
    ax1.set_title("ECG signal (normalized)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amp.")

    ax2 = fig.add_subplot(4,1,2)
    ax2.bar(np.arange(len(c)), c, width=0.8)
    ax2.set_title("|Walsh–Paley| coefficients")
    ax2.set_xlabel("Sequency index"); ax2.set_ylabel("|c_k|")

    ax3 = fig.add_subplot(4,1,3)
    ax3.bar(np.arange(len(z)), z, width=0.8)
    ax3.axhline(2.0, linestyle="--", lw=1.2, label="z=2.0")
    ax3.legend()
    ax3.set_title("|z|-profile")
    ax3.set_xlabel("Sequency index"); ax3.set_ylabel("|z_k|")

    ax4 = fig.add_subplot(4,1,4)
    idx = np.arange(len(c))
    ax4.bar(idx - 0.2, mu, width=0.4, label="reference μ")
    ax4.bar(idx + 0.2, c,  width=0.4, label="current c")
    ax4.set_title("Reference vs current coefficients")
    ax4.set_xlabel("Sequency index"); ax4.set_ylabel("magnitude")
    ax4.legend(ncols=2, fontsize=8)

    _save_tight(fig, Path(path_pdf))
