# Walsh-ECG: Fast, interpretable ECG anomaly detection with Walsh–Hadamard features

This repository contains a CPU-only, reproducible implementation of an interpretable ECG anomaly detection method based on the Fast Walsh–Hadamard Transform (FWHT) with Paley (sequency) ordering.

**Highlights**
- FWHT (O(N log N)) with Paley (sequency) ordering
- Compact per-window features by |coefficients| and window averaging
- Reference "normal" statistical model (mean, covariance + λI regularization)
- Mahalanobis-like distance with pseudo-inverse + z-score fallback
- Per-coefficient z-score explanations; sequency↔morphology mapping
- Figure generation (localization + examples)
- Optional Streamlit demo app
- Works with local WFDB datasets (e.g., PTB-XL, CEBSDB); synthetic fallback provided

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```
streamlit run walsh-ecg/app/app.py

```

