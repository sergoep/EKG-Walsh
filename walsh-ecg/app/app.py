import streamlit as st
import sys
import os
import numpy as np
import wfdb
from pathlib import Path

# -------------------- Adjust Python path --------------------
# Add the 'src' folder to sys.path so Python can find walsh_ecg_files
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import the analyzer class from your package
from walsh_ecg_files.analyzer import WalshECGAnalyzer

# -------------------- Streamlit setup --------------------
st.set_page_config(page_title="Walsh-ECG Demo", layout="wide")
st.title("Walsh–ECG: FWHT-based anomaly detection (CPU-only)")

# -------------------- Load reference --------------------
ref_file = st.file_uploader(
    "Upload reference .npz (from scripts/build_reference.py)", type=["npz"]
)
analyzer = WalshECGAnalyzer()
analyzer.ref = None  # initialize reference

if ref_file is not None:
    d = np.load(ref_file)
    analyzer.ref = type("Ref", (), {k: d[k] for k in d.files})()
    st.success("Reference loaded.")

# -------------------- Input signal --------------------
st.subheader("Input signal")
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader(
        "Upload a 1-column CSV of ECG (amplitude)", type=["csv", "txt"]
    )
with col2:
    wfdb_dir = st.text_input("Or path to WFDB directory on this machine (server)", value="")
    rec_name = st.text_input("WFDB record base name (e.g., b001)", value="")

x = None
fs = None

# -------------------- Load uploaded CSV --------------------
if uploaded is not None:
    try:
        arr = np.loadtxt(uploaded, delimiter=",")
    except Exception:
        arr = np.loadtxt(uploaded)
    x = arr.astype(float).ravel()
    fs = st.number_input(
        "Sampling rate of uploaded CSV", min_value=10, max_value=5000, value=256, step=1
    )

# -------------------- Load WFDB record --------------------
elif wfdb_dir and rec_name:
    try:
        rec = wfdb.rdrecord(str(Path(wfdb_dir) / rec_name))
        x = rec.p_signal[:, 0].astype(float)
        fs = float(rec.fs)
        st.success(f"Loaded WFDB {rec_name} at fs={fs} Hz, channels={rec.n_sig}")
    except Exception as e:
        st.error(f"WFDB load error: {e}")

# -------------------- Process signal --------------------
if x is not None:
    x_proc = analyzer.preprocess(x, fs)
    c = analyzer.features(x, fs)
    st.line_chart(x_proc[: min(5000, len(x_proc))])

    if analyzer.ref is not None:
        # compute distance and z-scores if implemented
        if hasattr(analyzer, "distance") and hasattr(analyzer, "zscores"):
            d = analyzer.distance(c)
            z = analyzer.zscores(c)
            st.markdown(
                f"**Distance D(c):** `{d:.4f}`  |  **Threshold θ:** `{analyzer.ref.theta:.4f}`"
            )
            st.caption("Features (|Walsh–Paley|)")
            st.bar_chart(c)
            st.caption("|z|-scores")
            st.bar_chart(z)
            st.write("Anomalous coeff indices (z>2):", np.flatnonzero(z > 2.0)[:100])
        else:
            st.warning("distance() and zscores() not implemented in analyzer.")
    else:
        st.warning("Reference not loaded; showing features only.")
