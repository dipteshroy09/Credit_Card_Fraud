from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.inference import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_metadata,
    load_model,
    predict_dataframe,
    summarize_predictions,
)


st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def get_artifacts():
    model = load_model()
    metadata = load_metadata()
    return model, metadata


st.title("Credit Card Fraud Detection")

try:
    model, metadata = get_artifacts()
except FileNotFoundError:
    st.error("Model not found. Train it first by running: python src/train.py")
    st.stop()

threshold_default = float(metadata.get("threshold", 0.5))
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=threshold_default,
    step=0.01,
)

top_n = st.sidebar.number_input("Top N risky rows", min_value=5, max_value=200, value=25, step=5)

st.subheader("Model summary")
metrics = metadata.get("metrics", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Threshold", f"{threshold:.2f}")
col2.metric("Test PR-AUC", f"{float(metrics.get('test_pr_auc', 0.0)):.4f}")
col3.metric("Test Recall", f"{float(metrics.get('test_recall', 0.0)):.4f}")
col4.metric("Test F1", f"{float(metrics.get('test_f1', 0.0)):.4f}")

st.subheader("Input")
mode = st.radio("Choose input method", options=["Upload CSV", "Use bundled dataset sample"], horizontal=True)


def bundled_data_path() -> Path:
    """Returns path to sample data file, or full dataset if available."""
    root = Path(__file__).resolve().parents[1] / "data"
    # Prefer sample file if available (for deployment)
    sample_path = root / "creditcard_sample.csv"
    if sample_path.exists():
        return sample_path
    # Fallback to full dataset
    return root / "creditcard.csv"


@st.cache_data
def load_sample(nrows: int) -> pd.DataFrame:
    return pd.read_csv(bundled_data_path(), nrows=nrows)


df_input: pd.DataFrame | None = None

if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
else:
    nrows = st.number_input("Rows to load", min_value=100, max_value=50000, value=5000, step=100)
    sample_df = load_sample(int(nrows))
    idx = st.number_input("Row index", min_value=0, max_value=max(len(sample_df) - 1, 0), value=0, step=1)
    df_input = sample_df.iloc[[int(idx)]]

if df_input is None:
    st.info("Provide input data to run predictions.")
    st.stop()

missing_cols = [c for c in FEATURE_COLUMNS if c not in df_input.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.write("Expected columns:")
    st.write(FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df_input.columns else []))
    st.stop()

st.subheader("Preview")
st.dataframe(df_input.head(50), use_container_width=True)

st.subheader("Predictions")
pred_df = predict_dataframe(model, df_input, threshold)
summary = summarize_predictions(pred_df)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", int(summary["total"]))
c2.metric("Predicted fraud", int(summary["predicted_fraud"]))
c3.metric("Predicted fraud rate", f"{summary['predicted_fraud_rate']:.4%}")

if len(pred_df) > 1:
    st.write("Top risky rows")
    st.dataframe(
        pred_df.sort_values("fraud_probability", ascending=False).head(int(top_n)),
        use_container_width=True,
    )
else:
    st.dataframe(pred_df, use_container_width=True)
