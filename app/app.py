from __future__ import annotations

import sys
import time
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


def _inject_css(theme: str) -> None:
    palette = {
        "dark": {
            "bg": "#070A12",
            "panel": "rgba(255,255,255,0.06)",
            "panel_border": "rgba(255,255,255,0.10)",
            "text": "#E9ECF5",
            "muted": "rgba(233,236,245,0.70)",
            "primary": "#7C5CFF",
            "primary_2": "#35D1FF",
            "success": "#22C55E",
            "danger": "#EF4444",
            "warning": "#F59E0B",
            "shadow": "0 14px 50px rgba(0,0,0,0.45)",
        },
        "light": {
            "bg": "#F6F8FF",
            "panel": "rgba(255,255,255,0.92)",
            "panel_border": "rgba(16,24,40,0.10)",
            "text": "#0B1220",
            "muted": "rgba(11,18,32,0.65)",
            "primary": "#635BFF",
            "primary_2": "#00B3FF",
            "success": "#16A34A",
            "danger": "#DC2626",
            "warning": "#D97706",
            "shadow": "0 18px 55px rgba(17, 24, 39, 0.12)",
        },
    }["dark" if theme.lower().startswith("dark") else "light"]

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg: {palette['bg']};
  --panel: {palette['panel']};
  --panel-border: {palette['panel_border']};
  --text: {palette['text']};
  --muted: {palette['muted']};
  --primary: {palette['primary']};
  --primary-2: {palette['primary_2']};
  --success: {palette['success']};
  --danger: {palette['danger']};
  --warning: {palette['warning']};
  --shadow: {palette['shadow']};
  --radius: 18px;
}}

html, body, [class*="css"] {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}

.stApp {{
  background: radial-gradient(1200px 800px at 12% 8%, rgba(124,92,255,0.26), rgba(124,92,255,0) 55%),
              radial-gradient(1200px 900px at 92% 12%, rgba(53,209,255,0.18), rgba(53,209,255,0) 55%),
              radial-gradient(1000px 800px at 70% 100%, rgba(34,197,94,0.10), rgba(34,197,94,0) 55%),
              var(--bg);
  color: var(--text);
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

section[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, rgba(124,92,255,0.10), rgba(53,209,255,0.06));
  border-right: 1px solid var(--panel-border);
}}

.block-container {{
  padding-top: 1.1rem;
  padding-bottom: 3rem;
  max-width: 1250px;
}}

.fg-hero {{
  position: relative;
  border-radius: calc(var(--radius) + 6px);
  padding: 30px 28px;
  border: 1px solid var(--panel-border);
  background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(53,209,255,0.12));
  box-shadow: var(--shadow);
  overflow: hidden;
}}

.fg-hero:after {{
  content: "";
  position: absolute;
  inset: -60px;
  background: radial-gradient(circle at 18% 20%, rgba(255,255,255,0.16), rgba(255,255,255,0) 52%);
  pointer-events: none;
}}

.fg-badge {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--panel-border);
  background: rgba(255,255,255,0.08);
  color: var(--text);
  font-size: 0.92rem;
  font-weight: 600;
}}

.fg-title {{
  margin: 14px 0 8px 0;
  font-size: 2.25rem;
  line-height: 1.15;
  letter-spacing: -0.02em;
  font-weight: 700;
}}

.fg-tagline {{
  margin: 0;
  font-size: 1.02rem;
  color: var(--muted);
  max-width: 68ch;
}}

.fg-card {{
  border-radius: var(--radius);
  border: 1px solid var(--panel-border);
  background: var(--panel);
  padding: 18px 18px;
  box-shadow: var(--shadow);
}}

.fg-card-title {{
  margin: 0 0 8px 0;
  font-weight: 700;
  letter-spacing: -0.01em;
  font-size: 1.05rem;
}}

.fg-card-sub {{
  margin: 0 0 14px 0;
  color: var(--muted);
  font-size: 0.95rem;
}}

.fg-metric-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}}

@media (max-width: 1100px) {{
  .fg-metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

.fg-metric {{
  border-radius: 16px;
  border: 1px solid var(--panel-border);
  background: rgba(255,255,255,0.06);
  padding: 12px 14px;
}}

.fg-metric-label {{
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 600;
  margin: 0;
}}

.fg-metric-value {{
  margin: 4px 0 0 0;
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}}

.fg-cta .stButton > button {{
  width: 100%;
  border-radius: 14px;
  padding: 0.72rem 0.95rem;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.14);
  background: linear-gradient(135deg, var(--primary), var(--primary-2));
  color: white;
  transition: transform .06s ease-in-out, box-shadow .15s ease;
  box-shadow: 0 12px 30px rgba(99,91,255,0.26);
}}

.fg-cta .stButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 16px 36px rgba(99,91,255,0.32);
}}

.stButton > button {{
  border-radius: 14px;
}}

div[data-testid="stFileUploader"] section {{
  border-radius: 16px;
  border: 1px dashed rgba(255,255,255,0.22);
  background: rgba(255,255,255,0.04);
}}

.stTextInput input, .stNumberInput input {{
  border-radius: 14px !important;
}}

div[data-testid="stSlider"] > div {{
  padding-top: 0.25rem;
}}

.stProgress > div > div > div {{
  background: linear-gradient(90deg, var(--primary), var(--primary-2)) !important;
}}

.fg-status {{
  border-radius: var(--radius);
  padding: 16px 16px;
  border: 1px solid var(--panel-border);
  background: rgba(255,255,255,0.06);
}}

.fg-status-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}

.fg-status-title {{
  margin: 0;
  font-weight: 800;
  letter-spacing: -0.01em;
  font-size: 1.1rem;
}}

.fg-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.9rem;
  font-weight: 700;
  border: 1px solid var(--panel-border);
}}

.fg-pill.success {{
  background: rgba(34,197,94,0.12);
  color: var(--success);
}}

.fg-pill.danger {{
  background: rgba(239,68,68,0.12);
  color: var(--danger);
}}

.fg-pill.warning {{
  background: rgba(245,158,11,0.12);
  color: var(--warning);
}}

.fg-muted {{
  margin: 10px 0 0 0;
  color: var(--muted);
  font-size: 0.93rem;
}}

.fg-divider {{
  height: 1px;
  background: var(--panel-border);
  margin: 14px 0;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, icon: str = "") -> None:
    st.markdown(
        f"""
<div class="fg-metric">
  <p class="fg-metric-label">{icon} {label}</p>
  <p class="fg-metric-value">{value}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_artifacts():
    model = load_model()
    metadata = load_metadata()
    return model, metadata


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

try:
    model, metadata = get_artifacts()
except FileNotFoundError:
    st.error("Model not found. Train it first by running: python src/train.py")
    st.stop()

threshold_default = float(metadata.get("threshold", 0.5))
theme = st.sidebar.radio("Theme", options=["Dark", "Light"], horizontal=True, help="Toggle a premium fintech theme.")
_inject_css(theme)

st.markdown(
    """
<div class="fg-hero">
  <div class="fg-badge">🛡️ <span>FraudGuard</span> <span style="opacity:.65; font-weight:600;">•</span> <span style="opacity:.85; font-weight:600;">Risk Scoring</span></div>
  <div class="fg-title">Credit Card Fraud Detection</div>
  <p class="fg-tagline">Trustworthy, fintech-grade scoring for transaction batches or single-row checks. Tune the decision threshold and review the highest-risk items instantly.</p>
</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=threshold_default,
    step=0.01,
    help="Higher threshold reduces false positives but may miss some fraud.",
)

top_n = st.sidebar.number_input(
    "Top N risky rows",
    min_value=5,
    max_value=200,
    value=25,
    step=5,
    help="When scoring multiple rows, show the N highest-risk transactions.",
)

metrics = metadata.get("metrics", {})
st.markdown("<div style=\"height: 18px\"></div>", unsafe_allow_html=True)
st.markdown('<div class="fg-card">', unsafe_allow_html=True)
st.markdown('<div class="fg-card-title">📈 Model health</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="fg-card-sub">Quick checks to build trust: current threshold and offline test metrics.</div>',
    unsafe_allow_html=True,
)
cols = st.columns(4)
with cols[0]:
    _metric_card("Threshold", f"{threshold:.2f}", "🎚️")
with cols[1]:
    _metric_card("Test PR-AUC", f"{float(metrics.get('test_pr_auc', 0.0)):.4f}", "✅")
with cols[2]:
    _metric_card("Test Recall", f"{float(metrics.get('test_recall', 0.0)):.4f}", "🎯")
with cols[3]:
    _metric_card("Test F1", f"{float(metrics.get('test_f1', 0.0)):.4f}", "⚖️")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style=\"height: 16px\"></div>", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-title">🧾 Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fg-card-sub">Upload a CSV (batch scoring) or pull a row from the bundled dataset sample.</div>',
        unsafe_allow_html=True,
    )

    with st.form("fg_input_form", clear_on_submit=False):
        mode = st.radio(
            "Input method",
            options=["Upload CSV", "Use bundled dataset sample"],
            horizontal=True,
            help="Upload a CSV to score multiple rows, or pick a single sample row for a quick check.",
        )

        df_input: pd.DataFrame | None = None
        preview_caption = ""

        if mode == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="The CSV must include the required feature columns (Time, V1–V28, Amount).",
            )
        else:
            nrows = st.number_input(
                "Rows to load",
                min_value=100,
                max_value=50000,
                value=5000,
                step=100,
                help="Loads only the first N rows from the bundled dataset file.",
            )
            sample_df = load_sample(int(nrows))
            idx = st.number_input(
                "Row index",
                min_value=0,
                max_value=max(len(sample_df) - 1, 0),
                value=0,
                step=1,
                help="Pick a single transaction row to score.",
            )
            preview_caption = f"Bundled sample • row {int(idx)}"

        st.markdown("<div style=\"height: 6px\"></div>", unsafe_allow_html=True)
        cta = st.columns([1, 1])
        with cta[0]:
            st.markdown('<div class="fg-cta">', unsafe_allow_html=True)
            run = st.form_submit_button("🔎 Run risk scoring")
            st.markdown("</div>", unsafe_allow_html=True)
        with cta[1]:
            show_preview = st.checkbox("Show preview", value=True, help="Preview the input rows before scoring.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-title">🧠 Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fg-card-sub">Fraud status with confidence and clear feedback. Use the threshold in the sidebar to tune sensitivity.</div>',
        unsafe_allow_html=True,
    )

    if not "run" in locals() or not run:
        st.info("Set your input options and click **Run risk scoring**.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    if mode == "Upload CSV":
        if uploaded is None:
            st.warning("Upload a CSV to continue.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        df_input = pd.read_csv(uploaded)
        preview_caption = "Uploaded CSV"
    else:
        df_input = sample_df.iloc[[int(idx)]]

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_input.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Expected columns:")
        st.write(FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df_input.columns else []))
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Running risk checks…"):
        p = st.progress(0)
        for i in range(1, 6):
            time.sleep(0.08)
            p.progress(i * 20)
        pred_df = predict_dataframe(model, df_input, threshold)
        summary = summarize_predictions(pred_df)
        p.empty()

    total = int(summary["total"])
    predicted_fraud = int(summary["predicted_fraud"])
    fraud_rate = float(summary["predicted_fraud_rate"]) if total > 0 else 0.0

    if total == 1:
        proba = float(pred_df.iloc[0]["fraud_probability"])
        pred = int(pred_df.iloc[0]["fraud_prediction"])
        is_fraud = pred == 1
        pill_class = "danger" if is_fraud else "success"
        pill_text = "Fraud suspected" if is_fraud else "Not fraud"
        pill_icon = "⚠️" if is_fraud else "✅"

        st.markdown(
            f"""
<div class="fg-status">
  <div class="fg-status-head">
    <p class="fg-status-title">Decision</p>
    <span class="fg-pill {pill_class}">{pill_icon} {pill_text}</span>
  </div>
  <p class="fg-muted">Fraud probability: <b>{proba:.3f}</b> • Threshold: <b>{threshold:.2f}</b></p>
  <div class="fg-divider"></div>
  <p class="fg-muted" style="margin-top:0;">Confidence</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"{preview_caption}")
    else:
        if predicted_fraud == 0:
            pill_class = "success"
            pill_text = "No suspicious rows"
            pill_icon = "✅"
        elif fraud_rate >= 0.05:
            pill_class = "danger"
            pill_text = "Elevated fraud risk"
            pill_icon = "⚠️"
        else:
            pill_class = "warning"
            pill_text = "Some risk detected"
            pill_icon = "🟡"

        st.markdown(
            f"""
<div class="fg-status">
  <div class="fg-status-head">
    <p class="fg-status-title">Batch summary</p>
    <span class="fg-pill {pill_class}">{pill_icon} {pill_text}</span>
  </div>
  <p class="fg-muted">Rows: <b>{total}</b> • Predicted fraud: <b>{predicted_fraud}</b> • Fraud rate: <b>{fraud_rate:.2%}</b></p>
  <div class="fg-divider"></div>
  <p class="fg-muted" style="margin-top:0;">Fraud-rate indicator</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(fraud_rate, 0.0), 1.0))
        st.caption(f"{preview_caption}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style=\"height: 16px\"></div>", unsafe_allow_html=True)

if "pred_df" in locals() and pred_df is not None:
    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    st.markdown('<div class="fg-card-title">🧩 Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fg-card-sub">Review the input preview and the highest-risk rows based on predicted fraud probability.</div>',
        unsafe_allow_html=True,
    )

    if show_preview:
        st.dataframe(df_input.head(50), use_container_width=True)

    if len(pred_df) > 1:
        st.markdown("<div style=\"height: 10px\"></div>", unsafe_allow_html=True)
        st.dataframe(
            pred_df.sort_values("fraud_probability", ascending=False).head(int(top_n)),
            use_container_width=True,
        )
    else:
        st.markdown("<div style=\"height: 10px\"></div>", unsafe_allow_html=True)
        st.dataframe(pred_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
