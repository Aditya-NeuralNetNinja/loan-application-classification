from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path when run as: streamlit run app/streamlit_app.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.data_loader import (
    build_leaderboard,
    business_cost_table,
    load_model_results,
    load_optimal_threshold,
    recommend_model,
)
from app.inference import InferenceError, get_required_columns, run_gbt_inference_from_pandas

st.set_page_config(page_title="HMDA Model Dashboard", page_icon="🏦", layout="wide")

st.title("HMDA Loan Denial Model Deployment Dashboard")
st.caption("Deployable dashboard for model selection, threshold policy, and business trade-offs.")

model_results, model_source = load_model_results()
threshold_data, threshold_source = load_optimal_threshold()
leaderboard, lb_source = build_leaderboard()
reco = recommend_model(model_results)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Models Available", len(model_results))
with col2:
    st.metric("Recommended Model", reco.get("recommended_model") or "N/A")
with col3:
    st.metric("Suggested Threshold", f"{threshold_data.get('optimal_threshold_f1', 0.5):.2f}")

with st.expander("Artifact Sources"):
    st.write("- Model Results:", model_source)
    st.write("- Leaderboard:", lb_source)
    st.write("- Threshold:", threshold_source)

st.subheader("Leaderboard")
if "Rank" in leaderboard.columns:
    display_cols = [
        c
        for c in ["Rank", "Model", "PR-AUC", "ROC-AUC", "Denial_F1", "D_Prec", "D_Rec", "Accuracy", "Time_s"]
        if c in leaderboard.columns
    ]
else:
    display_cols = [
        c
        for c in ["Model", "PR-AUC", "ROC-AUC", "Denial_F1", "D_Prec", "D_Rec", "Accuracy", "Time_s"]
        if c in leaderboard.columns
    ]

st.dataframe(leaderboard[display_cols], use_container_width=True)

st.subheader("Business Cost Simulation")
c1, c2 = st.columns(2)
with c1:
    cost_fp = st.slider("Cost of False Positive", min_value=50, max_value=1000, value=250, step=50)
with c2:
    cost_fn = st.slider("Cost of False Negative", min_value=500, max_value=10000, value=2500, step=100)

biz_df = business_cost_table(model_results, cost_fp=float(cost_fp), cost_fn=float(cost_fn))

st.dataframe(biz_df, use_container_width=True)

if not biz_df.empty:
    st.bar_chart(
        biz_df.set_index("Model")["Cost_per_1k_apps"],
        use_container_width=True,
    )

st.subheader("Recommendation")
st.json(reco)

st.subheader("Threshold Artifact")
st.json(threshold_data)

st.info(
    "This dashboard is artifact-driven for lightweight deployment. "
    "Full Spark inference services should run separately on larger infrastructure."
)

st.divider()
st.subheader("Batch Inference: Upload Dataset -> Preprocess -> GBT Scoring")
st.caption(
    "This runs your saved Spark preprocessing pipeline and optimal GBT model on uploaded CSV data."
)

uploaded_file = st.file_uploader(
    "Upload HMDA-like CSV for inference",
    type=["csv"],
    help="Column names are normalized automatically (lowercase, non-alphanumeric -> underscore).",
)

default_threshold = float(threshold_data.get("optimal_threshold_f1", 0.5))
inference_threshold = st.number_input(
    "Decision Threshold for Predicted Denial",
    min_value=0.01,
    max_value=0.99,
    value=float(default_threshold),
    step=0.01,
)

with st.expander("Required Input Columns (minimum contract)"):
    st.json(get_required_columns())

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as exc:
        st.error(f"Failed to read uploaded CSV: {exc}")
        st.stop()

    st.write(f"Uploaded rows: **{len(upload_df):,}**, columns: **{len(upload_df.columns):,}**")
    st.dataframe(upload_df.head(20), use_container_width=True)

    if st.button("Run Preprocessing + GBT Inference", type="primary"):
        with st.spinner("Running Spark preprocessing and GBT inference..."):
            try:
                scored_df, inference_summary = run_gbt_inference_from_pandas(
                    upload_df,
                    threshold=float(inference_threshold),
                )
            except InferenceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)
            else:
                st.success("Inference complete.")
                st.json(inference_summary)
                st.dataframe(scored_df.head(200), use_container_width=True)

                csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Scored Results CSV",
                    data=csv_bytes,
                    file_name="hmda_scored_output.csv",
                    mime="text/csv",
                )
