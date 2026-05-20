import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "data" / "predictions" / "validation_predictions.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
MODEL_COMPARISON_PATH = ROOT / "reports" / "figures" / "model_comparison.png"


st.set_page_config(
    page_title="CPG Demand Forecasting",
    layout="wide",
)

st.title("CPG Demand Forecasting Dashboard")

st.markdown(
    """
    Interactive dashboard for store-item level demand forecasting.
    The model compares a LightGBM forecasting approach against a 28-day moving average baseline.
    """
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


df = load_predictions()
metrics = load_metrics()

baseline = metrics["baseline_28_day_moving_average"]
lightgbm = metrics["lightgbm"]

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{lightgbm['mae']:.2f}",
    delta=f"{baseline['mae'] - lightgbm['mae']:.2f} improvement",
)

col2.metric(
    "RMSE",
    f"{lightgbm['rmse']:.2f}",
    delta=f"{baseline['rmse'] - lightgbm['rmse']:.2f} improvement",
)

col3.metric(
    "SMAPE",
    f"{lightgbm['smape']:.2f}%",
    delta=f"{baseline['smape'] - lightgbm['smape']:.2f} pp improvement",
)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Actual vs Prediction")

    stores = sorted(df["store"].unique())
    selected_store = st.selectbox("Store", stores)

    items = sorted(df[df["store"] == selected_store]["item"].unique())
    selected_item = st.selectbox("Item", items)

    subset = df[
        (df["store"] == selected_store)
        & (df["item"] == selected_item)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(subset["date"], subset["sales"], label="Actual")
    ax.plot(subset["date"], subset["prediction"], label="Prediction")

    ax.set_title(f"Store {selected_store} | Item {selected_item}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

with right:
    st.subheader("Selected Series Summary")

    st.write(f"**Store:** {selected_store}")
    st.write(f"**Item:** {selected_item}")
    st.write(f"**Average actual sales:** {subset['sales'].mean():.2f}")
    st.write(f"**Average prediction:** {subset['prediction'].mean():.2f}")
    st.write(f"**Max actual sales:** {subset['sales'].max():.2f}")
    st.write(f"**Min actual sales:** {subset['sales'].min():.2f}")

st.divider()

st.subheader("Model Comparison")

if MODEL_COMPARISON_PATH.exists():
    st.image(str(MODEL_COMPARISON_PATH))
else:
    st.warning("Model comparison chart not found. Run `python run_pipeline.py` first.")

with st.expander("Raw metrics JSON"):
    st.json(metrics)