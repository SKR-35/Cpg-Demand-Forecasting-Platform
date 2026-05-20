import pandas as pd

from src.config import DATE_COL, TARGET, VALIDATION_START_DATE
from src.evaluate import regression_metrics
from src.load_data import load_train


def evaluate_baseline(window: int = 28) -> dict:
    df = load_train()
    df = df.sort_values(["store", "item", DATE_COL]).copy()

    df["baseline_pred"] = (
        df.groupby(["store", "item"])[TARGET]
        .transform(lambda x: x.shift(1).rolling(window=window).mean())
    )

    valid_df = df[df[DATE_COL] >= VALIDATION_START_DATE].dropna(subset=["baseline_pred"])

    metrics = regression_metrics(
        valid_df[TARGET],
        valid_df["baseline_pred"]
    )

    return metrics