import joblib
import pandas as pd

from src.config import (
    DATE_COL,
    MODEL_PATH,
    PREDICTION_PATH,
    PREDICTIONS_DIR,
    TARGET,
)
from src.features import add_date_features
from src.load_data import load_train, load_test


def make_recursive_features(history: pd.DataFrame, forecast_rows: pd.DataFrame) -> pd.DataFrame:
    row = forecast_rows.copy()
    row = add_date_features(row)

    group_history = history.sort_values(DATE_COL)

    for lag in [7, 14, 28]:
        row[f"sales_lag_{lag}"] = group_history[TARGET].iloc[-lag]

    for window in [7, 28]:
        row[f"sales_rolling_mean_{window}"] = (
            group_history[TARGET].iloc[-window:].mean()
        )

    return row


def make_predictions() -> pd.DataFrame:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    train_df = load_train()
    test_df = load_test()

    train_df = train_df.sort_values(["store", "item", DATE_COL]).copy()
    test_df = test_df.sort_values(["store", "item", DATE_COL]).copy()

    predictions = []

    for (store, item), test_group in test_df.groupby(["store", "item"]):
        history = train_df[
            (train_df["store"] == store) &
            (train_df["item"] == item)
        ][[DATE_COL, "store", "item", TARGET]].copy()

        for _, test_row in test_group.iterrows():
            current_row = pd.DataFrame([test_row])

            feature_row = make_recursive_features(
                history=history,
                forecast_rows=current_row
            )

            for col in feature_cols:
                if col not in feature_row.columns:
                    feature_row[col] = 0

            X = feature_row[feature_cols]
            pred = float(model.predict(X)[0])
            pred = max(pred, 0)

            predictions.append({
                "id": int(test_row["id"]),
                "sales": pred,
                "date": test_row[DATE_COL],
                "store": store,
                "item": item,
            })

            new_history_row = pd.DataFrame([{
                DATE_COL: test_row[DATE_COL],
                "store": store,
                "item": item,
                TARGET: pred,
            }])

            history = pd.concat(
                [history, new_history_row],
                ignore_index=True
            )

    prediction_df = pd.DataFrame(predictions)
    prediction_df = prediction_df.sort_values("id")

    submission = prediction_df[["id", "sales"]].copy()
    submission.to_csv(PREDICTION_PATH, index=False)

    detailed_path = PREDICTIONS_DIR / "recursive_predictions_detailed.csv"
    prediction_df.to_csv(detailed_path, index=False)

    return submission


if __name__ == "__main__":
    output = make_predictions()
    print(output.head())
    print(f"Saved predictions to: {PREDICTION_PATH}")