import matplotlib.pyplot as plt
import pandas as pd

from src.config import PREDICTIONS_DIR, REPORTS_DIR


def plot_actual_vs_prediction(store: int = 1, item: int = 1) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_DIR / "validation_predictions.csv")
    df["date"] = pd.to_datetime(df["date"])

    subset = df[(df["store"] == store) & (df["item"] == item)].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(subset["date"], subset["sales"], label="Actual")
    plt.plot(subset["date"], subset["prediction"], label="Prediction")

    plt.title(f"Actual vs Prediction - Store {store}, Item {item}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    output_path = figures_dir / f"actual_vs_prediction_store_{store}_item_{item}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved chart to: {output_path}")