import json
import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORTS_DIR


def plot_model_comparison():
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORTS_DIR / "metrics.json"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.DataFrame(metrics).T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metric_names = ["mae", "rmse", "smape"]

    for ax, metric in zip(axes, metric_names):
        df[metric].plot(kind="bar", ax=ax)

        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    output_path = figures_dir / "model_comparison.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Saved comparison plot to: {output_path}")