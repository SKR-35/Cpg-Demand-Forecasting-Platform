import json

from src.baseline import evaluate_baseline
from src.config import REPORTS_DIR
from src.predict import make_predictions
from src.train import train_model
from src.visualize import plot_actual_vs_prediction
from src.model_comparison import plot_model_comparison


def save_metrics(metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved metrics to: {output_path}")


def main():
    baseline_metrics = evaluate_baseline()
    model_metrics = train_model()

    all_metrics = {
        "baseline_28_day_moving_average": baseline_metrics,
        "lightgbm": model_metrics,
    }

    print("Model comparison:")
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    save_metrics(all_metrics)
    
    plot_actual_vs_prediction(store=1, item=1)
    
    plot_model_comparison()

    submission = make_predictions()
    print("\nPrediction sample:")
    print(submission.head())


if __name__ == "__main__":
    main()