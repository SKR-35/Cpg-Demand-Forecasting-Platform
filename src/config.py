from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

MODEL_PATH = MODELS_DIR / "lightgbm_demand_model.joblib"
PREDICTION_PATH = PREDICTIONS_DIR / "submission.csv"

TARGET = "sales"
DATE_COL = "date"

RANDOM_STATE = 42
VALIDATION_START_DATE = "2017-10-01"
FORECAST_HORIZON = 90