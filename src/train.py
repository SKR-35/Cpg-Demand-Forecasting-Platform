import joblib
import lightgbm as lgb

from src.config import MODEL_PATH, TARGET, VALIDATION_START_DATE, MODELS_DIR, PREDICTIONS_DIR
from src.evaluate import regression_metrics
from src.features import get_feature_columns, make_features
from src.load_data import load_train

def train_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train()
    df = make_features(df, is_train=True)

    train_df = df[df["date"] < VALIDATION_START_DATE]
    valid_df = df[df["date"] >= VALIDATION_START_DATE]

    feature_cols = get_feature_columns(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)    

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    validation_output = valid_df[["date", "store", "item", "sales"]].copy()
    validation_output["prediction"] = valid_pred
    validation_output.to_csv(
        PREDICTIONS_DIR / "validation_predictions.csv",
        index=False
        )
    
    metrics = regression_metrics(y_valid, valid_pred)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    return metrics