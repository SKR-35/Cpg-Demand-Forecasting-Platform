import pandas as pd

from src.config import DATE_COL, TARGET


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month
    df["year"] = df[DATE_COL].dt.year
    df["dayofyear"] = df[DATE_COL].dt.dayofyear
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["store", "item", DATE_COL])

    group_cols = ["store", "item"]

    for lag in [7, 14, 28]:
        df[f"sales_lag_{lag}"] = df.groupby(group_cols)[TARGET].shift(lag)

    for window in [7, 28]:
        df[f"sales_rolling_mean_{window}"] = (
        df.groupby(group_cols)[TARGET]
        .transform(lambda x: x.shift(1).rolling(window=window).mean())
        )

    return df


def make_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = add_date_features(df)

    if is_train:
        df = add_lag_features(df)
        df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {DATE_COL, TARGET}
    return [col for col in df.columns if col not in excluded]