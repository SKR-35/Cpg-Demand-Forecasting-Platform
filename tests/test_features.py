import pandas as pd

from src.features import add_date_features, add_lag_features, make_features


def test_add_date_features_creates_expected_columns():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "store": [1, 1],
        "item": [1, 1],
        "sales": [10, 12],
    })

    result = add_date_features(df)

    expected_columns = {
        "dayofweek",
        "month",
        "year",
        "dayofyear",
        "weekofyear",
        "is_weekend",
    }

    assert expected_columns.issubset(result.columns)


def test_add_lag_features_creates_expected_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=40),
        "store": [1] * 40,
        "item": [1] * 40,
        "sales": list(range(40)),
    })

    result = add_lag_features(df)

    expected_columns = {
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_rolling_mean_7",
        "sales_rolling_mean_28",
    }

    assert expected_columns.issubset(result.columns)


def test_lag_values_are_correct():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=40),
        "store": [1] * 40,
        "item": [1] * 40,
        "sales": list(range(40)),
    })

    result = add_lag_features(df)

    # Row index 7 should have lag_7 equal to original sales at index 0
    assert result.loc[7, "sales_lag_7"] == 0

    # Row index 28 should have lag_28 equal to original sales at index 0
    assert result.loc[28, "sales_lag_28"] == 0


def test_make_features_drops_na_rows_for_training():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=40),
        "store": [1] * 40,
        "item": [1] * 40,
        "sales": list(range(40)),
    })

    result = make_features(df, is_train=True)

    assert result.isna().sum().sum() == 0
    assert len(result) < len(df)