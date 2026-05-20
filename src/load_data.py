import pandas as pd

from src.config import TRAIN_PATH, TEST_PATH, DATE_COL


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df