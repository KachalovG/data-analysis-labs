from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "total"
SOURCE_COLUMNS = ["total", "food", "non_food", "services"]


def _to_float(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        return series.str.replace(",", ".", regex=False).astype(float)
    return series.astype(float)


def prepare_data():
    data_path = Path(__file__).resolve().parent / "ipc_dataset.csv"
    df = pd.read_csv(data_path, sep=";")

    for column in SOURCE_COLUMNS:
        df[column] = _to_float(df[column])

    df["month_num"] = df["month_num"].astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    for column in SOURCE_COLUMNS:
        for lag in range(1, 4):
            df[f"{column}_lag{lag}"] = df[column].shift(lag)

    df["target"] = df[TARGET_COLUMN]
    df = df.dropna().reset_index(drop=True)

    feature_columns = ["year", "month_num", "month_sin", "month_cos"] + [
        f"{column}_lag{lag}" for column in SOURCE_COLUMNS for lag in range(1, 4)
    ]

    X = df[feature_columns]
    y = df["target"]
    return X, y, feature_columns
