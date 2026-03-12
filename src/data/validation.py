# src/data/validation.py

import pandas as pd
from src.features.subfolder.schema import FEATURES, TARGET

def validate_dataset(df: pd.DataFrame) -> None:
    expected = set(FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # No NaNs
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("NaNs detected in dataset")

    # Label must be binary
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must be 0/1")

    # Range / domain checks
    if (df["hour"] < 0).any() or (df["hour"] > 23).any():
        raise ValueError("hour out of range 0–23")

    if (df["country_risk"] < 1).any() or (df["country_risk"] > 5).any():
        raise ValueError("country_risk out of range 1–5")

    if not set(df["is_cross_border"].unique()).issubset({0, 1}):
        raise ValueError("is_cross_border must be 0/1")

    if (df["amount"] < 0).any():
        raise ValueError("amount must be >= 0")
