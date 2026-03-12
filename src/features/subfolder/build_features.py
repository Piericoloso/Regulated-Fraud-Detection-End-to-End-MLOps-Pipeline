import pandas as pd
from src.features.subfolder.schema import FEATURES, TARGET

def split_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
