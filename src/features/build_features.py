import pandas as pd

def add_lag_features(df: pd.DataFrame, group_cols=("store_id", "item_id")) -> pd.DataFrame:
    df = df.sort_values(list(group_cols) + ["date"]).copy()
    g = df.groupby(list(group_cols), sort=False)

    # lags
    for lag in (7, 14, 28):
        df[f"lag_{lag}"] = g["y"].shift(lag)

    # rolling mean/std avec transform (index aligné, no reset_index)
    df["roll_mean_7"] = g["y"].transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_mean_28"] = g["y"].transform(lambda s: s.shift(1).rolling(28).mean())
    df["roll_std_28"] = g["y"].transform(lambda s: s.shift(1).rolling(28).std())

    df["roll_std_28"] = df["roll_std_28"].fillna(0)
    return df

def select_features(df: pd.DataFrame):
    feature_cols = [
        "promo", "is_holiday", "oil_price",
        "dow", "week", "month", "year", "is_weekend",
        "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28", "roll_std_28"
    ]
    if "oil_price" in df.columns and df["oil_price"].isna().all():
        feature_cols.remove("oil_price")
    if "oil_price" not in df.columns and "oil_price" in feature_cols:
        feature_cols.remove("oil_price")
    return feature_cols