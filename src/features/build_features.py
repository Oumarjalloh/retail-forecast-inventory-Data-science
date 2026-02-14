import pandas as pd

def add_lag_features(df: pd.DataFrame, group_cols=("store_id", "item_id")) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "date"]).copy()

    g = df.groupby(list(group_cols), sort=False)

    # lags
    for lag in (7, 14, 28):
        df[f"lag_{lag}"] = g["y"].shift(lag)

    # rolling (shift 1 pour éviter leakage)
    df["roll_mean_7"] = g["y"].shift(1).rolling(7).mean().reset_index(level=[0,1], drop=True)
    df["roll_mean_28"] = g["y"].shift(1).rolling(28).mean().reset_index(level=[0,1], drop=True)
    df["roll_std_28"] = g["y"].shift(1).rolling(28).std().reset_index(level=[0,1], drop=True)

    # remplacer NaN
    df["roll_std_28"] = df["roll_std_28"].fillna(0)
    return df

def select_features(df: pd.DataFrame):
    feature_cols = [
        "promo", "is_holiday", "oil_price",
        "dow", "week", "month", "year", "is_weekend",
        "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_28", "roll_std_28"
    ]
    # oil_price peut être None si pas de fichier oil.csv
    if df["oil_price"].isna().all():
        feature_cols.remove("oil_price")
    return feature_cols