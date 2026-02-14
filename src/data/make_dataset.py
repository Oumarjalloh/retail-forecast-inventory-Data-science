import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED

def _read_csv(name: str) -> pd.DataFrame | None:
    path = DATA_RAW / name
    if not path.exists():
        return None
    return pd.read_csv(path)

def main():
    train = _read_csv("train.csv")
    if train is None:
        raise FileNotFoundError("data/raw/train.csv introuvable (mets le dataset Kaggle dans data/raw).")

    train["date"] = pd.to_datetime(train["date"])
    # Colonnes attendues dans Store Sales Kaggle:
    # date, store_nbr, family, sales, onpromotion

    # Holidays (optionnel)
    hol = _read_csv("holidays_events.csv")
    if hol is not None:
        hol["date"] = pd.to_datetime(hol["date"])
        hol = hol[["date", "type", "locale", "transferred"]].copy()
        hol["is_holiday"] = (hol["type"].isin(["Holiday", "Event"])) & (~hol["transferred"].fillna(False))
        hol = hol.groupby("date", as_index=False)["is_holiday"].max()
    else:
        hol = pd.DataFrame({"date": train["date"].unique(), "is_holiday": False})

    # Oil (optionnel)
    oil = _read_csv("oil.csv")
    if oil is not None:
        oil["date"] = pd.to_datetime(oil["date"])
        oil = oil.rename(columns={"dcoilwtico": "oil_price"})
        oil = oil.sort_values("date")
        oil["oil_price"] = oil["oil_price"].interpolate().ffill().bfill()
    else:
        oil = pd.DataFrame({"date": train["date"].unique(), "oil_price": None})

    df = train.copy()
    df = df.rename(columns={"store_nbr": "store_id", "family": "item_id", "sales": "y", "onpromotion": "promo"})
    df["promo"] = df["promo"].fillna(0).astype(int)

    df = df.merge(hol, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(False).astype(int)

    df = df.merge(oil[["date", "oil_price"]], on="date", how="left")

    # calendrier
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Option: retirer les ventes négatives / anomalies
    df["y"] = df["y"].clip(lower=0)

    out = DATA_PROCESSED / "sales_long.parquet"
    df.to_parquet(out, index=False)
    print(f"[OK] Saved: {out} rows={len(df):,} cols={df.shape[1]}")

if __name__ == "__main__":
    main()