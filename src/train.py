import pandas as pd
import numpy as np
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from src.features.build_features import add_lag_features, select_features
from src.backtest.rolling_cv import rolling_splits
from src.backtest.metrics import smape, wape
from src.models.baseline_ma import predict_ma
from src.models.arima_sarimax import predict_sarimax
from src.models.lgbm_model import LGBMForecaster

def pick_top_series(df, top_n=10):
    # choisir les top séries par volume (plus rapide)
    agg = df.groupby(["store_id","item_id"])["y"].sum().sort_values(ascending=False)
    top = agg.head(top_n).reset_index()[["store_id","item_id"]]
    return df.merge(top, on=["store_id","item_id"], how="inner")

def backtest_one_series(df_series, train_days=180, horizon=14, step=14):
    # df_series: une série (store,item)
    df_series = df_series.sort_values("date").copy()
    dates = df_series["date"]

    rows = []
    for train_end, test_start, test_end in rolling_splits(dates, train_days, horizon, step):
        train_mask = df_series["date"] <= train_end
        test_mask = (df_series["date"] >= test_start) & (df_series["date"] <= test_end)

        y_train = df_series.loc[train_mask, "y"]
        y_test = df_series.loc[test_mask, "y"].values
        h = len(y_test)

        # MA
        pred_ma = np.array(predict_ma(y_train, h, window=28))

        # SARIMAX
        pred_ar = np.array(predict_sarimax(y_train, h))

        rows.append({
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "h": h,
            "smape_ma": smape(y_test, pred_ma),
            "wape_ma": wape(y_test, pred_ma),
            "smape_sarimax": smape(y_test, pred_ar),
            "wape_sarimax": wape(y_test, pred_ar),
        })
    return pd.DataFrame(rows)

def main():
    df = pd.read_parquet(DATA_PROCESSED / "sales_long.parquet")
    df = pick_top_series(df, top_n=50)  # pour que ça tourne vite

    # features pour LGBM (global model)
    df_feat = add_lag_features(df)
    feature_cols = select_features(df_feat)

    # split simple chrono pour fit global model
    df_feat = df_feat.sort_values("date")
    cutoff = df_feat["date"].quantile(0.9)  # 90% train
    train_df = df_feat[df_feat["date"] <= cutoff].copy()
    valid_df = df_feat[df_feat["date"] > cutoff].copy()

    lgbm = LGBMForecaster(feature_cols)
    lgbm.fit(train_df)

    # eval global sur valid
    valid_df = valid_df.dropna(subset=feature_cols + ["y"])
    pred = lgbm.predict(valid_df)
    global_smape = float(np.round(smape(valid_df["y"].values, pred), 4))
    global_wape = float(np.round(wape(valid_df["y"].values, pred), 4))

    # backtest baselines sur un sous-ensemble de séries
    bt_list = []
    for (store_id, item_id), g in df.groupby(["store_id","item_id"]):
        bt = backtest_one_series(g, train_days=180, horizon=14, step=14)
        if len(bt):
            bt["store_id"] = store_id
            bt["item_id"] = item_id
            bt_list.append(bt)

    bt_all = pd.concat(bt_list, ignore_index=True) if bt_list else pd.DataFrame()

    # sauvegardes
    model_path = MODELS_DIR / "lgbm_forecaster.joblib"
    lgbm.save(model_path)

    # residual std (pour intervalle + stock reco), basé sur valid
    residuals = (valid_df["y"].values - pred)
    residual_std = float(np.std(residuals)) if len(residuals) else 0.0

    summary = {
        "lgbm_smape_valid": global_smape,
        "lgbm_wape_valid": global_wape,
        "residual_std": residual_std,
        "feature_cols": feature_cols,
    }
    pd.Series(summary).to_json(REPORTS_DIR / "train_summary.json")

    if len(bt_all):
        bt_all.to_csv(REPORTS_DIR / "backtest_baselines.csv", index=False)

    print("[OK] Model saved:", model_path)
    print("[OK] Summary:", summary)
    if len(bt_all):
        print("[OK] Baseline backtest saved:", REPORTS_DIR / "backtest_baselines.csv")

if __name__ == "__main__":
    main()