import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException

from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from src.features.build_features import add_lag_features, select_features
from src.models.lgbm_model import LGBMForecaster
from src.inventory.stock_reco import recommend_order
from src.api.schemas import ForecastRequest, ForecastResponse, ForecastPoint

app = FastAPI(title="Retail Forecast + Stock Reco API")

# Load artifacts
MODEL_PATH = MODELS_DIR / "lgbm_forecaster.joblib"
SUMMARY_PATH = REPORTS_DIR / "train_summary.json"

df_all = pd.read_parquet(DATA_PROCESSED / "sales_long.parquet").copy()

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run: python -m src.train")

forecaster = LGBMForecaster.load(MODEL_PATH)

residual_std = 0.0
if SUMMARY_PATH.exists():
    s = pd.read_json(SUMMARY_PATH, typ="series")
    residual_std = float(s.get("residual_std", 0.0))

def build_future_frame(hist: pd.DataFrame, horizon: int):
    last_date = hist["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    base = pd.DataFrame({
        "date": future_dates,
        "store_id": hist["store_id"].iloc[0],
        "item_id": hist["item_id"].iloc[0],
        "y": np.nan,
        "promo": 0,
        "is_holiday": 0,
        "oil_price": hist["oil_price"].iloc[-1] if "oil_price" in hist.columns else np.nan
    })
    # calendrier
    base["dow"] = base["date"].dt.dayofweek
    base["week"] = base["date"].dt.isocalendar().week.astype(int)
    base["month"] = base["date"].dt.month
    base["year"] = base["date"].dt.year
    base["is_weekend"] = (base["dow"] >= 5).astype(int)
    return base

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    hist = df_all[(df_all["store_id"] == req.store_id) & (df_all["item_id"] == req.item_id)].copy()
    if hist.empty:
        raise HTTPException(status_code=404, detail="Series not found (store_id/item_id).")

    hist = hist.sort_values("date").tail(365)  # limiter pour vitesse
    future = build_future_frame(hist, req.horizon_days)

    # concat hist+future pour recalculer les lags correctement
    full = pd.concat([hist, future], ignore_index=True)
    full = add_lag_features(full)

    feature_cols = forecaster.feature_cols
    # intervalle simple: yhat +/- 1.96*residual_std
    preds = forecaster.predict(full.tail(req.horizon_days))

    z = 1.96
    lower = np.maximum(preds - z * residual_std, 0)
    upper = np.maximum(preds + z * residual_std, 0)

    out_points = []
    for d, yhat, lo, up in zip(full.tail(req.horizon_days)["date"], preds, lower, upper):
        out_points.append(ForecastPoint(
            date=d.strftime("%Y-%m-%d"),
            yhat=float(yhat),
            yhat_lower=float(lo),
            yhat_upper=float(up)
        ))

    # stock reco sur lead time
    stock = recommend_order(
        forecast_next_L_days=list(preds),
        on_hand=req.on_hand,
        lead_time_days=req.lead_time_days,
        service_level=req.service_level,
        residual_std=residual_std
    )

    meta = {
        "model": "lightgbm_global",
        "residual_std": residual_std,
        "horizon_days": req.horizon_days
    }

    return ForecastResponse(series=out_points, stock=stock, meta=meta)

@app.get("/health")
def health():
    return {"ok": True}