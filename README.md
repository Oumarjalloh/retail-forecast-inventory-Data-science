# Retail Forecast & Stock Optimization (CDI-ready)

Projet **Data Science + Engineering** end-to-end pour le retail :
- **Forecast multi-produits** (daily) : baseline MA, SARIMAX, ML tabulaire (LightGBM)
- **Backtesting** propre (rolling windows) + métriques **sMAPE / WAPE**
- **Module “reco stock”** : niveau de stock recommandé selon **service level**
- **API FastAPI** : renvoie forecast + intervalle + recommandation stock
- **Dashboard Streamlit** : vision “Supply” (forecast, risque de rupture, produits à risque)

---

## 1) Valeur business

Objectif : aider une équipe Supply/Inventory à **prendre de meilleures décisions** :
- anticiper la demande (prévisions daily)
- **prioriser** les produits à risque de rupture
- calculer une **quantité à commander** en fonction de :
  - stock actuel (`on_hand`)
  - délai d’approvisionnement (`lead_time_days`)
  - niveau de service souhaité (`service_level` ex: 95%)

---

## 2) Dataset

Retail time series (Kaggle Store Sales).  
Les CSV sont attendus dans `data/raw/` (minimum : `train.csv`).

Le pipeline prépare un dataset “long format” :
| date | store_id | item_id | y | promo | is_holiday | oil_price | dow | week | month | year | is_weekend |
|------|----------|---------|---|-------|------------|-----------|-----|------|-------|------|-----------|

Sortie ETL :
- `data/processed/sales_long.parquet`

---

## 3) Installation (Windows)

### 3.1 Créer et activer l’environnemen
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

### 3.2 Installer les dépendances  Générer le parquet traité
pip install numpy pandas scikit-learn lightgbm matplotlib fastapi uvicorn streamlit pydantic joblib pyarrow tqdm statsmodels scipy

---

### 4) Générer le parquet traité

python -m src.data.make_dataset

Sortie :

data/processed/sales_long.parquet

---

### 5) Entraînement + Backtesting (rolling)

### Lancer l'entraînement

python -m src.train

Sorties attendues :

models/lgbm_forecaster.joblib
reports/train_summary.json
reports/backtest_baselines.csv (optionnel)

---

### 6) Stock Recommendation (service level)

Module : src/inventory/stock_reco.py

Politique base-stock (simple & pro) :

mu_L = somme des forecasts sur L jours (lead time)
sigma_L ≈ sqrt(L) * residual_std
z = quantile(service_level)
reorder_point = mu_L + z * sigma_L
recommended_order_qty = max(0, reorder_point - on_hand)

Sorties principales :
recommended_order_qty
prob_stockout (approx demande ~ Normal)

---

### 7) API (FastAPI)

### 7.1 Lancer l’API

uvicorn src.api.main:app --reload

Endpoints :

GET /health → check API
POST /forecast → forecast + intervalle + stock reco
Swagger UI : http://127.0.0.1:8000/docs

### 7.2 Exemple de requête /forecast

{
  "store_id": 1,
  "item_id": "AUTOMOTIVE",
  "horizon_days": 14,
  "on_hand": 200,
  "lead_time_days": 7,
  "service_level": 0.95
}

---

### 8) Dashboard (Streamlit)

### 8.1 Lancer le dashboard

streamlit run src\dashboard\app.py

Puis ouvrir :
http://localhost:8501

Le dashboard affiche :

forecast (yhat + lower/upper)
probabilité de rupture
reorder point
recommended order qty

---

### 10) ordre exact d’exécution

### 10.1 Activer venv :
.\.venv\Scripts\activate

### 10.2 Préparer dataset :
python -m src.data.make_dataset

### 10.3 Train + save model :
python -m src.train

### 10.4 Lancer API :
uvicorn src.api.main:app --reload

### 10.5 Lancer dashboard :
streamlit run src.dashboard.app
