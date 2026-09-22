# ₿ Bitcoin Volatility Forecaster (Flask)

A small Flask site that turns the *Bitcoin Price Prediction & Feature Analysis*
notebook into a working app — with an honest take: **direction is unpredictable,
magnitude is not.**

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2. Add your data (or let the app generate synthetic demo data)
mkdir -p data
cp /path/to/bitcoin_daily_master_360.csv data/

# 3. Run
python app.py
# → http://localhost:5000
```

The model auto-trains the first time `app.py` runs and caches to
`models/volatility_model.pkl`.

## What the site shows

| Page       | Purpose |
|------------|---------|
| `/`        | Live snapshot + tomorrow's volatility forecast (68/90/95% ranges) |
| `/analysis`| Correlations, weekday effect, Fear & Greed regime table |
| `/predict` | What-if sandbox for the 3 model inputs |
| `/api/*`   | JSON endpoints: `/api/current`, `/api/predict`, `/api/history` |

## Model

- **Target:** `E[ |return_{t+1}| ]` (volatility forecast)
- **Features:** `vol30`, `abs_ret1`, `vol_rel`
- **Estimator:** OLS linear regression
- **Sigma conversion:** `σ = E|X| · √(π/2)` (normal identity)
- **Holdout R² ≈ 0.08** — small but real, and beats the naive mean baseline

## Why not predict price or direction?

Reproduced in `analysis.html`:

1. Copying today's close gives **R² ≈ 0.99** on tomorrow's close — no skill needed.
2. Walk-forward Ridge & RandomForest never beat **~50%** on next-day sign.
3. Returns are **fat-tailed**; normal 90% bands under-cover.

So the app targets the one thing that *is* predictable: the size of the move.

## Environment variables

| Var          | Default                         | Meaning |
|--------------|---------------------------------|---------|
| `BTC_DATA`   | `data/bitcoin_daily_master_360.csv` | CSV path |
| `BTC_MODEL`  | `models/volatility_model.pkl`   | Cached model path |