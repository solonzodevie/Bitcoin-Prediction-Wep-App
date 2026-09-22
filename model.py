"""
Model training + prediction utilities for the Bitcoin volatility predictor.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------- paths
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = Path(os.getenv("BTC_DATA",  BASE_DIR / "data" / "bitcoin_daily_master_360.csv"))
MODEL_PATH = Path(os.getenv("BTC_MODEL", BASE_DIR / "models" / "volatility_model.pkl"))
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURES: List[str] = ["vol30", "abs_ret1", "vol_rel"]

# ---------------------------------------------------------------- data
def load_raw() -> pd.DataFrame:
    """Load the raw CSV. Generates a synthetic fallback if it is missing."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    print(f"[model] {DATA_PATH} not found -> generating synthetic demo data.")
    return _synthetic_data()


def _synthetic_data(n_days: int = 4385, seed: int = 7) -> pd.DataFrame:
    """Random-walk BTC-like series so the site runs out of the box."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-09-17", periods=n_days, freq="D")
    vol = 0.03 + 0.02 * np.abs(np.sin(np.arange(n_days) / 250))
    rets = rng.normal(0.001, 1, n_days) * vol
    close = 450 * np.exp(np.cumsum(rets))
    volume = np.abs(rng.normal(2e10, 8e9, n_days))
    return pd.DataFrame({
        "date": dates,
        "btc_close": close,
        "btc_volume": volume,
        "btc_return_pct": np.concatenate([[0.0], np.diff(np.log(close)) * 100]),
        "btc_volatility_30d": pd.Series(rets).rolling(30).std().to_numpy() * 100,
        "btc_sma_50":  pd.Series(close).rolling(50).mean(),
        "btc_sma_200": pd.Series(close).rolling(200).mean(),
        "fear_greed_value": np.clip(rng.normal(45, 22, n_days), 5, 95),
        "coinbase_reviews_count": rng.integers(0, 30, n_days),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the stationary features used in the notebook."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = (df.sort_values("date")
            .drop_duplicates("date", keep="last")
            .set_index("date"))

    # Every feature only uses info available at time t
    df["ret1"]     = df["btc_return_pct"]
    df["abs_ret1"] = df["ret1"].abs()
    df["vol30"]    = df["btc_volatility_30d"]

    log_vol = np.log(df["btc_volume"].replace(0, np.nan))
    df["vol_rel"] = log_vol - log_vol.rolling(30).mean()

    # Targets (shifted so row t holds tomorrow's outcome)
    df["y_ret"] = df["btc_return_pct"].shift(-1)
    df["y_abs"] = df["y_ret"].abs()
    return df


# ---------------------------------------------------------------- training
def train() -> Dict:
    """Fit the volatility model on all clean rows. Returns a bundle."""
    d = engineer_features(load_raw())
    train_df = d.dropna(subset=FEATURES + ["y_abs"])
    model = LinearRegression().fit(train_df[FEATURES], train_df["y_abs"])

    # Small holdout evaluation (last 20%) for reporting on the site
    cut = int(len(train_df) * 0.8)
    tr, te = train_df.iloc[:cut], train_df.iloc[cut:]
    hold = LinearRegression().fit(tr[FEATURES], tr["y_abs"])
    p = hold.predict(te[FEATURES])
    y = te["y_abs"].to_numpy()
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    baseline = float(np.sqrt(((y - y.mean()) ** 2).mean()))

    bundle = {
        "model": model,
        "features": FEATURES,
        "holdout": {
            "n": int(len(te)),
            "r2": float(r2),
            "rmse": rmse,
            "baseline_rmse": baseline,
        },
        "last_date": d.index[-1].isoformat(),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[model] trained. holdout R2 = {r2:.3f}, RMSE = {rmse:.3f}")
    return bundle


def load_or_train() -> Dict:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train()


# ---------------------------------------------------------------- prediction
def predict_range(bundle: Dict, features_row: Dict[str, float]) -> Dict[str, float]:
    """
    Turn an expected absolute return into a sigma + confidence ranges.
    For a normal X:  E|X| = sigma * sqrt(2/pi)  =>  sigma = E|X| * sqrt(pi/2).
    """
    x = np.array([[features_row[f] for f in bundle["features"]]], dtype=float)
    e_abs = float(bundle["model"].predict(x)[0])
    e_abs = max(e_abs, 0.05)  # floor to avoid degenerate zero
    sigma = e_abs * np.sqrt(np.pi / 2)
    return {
        "expected_abs_return": e_abs,
        "sigma": sigma,
        "range_68": (-1.0 * sigma, 1.0 * sigma),
        "range_90": (-1.645 * sigma, 1.645 * sigma),
        "range_95": (-1.96 * sigma, 1.96 * sigma),
    }


def latest_features(d: pd.DataFrame) -> Dict[str, float]:
    row = d[FEATURES].dropna().iloc[-1]
    return {k: float(v) for k, v in row.items()}


def latest_snapshot(d: pd.DataFrame) -> Dict:
    """Everything the dashboard needs about 'today'."""
    last = d.iloc[-1]
    prev = d.iloc[-2]
    return {
        "date": d.index[-1].date().isoformat(),
        "close": float(last["btc_close"]),
        "prev_close": float(prev["btc_close"]),
        "change_pct": float(last["btc_return_pct"]),
        "volume": float(last["btc_volume"]),
        "volatility_30d": float(last["btc_volatility_30d"]) if pd.notna(last["btc_volatility_30d"]) else None,
        "sma_50":  float(last["btc_sma_50"])  if pd.notna(last["btc_sma_50"])  else None,
        "sma_200": float(last["btc_sma_200"]) if pd.notna(last["btc_sma_200"]) else None,
        "fear_greed": float(last["fear_greed_value"]) if pd.notna(last.get("fear_greed_value", np.nan)) else None,
    }


# ---------------------------------------------------------------- analytics
def analytics(d: pd.DataFrame) -> Dict:
    """Pre-compute the stats shown on the Analysis page."""
    # Correlation table
    cands = ["vol30", "abs_ret1", "vol_rel", "ret1",
             "btc_close", "btc_volume", "btc_sma_50", "btc_sma_200"]
    rows = []
    for c in cands:
        if c not in d.columns:
            continue
        sub = d[[c, "y_ret", "y_abs"]].dropna()
        if len(sub) < 50:
            continue
        rows.append({
            "feature": c,
            "n": int(len(sub)),
            "corr_next_ret": float(sub[c].corr(sub["y_ret"])),
            "corr_next_abs": float(sub[c].corr(sub["y_abs"])),
        })

    # ---- Weekday effect (FIXED for pandas 2+ / 3+) ----
    d_wd = d.dropna(subset=["y_ret"]).copy()
    d_wd["_weekday"] = d_wd.index.dayofweek
    wd = (d_wd.groupby("_weekday")
              .agg(mean_ret=("btc_return_pct", "mean"),
                   mean_abs=("abs_ret1", "mean")))
    wd = wd.reindex(range(7))  # guarantee all 7 days present
    wd.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = [
        {"day": i,
         "mean_ret": float(r["mean_ret"]) if pd.notna(r["mean_ret"]) else 0.0,
         "mean_abs": float(r["mean_abs"]) if pd.notna(r["mean_abs"]) else 0.0}
        for i, r in wd.iterrows()
    ]

    # ---- Fear & Greed buckets (optional) ----
    fg = []
    if "fear_greed_classification" in d.columns:
        order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        g = (d.dropna(subset=["fear_greed_classification", "y_ret"])
               .groupby("fear_greed_classification")["y_ret"]
               .agg(["mean", "count"])
               .reindex(order)
               .dropna())
        fg = [{"label": i,
               "mean_ret": float(r["mean"]),
               "count": int(r["count"])} for i, r in g.iterrows()]

    return {
        "correlations": rows,
        "weekday": weekday,
        "fear_greed": fg,
    }