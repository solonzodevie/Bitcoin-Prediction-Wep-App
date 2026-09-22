from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
import os

from model import (
    analytics,
    engineer_features,
    latest_features,
    latest_snapshot,
    load_or_train,
    load_raw,
    predict_range,
)

app = Flask(__name__)

# ---- Load once at startup -------------------------------------------------
BUNDLE = load_or_train()
DF = engineer_features(load_raw())
FEATURES_TODAY = latest_features(DF)
SNAPSHOT = latest_snapshot(DF)
ANALYTICS = analytics(DF)


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    """Dashboard: current market + next-day volatility forecast."""
    forecast = predict_range(BUNDLE, FEATURES_TODAY)
    return render_template(
        "index.html",
        snapshot=SNAPSHOT,
        features=FEATURES_TODAY,
        forecast=forecast,
        holdout=BUNDLE["holdout"],
    )


@app.route("/analysis")
def analysis_page():
    """Honest findings from the notebook."""
    return render_template(
        "analysis.html",
        stats=ANALYTICS,
        holdout=BUNDLE["holdout"],
    )


@app.route("/predict")
def predict_page():
    """Interactive form for what-if scenarios."""
    return render_template("predict.html", features=FEATURES_TODAY)


# ---------------------------------------------------------------- API
@app.route("/api/current")
def api_current():
    forecast = predict_range(BUNDLE, FEATURES_TODAY)
    return jsonify({
        "snapshot": SNAPSHOT,
        "features": FEATURES_TODAY,
        "forecast": forecast,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    # Fall back to current values for any missing inputs
    row = {}
    for f in BUNDLE["features"]:
        try:
            row[f] = float(payload.get(f, FEATURES_TODAY[f]))
        except (TypeError, ValueError):
            return jsonify({"error": f"invalid value for '{f}'"}), 400

    forecast = predict_range(BUNDLE, row)
    return jsonify({"inputs": row, "forecast": forecast})


@app.route("/api/history")
def api_history():
    """Last N days of close + |return| for the chart."""
    n = int(request.args.get("n", 180))
    tail = DF.tail(n)
    return jsonify({
        "dates":  [d.date().isoformat() for d in tail.index],
        "close":  tail["btc_close"].round(2).tolist(),
        "absret": tail["abs_ret1"].round(3).tolist(),
        "vol30":  tail["vol30"].round(3).tolist(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)