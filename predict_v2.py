"""
predict_v2.py
─────────────
Multi-step PM2.5 inference for 1–24h ahead at any Indian station.

Two modes
─────────
1. LIVE: Fetches real-time data (last 72h) from OpenAQ + Open-Meteo + News.
2. STORED: Uses the last SEQ_LEN rows from the training parquet (offline test).

Output
──────
JSON with full 24-hour forecast per station, AQI labels, confidence band,
and active news/event signals.

Usage
─────
  # Live prediction for one station:
  python predict_v2.py --station rk_puram --live

  # Offline prediction (all stations in parquet):
  python predict_v2.py --all

  # Offline prediction for one station:
  python predict_v2.py --station delhi

  # Save output to JSON file:
  python predict_v2.py --station mumbai --live --out forecast.json
"""

import os, sys, json, pickle, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_PARQUET, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH,
    FEAT_PATH, STATION_ID_PATH, STATIONS_JSON,
    FEATURE_COLS, N_HORIZONS, SEQ_LEN,
    AQI_BREAKPOINTS,
)

# Force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── AQI Helper ───────────────────────────────────────────────────────────────

def classify_pm25(value: float) -> tuple:
    """Return (label, color_tag, category_int) for a PM2.5 value (CPCB scale)."""
    for lo, hi, label, cat, tag in AQI_BREAKPOINTS:
        if lo <= value < hi:
            return label, tag, cat
    return "Severe", "[MAROON]", 5


# ─── Single-Station Inference ────────────────────────────────────────────────

def predict_from_window(
    window_df: pd.DataFrame,
    station_name: str,
    station_id: int,
    model,
    scaler_X,
    scaler_y,
    feature_cols: list,
) -> dict:
    """
    Run the model on a pre-assembled feature window.

    Args
    ----
    window_df    : DataFrame with >= SEQ_LEN rows and all feature_cols
    station_name : display name
    station_id   : integer station index (from station_id_map)
    model        : loaded Keras model (inputs: [features, station_id])
    scaler_X     : RobustScaler for features
    scaler_y     : RobustScaler for PM2.5 targets
    feature_cols : ordered feature list

    Returns
    -------
    dict with forecast data (JSON-serialisable)
    """
    if len(window_df) < SEQ_LEN:
        raise ValueError(
            f"Need {SEQ_LEN} rows for station '{station_name}', got {len(window_df)}"
        )

    # ── Build scaled tensor (1, SEQ_LEN, n_features)
    raw   = window_df[feature_cols].values[-SEQ_LEN:].astype(np.float32)
    X_s   = scaler_X.transform(raw)[np.newaxis, ...]              # (1, 72, F)
    sid   = np.array([[station_id]], dtype=np.int32)              # (1, 1)

    # ── Predict
    pred_s    = model.predict([X_s, sid], verbose=0)              # (1, 24)
    pred_s    = pred_s.reshape(1, N_HORIZONS)

    # ── Inverse-transform back to µg/m³
    preds_inv = scaler_y.inverse_transform(
        pred_s.reshape(-1, 1)
    ).flatten()                                                   # (24,)
    preds_inv = np.clip(preds_inv, 0, 1500)

    # ── Current PM2.5
    current   = float(window_df["pm25"].iloc[-1])
    ts_now    = window_df["timestamp"].iloc[-1]
    ts_str    = str(ts_now)

    # ── Build forecast dict
    forecast  = {}
    for h in range(N_HORIZONS):
        val   = float(round(preds_inv[h], 1))
        label, tag, cat = classify_pm25(val)
        delta = round(val - current, 1)
        # Crude confidence: decays from ~90% at +1h to ~60% at +24h
        conf  = round(0.90 - (h / N_HORIZONS) * 0.30, 2)
        forecast[f"+{h+1}h"] = {
            "pm25":       val,
            "aqi":        label,
            "aqi_color":  tag,
            "delta":      delta,
            "confidence": conf,
        }

    # ── Active event signals
    signals = {}
    for col in [
        "news_dust_storm", "news_industrial_event", "news_crop_burning",
        "news_calamity", "news_fireworks", "news_smog_alert",
        "is_harvest_season", "is_diwali_period", "is_holi_period", "is_monsoon",
    ]:
        if col in window_df.columns:
            val_sig = float(window_df[col].iloc[-1])
            if val_sig > 0.1:
                signals[col] = round(val_sig, 2)

    return {
        "station":      station_name,
        "current_pm25": round(current, 1),
        "current_aqi":  classify_pm25(current)[0],
        "data_ts":      ts_str,
        "forecast":     forecast,
        "signals":      signals,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Pretty Printer ──────────────────────────────────────────────────────────

def print_forecast(result: dict) -> None:
    curr = result["current_pm25"]
    _, curr_tag, _ = classify_pm25(curr)

    print("\n" + "─" * 62)
    print(f"  Station : {result['station'].upper()}")
    print(f"  Now     : {curr:>7.1f} µg/m³  {curr_tag}  {result['current_aqi']}")
    print(f"  Data ts : {result['data_ts']}")

    if result["signals"]:
        print(f"  Signals : {', '.join(f'{k}={v:.1f}' for k, v in result['signals'].items())}")

    print("\n  Forecast:")
    print(f"  {'Hour':>6}  {'PM2.5':>8}  {'AQI':>22}  {'Δ':>7}  {'Conf':>6}")
    print("  " + "─" * 56)

    # Print all 24 horizons
    for h in range(1, N_HORIZONS + 1):
        key = f"+{h}h"
        fc  = result["forecast"][key]
        pm  = fc["pm25"]
        aqi = fc["aqi"]
        dlt = fc["delta"]
        cof = fc["confidence"]
        arr = ("↑" if dlt > 0 else ("↓" if dlt < 0 else "→"))
        print(f"  {key:>6}  {pm:>8.1f}  {aqi:>22}  {arr}{abs(dlt):>5.1f}  {cof:>5.0%}")

    print("─" * 62)


# ─── Load Artifacts ───────────────────────────────────────────────────────────

def load_artifacts():
    """Load model, scalers, feature list, and station ID map."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun: python train_model.py"
        )

    print("Loading model artifacts …")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(SCALER_X_PATH, "rb") as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_PATH, "rb") as f:
        scaler_y = pickle.load(f)
    with open(FEAT_PATH) as f:
        feature_cols = json.load(f)
    with open(STATION_ID_PATH) as f:
        station_id_map = json.load(f)

    print(f"  Features  : {len(feature_cols)}")
    print(f"  Stations  : {len(station_id_map)}")
    print(f"  Seq len   : {SEQ_LEN}h  |  Horizons: {N_HORIZONS}h")
    return model, scaler_X, scaler_y, feature_cols, station_id_map


# ─── Offline Inference (from stored parquet) ─────────────────────────────────

def predict_from_parquet(station_query: str, model, scaler_X, scaler_y,
                         feature_cols, station_id_map) -> list:
    """
    Run offline inference using the last SEQ_LEN rows from the training parquet.
    """
    df = pd.read_parquet(DATASET_PARQUET)
    df = df.sort_values(["station", "timestamp"])

    # Filter
    q = station_query.lower()
    matching = [s for s in df["station"].unique() if q in s.lower()]
    if not matching:
        print(f"  [SKIP] No station matching '{station_query}' in parquet")
        return []

    results = []
    for sname in matching:
        sdf = df[df["station"] == sname].tail(SEQ_LEN + 5).copy()
        if len(sdf) < SEQ_LEN:
            print(f"  [SKIP] {sname}: only {len(sdf)} rows")
            continue

        sid = station_id_map.get(sname, 0)
        try:
            res = predict_from_window(sdf, sname, sid, model,
                                      scaler_X, scaler_y, feature_cols)
            results.append(res)
        except Exception as e:
            print(f"  [ERROR] {sname}: {e}")

    return results


# ─── Live Inference ───────────────────────────────────────────────────────────

def predict_live(station_query: str, model, scaler_X, scaler_y,
                 feature_cols, station_id_map) -> list:
    """
    Run live inference: fetch real-time window from OpenAQ + weather + news.
    """
    from realtime_fetcher import load_station_registry, find_station, fetch_window_for_station

    registry = load_station_registry()
    matches  = find_station(station_query, registry)

    if not matches:
        print(f"  [SKIP] No station matching '{station_query}' in registry")
        return []

    results = []
    for stn in matches[:5]:   # max 5 stations per query
        try:
            window = fetch_window_for_station(stn, hours=SEQ_LEN)
            if window.empty:
                print(f"  [SKIP] {stn['station']}: empty window")
                continue

            sid = station_id_map.get(stn["station"], 0)
            res = predict_from_window(window, stn["station"], sid, model,
                                      scaler_X, scaler_y, feature_cols)
            results.append(res)
        except Exception as e:
            print(f"  [ERROR] {stn['station']}: {e}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PM2.5 India — 1-24h Forecast")
    parser.add_argument("--station", type=str, default=None,
                        help="Station name or city to predict")
    parser.add_argument("--all",     action="store_true",
                        help="Predict all stations in the parquet (offline)")
    parser.add_argument("--live",    action="store_true",
                        help="Fetch real-time data (requires network)")
    parser.add_argument("--out",     type=str, default=None,
                        help="Save JSON output to this file")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress detailed table; print JSON only")
    args = parser.parse_args()

    # Load model artifacts
    model, scaler_X, scaler_y, feature_cols, station_id_map = load_artifacts()

    all_results = []

    if args.live and args.station:
        print(f"\nLive inference: {args.station}")
        all_results = predict_live(args.station, model, scaler_X, scaler_y,
                                   feature_cols, station_id_map)

    elif args.all:
        df = pd.read_parquet(DATASET_PARQUET)
        stations_in_df = sorted(df["station"].unique())
        print(f"\nOffline inference: {len(stations_in_df)} stations")
        for sname in stations_in_df:
            res = predict_from_parquet(sname, model, scaler_X, scaler_y,
                                       feature_cols, station_id_map)
            all_results.extend(res)

    elif args.station:
        print(f"\nOffline inference: {args.station}")
        all_results = predict_from_parquet(args.station, model, scaler_X, scaler_y,
                                           feature_cols, station_id_map)
    else:
        parser.print_help()
        return

    # ── Print / save
    if not all_results:
        print("No results produced.")
        return

    if not args.quiet:
        for res in all_results:
            print_forecast(res)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved → {args.out}")
    else:
        print("\nFull JSON:")
        print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
