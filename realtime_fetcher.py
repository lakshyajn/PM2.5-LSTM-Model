"""
realtime_fetcher.py
───────────────────
Assembles a real-time 72-hour data window for any station, ready to feed
directly into the trained model for inference.

Pipeline per station
────────────────────
1. Fetch last 72h of hourly PM2.5 from OpenAQ v3 (live measurements).
2. Fetch last 72h + next 24h of weather from Open-Meteo forecast API.
3. Fetch recent fire events from FIRMS (last 10 days, India bbox).
4. Fetch live news signals for the station's city (NewsAPI, last 48h).
5. Run the same feature engineering as training.
6. Return a 72-row DataFrame ready for model.predict().

Usage
─────
  from realtime_fetcher import fetch_window_for_station, load_station_registry

  stations = load_station_registry()
  stn = next(s for s in stations if "rk_puram" in s["station"])
  window_df = fetch_window_for_station(stn)
  # window_df has 72 rows with all FEATURE_COLS filled
"""

import os, sys, io, json, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OPENAQ_BASE, OPENAQ_HEADERS,
    OPEN_METEO_FORECAST,
    FIRMS_BASE, FIRMS_SOURCE,
    FIRMS_API_KEY, INDIA_BBOX,
    STATIONS_JSON, FEATURE_COLS,
    SEQ_LEN,
)
from utils import build_features, haversine_km
from news_fetcher import fetch_realtime_news, build_calendar_news_signals

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(url, params=None, headers=None, timeout=30):
    """Simple GET with basic error handling."""
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


# ─── 1. Real-Time PM2.5 ───────────────────────────────────────────────────────

def fetch_realtime_pm25(sensor_id: int, hours: int = 72) -> pd.DataFrame:
    """
    Fetch last `hours` of hourly PM2.5 from OpenAQ v3 for a sensor.
    Returns DataFrame [timestamp(UTC), pm25].
    """
    to_dt   = datetime.utcnow()
    from_dt = to_dt - timedelta(hours=hours + 2)   # small buffer

    all_results = []
    page = 1

    while True:
        try:
            r = _get(
                f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                headers=OPENAQ_HEADERS,
                params={
                    "datetime_from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "datetime_to":   to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 1000, "page": page,
                },
            )
        except Exception as e:
            print(f"  [WARN] PM2.5 fetch error page {page}: {e}")
            break

        results = r.json().get("results", [])
        if not results:
            break
        all_results.extend(results)
        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.3)

    if not all_results:
        return pd.DataFrame(columns=["timestamp", "pm25"])

    rows = []
    for item in all_results:
        period = item.get("period", {})
        ts = (period.get("datetimeFrom", {}).get("utc") or
              period.get("dateFrom",     {}).get("utc"))
        val = item.get("value")
        if ts and val is not None:
            rows.append({"timestamp": ts, "pm25": float(val)})

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    df = df.groupby("timestamp", as_index=False)["pm25"].mean()
    df = df[df["pm25"].between(0, 1500)]
    return df.sort_values("timestamp").reset_index(drop=True)


# ─── 2. Real-Time Weather ─────────────────────────────────────────────────────

def fetch_realtime_weather(lat: float, lon: float, past_hours: int = 80) -> pd.DataFrame:
    """
    Fetch recent + forecast weather from Open-Meteo forecast API.
    Uses `past_days` parameter to get historical weather.
    Returns DataFrame [timestamp(UTC), temperature_2m, ...].
    """
    past_days = max(3, int(np.ceil(past_hours / 24)) + 1)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "surface_pressure", "precipitation",
            "wind_speed_10m", "wind_direction_10m",
            "cloud_cover", "shortwave_radiation",
            "boundary_layer_height",
        ]),
        "past_days":    past_days,
        "forecast_days": 2,    # next 48h for future weather at inference
        "timezone": "UTC",
    }

    try:
        r = _get(OPEN_METEO_FORECAST, params=params, timeout=60)
        data = r.json().get("hourly", {})
    except Exception as e:
        print(f"  [WARN] Weather fetch failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "time" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)

    if "boundary_layer_height" not in df.columns:
        df["boundary_layer_height"] = 0.0

    return df.sort_values("timestamp").reset_index(drop=True)


# ─── 3. Real-Time Fire Data ───────────────────────────────────────────────────

def fetch_realtime_fires(days: int = 10) -> pd.DataFrame:
    """
    Fetch recent FIRMS fire detections for all India.
    Returns raw fire DataFrame [timestamp, latitude, longitude, frp, ...].
    """
    date_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"{FIRMS_BASE}/{FIRMS_API_KEY}/{FIRMS_SOURCE}/{INDIA_BBOX}/{days}/{date_str}"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and "latitude" in r.text:
            fire_df = pd.read_csv(io.StringIO(r.text))
            if not fire_df.empty:
                fire_df["acq_time"] = fire_df["acq_time"].astype(str).str.zfill(4)
                fire_df["timestamp"] = pd.to_datetime(
                    fire_df["acq_date"] + " " + fire_df["acq_time"],
                    format="%Y-%m-%d %H%M", utc=True,
                ).dt.floor("h")
                return fire_df
    except Exception as e:
        print(f"  [WARN] FIRMS real-time fetch failed: {e}")

    return pd.DataFrame(columns=["timestamp", "latitude", "longitude", "frp"])


# ─── 4. Aggregate Fires for Station ──────────────────────────────────────────

def aggregate_fires_for_station_rt(fire_df: pd.DataFrame, lat: float, lon: float,
                                   radius_km: float = 300.0) -> pd.DataFrame:
    """
    Filter and aggregate fire events within radius_km of (lat, lon).
    Returns hourly [timestamp, fire_count, frp_sum, frp_mean, frp_max,
                    bright_ti4_mean, bright_ti5_mean].
    """
    if fire_df.empty:
        return pd.DataFrame(columns=["timestamp", "fire_count", "frp_sum",
                                     "frp_mean", "frp_max",
                                     "bright_ti4_mean", "bright_ti5_mean"])

    dist    = haversine_km(lat, lon, fire_df["latitude"].values, fire_df["longitude"].values)
    nearby  = fire_df[dist <= radius_km].copy()

    if nearby.empty:
        return pd.DataFrame(columns=["timestamp", "fire_count", "frp_sum",
                                     "frp_mean", "frp_max",
                                     "bright_ti4_mean", "bright_ti5_mean"])

    agg = {
        "fire_count": ("frp", "count"),
        "frp_sum":    ("frp", "sum"),
        "frp_mean":   ("frp", "mean"),
        "frp_max":    ("frp", "max"),
    }
    if "bright_ti4" in nearby.columns:
        agg["bright_ti4_mean"] = ("bright_ti4", "mean")
    if "bright_ti5" in nearby.columns:
        agg["bright_ti5_mean"] = ("bright_ti5", "mean")

    hourly = nearby.groupby("timestamp").agg(**agg).reset_index()
    for c in ["bright_ti4_mean", "bright_ti5_mean"]:
        if c not in hourly.columns:
            hourly[c] = 0.0
    return hourly


# ─── 5. Assemble Full Window ──────────────────────────────────────────────────

def fetch_window_for_station(
    station_info: dict,
    hours: int = SEQ_LEN,
    use_live_news: bool = True,
) -> pd.DataFrame:
    """
    Fetch and assemble a `hours`-row window of feature data for one station.

    Args
    ----
    station_info  : dict with keys: station, sensor_id, lat, lon, city
    hours         : number of past hours needed (default SEQ_LEN = 72)
    use_live_news : if True, fetch news from NewsAPI; else use calendar proxy

    Returns
    -------
    DataFrame with exactly `hours` rows and all FEATURE_COLS populated.
    Returns empty DataFrame on failure.
    """
    name       = station_info["station"]
    lat        = station_info["lat"]
    lon        = station_info["lon"]
    city       = station_info.get("city", "Unknown")
    sensor_id  = station_info.get("sensor_id")

    print(f"\n[RT] Assembling window for: {name} ({city})")

    # ── PM2.5
    if sensor_id:
        print(f"  Fetching PM2.5 (sensor {sensor_id}) …")
        pm_df = fetch_realtime_pm25(sensor_id, hours=hours + 10)
    else:
        print(f"  [WARN] No sensor_id for {name}, PM2.5 unavailable")
        pm_df = pd.DataFrame(columns=["timestamp", "pm25"])

    if len(pm_df) < 12:
        print(f"  [SKIP] Insufficient PM2.5 data ({len(pm_df)} rows)")
        return pd.DataFrame()

    # ── Weather
    print("  Fetching weather …")
    weather_df = fetch_realtime_weather(lat, lon, past_hours=hours + 10)

    # ── Fire
    print("  Fetching fire data …")
    fire_raw    = fetch_realtime_fires(days=10)
    fire_hourly = aggregate_fires_for_station_rt(fire_raw, lat, lon)

    # ── News
    if use_live_news:
        print(f"  Fetching live news for {city} …")
        try:
            news_df = fetch_realtime_news(city, hours=hours + 10)
        except Exception:
            news_df = pd.DataFrame()
    else:
        news_df = pd.DataFrame()

    # ── Feature engineering
    sdf = build_features(pm_df, weather_df, fire_hourly, name, lat, lon)

    # Merge news signals
    news_cols = [
        "news_dust_storm", "news_industrial_event", "news_crop_burning",
        "news_calamity", "news_fireworks", "news_smog_alert",
    ]
    if not news_df.empty and "timestamp" in news_df.columns:
        for col in news_cols:
            if col not in news_df.columns:
                news_df[col] = 0.0
        sdf = sdf.merge(news_df[["timestamp"] + news_cols], on="timestamp", how="left")
        for col in news_cols:
            # Merge creates _x/_y suffix if column already existed
            if f"{col}_y" in sdf.columns:
                sdf[col] = sdf[f"{col}_y"].fillna(sdf.get(f"{col}_x", 0.0))
                sdf.drop(columns=[f"{col}_x", f"{col}_y"], errors="ignore", inplace=True)
            sdf[col] = sdf[col].fillna(0.0)
    else:
        # Calendar-based fallback
        cal = build_calendar_news_signals(sdf["timestamp"], lat=lat)
        for col in news_cols:
            sdf[col] = cal[col].values if col in cal.columns else 0.0

    # ── Take last `hours` rows
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)
    sdf = sdf.tail(hours).reset_index(drop=True)

    # ── Ensure all feature columns are present
    for col in FEATURE_COLS:
        if col not in sdf.columns:
            sdf[col] = 0.0

    # ── Forward/backward fill any remaining NaN
    num_cols = sdf.select_dtypes(include=[np.number]).columns.tolist()
    sdf[num_cols] = sdf[num_cols].interpolate(
        method="linear", limit_direction="both"
    ).fillna(0.0)

    print(f"  Window ready: {len(sdf)} rows × {len(FEATURE_COLS)} features")
    return sdf


# ─── Station Registry ─────────────────────────────────────────────────────────

def load_station_registry() -> list:
    """Load the station registry saved by data_pipeline.py."""
    if not os.path.exists(STATIONS_JSON):
        raise FileNotFoundError(
            f"Station registry not found: {STATIONS_JSON}\n"
            "Run: python data_pipeline.py --discover"
        )
    with open(STATIONS_JSON) as f:
        return json.load(f)


def find_station(name_or_city: str, registry: list) -> list:
    """
    Fuzzy-search the registry by station name or city name.
    Returns list of matching station dicts.
    """
    q = name_or_city.lower()
    return [
        s for s in registry
        if q in s["station"].lower() or q in s.get("city", "").lower()
    ]


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time PM2.5 Data Fetcher")
    parser.add_argument("--station", default="rk_puram",
                        help="Station name or city to search in registry")
    parser.add_argument("--hours", type=int, default=SEQ_LEN,
                        help=f"Hours of lookback (default {SEQ_LEN})")
    parser.add_argument("--no-news", action="store_true",
                        help="Use calendar news proxy instead of live NewsAPI")
    args = parser.parse_args()

    registry = load_station_registry()
    matches  = find_station(args.station, registry)

    if not matches:
        print(f"No station found matching '{args.station}'")
        sys.exit(1)

    stn = matches[0]
    print(f"Station: {stn['station']}  ({stn.get('city', '?')})  "
          f"lat={stn['lat']:.3f}  lon={stn['lon']:.3f}")

    window = fetch_window_for_station(stn, hours=args.hours,
                                      use_live_news=not args.no_news)

    if window.empty:
        print("Could not assemble window.")
    else:
        print(f"\nWindow shape : {window.shape}")
        print(f"Latest PM2.5 : {window['pm25'].iloc[-1]:.1f} µg/m³")
        print(f"Timestamps   : {window['timestamp'].iloc[0]}  →  {window['timestamp'].iloc[-1]}")
        print("\nTail of feature window:")
        print(window[["timestamp", "pm25", "temperature_2m", "wind_speed",
                       "news_crop_burning", "news_fireworks"]].tail(5).to_string(index=False))
