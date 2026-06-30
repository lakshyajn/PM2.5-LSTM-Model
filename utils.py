"""
utils.py
────────
Shared feature-engineering utilities used by data_pipeline.py, 
realtime_fetcher.py, and train_model.py.
"""

import sys
import numpy as np
import pandas as pd
from datetime import timedelta

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    AQI_BREAKPOINTS, RUSH_HOURS, INDIAN_HOLIDAYS,
    DIWALI_DATES, HOLI_DATES,
)


# ── AQI ───────────────────────────────────────────────────────────────────────

def classify_pm25_cpcb(value: float):
    """Return (label, category_int, color_tag) per CPCB PM2.5 breakpoints."""
    for lo, hi, label, cat_int, tag in AQI_BREAKPOINTS:
        if lo <= value < hi:
            return label, cat_int, tag
    return "Severe", 5, "[MAROON]"


def pm25_to_aqi_category(series: pd.Series) -> pd.Series:
    """Vectorised CPCB category integer (0–5) for a pm25 series."""
    cats = np.zeros(len(series), dtype=np.float32)
    vals = series.values
    for lo, hi, _, cat, _ in AQI_BREAKPOINTS:
        cats[(vals >= lo) & (vals < hi)] = cat
    return pd.Series(cats, index=series.index)


# ── Wind ──────────────────────────────────────────────────────────────────────

def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wind_speed_10m + wind_direction_10m → wind_u, wind_v, wind_speed.
    Handles missing direction gracefully (sets u/v to 0).
    """
    speed = df.get("wind_speed_10m", pd.Series(0.0, index=df.index))
    direc = df.get("wind_direction_10m", pd.Series(0.0, index=df.index)).fillna(0.0)

    theta = np.radians(direc)
    df["wind_speed"] = speed.fillna(0.0)
    df["wind_u"]     = df["wind_speed"] * np.cos(theta)
    df["wind_v"]     = df["wind_speed"] * np.sin(theta)
    return df


# ── Dew Point ─────────────────────────────────────────────────────────────────

def add_dew_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dew_point_2m from temperature_2m and relative_humidity_2m
    if not already present (Magnus formula approximation).
    """
    if "dew_point_2m" in df.columns:
        return df
    T  = df.get("temperature_2m", pd.Series(25.0, index=df.index))
    RH = df.get("relative_humidity_2m", pd.Series(60.0, index=df.index))
    # Magnus formula
    alpha = (17.27 * T) / (237.3 + T) + np.log(RH / 100.0 + 1e-9)
    df["dew_point_2m"] = (237.3 * alpha) / (17.27 - alpha)
    return df


# ── Traffic / Calendar ────────────────────────────────────────────────────────

def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute traffic proxy features from timestamps (IST-aware).
    Adds: rush_hour, is_weekday, is_holiday, traffic_index,
          day_of_week_sin, day_of_week_cos.
    """
    ist = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    hour    = ist.dt.hour
    weekday = ist.dt.weekday          # 0=Mon, 6=Sun
    date_str = ist.dt.strftime("%Y-%m-%d")

    df["rush_hour"]   = hour.isin(RUSH_HOURS).astype(np.int8)
    df["is_weekday"]  = (weekday < 5).astype(np.int8)
    df["is_holiday"]  = date_str.isin(INDIAN_HOLIDAYS).astype(np.int8)
    df["traffic_index"] = (
        df["rush_hour"] * df["is_weekday"] * (1 - df["is_holiday"])
    ).astype(np.float32)

    df["day_of_week_sin"] = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
    df["day_of_week_cos"] = np.cos(2 * np.pi * weekday / 7).astype(np.float32)

    # Store hour for time features
    df["_hour"]    = hour.values
    df["_weekday"] = weekday.values
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings for hour and month."""
    if "_hour" not in df.columns:
        ist = df["timestamp"].dt.tz_convert("Asia/Kolkata")
        df["_hour"] = ist.dt.hour

    hour  = df["_hour"]
    month = df["timestamp"].dt.month

    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24).astype(np.float32)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    return df


# ── Seasonal / Event Flags ────────────────────────────────────────────────────

def add_seasonal_flags(df: pd.DataFrame, lat: float = 28.0) -> pd.DataFrame:
    """
    Add season and known-event binary flags computed from the calendar.

    Flags:
        is_monsoon        : June–September
        is_winter         : December–February (north India inversion season)
        is_harvest_season : October–November (stubble burning, north India)
        is_diwali_period  : ±3 days of Diwali date
        is_holi_period    : ±1 day of Holi date
    """
    ist = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    month = ist.dt.month

    df["is_monsoon"]       = month.isin([6, 7, 8, 9]).astype(np.int8)
    df["is_winter"]        = month.isin([12, 1, 2]).astype(np.int8)
    # harvest season more relevant for north India (lat > 25)
    if lat > 25:
        df["is_harvest_season"] = month.isin([10, 11]).astype(np.int8)
    else:
        df["is_harvest_season"] = (month.isin([1, 2]) | month.isin([10, 11])).astype(np.int8)

    # Diwali period
    diwali_flag = pd.Series(0, index=df.index, dtype=np.int8)
    for year, date_str in DIWALI_DATES.items():
        center = pd.Timestamp(date_str).tz_localize("Asia/Kolkata")
        mask = (ist >= center - pd.Timedelta(days=3)) & (ist <= center + pd.Timedelta(days=2))
        diwali_flag |= mask.astype(np.int8)
    df["is_diwali_period"] = diwali_flag

    # Holi period
    holi_flag = pd.Series(0, index=df.index, dtype=np.int8)
    for year, date_str in HOLI_DATES.items():
        center = pd.Timestamp(date_str).tz_localize("Asia/Kolkata")
        mask = (ist >= center - pd.Timedelta(days=1)) & (ist <= center + pd.Timedelta(days=1))
        holi_flag |= mask.astype(np.int8)
    df["is_holi_period"] = holi_flag

    return df


# ── Lag / Rolling PM2.5 Features ─────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Autoregressive lag and rolling features for pm25.
    Must be called on a SINGLE station's sorted DataFrame.
    """
    pm = df["pm25"]

    for lag in [1, 3, 6, 12, 24, 48, 72]:
        df[f"pm25_lag{lag}"] = pm.shift(lag)

    # 24-hour rolling stats (shifted by 1 to avoid look-ahead)
    s1 = pm.shift(1)
    df["pm25_rmean24"] = s1.rolling(24, min_periods=3).mean()
    df["pm25_rmax24"]  = s1.rolling(24, min_periods=3).max()
    df["pm25_rmin24"]  = s1.rolling(24, min_periods=3).min()
    df["pm25_rstd24"]  = s1.rolling(24, min_periods=3).std().fillna(0.0)

    # 6-hour linear trend (slope via polyfit proxy)
    df["pm25_trend6h"] = (
        s1.rolling(6, min_periods=2).mean() - s1.rolling(12, min_periods=3).mean()
    ).fillna(0.0)

    return df


# ── Co-pollutant Defaults ─────────────────────────────────────────────────────

def ensure_copollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill co-pollutant columns if not present."""
    for col in ["pm10", "no2", "so2", "o3", "co"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ── News Defaults ─────────────────────────────────────────────────────────────

def ensure_news_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill news signal columns if not present (historical data)."""
    for col in [
        "news_dust_storm", "news_industrial_event",
        "news_crop_burning", "news_calamity",
        "news_fireworks", "news_smog_alert",
    ]:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ── Fire Helpers ──────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorised haversine distance (km) from (lat1, lon1) to arrays."""
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2_arr))
        * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Full Feature Pipeline (single station) ────────────────────────────────────

def build_features(
    pm_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fire_hourly: pd.DataFrame,
    station_name: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Merge PM2.5, weather and fire data for ONE station and compute all features.

    Args
    ----
    pm_df        : DataFrame with [timestamp, pm25]
    weather_df   : DataFrame with [timestamp, temperature_2m, ...]
    fire_hourly  : DataFrame with [timestamp, fire_count, frp_sum, frp_mean,
                                   frp_max, bright_ti4_mean, bright_ti5_mean]
                  (pre-filtered to this station's radius)
    station_name : station identifier string
    lat, lon     : station coordinates

    Returns
    -------
    DataFrame with all feature columns + target columns
    """
    sdf = pm_df.copy()

    # ── Weather merge
    if not weather_df.empty:
        sdf = sdf.merge(weather_df, on="timestamp", how="left")

    # ── Wind components
    sdf = add_wind_components(sdf)
    sdf = add_dew_point(sdf)

    # ── Fire merge
    fire_cols = ["fire_count", "frp_sum", "frp_mean", "frp_max",
                 "bright_ti4_mean", "bright_ti5_mean"]
    if not fire_hourly.empty:
        sdf = sdf.merge(fire_hourly, on="timestamp", how="left")
    for c in fire_cols:
        if c not in sdf.columns:
            sdf[c] = 0.0
        sdf[c] = sdf[c].fillna(0.0)

    # ── Calendar features
    sdf = add_traffic_features(sdf)
    sdf = add_time_features(sdf)
    sdf = add_seasonal_flags(sdf, lat=lat)

    # ── Station identity
    sdf["station"] = station_name
    sdf["lat"]     = np.float32(lat)
    sdf["lon"]     = np.float32(lon)

    # ── AQI category
    sdf["aqi_category"] = pm25_to_aqi_category(sdf["pm25"])

    # ── Sort before lags
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)
    sdf = add_lag_features(sdf)

    # ── Co-pollutants and news defaults
    sdf = ensure_copollutants(sdf)
    sdf = ensure_news_cols(sdf)

    # ── Boundary layer height default
    if "boundary_layer_height" not in sdf.columns:
        sdf["boundary_layer_height"] = 0.0

    # ── 24-step targets: future PM2.5 values
    for h in range(1, 25):
        sdf[f"target_{h}h"] = sdf["pm25"].shift(-h)

    # ── Drop private helper columns
    sdf.drop(columns=["_hour", "_weekday", "wind_speed_10m",
                       "wind_direction_10m"], errors="ignore", inplace=True)

    return sdf


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("utils.py: OK")
    print(f"  Feature columns: {len(__import__('config').FEATURE_COLS)}")
