"""
historical_s3.py
────────────────
Fetches historical PM2.5 data from the OpenAQ S3 archive for Indian stations.
The S3 archive is publicly accessible (no auth needed) and contains data 
going back to 2016, organized by location_id / year / month.

Usage
─────
  # Fetch historical data for a single location
  python historical_s3.py --location-id 17 --start 2022-01-01 --end 2025-01-01

  # Fetch for ALL stations in the registry (fills gaps before OpenAQ v3 data)
  python historical_s3.py --all --start 2022-01-01 --end 2025-02-01

  # Merge S3 historical + OpenAQ v3 recent data into one parquet
  python historical_s3.py --merge
"""

import os, sys, io, gzip, json, time, argparse
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

from config import (
    DATA_DIR, STATIONS_JSON, DATASET_PARQUET,
    TRAIN_START, TRAIN_END,
)

S3_BASE = "https://openaq-data-archive.s3.amazonaws.com"
S3_PATH = "records/csv.gz/locationid={loc_id}/year={year}/month={month:02d}/location-{loc_id}-{year}{month:02d}{day:02d}.csv.gz"
HIST_PARQUET = os.path.join(DATA_DIR, "historical_s3.parquet")
COMBINED_PARQUET = os.path.join(DATA_DIR, "combined_pm25.parquet")


def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _fetch_day(loc_id: int, year: int, month: int, day: int) -> pd.DataFrame:
    """Download and parse one day's CSV.GZ from the S3 archive."""
    path = S3_PATH.format(loc_id=loc_id, year=year, month=month, day=day)
    url  = f"{S3_BASE}/{path}"

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 404:
            return pd.DataFrame()
        if r.status_code != 200:
            return pd.DataFrame()

        with gzip.open(io.BytesIO(r.content)) as f:
            df = pd.read_csv(f, low_memory=False)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_location_s3(loc_id: int, start_date: str, end_date: str,
                      param_filter="pm25") -> pd.DataFrame:
    """
    Fetch all daily CSV.GZ files for a location from S3 between start and end date.
    Returns hourly-resampled DataFrame [timestamp, pm25].
    """
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    date_range = pd.date_range(start_dt, end_dt, freq="D")
    all_dfs = []

    for dt in date_range:
        df_day = _fetch_day(loc_id, dt.year, dt.month, dt.day)
        if df_day.empty:
            continue

        # Filter to pm25 parameter
        param_col = None
        for c in ["parameter", "parameterId", "Parameter"]:
            if c in df_day.columns:
                param_col = c
                break

        if param_col:
            mask = df_day[param_col].astype(str).str.lower().str.contains("pm25|pm2.5")
            df_day = df_day[mask]

        if df_day.empty:
            continue

        # Find value column
        val_col = None
        for c in ["value", "Value", "average"]:
            if c in df_day.columns:
                val_col = c
                break
        if val_col is None:
            continue

        # Find datetime column
        ts_col = None
        for c in ["date", "datetime", "Date", "utc", "date_utc", "dateTimeFrom"]:
            if c in df_day.columns:
                ts_col = c
                break
        if ts_col is None:
            continue

        df_day = df_day[[ts_col, val_col]].copy()
        df_day.columns = ["timestamp", "pm25"]
        df_day["pm25"] = pd.to_numeric(df_day["pm25"], errors="coerce")
        df_day = df_day.dropna()
        df_day["pm25"] = df_day["pm25"].clip(0, 1500)
        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"], utc=True, errors="coerce")
        df_day = df_day.dropna(subset=["timestamp"])
        all_dfs.append(df_day)

    if not all_dfs:
        return pd.DataFrame(columns=["timestamp", "pm25"])

    df = pd.concat(all_dfs, ignore_index=True)
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.groupby("timestamp", as_index=False)["pm25"].mean()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _worker_s3(station_info, start_date, end_date):
    """Worker for parallel S3 fetch."""
    loc_id = station_info.get("location_id")
    name   = station_info["station"]
    if loc_id is None:
        return name, pd.DataFrame(columns=["timestamp", "pm25"])
    df = fetch_location_s3(loc_id, start_date, end_date)
    return name, df


def fetch_all_s3(stations, start_date, end_date, max_workers=4):
    """Parallel S3 fetch for all stations."""
    pm25_dict = {}
    total = len(stations)
    _print(f"  Fetching S3 historical data: {total} stations, {max_workers} workers ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_s3, s, start_date, end_date): s
            for s in stations
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                name, df = future.result()
                pm25_dict[name] = df
                status = f"{len(df):>6,} hrs" if not df.empty else "  NO DATA"
                _print(f"  [{i:>3}/{total}] {name:<45} {status}")
            except Exception as e:
                stn = futures[future]
                pm25_dict[stn["station"]] = pd.DataFrame(columns=["timestamp", "pm25"])
                _print(f"  [{i:>3}/{total}] {stn['station']:<45} ERROR: {e}")

    return pm25_dict


def build_historical_parquet(start_date=None, end_date=None, city_filter=None):
    """
    Fetch historical S3 data for all/filtered stations and save to parquet.
    The date range should cover ONLY the historical period not covered by v3 API
    (typically before Feb 2025).
    """
    start_date = start_date or "2022-01-01"
    end_date   = end_date   or "2025-02-01"

    _print("=" * 60)
    _print(f"  OpenAQ S3 Historical Fetch: {start_date} -> {end_date}")
    _print("=" * 60)

    with open(STATIONS_JSON, encoding="utf-8") as f:
        stations = json.load(f)

    if city_filter:
        city_set = {c.lower() for c in city_filter}
        stations = [s for s in stations if s.get("city", "").lower() in city_set]
        _print(f"  City filter: {city_filter} -> {len(stations)} stations")

    # Only stations with a location_id
    stations = [s for s in stations if s.get("location_id") is not None]
    _print(f"  Stations with location_id: {len(stations)}")

    pm25_dict = fetch_all_s3(stations, start_date, end_date, max_workers=6)

    # Build station DataFrames (just timestamp + pm25 + station + city + lat/lon)
    dfs = []
    for stn in stations:
        name = stn["station"]
        df   = pm25_dict.get(name)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["station"] = name
        df["city"]    = stn.get("city", "Unknown")
        df["lat"]     = float(stn["lat"])
        df["lon"]     = float(stn["lon"])
        df["location_id"] = stn.get("location_id")
        dfs.append(df)

    if not dfs:
        _print("[WARN] No S3 data returned. The archive may not have Indian data before Feb 2025.")
        _print("       Try individual location URLs manually to verify:")
        sample_id = stations[0]["location_id"] if stations else 17
        _print(f"       {S3_BASE}/records/csv.gz/locationid={sample_id}/year=2023/month=01/")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["station", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_parquet(HIST_PARQUET, index=False, compression="snappy")
    _print(f"\n  S3 historical data: {len(combined):,} rows, {combined['station'].nunique()} stations")
    _print(f"  Saved -> {HIST_PARQUET}")
    return combined


def merge_with_v3(v3_parquet=DATASET_PARQUET, hist_parquet=HIST_PARQUET,
                  out_parquet=COMBINED_PARQUET):
    """
    Merge historical S3 PM2.5 with the v3 API dataset.
    Deduplicates by (station, timestamp), keeping v3 values for overlapping hours.
    """
    _print("Merging historical S3 + OpenAQ v3 datasets ...")

    if not os.path.exists(v3_parquet):
        _print(f"  [SKIP] v3 parquet not found: {v3_parquet}")
        return

    df_v3   = pd.read_parquet(v3_parquet)
    df_hist = pd.read_parquet(hist_parquet) if os.path.exists(hist_parquet) else pd.DataFrame()

    if df_hist.empty:
        _print("  [SKIP] No historical data to merge.")
        return

    # Align columns: hist only has timestamp/pm25/station/city/lat/lon
    common_cols = [c for c in df_hist.columns if c in df_v3.columns]

    # Mark source
    df_v3["_src"]   = "v3"
    df_hist["_src"] = "s3"

    # Stack, deduplicate — keep v3 where overlap
    combined = pd.concat([df_v3[common_cols + ["_src"]],
                          df_hist[common_cols + ["_src"]]], ignore_index=True)
    combined.sort_values(["station", "timestamp", "_src"], ascending=[True, True, True], inplace=True)
    combined.drop_duplicates(subset=["station", "timestamp"], keep="first", inplace=True)
    combined.drop(columns=["_src"], inplace=True)
    combined.sort_values(["station", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_parquet(out_parquet, index=False, compression="snappy")
    _print(f"  Combined: {len(combined):,} rows, {combined['station'].nunique()} stations")
    _print(f"  Saved -> {out_parquet}")
    return combined


def check_s3_availability(sample_loc_ids=None):
    """
    Quick test: check if S3 has data for a few Indian location IDs.
    """
    if sample_loc_ids is None:
        # Try a few known Indian location IDs
        sample_loc_ids = [17, 5586, 5639, 7044, 2597, 11607, 3409496, 6148686]

    _print("Testing S3 archive availability for Indian locations:")
    _print(f"{'LocID':>8}  {'2022':>8}  {'2023':>8}  {'2024':>8}  {'2025-01':>10}")
    _print("-" * 50)

    for loc_id in sample_loc_ids:
        row = [f"{loc_id:>8}"]
        for year, month, day in [(2022,6,1), (2023,6,1), (2024,6,1), (2025,1,15)]:
            df = _fetch_day(loc_id, year, month, day)
            status = f"{len(df):>4} rows" if not df.empty else "  empty"
            row.append(status)
            time.sleep(0.2)
        _print("  ".join(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAQ S3 Historical Fetcher")
    parser.add_argument("--check",       action="store_true",
                        help="Test S3 availability for sample Indian locations")
    parser.add_argument("--all",         action="store_true",
                        help="Fetch historical S3 data for all stations")
    parser.add_argument("--merge",       action="store_true",
                        help="Merge S3 historical with v3 API dataset")
    parser.add_argument("--cities",      nargs="+", default=None)
    parser.add_argument("--start",       default="2022-01-01")
    parser.add_argument("--end",         default="2025-02-01")
    args = parser.parse_args()

    if args.check:
        check_s3_availability()

    if args.all:
        build_historical_parquet(args.start, args.end, city_filter=args.cities)

    if args.merge:
        merge_with_v3()
