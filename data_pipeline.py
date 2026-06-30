"""
data_pipeline.py
────────────────
Nationwide India PM2.5 dataset builder.

Steps
─────
1. Discover all operational CPCB PM2.5 stations in India via OpenAQ v3.
2. Fetch historical PM2.5 hourly data per station (2022 → today).
3. Fetch ERA5 weather data per unique city via Open-Meteo archive.
4. Fetch NASA FIRMS fire data for all-India bounding box (chunked).
5. Associate fires with each station (within FIRE_RADIUS_KM).
6. Run feature engineering pipeline (utils.build_features).
7. Save long-format Parquet dataset.

Usage
─────
  # Auto-discover stations and save registry:
  python data_pipeline.py --discover

  # Build full dataset (uses saved registry):
  python data_pipeline.py --build

  # Build for specific cities only (testing):
  python data_pipeline.py --build --cities Delhi Mumbai Kolkata

  # Full pipeline in one go:
  python data_pipeline.py --discover --build
"""

import os, sys, io, json, time, argparse, math
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered stdout for background task visibility
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import build_features, haversine_km

def _print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(url, params=None, headers=None, retries=4, delay=2, timeout=60):
    """GET with exponential-backoff retry and rate-limit handling."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  [RATE LIMIT] sleeping {wait}s …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))
    return None


# ─── 1. STATION DISCOVERY ─────────────────────────────────────────────────────

def discover_india_stations(force_refresh=False):
    """
    Query OpenAQ v3 for all Indian PM2.5 locations.
    Returns list of station dicts with sensor_id, lat, lon, city, state.
    Results are cached to STATIONS_JSON.
    """
    if os.path.exists(STATIONS_JSON) and not force_refresh:
        print(f"Loading cached station registry: {STATIONS_JSON}")
        with open(STATIONS_JSON, encoding="utf-8") as f:
            return json.load(f)

    print("=" * 60)
    print("  Discovering India PM2.5 stations via OpenAQ v3 …")
    print("=" * 60)

    all_locations = []
    page = 1
    while True:
        try:
            r = _get(
                f"{OPENAQ_BASE}/locations",
                headers=OPENAQ_HEADERS,
                params={
                    "parameters_name": "pm25",
                    "bbox": "65.0,5.0,100.0,40.0",   # India bounding box
                    "limit": 1000, "page": page,
                },
            )
        except Exception as e:
            print(f"  [ERROR] discovery page {page}: {e}")
            break

        results = r.json().get("results", [])
        if not results:
            break
        all_locations.extend(results)
        print(f"  Page {page}: {len(results)} locations  (total {len(all_locations)})")
        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.5)

    stations = []
    for loc in all_locations:
        coords = loc.get("coordinates", {})
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        if lat is None or lon is None:
            continue

        # Filter to India proper
        if not (5 <= lat <= 40 and 65 <= lon <= 100):
            continue

        city  = loc.get("locality") or loc.get("city") or "Unknown"
        owner = ""
        if loc.get("providers"):
            owner = loc["providers"][0].get("name", "")

        # Find ALL pm25 sensor ids (newer sensors tend to have hourly data)
        pm25_sids = []
        for s in loc.get("sensors", []):
            param_name = (s.get("parameter", {}).get("name", "") or "").lower()
            if "pm25" in param_name or "pm2.5" in param_name:
                pm25_sids.append(s["id"])

        if not pm25_sids:
            continue

        # Use the highest sensor ID as default (tends to be newer with more data)
        pm25_sids.sort(reverse=True)

        # Safe name: use loc name if not None, else fall back to id
        raw_name = loc.get("name") or f"loc_{loc['id']}"
        safe_name = (raw_name + f"_{loc['id']}").replace(" ", "_").replace(",", "").lower()[:50]

        stations.append({
            "station":     safe_name,
            "location_id": loc["id"],
            "sensor_id":   pm25_sids[0],         # best candidate
            "sensor_ids":  pm25_sids,             # all pm25 sensors
            "lat":         lat,
            "lon":         lon,
            "city":        city,
            "name":        raw_name,
            "owner":       owner,
        })

    # Add fallback major-city stations (if not already discovered)
    discovered_cities = {s["city"].lower() for s in stations}
    for key, info in MAJOR_CITIES_FALLBACK.items():
        if info["city"].lower() not in discovered_cities:
            stations.append({
                "station":     key,
                "location_id": None,
                "sensor_id":   None,   # will attempt discovery at fetch time
                "lat":         info["lat"],
                "lon":         info["lon"],
                "city":        info["city"],
                "name":        key,
                "owner":       "fallback",
            })

    print(f"\n  Total stations: {len(stations)}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(stations, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {STATIONS_JSON}")
    return stations


def _resolve_sensor_id(lat, lon, radius_km=5):
    """
    Attempt to find a pm25 sensor_id for a fallback station using coordinates.
    Returns sensor_id or None.
    """
    try:
        r = _get(
            f"{OPENAQ_BASE}/locations",
            headers=OPENAQ_HEADERS,
            params={"coordinates": f"{lat},{lon}",
                    "radius": int(radius_km * 1000),
                    "parameters_name": "pm25", "limit": 3},
        )
        for loc in r.json().get("results", []):
            for s in loc.get("sensors", []):
                pn = (s.get("parameter", {}).get("name", "") or "").lower()
                if "pm25" in pn or "pm2.5" in pn:
                    return s["id"]
    except Exception:
        pass
    return None


# ─── 2. PM2.5 FETCH ───────────────────────────────────────────────────────────

def fetch_pm25_for_sensor(sensor_id, station_name, start_date, end_date):
    """
    Fetch hourly PM2.5 data for one sensor via OpenAQ v3 /sensors/{id}/hours.
    
    Uses the /hours endpoint (pre-aggregated hourly averages) instead of
    /measurements, because most Indian CPCB stations only expose raw 15-min
    data on /measurements and the /hours endpoint is the proper hourly source.
    
    Chunks requests into 6-month windows to handle API limitations.
    Returns DataFrame [timestamp, pm25].
    """
    all_rows = []
    
    # Chunk by 6-month windows
    chunk_start = pd.Timestamp(start_date, tz="UTC")
    chunk_end_final = pd.Timestamp(end_date, tz="UTC")
    
    while chunk_start < chunk_end_final:
        chunk_end = min(chunk_start + pd.DateOffset(months=6), chunk_end_final)
        
        page = 1
        while True:
            try:
                r = _get(
                    f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                    headers=OPENAQ_HEADERS,
                    params={
                        "datetime_from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "datetime_to":   chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "limit": 1000, "page": page,
                    },
                )
            except Exception as e:
                _print(f"    [ERROR] {station_name} chunk {chunk_start.date()} p{page}: {e}")
                break

            results = r.json().get("results", [])
            if not results:
                break
            
            for item in results:
                period = item.get("period", {})
                ts = (period.get("datetimeFrom", {}).get("utc")
                      or period.get("dateFrom", {}).get("utc"))
                val = item.get("value")
                if ts and val is not None:
                    all_rows.append({"timestamp": ts, "pm25": float(val)})
            
            if len(results) < 1000:
                break
            page += 1
            time.sleep(OPENAQ_DELAY_SECONDS)
        
        chunk_start = chunk_end
        time.sleep(OPENAQ_DELAY_SECONDS)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "pm25"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    df = df.groupby("timestamp", as_index=False)["pm25"].mean()
    df = df[df["pm25"].between(0, 1500)]   # physical validity filter
    return df


def _fetch_station_worker(station_info, start_date, end_date):
    """Worker function for ThreadPoolExecutor. Tries all sensor IDs.
    
    Uses station's known data_start (from screener) to avoid wasting API
    calls on empty date ranges before data actually exists.
    """
    name = station_info["station"]
    sids = station_info.get("sensor_ids", [])
    sid  = station_info.get("sensor_id")

    # Build list of sensor IDs to try (primary first, then alternatives)
    try_sids = []
    if sid is not None:
        try_sids.append(sid)
    for s in sids:
        if s not in try_sids:
            try_sids.append(s)

    # Resolve sensor if no IDs at all
    if not try_sids:
        resolved = _resolve_sensor_id(station_info["lat"], station_info["lon"])
        if resolved:
            try_sids.append(resolved)

    if not try_sids:
        return name, pd.DataFrame(columns=["timestamp", "pm25"])

    # Determine effective start date:
    # If screener recorded data_start AND it's not a single-day record,
    # use the earlier of (data_start - 1 day) vs global start_date.
    effective_start = start_date
    known_start = station_info.get("data_start")
    known_end   = station_info.get("data_end")
    if known_start and known_end and known_start != known_end:
        # Real continuous data exists before our global start
        station_start = known_start[:10]   # YYYY-MM-DD
        if station_start < start_date:
            effective_start = station_start
            _print(f"    [EARLY DATA] {name}: using {effective_start} (vs global {start_date})")

    # Try each sensor ID until we get data
    for attempt_sid in try_sids:
        df = fetch_pm25_for_sensor(attempt_sid, name, effective_start, end_date)
        if not df.empty:
            return name, df

    return name, pd.DataFrame(columns=["timestamp", "pm25"])



def fetch_all_pm25_parallel(stations, start_date, end_date, max_workers=OPENAQ_MAX_WORKERS):
    """Parallel PM2.5 fetch for all stations. Saves checkpoints every 50 stations."""
    import pickle
    checkpoint_path = os.path.join(DATA_DIR, "_pm25_checkpoint.pkl")

    # Resume from checkpoint if available
    pm25_dict = {}
    done_names = set()
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                pm25_dict = pickle.load(f)
            done_names = set(pm25_dict.keys())
            _print(f"  [RESUME] Loaded checkpoint: {len(done_names)} stations already done")
        except Exception:
            pm25_dict = {}

    # Filter out already-completed stations
    remaining = [s for s in stations if s["station"] not in done_names]
    total = len(stations)
    _print(f"  Launching {len(remaining)}/{total} station fetches with {max_workers} parallel workers …")

    completed_count = len(done_names)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_station_worker, s, start_date, end_date): s
            for s in remaining
        }
        for future in as_completed(futures):
            completed_count += 1
            try:
                name, df = future.result()
                pm25_dict[name] = df
                status = f"{len(df):>6,} hrs" if not df.empty else "  NO DATA"
                _print(f"  [{completed_count:>3}/{total}] {name:<45} {status}")
            except Exception as e:
                stn = futures[future]
                pm25_dict[stn["station"]] = pd.DataFrame(columns=["timestamp", "pm25"])
                _print(f"  [{completed_count:>3}/{total}] {stn['station']:<45} ERROR: {e}")

            # Save checkpoint every 50 stations
            if completed_count % 50 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(pm25_dict, f)
                _print(f"  [CHECKPOINT] Saved {completed_count} stations → {checkpoint_path}")

    # Final checkpoint save
    with open(checkpoint_path, "wb") as f:
        pickle.dump(pm25_dict, f)

    return pm25_dict


# ─── 3. WEATHER FETCH ─────────────────────────────────────────────────────────

def fetch_weather_for_city(lat, lon, start_date, end_date, city_label=""):
    """
    Fetch ERA5 reanalysis weather via Open-Meteo archive API.
    Returns DataFrame with [timestamp, *WEATHER_VARS].
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ",".join(WEATHER_VARS),
    }
    try:
        r = _get(OPEN_METEO_ARCHIVE, params=params, timeout=120)
        data = r.json().get("hourly", {})
    except Exception as e:
        print(f"  [WARN] Weather fetch failed for {city_label}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "time" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)
    df = df.dropna(subset=["temperature_2m"])   # drop rows with no weather at all

    # If boundary_layer_height wasn't returned, zero-fill
    if "boundary_layer_height" not in df.columns:
        df["boundary_layer_height"] = 0.0

    return df


def fetch_weather_for_cities(stations, start_date, end_date):
    """
    Fetch weather once per unique (rounded) city coordinate.
    Returns dict: station_name → weather_df.
    """
    # Group stations into city clusters (≈ 0.5° grid)
    city_groups = {}
    for s in stations:
        key = (round(s["lat"] * 2) / 2, round(s["lon"] * 2) / 2)
        city_groups.setdefault(key, []).append(s["station"])

    print(f"\n  Fetching weather for {len(city_groups)} city clusters …")
    weather_cache = {}   # (lat, lon) → df

    for i, ((lat, lon), stn_list) in enumerate(city_groups.items(), 1):
        label = ", ".join(stn_list[:3])
        print(f"  [{i:>3}/{len(city_groups)}] ({lat:.1f}, {lon:.1f}) → {label} …")
        df = fetch_weather_for_city(lat, lon, start_date, end_date, label)
        for sname in stn_list:
            weather_cache[sname] = df

    return weather_cache


# ─── 4. FIRMS FIRE FETCH ──────────────────────────────────────────────────────

def fetch_firms_india(start_date, end_date):
    """
    Fetch VIIRS fire data for all of India in 5-day chunks.
    Returns a single DataFrame with columns:
      [timestamp, latitude, longitude, frp, bright_ti4, bright_ti5]
    """
    print(f"\n  Fetching FIRMS fire data for India ({start_date} → {end_date}) …")

    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)
    chunks   = []
    current  = start_dt

    total_days = (end_dt - start_dt).days + 1
    n_chunks   = math.ceil(total_days / FIRMS_CHUNK_DAYS)

    for chunk_i in range(n_chunks):
        days     = min(FIRMS_CHUNK_DAYS, (end_dt - current).days + 1)
        date_str = current.strftime("%Y-%m-%d")

        url = (
            f"{FIRMS_BASE}/{FIRMS_API_KEY}/{FIRMS_SOURCE}/"
            f"{INDIA_BBOX}/{days}/{date_str}"
        )
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "latitude" in r.text:
                chunk = pd.read_csv(io.StringIO(r.text))
                if not chunk.empty:
                    chunks.append(chunk)
        except Exception as e:
            pass   # silently skip failed chunks

        current += timedelta(days=days)
        if chunk_i % 20 == 0:
            print(f"    chunk {chunk_i+1}/{n_chunks}  ({date_str})")
        time.sleep(1.0)

    if not chunks:
        print("  [WARN] No FIRMS data returned for India")
        return pd.DataFrame(columns=["timestamp", "latitude", "longitude",
                                     "frp", "bright_ti4", "bright_ti5"])

    fire_df = pd.concat(chunks, ignore_index=True)

    fire_df["acq_time"] = fire_df["acq_time"].astype(str).str.zfill(4)
    fire_df["timestamp"] = pd.to_datetime(
        fire_df["acq_date"] + " " + fire_df["acq_time"],
        format="%Y-%m-%d %H%M", utc=True,
    ).dt.floor("h")

    keep = ["timestamp", "latitude", "longitude", "frp"]
    if "bright_ti4" in fire_df.columns:
        keep.append("bright_ti4")
    if "bright_ti5" in fire_df.columns:
        keep.append("bright_ti5")

    fire_df = fire_df[keep].dropna(subset=["frp"])
    print(f"  FIRMS total fire events: {len(fire_df):,}")
    return fire_df


def aggregate_fires_for_station(fire_df, lat, lon, radius_km=FIRE_RADIUS_KM):
    """
    Aggregate hourly fire metrics within radius_km of (lat, lon).
    Returns DataFrame [timestamp, fire_count, frp_sum, frp_mean, frp_max,
                       bright_ti4_mean, bright_ti5_mean].
    """
    if fire_df.empty:
        return pd.DataFrame(columns=["timestamp", "fire_count", "frp_sum",
                                     "frp_mean", "frp_max",
                                     "bright_ti4_mean", "bright_ti5_mean"])

    dist = haversine_km(lat, lon, fire_df["latitude"].values, fire_df["longitude"].values)
    nearby = fire_df[dist <= radius_km].copy()

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


# ─── 5. CO-POLLUTANT FETCH (PM10, NO2, SO2, O3, CO) ─────────────────────────

def fetch_copollutants(location_id, start_date, end_date):
    """
    Fetch hourly averages for PM10, NO2, SO2, O3, CO from OpenAQ v3.
    Returns DataFrame [timestamp, pm10, no2, so2, o3, co].
    """
    param_map = {"pm10": "pm10", "no2": "no2", "so2": "so2", "o3": "o3", "co": "co"}
    dfs = []

    try:
        r = _get(
            f"{OPENAQ_BASE}/locations/{location_id}/sensors",
            headers=OPENAQ_HEADERS,
        )
        sensors = r.json().get("results", [])
    except Exception:
        return pd.DataFrame()

    for s in sensors:
        pname = (s.get("parameter", {}).get("name", "") or "").lower()
        col = param_map.get(pname)
        if col is None:
            continue

        sid = s["id"]
        page = 1
        rows = []
        while True:
            try:
                r2 = _get(
                    f"{OPENAQ_BASE}/sensors/{sid}/hours",
                    headers=OPENAQ_HEADERS,
                    params={
                        "datetime_from": f"{start_date}T00:00:00Z",
                        "datetime_to":   f"{end_date}T23:59:59Z",
                        "limit": 1000, "page": page,
                    },
                )
                res = r2.json().get("results", [])
                if not res:
                    break
                for item in res:
                    period = item.get("period", {})
                    ts = (period.get("datetimeFrom", {}).get("utc") or
                          period.get("dateFrom", {}).get("utc"))
                    val = item.get("value")
                    if ts and val is not None:
                        rows.append({"timestamp": ts, col: float(val)})
                if len(res) < 1000:
                    break
                page += 1
                time.sleep(0.3)
            except Exception:
                break

        if rows:
            tmp = pd.DataFrame(rows)
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.floor("h")
            tmp = tmp.groupby("timestamp", as_index=False)[col].mean()
            dfs.append(tmp)

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="timestamp", how="outer")
    return merged


# ─── 6. DATASET BUILDER ───────────────────────────────────────────────────────

def build_station_dataframe(
    station_info,
    pm_df,
    weather_df,
    fire_df_india,
    start_date,
    end_date,
    fetch_copollutants_flag=True,
):
    """
    Build the full feature-engineered DataFrame for one station.
    Returns None if the station has insufficient data.
    """
    if pm_df is None or pm_df.empty or len(pm_df) < MIN_STATION_RECORDS:
        return None

    lat    = station_info["lat"]
    lon    = station_info["lon"]
    name   = station_info["station"]
    loc_id = station_info.get("location_id")

    # Fire aggregation for this station
    fire_hourly = aggregate_fires_for_station(fire_df_india, lat, lon)

    # Optional: co-pollutants
    copoll_df = pd.DataFrame()
    if fetch_copollutants_flag and loc_id is not None:
        try:
            copoll_df = fetch_copollutants(loc_id, start_date, end_date)
        except Exception:
            pass

    # Merge co-pollutants into pm_df
    if not copoll_df.empty:
        pm_df = pm_df.merge(copoll_df, on="timestamp", how="left")

    # Run full feature pipeline
    sdf = build_features(pm_df, weather_df, fire_hourly, name, lat, lon)

    # Drop rows where key lags / targets are missing
    drop_cols = ["pm25_lag24"] + [f"target_{h}h" for h in range(1, 25)]
    sdf.dropna(subset=[c for c in drop_cols if c in sdf.columns], inplace=True)

    # Interpolate remaining NaN in numeric cols
    num_cols = sdf.select_dtypes(include=[np.number]).columns.tolist()
    sdf[num_cols] = (
        sdf.sort_values("timestamp")
        [num_cols]
        .interpolate(method="linear", limit_direction="both")
    )
    sdf.dropna(inplace=True)

    return sdf if len(sdf) >= 100 else None


def build_dataset(stations, start_date, end_date,
                  city_filter=None, fetch_copollutants_flag=False):
    """
    Build the full long-format dataset for all stations.

    Args
    ----
    stations            : list of station dicts from discover_india_stations()
    start_date / end_date : ISO date strings
    city_filter         : list of city names to restrict (None = all)
    fetch_copollutants_flag : fetch PM10/NO2/etc. (slower, more features)
    """
    _print("=" * 60)
    _print("  PM2.5 India Dataset Builder")
    _print("=" * 60)

    # Filter by city if requested
    if city_filter:
        city_set = {c.lower() for c in city_filter}
        stations = [s for s in stations if s.get("city", "").lower() in city_set]
        _print(f"  City filter active: {city_filter} -> {len(stations)} stations")

    _print(f"  Stations: {len(stations)}")
    _print(f"  Date range: {start_date}  ->  {end_date}")
    _print(f"  Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # ── Fetch PM2.5 for all stations (parallel)
    _print(f"\n[Step 1/4] Fetching PM2.5 data ({OPENAQ_MAX_WORKERS} workers) …")
    pm25_dict = fetch_all_pm25_parallel(stations, start_date, end_date)

    # Filter out stations with no data
    valid_stations = [s for s in stations if
                      pm25_dict.get(s["station"]) is not None and
                      len(pm25_dict[s["station"]]) >= MIN_STATION_RECORDS]
    print(f"\n  Stations with sufficient data: {len(valid_stations)}/{len(stations)}")

    if not valid_stations:
        print("[FATAL] No valid stations found. Check API keys and date range.")
        return

    # ── Fetch weather per city cluster
    print("\n[Step 2/4] Fetching weather data …")
    weather_dict = fetch_weather_for_cities(valid_stations, start_date, end_date)

    # ── Fetch FIRMS fire data for all India
    print("\n[Step 3/4] Fetching FIRMS fire data …")
    fire_df_india = fetch_firms_india(start_date, end_date)

    # ── Build per-station DataFrames
    print("\n[Step 4/4] Building station features …")
    station_dfs = []

    for i, stn in enumerate(valid_stations, 1):
        name       = stn["station"]
        pm_df      = pm25_dict[name]
        weather_df = weather_dict.get(name, pd.DataFrame())

        print(f"  [{i:>3}/{len(valid_stations)}] {name:<45}", end=" ")
        try:
            sdf = build_station_dataframe(
                stn, pm_df, weather_df, fire_df_india,
                start_date, end_date, fetch_copollutants_flag,
            )
            if sdf is not None:
                station_dfs.append(sdf)
                print(f"→ {len(sdf):>6,} rows")
            else:
                print("→ SKIP (insufficient)")
        except Exception as e:
            print(f"→ ERROR: {e}")

    if not station_dfs:
        print("[FATAL] No station DataFrames produced.")
        return

    # ── Combine
    df = pd.concat(station_dfs, ignore_index=True)
    df.sort_values(["station", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Cast to float32 to reduce disk/memory footprint
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].astype(np.float32)

    # ── Summary
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Shape       : {df.shape}")
    print(f"  Stations    : {df['station'].nunique()}")
    cities = df["station"].unique()[:5].tolist()
    print(f"  Sample stns : {cities}")
    print(f"  Date range  : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  PM2.5 range : {df['pm25'].min():.1f} – {df['pm25'].max():.1f} µg/m³")

    # ── Save
    df.to_parquet(DATASET_PARQUET, index=False, compression="snappy")
    print(f"\n  Saved → {DATASET_PARQUET}  ({os.path.getsize(DATASET_PARQUET)/1e6:.1f} MB)")

    # Save station metadata
    meta = {
        "station_col": "station",
        "target_cols": [f"target_{h}h" for h in range(1, 25)],
        "feature_cols": FEATURE_COLS,
        "all_columns": list(df.columns),
        "n_stations": df["station"].nunique(),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "built_at": datetime.utcnow().isoformat(),
    }
    meta_path = os.path.join(DATA_DIR, "dataset_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta  → {meta_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PM2.5 India Data Pipeline")
    parser.add_argument("--discover",        action="store_true",
                        help="Discover and cache India station registry")
    parser.add_argument("--build",           action="store_true",
                        help="Build the training dataset")
    parser.add_argument("--force-refresh",   action="store_true",
                        help="Force re-discovery even if registry exists")
    parser.add_argument("--cities",          nargs="+", default=None,
                        help="Filter to specific cities (space-separated)")
    parser.add_argument("--screened",        action="store_true",
                        help="Use pre-screened stations from filter_stations.py (faster)")
    parser.add_argument("--start",           default=TRAIN_START)
    parser.add_argument("--end",             default=TRAIN_END)
    parser.add_argument("--copollutants",    action="store_true",
                        help="Also fetch PM10, NO2, SO2, O3, CO per station")
    args = parser.parse_args()

    if not args.discover and not args.build:
        parser.print_help()
        return

    screened_path = os.path.join(DATA_DIR, "screened_stations_clean.json")
    if not os.path.exists(screened_path):
        screened_path = os.path.join(DATA_DIR, "screened_stations.json")  # fallback
    if args.screened and os.path.exists(screened_path):
        _print(f"Using pre-screened station list: {screened_path}")
        with open(screened_path, encoding="utf-8") as f:
            stations = json.load(f)
        _print(f"Loaded {len(stations)} pre-screened stations with confirmed /hours data")
    else:
        stations = discover_india_stations(force_refresh=args.force_refresh)
        _print(f"\nRegistry loaded: {len(stations)} stations")

    if args.build:
        build_dataset(
            stations,
            start_date=args.start,
            end_date=args.end,
            city_filter=args.cities if not args.screened else None,
            fetch_copollutants_flag=args.copollutants,
        )


if __name__ == "__main__":
    main()
