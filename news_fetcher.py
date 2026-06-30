"""
news_fetcher.py
───────────────
Fetches PM2.5-relevant news signals from NewsAPI and converts them into
hourly feature vectors for each city.

Two modes
─────────
1. HISTORICAL (training): Uses only calendar-based event signals since
   NewsAPI free tier only allows ~30 days of history.

2. REAL-TIME (inference): Fetches recent news (last 48h) from NewsAPI
   and returns per-hour signal intensities for a given city.

Usage
─────
  # Generate historical news signals parquet from calendar:
  python news_fetcher.py --historical --start 2022-01-01 --end 2026-06-20

  # Fetch live news for a city (used by realtime_fetcher.py):
  python news_fetcher.py --live --city Delhi --hours 48
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NEWSAPI_KEY, NEWSAPI_BASE, NEWS_PARQUET, DATA_DIR, DIWALI_DATES, HOLI_DATES

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── Signal Definitions ───────────────────────────────────────────────────────

# NewsAPI query templates → signal column name
NEWS_QUERIES = {
    "news_dust_storm":        "dust storm OR sandstorm OR andhi OR dust haze India",
    "news_industrial_event":  "factory fire OR industrial blast OR chemical explosion India",
    "news_crop_burning":      "stubble burning OR parali OR crop burning OR farm fire Punjab Haryana",
    "news_calamity":          "cyclone OR earthquake OR flood pollution India air quality",
    "news_fireworks":         "Diwali crackers OR New Year fireworks OR firecracker ban India",
    "news_smog_alert":        "smog alert OR pollution emergency OR GRAP OR red alert air quality India",
}

# City-to-search-term mapping for geographic relevance filtering
CITY_SEARCH_TERMS = {
    "Delhi":          ["Delhi", "NCR", "Gurgaon", "Noida"],
    "Mumbai":         ["Mumbai", "Maharashtra"],
    "Kolkata":        ["Kolkata", "West Bengal"],
    "Chennai":        ["Chennai", "Tamil Nadu"],
    "Bengaluru":      ["Bengaluru", "Bangalore", "Karnataka"],
    "Hyderabad":      ["Hyderabad", "Telangana"],
    "Ahmedabad":      ["Ahmedabad", "Gujarat"],
    "Lucknow":        ["Lucknow", "Uttar Pradesh", "UP"],
    "Patna":          ["Patna", "Bihar"],
    "Jaipur":         ["Jaipur", "Rajasthan"],
    "Amritsar":       ["Amritsar", "Ludhiana", "Punjab"],
    "Chandigarh":     ["Chandigarh", "Punjab", "Haryana"],
    "Bhopal":         ["Bhopal", "Madhya Pradesh"],
    "Bhubaneswar":    ["Bhubaneswar", "Odisha"],
    "Raipur":         ["Raipur", "Chhattisgarh"],
    "Ranchi":         ["Ranchi", "Jharkhand"],
    "Visakhapatnam":  ["Visakhapatnam", "Vizag", "Andhra Pradesh"],
    "Kochi":          ["Kochi", "Kerala"],
    "Guwahati":       ["Guwahati", "Assam"],
    "Dehradun":       ["Dehradun", "Uttarakhand"],
}


# ─── Calendar-Based Historical Signals ───────────────────────────────────────

def build_calendar_news_signals(timestamps_utc: pd.Series, lat: float = 28.0) -> pd.DataFrame:
    """
    Generate deterministic news-signal features from the calendar.
    These proxy the impact of recurring events on PM2.5 during historical training.

    Returns a DataFrame with one row per hour (aligned to timestamps_utc)
    and columns: news_dust_storm, news_industrial_event, news_crop_burning,
                 news_calamity, news_fireworks, news_smog_alert.
    """
    ist = timestamps_utc.dt.tz_convert("Asia/Kolkata")
    month  = ist.dt.month
    hour   = ist.dt.hour

    df_out = pd.DataFrame(index=timestamps_utc.index)
    df_out["timestamp"] = timestamps_utc

    # ── Dust storm: pre-monsoon season (April–June), afternoon hours
    df_out["news_dust_storm"] = (
        month.isin([4, 5, 6]) & hour.isin(range(12, 21))
    ).astype(np.float32) * 0.3   # base probability, not definite event

    # ── Industrial events: low background, slightly higher on weekdays
    df_out["news_industrial_event"] = np.float32(0.05)

    # ── Crop burning: Oct 15 – Nov 30 for north India (lat > 25)
    if lat > 25.0:
        burn_season = (
            ((month == 10) & (ist.dt.day >= 15)) |
            (month == 11)
        )
        df_out["news_crop_burning"] = burn_season.astype(np.float32) * 0.8
    else:
        df_out["news_crop_burning"] = np.float32(0.0)

    # ── Calamity: monsoon season enhanced risk
    df_out["news_calamity"] = month.isin([6, 7, 8, 9]).astype(np.float32) * 0.1

    # ── Fireworks: Diwali ±3 days + New Year's Eve
    diwali_flag = pd.Series(False, index=df_out.index)
    for year, date_str in DIWALI_DATES.items():
        center = pd.Timestamp(date_str).tz_localize("Asia/Kolkata")
        mask   = ((ist >= center - pd.Timedelta(days=3)) &
                  (ist <= center + pd.Timedelta(days=2)))
        diwali_flag |= mask

    new_year_flag = (
        ((month == 12) & (ist.dt.day == 31) & hour.isin(range(20, 24))) |
        ((month == 1)  & (ist.dt.day == 1)  & hour.isin(range(0, 3)))
    )
    df_out["news_fireworks"] = (diwali_flag | new_year_flag).astype(np.float32)

    # ── Smog alert: winter (Nov–Feb) + very high PM2.5 season (Delhi especially)
    winter_smog = month.isin([11, 12, 1, 2]).astype(np.float32)
    if lat > 25.0:
        df_out["news_smog_alert"] = winter_smog * 0.4
    else:
        df_out["news_smog_alert"] = winter_smog * 0.1

    signal_cols = [
        "news_dust_storm", "news_industrial_event", "news_crop_burning",
        "news_calamity", "news_fireworks", "news_smog_alert",
    ]
    df_out[signal_cols] = df_out[signal_cols].astype(np.float32)
    return df_out


# ─── Real-Time NewsAPI Fetcher ────────────────────────────────────────────────

def _newsapi_request(query, from_dt, to_dt, page=1, page_size=100):
    """Single NewsAPI /everything request. Returns list of articles."""
    params = {
        "q":          query,
        "from":       from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "to":         to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "language":   "en",
        "sortBy":     "publishedAt",
        "pageSize":   page_size,
        "page":       page,
        "apiKey":     NEWSAPI_KEY,
    }
    try:
        r = requests.get(NEWSAPI_BASE, params=params, timeout=15)
        if r.status_code == 426:
            print("  [NewsAPI] Upgrade required for older dates (free tier)")
            return []
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        print(f"  [NewsAPI] Error: {e}")
        return []


def fetch_realtime_news(city: str, hours: int = 48) -> pd.DataFrame:
    """
    Fetch last `hours` of news for `city` and compute hourly signal scores.

    Returns DataFrame [timestamp(UTC), news_dust_storm, news_industrial_event,
                       news_crop_burning, news_calamity, news_fireworks, news_smog_alert]
    """
    to_dt   = datetime.utcnow()
    from_dt = to_dt - timedelta(hours=hours)

    city_terms = CITY_SEARCH_TERMS.get(city, [city])

    # Per-signal article counts per hour
    hourly_counts = {}

    for signal_col, base_query in NEWS_QUERIES.items():
        # Add city filter to query
        city_clause = " OR ".join(f'"{t}"' for t in city_terms)
        query = f"({base_query}) AND ({city_clause})"

        articles = _newsapi_request(query, from_dt, to_dt)
        time.sleep(0.2)

        for article in articles:
            pub = article.get("publishedAt", "")
            try:
                ts = pd.Timestamp(pub).floor("h").tz_convert("UTC")
            except Exception:
                continue

            key = (ts, signal_col)
            hourly_counts[key] = hourly_counts.get(key, 0) + 1

    # Build hourly timestamp grid
    freq_h = pd.date_range(
        start=pd.Timestamp(from_dt, tz="UTC").floor("h"),
        end=pd.Timestamp(to_dt, tz="UTC").floor("h"),
        freq="h",
    )

    result = pd.DataFrame({"timestamp": freq_h})

    for col in NEWS_QUERIES:
        counts = []
        for ts in freq_h:
            c = hourly_counts.get((ts, col), 0)
            # Normalise: 0 → 0, 1 → 0.5, 3+ → 1.0
            score = min(c / 3.0, 1.0)
            counts.append(np.float32(score))
        result[col] = counts

    return result


# ─── Generate and Save Historical News Parquet ───────────────────────────────

def generate_historical_news_parquet(start_date: str, end_date: str):
    """
    Build a parquet file of calendar-derived news signal features for all
    cities from start_date to end_date (hourly frequency).

    This is used by train_model.py to merge into the main dataset.
    """
    print("Building calendar-based historical news signals …")

    timestamps = pd.date_range(
        start=pd.Timestamp(start_date, tz="UTC"),
        end=pd.Timestamp(end_date, tz="UTC"),
        freq="h",
    )
    ts_series = pd.Series(timestamps)

    rows_by_city = {}
    for city, terms in CITY_SEARCH_TERMS.items():
        # Rough lat: north cities (>25) vs south (<25)
        lat = 28.0 if city in [
            "Delhi", "Lucknow", "Amritsar", "Chandigarh",
            "Patna", "Jaipur", "Dehradun"
        ] else 18.0

        df_signals = build_calendar_news_signals(ts_series, lat=lat)
        df_signals["city"] = city
        rows_by_city[city] = df_signals

    combined = pd.concat(list(rows_by_city.values()), ignore_index=True)
    combined.sort_values(["city", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_parquet(NEWS_PARQUET, index=False, compression="snappy")
    print(f"  Saved → {NEWS_PARQUET}  ({os.path.getsize(NEWS_PARQUET)/1e6:.1f} MB)")
    return combined


def get_news_for_station(city: str, timestamps: pd.Series, lat: float = 28.0) -> pd.DataFrame:
    """
    Get news signals for a specific station's timestamps.
    Uses the pre-built news parquet if available, else builds calendar signals.
    """
    if os.path.exists(NEWS_PARQUET):
        news_df = pd.read_parquet(NEWS_PARQUET)
        city_news = news_df[news_df["city"] == city].drop(columns=["city"])
        if not city_news.empty:
            return city_news

    # Fallback: compute on-the-fly from calendar
    return build_calendar_news_signals(timestamps, lat=lat)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    from config import TRAIN_START, TRAIN_END

    parser = argparse.ArgumentParser(description="PM2.5 News Signal Fetcher")
    parser.add_argument("--historical", action="store_true",
                        help="Generate calendar-based historical signals parquet")
    parser.add_argument("--live", action="store_true",
                        help="Fetch real-time news for a city")
    parser.add_argument("--city", default="Delhi")
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--start", default=TRAIN_START)
    parser.add_argument("--end",   default=TRAIN_END)
    args = parser.parse_args()

    if args.historical:
        generate_historical_news_parquet(args.start, args.end)

    if args.live:
        print(f"\nFetching live news for {args.city} (last {args.hours}h) …")
        df = fetch_realtime_news(args.city, hours=args.hours)
        print(df.tail(10).to_string())
        print(f"\nSignal averages:\n{df.drop(columns='timestamp').mean().round(3)}")


if __name__ == "__main__":
    main()
