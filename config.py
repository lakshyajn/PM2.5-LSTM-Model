"""
config.py
─────────
Central configuration for the PM2.5 India Forecasting System.
All API keys, file paths, model hyperparameters, and feature lists live here.
"""

import os
import numpy as np

# ─── API Keys ──────────────────────────────────────────────────────────────────
OPENAQ_API_KEY = "97df45e8a9386c64e83d37ce26c23254d42948567f4d48d1ecef928152abea98"
FIRMS_API_KEY  = "4d998c8cc19c96e93a7b4971574cd14f"
NEWSAPI_KEY    = "0335b3e9782542228b4c156a06ddd4fd"

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")

STATIONS_JSON    = os.path.join(DATA_DIR, "india_stations.json")
DATASET_PARQUET  = os.path.join(DATA_DIR, "dataset_v3.parquet")
NEWS_PARQUET     = os.path.join(DATA_DIR, "news_signals.parquet")
MODEL_PATH       = os.path.join(DATA_DIR, "model_v2.keras")
SCALER_X_PATH    = os.path.join(DATA_DIR, "scaler_X.pkl")
SCALER_Y_PATH    = os.path.join(DATA_DIR, "scaler_y.pkl")
FEAT_PATH        = os.path.join(DATA_DIR, "feature_columns.json")
STATION_ID_PATH  = os.path.join(DATA_DIR, "station_id_map.json")
METRICS_PATH     = os.path.join(DATA_DIR, "metrics.json")
LOSS_PLOT        = os.path.join(DATA_DIR, "loss_curve.png")
PRED_PLOT        = os.path.join(DATA_DIR, "predictions_test.png")

os.makedirs(DATA_DIR, exist_ok=True)

# ─── Training Data Range ───────────────────────────────────────────────────────
TRAIN_START = "2022-01-01"
TRAIN_END   = "2026-06-20"   # update to near-current date

# ─── Model Hyperparameters ─────────────────────────────────────────────────────
SEQ_LEN       = 72    # 72-hour lookback window
N_HORIZONS    = 24   # predict t+1h through t+24h
EMBED_DIM     = 16   # station embedding dimension
BATCH_SIZE    = 128
EPOCHS        = 120
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
HUBER_DELTA   = 15.0  # µg/m³ Huber loss threshold
MAX_SEQS_TRAIN = 300_000  # 300k/epoch ~4-5min on GTX1650; shuffled fresh each epoch

# ─── LSTM Architecture ─────────────────────────────────────────────────────────
LSTM1_UNITS  = 256
LSTM2_UNITS  = 128
DENSE1_UNITS = 128
DENSE2_UNITS = 64
N_ATTN_HEADS = 4
ATTN_KEY_DIM = 64
DROPOUT_RATE = 0.25

# ─── Horizon Loss Weights (exponential decay, near-term weighted more) ─────────
HORIZON_WEIGHTS = np.exp(-np.arange(N_HORIZONS) * 0.04).astype(np.float32)
HORIZON_WEIGHTS /= HORIZON_WEIGHTS.sum()

# ─── Feature Columns (order matters — model trained in this order) ─────────────
FEATURE_COLS = [
    # Location embedding proxy
    "lat", "lon",

    # PM2.5 autoregressive
    "pm25",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_lag24", "pm25_lag48", "pm25_lag72",
    "pm25_rmean24", "pm25_rmax24", "pm25_rmin24", "pm25_rstd24",
    "pm25_trend6h",
    "aqi_category",

    # Co-pollutants (OpenAQ; zero-filled if unavailable)
    "pm10", "no2", "so2", "o3", "co",

    # Weather (Open-Meteo ERA5 archive)
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "precipitation",
    "wind_speed", "wind_u", "wind_v",
    "cloud_cover", "shortwave_radiation",
    "boundary_layer_height",

    # Fire (NASA FIRMS VIIRS)
    "fire_count", "frp_sum", "frp_mean", "frp_max",
    "bright_ti4_mean", "bright_ti5_mean",

    # Traffic / calendar proxy
    "rush_hour", "is_weekday", "is_holiday", "traffic_index",
    "day_of_week_sin", "day_of_week_cos",

    # Cyclic time encodings
    "hour_sin", "hour_cos", "month_sin", "month_cos",

    # Seasonal / known event flags (computed from calendar)
    "is_monsoon", "is_winter", "is_harvest_season",
    "is_diwali_period", "is_holi_period",

    # News signals (0 for historical; populated at real-time inference)
    "news_dust_storm", "news_industrial_event",
    "news_crop_burning", "news_calamity",
    "news_fireworks", "news_smog_alert",
]

# ─── API Base URLs ──────────────────────────────────────────────────────────────
OPENAQ_BASE     = "https://api.openaq.org/v3"
OPENAQ_HEADERS  = {"X-API-Key": OPENAQ_API_KEY}

OPEN_METEO_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

FIRMS_BASE   = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE = "VIIRS_SNPP_SP"

NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# Open-Meteo hourly weather variables to request
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "precipitation",
    "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "shortwave_radiation",
    "boundary_layer_height",
]

# ─── Rate Limiting ──────────────────────────────────────────────────────────────
OPENAQ_MAX_WORKERS   = 2    # conservative: avoids 429 rate limits
OPENAQ_DELAY_SECONDS = 1.5  # 1.5s per request = ~1.3 req/s per worker
FIRMS_CHUNK_DAYS     = 5

# ─── India Coverage ─────────────────────────────────────────────────────────────
INDIA_COUNTRY_CODE     = "IN"
MIN_STATION_RECORDS    = 200  # skip stations with fewer hourly records
FIRE_RADIUS_KM         = 300  # associate fires within this radius to a station

# India bounding box for FIRMS (one large fetch per date chunk)
INDIA_BBOX = "65.0,5.0,100.0,40.0"   # west,south,east,north

# ─── CPCB AQI Breakpoints for PM2.5 (µg/m³) ────────────────────────────────────
AQI_BREAKPOINTS = [
    (0,   30,   "Good",         0, "[GREEN]"),
    (30,  60,   "Satisfactory", 1, "[YELLOW]"),
    (60,  90,   "Moderate",     2, "[ORANGE]"),
    (90,  120,  "Poor",         3, "[RED]"),
    (120, 250,  "Very Poor",    4, "[PURPLE]"),
    (250, 9999, "Severe",       5, "[MAROON]"),
]

# ─── Traffic Rush Hours (IST) ───────────────────────────────────────────────────
RUSH_HOURS = {7, 8, 9, 10, 17, 18, 19, 20}

# ─── Indian Public Holidays ─────────────────────────────────────────────────────
INDIAN_HOLIDAYS = {
    # 2022
    "2022-01-26", "2022-03-18", "2022-04-14", "2022-04-15",
    "2022-05-03", "2022-07-10", "2022-08-15", "2022-10-02",
    "2022-10-05", "2022-10-24", "2022-11-08", "2022-12-25",
    # 2023
    "2023-01-26", "2023-03-08", "2023-04-14", "2023-04-22",
    "2023-06-29", "2023-08-15", "2023-10-02", "2023-10-24",
    "2023-11-12", "2023-11-13", "2023-11-27", "2023-12-25",
    # 2024
    "2024-01-26", "2024-03-25", "2024-03-29", "2024-04-10",
    "2024-04-14", "2024-06-17", "2024-08-15", "2024-10-02",
    "2024-10-12", "2024-11-01", "2024-11-15", "2024-12-25",
    # 2025
    "2025-01-26", "2025-03-14", "2025-03-25", "2025-04-14",
    "2025-04-18", "2025-06-07", "2025-08-15", "2025-10-02",
    "2025-10-20", "2025-10-21", "2025-11-05", "2025-12-25",
    # 2026
    "2026-01-26", "2026-03-04", "2026-03-14", "2026-04-03",
    "2026-04-14", "2026-08-15", "2026-10-02", "2026-10-09",
    "2026-10-28", "2026-12-25",
}

# Diwali central dates (±3 days = diwali_period)
DIWALI_DATES = {
    2022: "2022-10-24",
    2023: "2023-11-12",
    2024: "2024-11-01",
    2025: "2025-10-20",
    2026: "2026-11-08",
}

# Holi central dates (±1 day = holi_period)
HOLI_DATES = {
    2022: "2022-03-18",
    2023: "2023-03-08",
    2024: "2024-03-25",
    2025: "2025-03-14",
    2026: "2026-03-04",
}

# Major Indian cities with fallback station coordinates (used when OpenAQ has no data)
# Format: {city_key: {state, lat, lon, city_name}}
MAJOR_CITIES_FALLBACK = {
    "delhi_ito":          {"state": "Delhi",             "lat": 28.630, "lon": 77.240, "city": "Delhi"},
    "mumbai_bandra":      {"state": "Maharashtra",       "lat": 19.054, "lon": 72.840, "city": "Mumbai"},
    "kolkata_victoria":   {"state": "West Bengal",       "lat": 22.545, "lon": 88.342, "city": "Kolkata"},
    "chennai_alandur":    {"state": "Tamil Nadu",        "lat": 13.001, "lon": 80.208, "city": "Chennai"},
    "bengaluru_btm":      {"state": "Karnataka",         "lat": 12.912, "lon": 77.609, "city": "Bengaluru"},
    "hyderabad_bollaram": {"state": "Telangana",         "lat": 17.530, "lon": 78.380, "city": "Hyderabad"},
    "ahmedabad_ctm":      {"state": "Gujarat",           "lat": 23.041, "lon": 72.590, "city": "Ahmedabad"},
    "lucknow_talkatora":  {"state": "Uttar Pradesh",     "lat": 26.870, "lon": 80.870, "city": "Lucknow"},
    "kanpur_nehru":       {"state": "Uttar Pradesh",     "lat": 26.460, "lon": 80.310, "city": "Kanpur"},
    "patna_igsc":         {"state": "Bihar",             "lat": 25.612, "lon": 85.162, "city": "Patna"},
    "jaipur_mansarovar":  {"state": "Rajasthan",         "lat": 26.855, "lon": 75.800, "city": "Jaipur"},
    "amritsar_golden":    {"state": "Punjab",            "lat": 31.630, "lon": 74.870, "city": "Amritsar"},
    "ludhiana_civil":     {"state": "Punjab",            "lat": 30.907, "lon": 75.857, "city": "Ludhiana"},
    "gurugram_vikas":     {"state": "Haryana",           "lat": 28.450, "lon": 77.030, "city": "Gurugram"},
    "faridabad_sector":   {"state": "Haryana",           "lat": 28.410, "lon": 77.310, "city": "Faridabad"},
    "bhopal_ttt":         {"state": "Madhya Pradesh",    "lat": 23.259, "lon": 77.412, "city": "Bhopal"},
    "indore_polo":        {"state": "Madhya Pradesh",    "lat": 22.717, "lon": 75.857, "city": "Indore"},
    "bhubaneswar_camp":   {"state": "Odisha",            "lat": 20.296, "lon": 85.824, "city": "Bhubaneswar"},
    "raipur_telibanda":   {"state": "Chhattisgarh",      "lat": 21.248, "lon": 81.634, "city": "Raipur"},
    "ranchi_morabadi":    {"state": "Jharkhand",         "lat": 23.370, "lon": 85.342, "city": "Ranchi"},
    "visakhapatnam_gvm":  {"state": "Andhra Pradesh",    "lat": 17.720, "lon": 83.300, "city": "Visakhapatnam"},
    "kochi_ernakulam":    {"state": "Kerala",            "lat": 9.990,  "lon": 76.280, "city": "Kochi"},
    "guwahati_pan":       {"state": "Assam",             "lat": 26.146, "lon": 91.736, "city": "Guwahati"},
    "chandigarh_sec":     {"state": "Chandigarh",        "lat": 30.733, "lon": 76.779, "city": "Chandigarh"},
    "dehradun_isbt":      {"state": "Uttarakhand",       "lat": 30.326, "lon": 78.046, "city": "Dehradun"},
    "srinagar_barzulla":  {"state": "J&K",               "lat": 34.086, "lon": 74.805, "city": "Srinagar"},
    "pune_katraj":        {"state": "Maharashtra",       "lat": 18.445, "lon": 73.857, "city": "Pune"},
    "agra_sanjay":        {"state": "Uttar Pradesh",     "lat": 27.180, "lon": 78.001, "city": "Agra"},
    "varanasi_ardn":      {"state": "Uttar Pradesh",     "lat": 25.335, "lon": 83.007, "city": "Varanasi"},
    "nagpur_civil":       {"state": "Maharashtra",       "lat": 21.148, "lon": 79.082, "city": "Nagpur"},
}
