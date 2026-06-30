# PM2.5 India — Nationwide Real-Time Forecasting System

Predicts PM2.5 concentration for **1–24 hours ahead** at all major operational CPCB monitoring stations across India. Covers all states, integrates real-time data, news/event signals, and uses an improved deep learning architecture.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Discover all India PM2.5 stations
python data_pipeline.py --discover

# 3. Build training dataset (start with a few cities for a quick test)
python data_pipeline.py --build --cities Delhi Mumbai Kolkata Chennai

# 4. Generate historical news signals
python news_fetcher.py --historical

# 5. Train the model
python train_model.py

# 6. Offline prediction (from stored data)
python predict_v2.py --station delhi

# 7. Live prediction (fetches real-time data)
python predict_v2.py --station rk_puram --live
```

---

## File Structure

```
pm25_india/
├── config.py            Central configuration — API keys, paths, hyperparams
├── utils.py             Feature engineering utilities (shared)
├── data_pipeline.py     Nationwide station discovery + historical data fetch
├── news_fetcher.py      NewsAPI signals + calendar event features
├── realtime_fetcher.py  Real-time 72h window assembly for live inference
├── train_model.py       Model architecture + training loop
├── predict_v2.py        1–24h inference CLI (live or offline)
├── requirements.txt     Python dependencies
└── data/
    ├── india_stations.json    Auto-discovered station registry
    ├── dataset_v3.parquet     Training dataset (all India, 2022–now)
    ├── news_signals.parquet   Pre-computed historical news features
    ├── model_v2.keras         Trained model
    ├── scaler_X.pkl           Feature scaler (RobustScaler)
    ├── scaler_y.pkl           Target scaler (RobustScaler)
    ├── feature_columns.json   Ordered feature list
    ├── station_id_map.json    Station name → integer ID
    └── metrics.json           Per-horizon evaluation metrics
```

---

## Model Architecture

```
Input: (batch, 72h, ~60 features) + station_id

  Embedding(n_stations, 16)  →  station-specific bias
  Concatenate with features  →  (batch, 72, 76)

  BiLSTM(256, return_sequences=True)
  LayerNorm
  MultiHeadAttention(4 heads, key_dim=64)  [temporal]
  LayerNorm + residual
  LSTM(128)
  Dense(128, GELU) → BatchNorm
  Dense(64, GELU)
  Dense(24)  →  t+1h … t+24h PM2.5 predictions
```

**Loss**: Weighted Huber (exponential decay across horizons — near-term weighted more)  
**Optimizer**: AdamW + CosineDecay LR schedule  
**Lookback**: 72 hours | **Targets**: 24 hours | **Features**: ~60

---

## Features Used

| Category | Features |
|---|---|
| PM2.5 AR | pm25, lag1/3/6/12/24/48/72, rolling mean/max/min/std/trend |
| Co-pollutants | PM10, NO2, SO2, O3, CO (zero-filled if unavailable) |
| Weather | Temperature, Humidity, Dew Point, Pressure, Precipitation, Wind (u,v,speed), Cloud Cover, Solar Radiation, Boundary Layer Height |
| Fire (FIRMS) | Fire count, FRP sum/mean/max, Brightness Ti4/Ti5 |
| Traffic/Calendar | Rush hour, weekday, holiday, traffic index, day-of-week cyclics |
| Time | Hour sin/cos, month sin/cos |
| Seasonal | Monsoon, winter, harvest season, Diwali period, Holi period |
| News signals | Dust storm, industrial event, crop burning, calamity, fireworks, smog alert |

---

## Data Sources

| Source | Data | API |
|---|---|---|
| [OpenAQ v3](https://openaq.org) | PM2.5 + co-pollutants (hourly) | Free API key |
| [Open-Meteo](https://open-meteo.com) | ERA5 weather archive + forecast | No key needed |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) | VIIRS fire detections | Free API key |
| [NewsAPI](https://newsapi.org) | Environmental news signals | Free API key |

---

## Training Data

- **Range**: January 2022 → June 2026 (or as much as available per station)
- **Stations**: 100–300+ across India (auto-discovered from OpenAQ)
- **Coverage**: All states/UTs with at least one operational CPCB station

> **Note on Delhi**: OpenAQ may not have 2024 data for all Delhi stations. The system gracefully uses whatever data is available (minimum 200 hours per station).

---

## Prediction Output (JSON)

```json
{
  "station": "rk_puram_1234",
  "current_pm25": 87.3,
  "current_aqi": "Poor",
  "data_ts": "2026-06-28 12:00:00+00:00",
  "forecast": {
    "+1h":  { "pm25": 90.1, "aqi": "Poor", "delta": +2.8, "confidence": 0.90 },
    "+2h":  { "pm25": 91.4, "aqi": "Poor", "delta": +4.1, "confidence": 0.87 },
    ...
    "+24h": { "pm25": 78.2, "aqi": "Moderate", "delta": -9.1, "confidence": 0.60 }
  },
  "signals": {
    "news_crop_burning": 0.8,
    "is_harvest_season": 1.0
  },
  "generated_at": "2026-06-28T12:05:00+00:00"
}
```

---

## Target Metrics

| Horizon | Target MAE | Target R² |
|---|---|---|
| 1h | < 8 µg/m³ | > 0.80 |
| 6h | < 11 µg/m³ | > 0.72 |
| 12h | < 14 µg/m³ | > 0.65 |
| 24h | < 17 µg/m³ | > 0.58 |

---

## Quick Test (Small Dataset)

To verify everything works before a full run:
```bash
# Test with 3 cities, 1 epoch
python data_pipeline.py --build --cities Delhi Mumbai Chennai --start 2024-01-01 --end 2024-12-31
python news_fetcher.py --historical --start 2024-01-01 --end 2024-12-31
python train_model.py --quick
python predict_v2.py --station delhi
```
