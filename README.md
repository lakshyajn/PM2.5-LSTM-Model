# PM2.5 India — Nationwide Real-Time Forecasting System

Predicts PM2.5 concentration for **1–24 hours ahead** at 434 operational CPCB monitoring stations across India.
Covers all major states, integrates ERA5 weather, FIRMS fire data, traffic proxies, and news/event signals using a BiLSTM + Multi-Head Attention deep learning model.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# For GPU training (NVIDIA CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For Google Colab (GPU auto-detected):
pip install torch torchvision torchaudio

# 2. (Optional) Discover all India PM2.5 stations from OpenAQ
python data_pipeline.py --discover

# 3. Build training dataset using pre-screened stations (recommended — faster, 434 stations)
python data_pipeline.py --build --screened

# 4. Generate historical news/event signals
python news_fetcher.py --historical

# 5. Train the model on GPU
python train_torch.py --epochs 100 --batch 128

# 6. Resume training if interrupted
python train_torch.py --epochs 100 --batch 128          # auto-resumes from checkpoint

# 7. Live prediction (fetches real-time data)
python predict_v2.py --station delhi --live

# 8. Offline prediction (from stored data)
python predict_v2.py --station rk_puram
```

---

## Data Pipeline Flags

```bash
python data_pipeline.py --help

  --discover          Discover and cache India station registry from OpenAQ
  --build             Build the training dataset (PM2.5 + weather + fire)
  --screened          Use pre-screened stations (data/screened_stations_clean.json) — faster
  --force-refresh     Force re-discovery even if registry exists
  --cities X Y Z      Filter to specific cities (space-separated)
  --start YYYY-MM-DD  Data start date (default: 2025-02-18)
  --end   YYYY-MM-DD  Data end date   (default: today)
  --copollutants      Also fetch PM10, NO2, SO2, O3, CO per station
```

**Full rebuild** (434 stations, all data):
```bash
python data_pipeline.py --build --screened
```

**Quick test** (3 cities only):
```bash
python data_pipeline.py --build --cities Delhi Mumbai Chennai --start 2025-01-01
```

---

## File Structure

```
pm25_india/
├── config.py                 Central config — API keys, paths, hyperparameters
├── utils.py                  Feature engineering utilities (shared)
├── data_pipeline.py          Station discovery + historical data fetch
├── news_fetcher.py           NewsAPI signals + calendar event features
├── realtime_fetcher.py       Real-time 72h window assembly for live inference
├── train_torch.py            PyTorch GPU training (recommended)
├── train_model.py            TensorFlow training (legacy, CPU only on Windows)
├── predict_v2.py             1–24h inference CLI (live or offline)
├── filter_stations.py        Screen stations by data quality
├── clean_screened.py         Filter to productive sensor ID ranges
├── requirements.txt          Python dependencies
└── data/
    ├── screened_stations_clean.json   434 pre-screened stations (push to GitHub)
    ├── india_stations.json            Full station registry (~540 stations)
    ├── dataset_v3.parquet             Training dataset (~241 MB, NOT in GitHub)
    ├── news_signals.parquet           Pre-computed news features
    ├── model_torch_best.pt            Best PyTorch checkpoint (~8 MB)
    ├── scaler_X.pkl                   Feature scaler (RobustScaler)
    ├── scaler_y.pkl                   Target scaler (RobustScaler)
    ├── feature_columns.json           Ordered feature list (59 features)
    ├── station_id_map.json            Station name → integer ID (434 stations)
    └── metrics.json                   Per-horizon evaluation metrics
```

> **Note**: `dataset_v3.parquet` (241 MB) is excluded from GitHub. Rebuild it on the new machine with `python data_pipeline.py --build --screened` (takes ~2 hours with API rate limits).

---

## GPU Training

| GPU | Epoch time (300k seqs) | 50 epochs |
|---|---|---|
| GTX 1650 (no Tensor Cores) | ~15 min | ~13 hrs |
| Colab T4 (free) | ~4 min | ~3.5 hrs |
| RTX 3060 / 4060 | ~2.5 min | ~2 hrs |
| RTX 3090 / 4090 | ~1 min | ~50 min |
| Colab A100 (paid) | ~45s | ~40 min |

**Recommended batch sizes:**
- GTX 1650 (4 GB): `--batch 128`
- RTX 3060/3070 (8–12 GB): `--batch 256`
- A100 (40–80 GB): `--batch 512`

---

## Model Architecture

```
Input: (batch, 72h, 59 features) + station_id

  Embedding(434 stations, 16)  →  station-specific representation
  Concatenate with features    →  (batch, 72, 75)

  BiLSTM(256, bidirectional=True)  →  (batch, 72, 512)
  LayerNorm + Dropout(0.25)
  MultiHeadAttention(4 heads, key_dim=64)  [temporal self-attention]
  LayerNorm + residual
  LSTM(128)
  Dense(128, GELU) → BatchNorm
  Dense(64, GELU)
  Dense(24)  →  t+1h … t+24h PM2.5 predictions

Params: ~2.1M
Loss: Weighted Huber (exponential decay — near-term horizons weighted more)
Optimizer: AdamW + CosineAnnealing LR + AMP (mixed precision on GPU)
```

---

## Features (59 total)

| Category | Features |
|---|---|
| PM2.5 autoregressive | pm25, lag1/3/6/12/24/48/72, rolling mean/max/min/std/trend (14) |
| Co-pollutants | PM10, NO2, SO2, O3, CO (5) |
| Weather (ERA5) | Temperature, Humidity, Dew Point, Pressure, Precipitation, Wind u/v/speed, Cloud Cover, Solar Radiation, Boundary Layer Height (11) |
| Fire (NASA FIRMS) | Fire count, FRP sum/mean/max, Brightness Ti4/Ti5 (6) |
| Traffic/Calendar | Rush hour, weekday, holiday, traffic index, day-of-week sin/cos (6) |
| Cyclic time | Hour sin/cos, month sin/cos (4) |
| Seasonal/Events | Monsoon, winter, harvest season, Diwali, Holi (5) |
| News signals | Dust storm, industrial event, crop burning, calamity, fireworks, smog alert (6) |
| Location | lat, lon, aqi_category (3) |

---

## Data Sources

| Source | Data | API |
|---|---|---|
| [OpenAQ v3](https://openaq.org) | PM2.5 + co-pollutants (hourly) | Free key |
| [Open-Meteo](https://open-meteo.com) | ERA5 weather archive + forecast | No key needed |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) | VIIRS fire detections | Free key |
| [NewsAPI](https://newsapi.org) | Environmental news signals | Free key |

---

## Training Data

- **Range**: Feb 18, 2025 → present (continuous hourly)
- **Stations**: 434 across 29 cities in India
- **Total rows**: ~3.3 million
- **Train / Val / Test split**: 70% / 15% / 15% (chronological)

---

## Prediction Output (JSON)

```json
{
  "station": "rk_puram_1234",
  "current_pm25": 87.3,
  "current_aqi": "Poor",
  "forecast": {
    "+1h":  { "pm25": 90.1, "aqi": "Poor",     "delta": +2.8 },
    "+6h":  { "pm25": 95.2, "aqi": "Very Poor", "delta": +7.9 },
    "+24h": { "pm25": 78.2, "aqi": "Moderate",  "delta": -9.1 }
  },
  "signals": {
    "news_crop_burning": 0.8,
    "is_harvest_season": 1.0
  }
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
