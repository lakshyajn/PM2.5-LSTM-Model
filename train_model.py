"""
train_model.py
──────────────
Improved PM2.5 forecasting model for all-India, multi-station, 1–24h ahead.

Architecture (LSTM++ with Temporal Attention)
─────────────────────────────────────────────
  Inputs:
    features  : (batch, SEQ_LEN, n_features)   — scaled feature window
    station_id: (batch, 1)                      — integer station index

  Body:
    Embedding(n_stations+1, EMBED_DIM)          — station-specific bias
    RepeatVector(SEQ_LEN) → Concat with features
    BiLSTM(256, return_sequences=True) + LayerNorm
    MultiHeadAttention(4 heads, key_dim=64)  + residual + LayerNorm
    LSTM(128, return_sequences=False)
    Dense(128, gelu) → BatchNorm
    Dense(64,  gelu)
    Dense(24)                                   — t+1h … t+24h PM2.5

Loss:
    Weighted Huber — exponential decay over horizons (near-term weighted more)

Training improvements:
    AdamW with CosineDecay LR schedule
    EarlyStopping (patience=15) + ReduceLROnPlateau
    Gradient clipping (max_norm=1.0)

Usage
─────
  python train_model.py                  # train with defaults
  python train_model.py --epochs 150 --batch 64
  python train_model.py --quick          # fast test (1 epoch, 10 % data)
"""

import os, sys, json, pickle, argparse, ctypes
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Embedding,
    Reshape, RepeatVector, Concatenate,
    MultiHeadAttention, LayerNormalization, BatchNormalization,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_PARQUET, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH,
    FEAT_PATH, STATION_ID_PATH, METRICS_PATH, LOSS_PLOT, PRED_PLOT,
    FEATURE_COLS, N_HORIZONS, SEQ_LEN, EMBED_DIM,
    BATCH_SIZE, EPOCHS, TRAIN_FRAC, VAL_FRAC,
    LSTM1_UNITS, LSTM2_UNITS, DENSE1_UNITS, DENSE2_UNITS,
    N_ATTN_HEADS, ATTN_KEY_DIM, DROPOUT_RATE,
    HUBER_DELTA, HORIZON_WEIGHTS, MAX_SEQS_TRAIN,
)

# Force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tf.random.set_seed(42)
np.random.seed(42)


# ─── Windows Sleep Prevention ─────────────────────────────────────────────────

def _prevent_sleep():
    """
    Tell Windows not to sleep/hibernate/turn off display while training.
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    """
    try:
        ES_CONTINUOUS       = 0x80000000
        ES_SYSTEM_REQUIRED  = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("  [Sleep Prevention] Windows sleep blocked for this session.")
    except Exception as e:
        print(f"  [Sleep Prevention] Could not block sleep: {e}")


def _allow_sleep():
    """Restore normal Windows power management after training."""
    try:
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("  [Sleep Prevention] Windows sleep restored.")
    except Exception:
        pass

TARGET_COLS = [f"target_{h}h" for h in range(1, N_HORIZONS + 1)]


# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_data(parquet_path=DATASET_PARQUET):
    print(f"Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # Ensure all FEATURE_COLS exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [FILL] '{col}' missing → 0")
            df[col] = 0.0

    # Ensure all TARGET_COLS exist
    for col in TARGET_COLS:
        if col not in df.columns:
            print(f"  [FILL] target '{col}' missing → NaN (will drop)")
            df[col] = np.nan

    df.dropna(subset=TARGET_COLS + ["pm25"], inplace=True)

    print(
        f"  Rows: {len(df):,}  |  Stations: {df['station'].nunique()}  "
        f"|  Features: {len(FEATURE_COLS)}"
    )
    print(f"  Date: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


# ─── 2. Station ID Map ────────────────────────────────────────────────────────

def build_station_id_map(df):
    """Map station string → integer index (0-indexed, 0 reserved for unknown)."""
    stations = sorted(df["station"].unique())
    sid_map  = {s: i + 1 for i, s in enumerate(stations)}   # 1-based
    with open(STATION_ID_PATH, "w") as f:
        json.dump(sid_map, f, indent=2)
    print(f"  Station map: {len(sid_map)} stations → {STATION_ID_PATH}")
    return sid_map


# ─── 3. Chronological Split ───────────────────────────────────────────────────

def chrono_split_df(df):
    """
    Sort by timestamp, split 70/15/15 by global time (not per-station).
    Returns three DataFrames.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))

    df_tr = df.iloc[:t1].copy()
    df_va = df.iloc[t1:t2].copy()
    df_te = df.iloc[t2:].copy()

    print(
        f"  Train: {len(df_tr):,} rows  |  Val: {len(df_va):,}  |  Test: {len(df_te):,}"
    )
    return df_tr, df_va, df_te


# ─── 4. Scalers ───────────────────────────────────────────────────────────────

def fit_scalers(df_tr, feature_cols):
    """
    Fit RobustScaler on training features and PM2.5 targets.
    RobustScaler (median/IQR) is more resilient to extreme PM2.5 spikes.
    """
    # Feature scaler: fit on a sample to handle large datasets
    sample_size = min(len(df_tr), 200_000)
    sample = df_tr[feature_cols].values[:sample_size].astype(np.float32)

    scaler_X = RobustScaler()
    scaler_X.fit(sample)

    # Target scaler: fit on all training PM2.5 values (all horizons pooled)
    all_target_vals = df_tr[TARGET_COLS].values.reshape(-1, 1).astype(np.float32)
    scaler_y = RobustScaler()
    scaler_y.fit(all_target_vals)

    # Save
    with open(SCALER_X_PATH, "wb") as f:
        pickle.dump(scaler_X, f)
    with open(SCALER_Y_PATH, "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"  Scalers saved: {SCALER_X_PATH}, {SCALER_Y_PATH}")
    return scaler_X, scaler_y


# ─── 5. Sequence Generation (memory-efficient generator) ─────────────────────

def _sequence_generator(df, station_id_map, scaler_X, scaler_y,
                         feature_cols, seq_len=SEQ_LEN, n_horizons=N_HORIZONS,
                         max_seqs=0, shuffle=True, seed=42):
    """
    Generator that yields (features, station_id), target one sequence at a time.
    Never materializes the full array — memory usage is O(seq_len * n_features).
    """
    rng = np.random.default_rng(seed)

    stations = sorted(df["station"].unique())
    per_station = max(1, max_seqs // len(stations)) if max_seqs > 0 else 0

    # Build index list first (light: just integers)
    all_indices = []  # (station_idx, row_i)
    station_arrays = {}  # cache scaled arrays per station

    for station in stations:
        sdf = df[df["station"] == station].sort_values("timestamp").reset_index(drop=True)
        sid = station_id_map.get(station, 0)
        n   = len(sdf)
        if n < seq_len + n_horizons:
            continue

        feat_s = scaler_X.transform(sdf[feature_cols].values.astype(np.float32))
        tgt_s  = scaler_y.transform(
            sdf[TARGET_COLS].values.astype(np.float32).reshape(-1, 1)
        ).reshape(n, n_horizons)

        station_arrays[station] = (feat_s, tgt_s, sid)

        valid = np.arange(seq_len, n - n_horizons + 1)
        if per_station > 0 and len(valid) > per_station:
            valid = rng.choice(valid, size=per_station, replace=False)

        for i in valid:
            all_indices.append((station, int(i)))

    if shuffle:
        rng.shuffle(all_indices)

    for station, i in all_indices:
        feat_s, tgt_s, sid = station_arrays[station]
        X_seq = feat_s[i - seq_len : i]          # (seq_len, n_features)
        y_seq = tgt_s[i]                           # (n_horizons,)
        yield (X_seq, np.array([sid], dtype=np.int32)), y_seq


def make_tf_dataset(df, station_id_map, scaler_X, scaler_y,
                    feature_cols, batch_size=BATCH_SIZE,
                    seq_len=SEQ_LEN, n_horizons=N_HORIZONS,
                    max_seqs=0, shuffle=True, seed=42):
    """
    Wraps _sequence_generator into a batched tf.data.Dataset.
    Memory usage: O(batch_size * seq_len * n_features) instead of O(N_total).
    """
    n_features = len(feature_cols)

    output_sig = (
        (
            tf.TensorSpec(shape=(seq_len, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(1,),                  dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(n_horizons,), dtype=tf.float32),
    )

    gen = lambda: _sequence_generator(
        df, station_id_map, scaler_X, scaler_y,
        feature_cols, seq_len, n_horizons, max_seqs, shuffle, seed,
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    ds = ds.batch(batch_size).repeat()  # loop so steps_per_epoch controls epoch end
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# Legacy wrapper kept for val/test eval (smaller sets, safe to materialize)
def make_sequences(df, station_id_map, scaler_X, scaler_y,
                   feature_cols, seq_len=SEQ_LEN, n_horizons=N_HORIZONS,
                   max_seqs=0, shuffle=True, seed=42):
    """
    Materializes sequences into numpy arrays. Only use for val/test
    (small enough to fit in RAM after applying max_seqs cap).
    """
    rng = np.random.default_rng(seed)
    X_list, sid_list, y_list = [], [], []
    stations = sorted(df["station"].unique())
    per_station = max(1, max_seqs // len(stations)) if max_seqs > 0 else 0

    for station in stations:
        sdf = df[df["station"] == station].sort_values("timestamp").reset_index(drop=True)
        sid = station_id_map.get(station, 0)
        n   = len(sdf)
        if n < seq_len + n_horizons:
            continue

        feat_s = scaler_X.transform(sdf[feature_cols].values.astype(np.float32))
        tgt_s  = scaler_y.transform(
            sdf[TARGET_COLS].values.astype(np.float32).reshape(-1, 1)
        ).reshape(n, n_horizons)

        valid = np.arange(seq_len, n - n_horizons + 1)
        if per_station > 0 and len(valid) > per_station:
            valid = rng.choice(valid, size=per_station, replace=False)

        for i in valid:
            X_list.append(feat_s[i - seq_len : i])
            y_list.append(tgt_s[i])
            sid_list.append(sid)

    X    = np.array(X_list,   dtype=np.float32)
    sids = np.array(sid_list, dtype=np.int32).reshape(-1, 1)
    y    = np.array(y_list,   dtype=np.float32)

    if shuffle:
        perm = rng.permutation(len(X))
        X, sids, y = X[perm], sids[perm], y[perm]

    return X, sids, y



# ─── 6. Model Architecture ────────────────────────────────────────────────────

def build_model(n_features: int, n_stations: int):
    """
    Bidirectional LSTM + Multi-Head Attention + station embedding.

    Inputs  : [features (B, SEQ_LEN, n_features), station_id (B, 1)]
    Outputs : pm25_forecast (B, N_HORIZONS) — scaled PM2.5 for t+1 … t+24
    """
    # ── Inputs
    feat_in = Input(shape=(SEQ_LEN, n_features), name="features")
    stn_in  = Input(shape=(1,), dtype="int32",   name="station_id")

    # ── Station embedding
    emb = Embedding(n_stations + 2, EMBED_DIM, name="station_embed")(stn_in)
    emb = Reshape((EMBED_DIM,))(emb)                    # (B, EMBED_DIM)
    emb = RepeatVector(SEQ_LEN)(emb)                    # (B, SEQ_LEN, EMBED_DIM)

    # ── Concatenate
    x = Concatenate(axis=-1, name="concat")([feat_in, emb])  # (B, SEQ_LEN, n+EMBED_DIM)

    # ── Bidirectional LSTM
    x = Bidirectional(
        LSTM(LSTM1_UNITS, return_sequences=True),
        name="bilstm_1",
    )(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = LayerNormalization(name="ln_1")(x)

    # ── Multi-head temporal attention (residual)
    attn_out, _ = MultiHeadAttention(
        num_heads=N_ATTN_HEADS, key_dim=ATTN_KEY_DIM, name="temporal_attn"
    )(x, x, return_attention_scores=True)
    x = LayerNormalization(name="ln_2")(x + attn_out)

    # ── Second LSTM (decode)
    x = LSTM(LSTM2_UNITS, return_sequences=False, name="lstm_2")(x)
    x = Dropout(DROPOUT_RATE)(x)

    # ── Dense head
    x = Dense(DENSE1_UNITS, activation="gelu", name="dense_1")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dense(DENSE2_UNITS, activation="gelu", name="dense_2")(x)

    # ── 24-horizon output
    out = Dense(N_HORIZONS, name="forecast")(x)

    model = Model(inputs=[feat_in, stn_in], outputs=out, name="pm25_lstm_attn")
    return model


# ─── 7. Custom Weighted Huber Loss ────────────────────────────────────────────

def make_weighted_huber_loss(weights=HORIZON_WEIGHTS, delta=HUBER_DELTA):
    """
    Returns a Keras loss function that applies per-horizon weighting.
    Near-term horizons are weighted more (exponential decay).
    """
    hw = tf.constant(weights, dtype=tf.float32)  # (N_HORIZONS,)

    @tf.function
    def weighted_huber(y_true, y_pred):
        err    = y_true - y_pred
        abs_e  = tf.abs(err)
        quad   = tf.minimum(abs_e, delta)
        lin    = abs_e - quad
        huber  = 0.5 * tf.square(quad) + delta * lin     # (B, N_HORIZONS)
        return tf.reduce_mean(huber * hw)

    weighted_huber.__name__ = "weighted_huber"
    return weighted_huber


# ─── 8. Metrics ───────────────────────────────────────────────────────────────

def compute_horizon_metrics(y_true, y_pred, scaler_y, label=""):
    """
    Compute MAE, RMSE, R² per horizon (original µg/m³ space).
    Also returns aggregate 1h and 24h metrics.
    """
    # Inverse transform: shape (N, N_HORIZONS)
    N, H = y_true.shape
    y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(N, H)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(N, H)

    metrics = {}
    for h in range(H):
        yt = y_true_inv[:, h]
        yp = y_pred_inv[:, h]
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = float(r2_score(yt, yp))
        metrics[f"t+{h+1}h"] = {"mae": round(mae, 3), "rmse": round(rmse, 3),
                                  "r2":  round(r2, 4)}

    # Print key horizons
    h_print = [0, 2, 5, 11, 23]   # +1h, +3h, +6h, +12h, +24h
    print(f"\n  {label} Test Metrics (selected horizons):")
    print(f"  {'Horizon':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    for h in h_print:
        m = metrics[f"t+{h+1}h"]
        print(f"  t+{h+1:>2}h      {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['r2']:>8.4f}")

    return metrics, y_true_inv, y_pred_inv


# ─── 9. Plots ─────────────────────────────────────────────────────────────────

def plot_loss(history):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train", lw=1.5)
    ax.plot(history.history["val_loss"], label="Val",   lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted Huber Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(LOSS_PLOT, dpi=120); plt.close(fig)
    print(f"  Loss plot → {LOSS_PLOT}")


def plot_predictions(y_true_inv, y_pred_inv, n=500):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    idx = range(min(n, len(y_true_inv)))

    axes[0].plot(idx, y_true_inv[:n, 0],  label="Actual",     alpha=0.8, lw=1)
    axes[0].plot(idx, y_pred_inv[:n, 0],  label="Pred +1h",   alpha=0.8, lw=1, ls="--")
    axes[0].set_ylabel("PM2.5 (µg/m³)"); axes[0].legend()
    axes[0].set_title("+1h Forecast — Test Set")

    axes[1].plot(idx, y_true_inv[:n, 23], label="Actual",     alpha=0.8, lw=1)
    axes[1].plot(idx, y_pred_inv[:n, 23], label="Pred +24h",  alpha=0.8, lw=1, ls="--")
    axes[1].set_ylabel("PM2.5 (µg/m³)"); axes[1].set_xlabel("Sample")
    axes[1].legend(); axes[1].set_title("+24h Forecast — Test Set")

    fig.tight_layout()
    fig.savefig(PRED_PLOT, dpi=120); plt.close(fig)
    print(f"  Prediction plot → {PRED_PLOT}")


def plot_horizon_r2(metrics: dict):
    """Plot R² across all 24 horizons."""
    horizons = list(range(1, N_HORIZONS + 1))
    r2_vals  = [metrics[f"t+{h}h"]["r2"] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(horizons, r2_vals, "o-", lw=2, color="#3a86ff")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Forecast Horizon (hours ahead)")
    ax.set_ylabel("R²")
    ax.set_title("R² Score vs Forecast Horizon (Test Set)")
    ax.set_ylim(min(r2_vals) - 0.05, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = LOSS_PLOT.replace("loss_curve", "horizon_r2")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  Horizon R² plot → {out}")


# ─── 10. Main Training Loop ───────────────────────────────────────────────────

def train(epochs=EPOCHS, batch_size=BATCH_SIZE, quick=False, resume=True):
    print("=" * 65)
    print("  PM2.5 India — LSTM++ Training")
    print("=" * 65)

    # ── Block Windows sleep immediately
    _prevent_sleep()

    # ── Load
    df = load_data()
    feature_cols = FEATURE_COLS
    n_features   = len(feature_cols)

    # ── Station map
    print("\nBuilding station ID map …")
    station_id_map = build_station_id_map(df)
    n_stations = len(station_id_map)

    # ── Split
    print("\nSplitting data (chronological) …")
    df_tr, df_va, df_te = chrono_split_df(df)

    # ── Scalers
    print("\nFitting scalers …")
    scaler_X, scaler_y = fit_scalers(df_tr, feature_cols)

    # ── Generate sequences
    max_tr = MAX_SEQS_TRAIN if not quick else 5_000
    max_ev = 50_000          if not quick else 2_000

    # Training: memory-efficient tf.data generator (never allocates full array)
    print("\nBuilding training tf.data pipeline (memory-efficient) …")
    ds_tr = make_tf_dataset(
        df_tr, station_id_map, scaler_X, scaler_y,
        feature_cols, batch_size=batch_size, max_seqs=max_tr, shuffle=True,
    )
    n_tr_seqs = min(max_tr, int(sum(
        max(0, len(df_tr[df_tr['station'] == s]) - SEQ_LEN - N_HORIZONS + 1)
        for s in df_tr['station'].unique()
    )))
    steps_per_epoch = max(1, n_tr_seqs // batch_size)
    print(f"  Est. sequences: {n_tr_seqs:,}  |  Steps/epoch: {steps_per_epoch:,}")

    # Val/test: capped at 50k so safe to materialize
    print("Generating validation sequences …")
    X_va, sid_va, y_va = make_sequences(
        df_va, station_id_map, scaler_X, scaler_y,
        feature_cols, max_seqs=max_ev, shuffle=False,
    )
    print(f"  X_va: {X_va.shape}")

    print("Generating test sequences …")
    X_te, sid_te, y_te = make_sequences(
        df_te, station_id_map, scaler_X, scaler_y,
        feature_cols, max_seqs=max_ev, shuffle=False,
    )
    print(f"  X_te: {X_te.shape}")

    # ── Build or resume model
    ckpt_path = MODEL_PATH.replace(".keras", "_best.keras")
    if resume and os.path.exists(ckpt_path):
        print(f"\n[RESUME] Loading checkpoint: {ckpt_path}")
        model = tf.keras.models.load_model(
            ckpt_path,
            custom_objects={"weighted_huber": make_weighted_huber_loss()},
            compile=False,
        )
        print("  Checkpoint loaded successfully.")
        model.summary()
    else:
        print(f"\nBuilding model … ({n_features} features, {n_stations} stations)")
        model = build_model(n_features, n_stations)
        model.summary()

    # ── Compile
    n_steps  = steps_per_epoch * epochs
    # When resuming, use a lower initial LR (warm restart from checkpoint)
    init_lr = 3e-4 if (resume and os.path.exists(ckpt_path)) else 1e-3
    lr_sched = CosineDecay(
        initial_learning_rate=init_lr,
        decay_steps=max(1, n_steps),
        alpha=1e-5,
    )
    loss_fn = make_weighted_huber_loss()

    model.compile(
        optimizer=AdamW(learning_rate=lr_sched, weight_decay=1e-4,
                        clipnorm=1.0),
        loss=loss_fn,
        metrics=["mae"],
    )

    # ── Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    # ── Train
    print(f"\nTraining … (epochs={epochs}, batch={batch_size}, steps/epoch~{steps_per_epoch:,})")
    history = model.fit(
        ds_tr,                                       # tf.data generator (no RAM spike)
        validation_data=([X_va, sid_va], y_va),
        epochs=1 if quick else epochs,
        steps_per_epoch=steps_per_epoch if not quick else None,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Restore sleep on training end
    _allow_sleep()

    # ── Evaluate
    print("\nEvaluating on test set …")
    y_pred_te = model.predict([X_te, sid_te], verbose=0, batch_size=512)
    metrics, y_true_inv, y_pred_inv = compute_horizon_metrics(
        y_te, y_pred_te, scaler_y, label="India"
    )

    # ── Save model
    model.save(MODEL_PATH)
    print(f"\nModel → {MODEL_PATH}")

    # Save feature columns
    with open(FEAT_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Features → {FEAT_PATH}")

    # Save full metrics
    summary = {
        "architecture": "BiLSTM + MultiHeadAttention + StationEmbed",
        "n_features":   n_features,
        "n_stations":   n_stations,
        "seq_len":      SEQ_LEN,
        "n_horizons":   N_HORIZONS,
        "resumed_from_checkpoint": resume and os.path.exists(ckpt_path),
        "val_seqs":     int(len(X_va)),
        "test_seqs":    int(len(X_te)),
        "epochs_run":   int(len(history.history["loss"])),
        "1h":  metrics["t+1h"],
        "6h":  metrics["t+6h"],
        "12h": metrics["t+12h"],
        "24h": metrics["t+24h"],
        "all_horizons": metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics → {METRICS_PATH}")

    # ── Plots
    plot_loss(history)
    plot_predictions(y_true_inv, y_pred_inv)
    plot_horizon_r2(metrics)

    print("\n" + "=" * 65)
    print("  Training complete.")
    print(f"  1h   → MAE={metrics['t+1h']['mae']:.2f}  R²={metrics['t+1h']['r2']:.4f}")
    print(f"  24h  → MAE={metrics['t+24h']['mae']:.2f}  R²={metrics['t+24h']['r2']:.4f}")
    print("=" * 65)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PM2.5 India Model Training")
    parser.add_argument("--epochs",    type=int,  default=EPOCHS)
    parser.add_argument("--batch",     type=int,  default=BATCH_SIZE)
    parser.add_argument("--quick",     action="store_true",
                        help="Quick smoke-test: 1 epoch, 5K sequences")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint and train from scratch")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch, quick=args.quick,
          resume=not args.no_resume)


if __name__ == "__main__":
    main()
