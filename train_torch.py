"""
train_torch.py
──────────────
PyTorch GPU training for PM2.5 India — identical architecture to TF version.
Runs on GTX 1650 via CUDA 12.1 (~10-15x faster than CPU TF).

Architecture (matches TF train_model.py exactly):
  BiLSTM(256) + LayerNorm
  MultiHeadAttention(4 heads, key_dim=64) + residual + LayerNorm
  LSTM(128)
  Dense(128, GELU) -> BatchNorm
  Dense(64, GELU)
  Dense(24)  -> t+1h ... t+24h PM2.5

Usage:
  python train_torch.py                    # train (auto-resume if checkpoint exists)
  python train_torch.py --epochs 100 --batch 128
  python train_torch.py --no-resume        # train from scratch
  python train_torch.py --quick            # smoke test
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_PARQUET, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH,
    FEAT_PATH, STATION_ID_PATH, METRICS_PATH,
    FEATURE_COLS, N_HORIZONS, SEQ_LEN, EMBED_DIM,
    BATCH_SIZE, EPOCHS, TRAIN_FRAC, VAL_FRAC,
    LSTM1_UNITS, LSTM2_UNITS, DENSE1_UNITS, DENSE2_UNITS,
    N_ATTN_HEADS, ATTN_KEY_DIM, DROPOUT_RATE, MAX_SEQS_TRAIN,
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_COLS  = [f"target_{h}h" for h in range(1, N_HORIZONS + 1)]
DATA_DIR     = os.path.dirname(DATASET_PARQUET)
TORCH_CKPT   = os.path.join(DATA_DIR, "model_torch_best.pt")
TORCH_FINAL  = os.path.join(DATA_DIR, "model_torch_final.pt")


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {name}  ({mem:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print("  No GPU found — using CPU")
    return dev


def auto_max_seqs(device, seq_len=SEQ_LEN, n_features=len(FEATURE_COLS),
                  n_splits=3, headroom_mb=600):
    """
    Compute how many sequences fit in GPU VRAM.

    Formula:
      usable_vram = total_vram - headroom (model + grads + optimizer + activations)
      bytes_per_seq = seq_len * n_features * 2  (float16)
      n_splits = 3  (train + val + test all live in VRAM simultaneously)
      max_seqs = usable_vram / (bytes_per_seq * n_splits)

    GPU          VRAM    headroom   -> max_seqs
    GTX 1650     4 GB    600 MB     -> ~143k
    RTX A2000    6 GB    600 MB     -> ~227k
    RTX 3060    12 GB    600 MB     -> ~481k
    RTX 3090    24 GB    600 MB     -> ~980k
    A100        40 GB    600 MB     -> ~1.65M (capped at actual data)
    """
    if device.type != "cuda":
        return MAX_SEQS_TRAIN  # CPU: use config value

    total_bytes  = torch.cuda.get_device_properties(0).total_memory
    usable_bytes = total_bytes - headroom_mb * 1024**2
    bytes_per_seq = seq_len * n_features * 2  # float16
    max_seqs = int(usable_bytes / (bytes_per_seq * n_splits))

    # Floor to nearest 10k for clean numbers
    max_seqs = (max_seqs // 10_000) * 10_000
    max_seqs = max(10_000, max_seqs)  # minimum 10k

    total_vram_gb = total_bytes / 1024**3
    print(f"  Auto VRAM budget: {total_vram_gb:.1f} GB total, "
          f"{headroom_mb} MB reserved  ->  max {max_seqs:,} seqs/split")
    return max_seqs


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_gpu_dataset(df, station_id_map, scaler_X, scaler_y,
                      feature_cols, device, seq_len=SEQ_LEN, n_horizons=N_HORIZONS,
                      max_seqs=0, shuffle=True, seed=42):
    """
    Builds all sequences into pre-allocated numpy arrays (fast, no list copies),
    then moves them to GPU VRAM as float16.
    Training loop: pure GPU VRAM -> GPU cores, zero CPU per batch.
    """
    rng      = np.random.default_rng(seed)
    stations = sorted(df["station"].unique())
    per_station = max(1, max_seqs // len(stations)) if max_seqs > 0 else 0

    # ── Pass 1: count total sequences (fast, no data copy) ───────────────────
    total = 0
    counts = {}   # station -> number of valid indices chosen
    chosen = {}   # station -> chosen indices array
    for station in stations:
        sdf = df[df["station"] == station]
        n   = len(sdf)
        if n < seq_len + n_horizons:
            continue
        valid = np.arange(seq_len, n - n_horizons + 1)
        if per_station > 0 and len(valid) > per_station:
            valid = rng.choice(valid, size=per_station, replace=False)
        chosen[station] = valid
        counts[station] = len(valid)
        total += len(valid)

    n_seqs = total
    # ── Pre-allocate arrays (one allocation, no doubling) ────────────────────
    X_arr   = np.empty((n_seqs, seq_len, len(feature_cols)), dtype=np.float32)
    y_arr   = np.empty((n_seqs, n_horizons),                 dtype=np.float32)
    sid_arr = np.empty(n_seqs,                               dtype=np.int64)

    # ── Pass 2: fill arrays ───────────────────────────────────────────────────
    ptr = 0
    for station in stations:
        if station not in chosen:
            continue
        sdf = (df[df["station"] == station]
               .sort_values("timestamp")
               .reset_index(drop=True))
        sid = station_id_map.get(station, 0)
        n   = len(sdf)

        feat_s = scaler_X.transform(sdf[feature_cols].values.astype(np.float32))
        tgt_s  = scaler_y.transform(
            sdf[TARGET_COLS].values.astype(np.float32).reshape(-1, 1)
        ).reshape(n, n_horizons)

        for i in chosen[station]:
            X_arr[ptr]   = feat_s[i - seq_len : i]
            y_arr[ptr]   = tgt_s[i]
            sid_arr[ptr] = sid
            ptr += 1

    # ── In-place shuffle (numpy, not Python lists) ───────────────────────────
    if shuffle:
        perm = rng.permutation(n_seqs)
        X_arr   = X_arr[perm]
        y_arr   = y_arr[perm]
        sid_arr = sid_arr[perm]

    vram_mb = n_seqs * seq_len * len(feature_cols) * 2 / 1024**2  # float16

    if device.type == "cuda":
        # One-shot CPU->GPU transfer as float16 — no per-batch transfer ever again
        X_gpu   = torch.from_numpy(X_arr).half().to(device)
        y_gpu   = torch.from_numpy(y_arr).to(device)
        sid_gpu = torch.from_numpy(sid_arr).to(device)
        del X_arr, y_arr, sid_arr  # free CPU RAM immediately
        print(f"    {n_seqs:,} sequences -> GPU VRAM as float16  ({vram_mb:.0f} MB used)")
        return torch.utils.data.TensorDataset(X_gpu, sid_gpu, y_gpu)
    else:
        print(f"    {n_seqs:,} sequences -> CPU RAM as float32")
        return torch.utils.data.TensorDataset(
            torch.from_numpy(X_arr), torch.from_numpy(sid_arr), torch.from_numpy(y_arr))




# ─── Model ────────────────────────────────────────────────────────────────────

class PM25Model(nn.Module):
    """BiLSTM + Multi-Head Attention + Station Embedding (matches TF version)."""
    def __init__(self, n_features, n_stations,
                 embed_dim=EMBED_DIM,
                 lstm1=LSTM1_UNITS, lstm2=LSTM2_UNITS,
                 d1=DENSE1_UNITS, d2=DENSE2_UNITS,
                 n_heads=N_ATTN_HEADS, dropout=DROPOUT_RATE,
                 n_horizons=N_HORIZONS):
        super().__init__()
        self.embed  = nn.Embedding(n_stations + 1, embed_dim)
        self.bilstm = nn.LSTM(n_features + embed_dim, lstm1,
                              batch_first=True, bidirectional=True)
        self.norm1  = nn.LayerNorm(lstm1 * 2)
        self.drop1  = nn.Dropout(dropout)
        self.attn   = nn.MultiheadAttention(lstm1 * 2, n_heads,
                                             dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(lstm1 * 2)
        self.lstm2  = nn.LSTM(lstm1 * 2, lstm2, batch_first=True)
        self.drop2  = nn.Dropout(dropout)
        self.fc1    = nn.Linear(lstm2, d1)
        self.bn1    = nn.BatchNorm1d(d1)
        self.act1   = nn.GELU()
        self.drop3  = nn.Dropout(dropout)
        self.fc2    = nn.Linear(d1, d2)
        self.act2   = nn.GELU()
        self.out    = nn.Linear(d2, n_horizons)

    def forward(self, x, station_ids):
        # x: (B, T, F),  station_ids: (B,)
        e   = self.embed(station_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        x   = torch.cat([x, e], dim=-1)          # (B, T, F+embed)
        x, _ = self.bilstm(x)
        x   = self.drop1(self.norm1(x))
        a, _ = self.attn(x, x, x)
        x   = self.norm2(x + a)
        x, _ = self.lstm2(x)
        x   = self.drop2(x[:, -1, :])            # last timestep
        x   = self.drop3(self.act1(self.bn1(self.fc1(x))))
        x   = self.act2(self.fc2(x))
        return self.out(x)


# ─── Loss ─────────────────────────────────────────────────────────────────────

class WeightedHuberLoss(nn.Module):
    def __init__(self, n_horizons=N_HORIZONS, delta=1.0, decay=0.95):
        super().__init__()
        w = torch.tensor([decay**(h-1) for h in range(1, n_horizons+1)],
                         dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())
        self.delta = delta

    def forward(self, pred, target):
        err  = pred - target
        loss = torch.where(err.abs() < self.delta,
                           0.5 * err**2,
                           self.delta * (err.abs() - 0.5 * self.delta))
        return (loss * self.weights).mean()


# ─── Data Helpers ─────────────────────────────────────────────────────────────

def load_data():
    print(f"Loading {DATASET_PARQUET} ...")
    df = pd.read_parquet(DATASET_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station", "timestamp"])
    print(f"  Rows: {len(df):,}  Stations: {df['station'].nunique()}  Features: {len(FEATURE_COLS)}")
    return df


def chrono_split(df):
    df = df.sort_values("timestamp")
    n  = len(df)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    df_tr, df_va, df_te = df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]
    print(f"  Train: {len(df_tr):,}  Val: {len(df_va):,}  Test: {len(df_te):,}")
    return df_tr, df_va, df_te


def fit_scalers(df_tr):
    sample = (df_tr[FEATURE_COLS].dropna()
              .sample(min(200_000, len(df_tr)), random_state=42)
              .values.astype(np.float32))
    scaler_X = RobustScaler().fit(sample)
    scaler_y = RobustScaler().fit(
        df_tr[TARGET_COLS].values.reshape(-1, 1).astype(np.float32))
    with open(SCALER_X_PATH, "wb") as f: pickle.dump(scaler_X, f)
    with open(SCALER_Y_PATH, "wb") as f: pickle.dump(scaler_y, f)
    print("  Scalers saved.")
    return scaler_X, scaler_y


def build_station_map(df):
    stations = sorted(df["station"].unique())
    smap = {s: i for i, s in enumerate(stations)}
    with open(STATION_ID_PATH, "w") as f: json.dump(smap, f, indent=2)
    print(f"  Station map: {len(smap)} stations")
    return smap


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true_sc, y_pred_sc, scaler_y):
    n = y_true_sc.shape[0]
    y_true_inv = scaler_y.inverse_transform(
        y_true_sc.reshape(-1, 1)).reshape(n, N_HORIZONS)
    y_pred_inv = scaler_y.inverse_transform(
        y_pred_sc.reshape(-1, 1)).reshape(n, N_HORIZONS)
    metrics = {}
    print(f"\n  {'Horizon':<8} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*36}")
    for h in [1, 3, 6, 12, 24]:
        mae  = mean_absolute_error(y_true_inv[:, h-1], y_pred_inv[:, h-1])
        rmse = mean_squared_error(y_true_inv[:, h-1], y_pred_inv[:, h-1])**0.5
        r2   = r2_score(y_true_inv[:, h-1], y_pred_inv[:, h-1])
        metrics[f"t+{h}h"] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  t+{h}h     {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f}")
    with open(METRICS_PATH, "w") as f: json.dump(metrics, f, indent=2)
    return metrics, y_true_inv, y_pred_inv


# ─── Epoch Loops ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, amp_scaler, epoch, epochs):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=f"  Ep {epoch:>3}/{epochs} [train]",
               ncols=80, leave=False, unit="step")
    for X, sids, y in bar:
        # Data already on GPU VRAM — just cast float16 -> float32 for model (GPU-only op)
        if X.dtype == torch.float16:
            X = X.float()
        optimizer.zero_grad(set_to_none=True)
        if amp_scaler:
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(X, sids), y)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss = loss_fn(model(X, sids), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item(); n += 1
        bar.set_postfix(loss=f"{total/n:.4f}")
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, n   = 0.0, 0
    yt_list, yp_list = [], []
    bar = tqdm(loader, desc="               [val]  ", ncols=80, leave=False, unit="step")
    for X, sids, y in bar:
        # Data already on GPU VRAM
        if X.dtype == torch.float16:
            X = X.float()
        p = model(X, sids)
        total += loss_fn(p, y).item(); n += 1
        yt_list.append(y.cpu().numpy())
        yp_list.append(p.cpu().numpy())
        bar.set_postfix(loss=f"{total/n:.4f}")
    return total / max(1, n), np.concatenate(yt_list), np.concatenate(yp_list)


# ─── Main Train ───────────────────────────────────────────────────────────────

def train(epochs=EPOCHS, batch_size=BATCH_SIZE, quick=False, resume=True):
    print("=" * 65)
    print("  PM2.5 India -- PyTorch GPU Training")
    print("=" * 65)
    device  = get_device()
    use_amp = device.type == "cuda"

    df          = load_data()
    station_map = build_station_map(df)
    n_stations  = len(station_map)

    print("\nSplitting ...")
    df_tr, df_va, df_te = chrono_split(df)

    if resume and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
        print("\nLoading existing scalers ...")
        with open(SCALER_X_PATH, "rb") as f: scaler_X = pickle.load(f)
        with open(SCALER_Y_PATH, "rb") as f: scaler_y = pickle.load(f)
    else:
        print("\nFitting scalers ...")
        scaler_X, scaler_y = fit_scalers(df_tr)

    with open(FEAT_PATH, "w") as f: json.dump(FEATURE_COLS, f, indent=2)

    if quick:
        max_tr, max_ev = 5_000, 2_000
    elif MAX_SEQS_TRAIN > 0:
        # Config override: user explicitly set a cap
        max_tr = MAX_SEQS_TRAIN
        max_ev = min(50_000, MAX_SEQS_TRAIN // 2)
    else:
        # Auto-scale: fill GPU VRAM as much as possible
        max_tr = auto_max_seqs(device)
        max_ev = min(50_000, max_tr // 3)

    print("\nBuilding datasets (GPU-resident for pure GPU training) ...")
    ds_tr = build_gpu_dataset(df_tr, station_map, scaler_X, scaler_y,
                               FEATURE_COLS, device, max_seqs=max_tr)
    ds_va = build_gpu_dataset(df_va, station_map, scaler_X, scaler_y,
                               FEATURE_COLS, device, max_seqs=max_ev, shuffle=False)
    ds_te = build_gpu_dataset(df_te, station_map, scaler_X, scaler_y,
                               FEATURE_COLS, device, max_seqs=max_ev, shuffle=False)

    # DataLoader just shuffles indices into GPU tensors — no CPU work per batch
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=512,        shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=512,        shuffle=False, num_workers=0)

    model     = PM25Model(len(FEATURE_COLS), n_stations).to(device)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0
    PATIENCE       = 15

    if resume and os.path.exists(TORCH_CKPT):
        print(f"\n[RESUME] Loading: {TORCH_CKPT}")
        ckpt = torch.load(TORCH_CKPT, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_epoch    = ckpt.get("epoch", 0)
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        patience_count = ckpt.get("patience_count", 0)
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val_loss:.5f}, patience={patience_count}")

    optimizer  = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=max(1, epochs - start_epoch), eta_min=1e-6)
    loss_fn    = WeightedHuberLoss().to(device)
    amp_scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if quick: epochs = start_epoch + 1

    print(f"\nTraining on {device} ... (epochs={epochs}, batch={batch_size}, AMP={use_amp})")
    print("=" * 65)

    for epoch in range(start_epoch, epochs):
        t0      = time.time()
        tr_loss = train_epoch(model, dl_tr, optimizer, loss_fn, device, amp_scaler,
                              epoch + 1, epochs)
        va_loss, _, _ = eval_epoch(model, dl_va, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1:>3}/{epochs}  "
              f"tr={tr_loss:.4f}  val={va_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"time={elapsed:.0f}s")

        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            patience_count = 0
            torch.save({"model": model.state_dict(), "epoch": epoch+1,
                        "best_val_loss": best_val_loss,
                        "patience_count": patience_count,
                        "n_features": len(FEATURE_COLS),
                        "n_stations": n_stations}, TORCH_CKPT)
            print(f"    -> Best checkpoint saved (val={best_val_loss:.5f})")
        else:
            patience_count += 1
            print(f"    patience {patience_count}/{PATIENCE}")
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping.")
                break

    # Final evaluation
    print("\nLoading best checkpoint ...")
    ckpt = torch.load(TORCH_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    print("Evaluating on test set ...")
    _, y_te_true, y_te_pred = eval_epoch(model, dl_te, loss_fn, device)
    metrics, _, _ = compute_metrics(y_te_true, y_te_pred, scaler_y)

    torch.save({"model": model.state_dict(),
                "n_features": len(FEATURE_COLS),
                "n_stations": n_stations}, TORCH_FINAL)

    print(f"\nFinal model -> {TORCH_FINAL}")
    print("=" * 65)
    print(f"  t+1h  MAE={metrics['t+1h']['mae']:.2f}  R2={metrics['t+1h']['r2']:.4f}")
    print(f"  t+24h MAE={metrics['t+24h']['mae']:.2f}  R2={metrics['t+24h']['r2']:.4f}")
    print("=" * 65)


def main():
    p = argparse.ArgumentParser(description="PM2.5 PyTorch GPU Training")
    p.add_argument("--epochs",    type=int,  default=EPOCHS)
    p.add_argument("--batch",     type=int,  default=BATCH_SIZE)
    p.add_argument("--quick",     action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch,
          quick=args.quick, resume=not args.no_resume)


if __name__ == "__main__":
    main()
