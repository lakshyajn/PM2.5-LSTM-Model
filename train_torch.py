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


def get_num_workers():
    """Safe worker count: 2 on GPU systems (shared memory), 0 on CPU."""
    if not torch.cuda.is_available():
        return 0
    import sys
    # Windows spawn is safe with shared-memory tensors; 2 workers = good overlap
    return 2


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
    print(f"  VRAM: {total_vram_gb:.1f} GB  |  recommended max_seqs ~{max_seqs:,}")
    return max_seqs


# ─── Dataset (Streaming) ──────────────────────────────────────────────────────
# ─── Dataset (Streaming, Flat-tensor) ────────────────────────────────────────────────
# Design:
#   1. Scale ALL rows once -> one contiguous shared-memory tensor per split
#      (434 scattered dicts -> one flat tensor: better cache locality)
#   2. Flat int32 numpy arrays as index (no Python tuples, no dict lookups)
#   3. No .clone() — row slices of a C-contiguous 2D tensor are contiguous;
#      DataLoader collation stacks them into a new owned tensor
#   4. When MAX_SEQS_TRAIN=0: DataLoader shuffle=True (PyTorch C++ RandomSampler)
#      When MAX_SEQS_TRAIN>0: ResampleSampler for capped random subsets
#   5. pin_memory + non_blocking -> async CPU->GPU DMA pipeline

def build_scaled_flat(df, scaler_X, scaler_y, feature_cols, station_id_map,
                      seq_len=SEQ_LEN, n_horizons=N_HORIZONS):
    """
    Scale all rows, concatenate into ONE flat contiguous shared-memory tensor.
    Eliminates per-station dict overhead and scattered memory allocations.

    Returns:
        flat_feats: Tensor (total_rows, n_feats)    — shared memory
        flat_tgts:  Tensor (total_rows, n_horizons) — shared memory
        offsets:    {station -> (start_row, n_rows)}
        sids:       {station -> int}
    """
    stations = sorted(df["station"].unique())
    feat_chunks, tgt_chunks = [], []
    offsets, sids = {}, {}
    ptr = 0

    for station in stations:
        sdf = (df[df["station"] == station]
               .sort_values("timestamp")
               .reset_index(drop=True))
        n = len(sdf)
        if n < seq_len + n_horizons:
            continue

        feat_np = scaler_X.transform(sdf[feature_cols].values.astype(np.float32))
        tgt_np  = scaler_y.transform(
            sdf[TARGET_COLS].values.astype(np.float32).reshape(-1, 1)
        ).reshape(n, n_horizons)

        feat_chunks.append(feat_np)
        tgt_chunks.append(tgt_np)
        offsets[station] = (ptr, n)
        sids[station]    = station_id_map.get(station, 0)
        ptr += n

    # Single np.vstack -> one contiguous block -> share_memory_()
    flat_feats = torch.from_numpy(np.vstack(feat_chunks)).share_memory_()
    flat_tgts  = torch.from_numpy(np.vstack(tgt_chunks)).share_memory_()
    ram_mb = (flat_feats.numel() + flat_tgts.numel()) * 4 / 1024**2
    print(f"  Scaled: {ptr:,} rows  ({ram_mb:.0f} MB RAM, flat contiguous shared tensor)")
    return flat_feats, flat_tgts, offsets, sids


def build_index_flat(offsets, sids, seq_len=SEQ_LEN, n_horizons=N_HORIZONS):
    """
    Build 3 flat int32/int64 numpy arrays instead of a list of Python tuples.
    ~10x less memory, no tuple overhead, no dict lookup per sample.

    Returns:
        arr_offsets: int64 (n_seqs,) — station's start row in flat tensor
        arr_rows:    int32 (n_seqs,) — local row_i within that station
        arr_sids:    int32 (n_seqs,) — station embedding ID
    """
    # Pre-count total sequences to allocate exactly
    n_total = sum(
        max(0, n - seq_len - n_horizons + 1)
        for _, (_, n) in offsets.items()
    )
    arr_offsets = np.empty(n_total, dtype=np.int64)
    arr_rows    = np.empty(n_total, dtype=np.int32)
    arr_sids    = np.empty(n_total, dtype=np.int32)

    ptr = 0
    for station in sorted(offsets.keys()):
        start, n = offsets[station]
        sid      = sids[station]
        valid    = np.arange(seq_len, n - n_horizons + 1, dtype=np.int32)
        k        = len(valid)
        arr_offsets[ptr:ptr+k] = start
        arr_rows[ptr:ptr+k]    = valid
        arr_sids[ptr:ptr+k]    = sid
        ptr += k

    return arr_offsets, arr_rows, arr_sids


class PM25StreamDataset(Dataset):
    """
    Flat-tensor streaming dataset.
    - No dict lookup: direct integer offset into one contiguous tensor
    - No .clone(): row-slices of C-contiguous 2D tensors are contiguous;
      DataLoader collation (torch.stack) copies them into a new batch tensor
    - No Python tuples: three flat int32/int64 numpy arrays as index
    """
    def __init__(self, flat_feats, flat_tgts,
                 arr_offsets, arr_rows, arr_sids, seq_len=SEQ_LEN):
        self.flat_feats  = flat_feats    # Tensor (total_rows, n_feats)
        self.flat_tgts   = flat_tgts     # Tensor (total_rows, n_horizons)
        self.arr_offsets = arr_offsets   # np.int64 (n_seqs,)
        self.arr_rows    = arr_rows      # np.int32 (n_seqs,)
        self.arr_sids    = arr_sids      # np.int32 (n_seqs,)
        self.seq_len     = seq_len

    def __len__(self):
        return len(self.arr_rows)

    def __getitem__(self, idx):
        off = int(self.arr_offsets[idx])
        i   = int(self.arr_rows[idx])
        sid = int(self.arr_sids[idx])
        # Direct integer slice into flat contiguous tensor — no dict, no clone
        X = self.flat_feats[off + i - self.seq_len : off + i]
        y = self.flat_tgts[off + i]
        return X, torch.tensor(sid, dtype=torch.long), y


class ResampleSampler(torch.utils.data.Sampler):
    """
    Only used when MAX_SEQS_TRAIN > 0 (capped epochs).
    Picks max_seqs random indices each epoch via set_epoch() -> different seed.
    When MAX_SEQS_TRAIN == 0, DataLoader's built-in C++ shuffle is used instead.
    """
    def __init__(self, n_total, max_seqs):
        self.n_total  = n_total
        self.max_seqs = max_seqs
        self.epoch    = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch * 1337 + 42)
        idx = torch.randperm(self.n_total, generator=g)[:self.max_seqs]
        return iter(idx.tolist())

    def __len__(self):
        return self.max_seqs


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
        # Async CPU->GPU DMA (non_blocking=True + pin_memory=True in DataLoader)
        X    = X.to(device, non_blocking=True)
        sids = sids.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)
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
        X    = X.to(device, non_blocking=True)
        sids = sids.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)
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

    max_tr = 5_000 if quick else (MAX_SEQS_TRAIN if MAX_SEQS_TRAIN > 0 else 0)
    max_ev = 2_000 if quick else 50_000

    if not quick and MAX_SEQS_TRAIN == 0:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device.type == "cuda" else 0
        rec = auto_max_seqs(device)
        print(f"  MAX_SEQS_TRAIN=0: using ALL sequences per epoch")
        print(f"  (Set MAX_SEQS_TRAIN={rec:,} in config.py to cap epoch time on this GPU)")

    # ── Scale ALL data once -> flat contiguous shared tensors ───────────────────
    print("\nScaling all features (once) ...")
    ff_tr, ft_tr, off_tr, sid_tr = build_scaled_flat(
        df_tr, scaler_X, scaler_y, FEATURE_COLS, station_map)
    ff_va, ft_va, off_va, sid_va = build_scaled_flat(
        df_va, scaler_X, scaler_y, FEATURE_COLS, station_map)
    ff_te, ft_te, off_te, sid_te = build_scaled_flat(
        df_te, scaler_X, scaler_y, FEATURE_COLS, station_map)
    del df, df_tr, df_va, df_te   # free parquet RAM (~1.5 GB)

    # ── Build flat int32 index arrays (no Python tuples) ────────────────────────
    print("Building sequence indices ...")
    ao_tr, ar_tr, as_tr = build_index_flat(off_tr, sid_tr)
    ao_va, ar_va, as_va = build_index_flat(off_va, sid_va)
    ao_te, ar_te, as_te = build_index_flat(off_te, sid_te)
    print(f"  Train: {len(ar_tr):,}  Val: {len(ar_va):,}  Test: {len(ar_te):,} valid sequences")

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_tr = PM25StreamDataset(ff_tr, ft_tr, ao_tr, ar_tr, as_tr)
    ds_va = PM25StreamDataset(ff_va, ft_va, ao_va, ar_va, as_va)
    ds_te = PM25StreamDataset(ff_te, ft_te, ao_te, ar_te, as_te)

    nw  = get_num_workers()
    pin = device.type == "cuda"
    pw  = (nw > 0)
    pf  = 2 if nw > 0 else None

    # When max_tr==0 use all seqs: DataLoader's built-in C++ RandomSampler (no Python permutation)
    # When max_tr>0 cap with ResampleSampler: fresh random subset each epoch via set_epoch()
    if max_tr == 0:
        dl_tr      = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                                num_workers=nw, pin_memory=pin,
                                persistent_workers=pw, prefetch_factor=pf)
        sampler_tr = None   # DataLoader handles shuffling internally
    else:
        sampler_tr = ResampleSampler(len(ds_tr), max_seqs=max_tr)
        dl_tr      = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler_tr,
                                num_workers=nw, pin_memory=pin,
                                persistent_workers=pw, prefetch_factor=pf)

    sampler_va = ResampleSampler(len(ds_va), max_seqs=max_ev)
    dl_va = DataLoader(ds_va, batch_size=512, sampler=sampler_va,
                       num_workers=nw, pin_memory=pin,
                       persistent_workers=pw, prefetch_factor=pf)
    dl_te = DataLoader(ds_te, batch_size=512, shuffle=False,
                       num_workers=nw, pin_memory=pin,
                       persistent_workers=pw, prefetch_factor=pf)
    steps = len(dl_tr)
    print(f"  DataLoader: workers={nw}  pin={pin}  prefetch={pf}  steps/epoch={steps:,}")


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
        t0 = time.time()

        # Advance sampler seed for fresh random subset (only when capping)
        if sampler_tr is not None:
            sampler_tr.set_epoch(epoch)
        sampler_va.set_epoch(epoch)

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
