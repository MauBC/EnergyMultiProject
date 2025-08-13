# app.py
# -*- coding: utf-8 -*-
"""
Multi-step forecasting (e.g., 48h) for one or multiple solar plants.

Models:
  1) MLP (feed-forward): input = flattened window [look_back * n_features]
  2) GRU (recurrent):    input = sequence [look_back, n_features]
Both output a single vector of size (horizon * n_targets) predicting the full future horizon at once.

Key improvements in this version:
  - Add cyc features: hour sin/cos, day-of-year sin/cos (optional weekday sin/cos)
  - Add lag features for DNI, Temperatura, Humedad: 24,48,72 horas
  - Drop/Downweight night hours in loss; option to drop night-only sequences
  - Safe MinMax inverse, clamps, zero-span guard
  - Step weights across horizon
  - Better schedulers: ReduceLROnPlateau (default) or Exponential
  - Day-only metrics (same threshold as training mask)

Usage example (48h horizon, 168h look-back):
  python app.py ^
    --csv COLCA_SOLAR_MERGED.csv ^
    --plants "COLCA_SOLAR" ^
    --horizon 48 ^
    --look-back 168 ^
    --epochs 100 ^
    --models-dir models --outputs-dir outputs --tag COLCA ^
    --loss huber --huber-delta 0.5 ^
    --mask-mode drop --day-thresh 0.10 --drop-night-seqs ^
    --gru-layers 2 --hidden 192 --dropout 0.10 ^
    --step-weights frontloaded ^
    --use-lags --lags "24,48,72" --add-doy
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================ Utils ============================

def detect_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] No se detecto GPU. Usando CPU.")
    return device

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def slugify(s):
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')

def derive_dirs(csv_path, base_models_dir, base_outputs_dir, look_back, horizon, tag=None):
    if not tag:
        tag = slugify(Path(csv_path).stem)
    models_dir = os.path.join(base_models_dir, tag, f"lb{look_back}_hz{horizon}")
    outputs_dir = os.path.join(base_outputs_dir, tag, f"lb{look_back}_hz{horizon}")
    return tag, models_dir, outputs_dir

def add_time_features(df, dt_col="datetime", add_doy=True, add_week=False):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)

    df["hour"] = df[dt_col].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)

    if add_doy:
        df["doy"] = df[dt_col].dt.dayofyear
        df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365.0)
        df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365.0)

    if add_week:
        df["dow"] = df[dt_col].dt.dayofweek
        df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7.0)
        df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7.0)

    return df

def add_lag_features(df, cols, lags):
    df = df.copy()
    for c in cols:
        if c not in df.columns: 
            continue
        for L in lags:
            df[f"{c}_lag_{L}"] = df[c].shift(L)
    return df

def time_split_indices(n, train_ratio=0.7, val_ratio=0.15):
    i_train_end = int(n * train_ratio)
    i_val_end = int(n * (train_ratio + val_ratio))
    return i_train_end, i_val_end

def check_finite(name, arr):
    arr = np.asarray(arr)
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    is_ok = np.isfinite(arr).all()
    print(f"[CHECK] {name}: shape={arr.shape} finite={is_ok} nan={n_nan} inf={n_inf}")
    return is_ok

def mape_safe(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0

def metrics_row(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    sm = smape(y_true, y_pred)
    mp = mape_safe(y_true, y_pred)
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "sMAPE_%": sm, "MAPE_safe_%": mp}


# ============================ Sequences ============================

def make_multistep_sequences(X, Y, look_back, horizon, drop_night_seqs=False, day_thresh_scaled=0.05):
    """
    Returns:
      Xs_mlp: (n_seq, look_back*n_features)
      Xs_gru: (n_seq, look_back, n_features)
      Ys:     (n_seq, horizon*n_targets)
    If drop_night_seqs=True, drops sequences whose target window (Y future) is all below threshold.
    day_thresh_scaled applies to Y already in scaled space (0..1).
    """
    Xs_mlp, Xs_gru, Ys = [], [], []
    n = len(X)
    end = n - look_back - horizon + 1
    for i in range(max(0, end)):
        x_block = X[i:i+look_back, :]                    # (Lb, F)
        y_block = Y[i+look_back:i+look_back+horizon, :]  # (H, T)

        if drop_night_seqs:
            if not np.any(y_block > day_thresh_scaled):
                continue  # all-future steps are night-like -> skip

        Xs_gru.append(x_block)
        Xs_mlp.append(x_block.reshape(-1))
        Ys.append(y_block.reshape(-1))
    return np.array(Xs_mlp), np.array(Xs_gru), np.array(Ys)


# ============================ Datasets ============================

class DatasetMLP(Dataset):
    def __init__(self, X_mlp, Y):
        self.X = X_mlp.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class DatasetGRU(Dataset):
    def __init__(self, X_gru, Y):
        self.X = X_gru.astype(np.float32)  # (n_seq, L, F)
        self.Y = Y.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class LSTMModel(nn.Module):
    def __init__(self, n_features, look_back, output_dim, hidden_size=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=(dropout if n_layers > 1 else 0.0)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)      # (B, L, H)
        last = out[:, -1, :]       # (B, H)
        return self.head(last)     # (B, out_dim)


# ============================ Models ============================

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.net(x)

class GRUModel(nn.Module):
    def __init__(self, n_features, look_back, output_dim, hidden_size=192, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=(dropout if n_layers > 1 else 0.0)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        out, _ = self.gru(x)   # (B, L, H)
        last = out[:, -1, :]   # (B, H)
        return self.head(last) # (B, out_dim)


# ============================ Loss helpers ============================

def make_step_weights(h, mode="frontloaded"):
    if mode == "none":
        return None
    if mode == "frontloaded":
        w = np.linspace(1.0, 0.6, h, dtype=np.float32)
    elif mode == "linear":
        w = np.linspace(1.0, 0.8, h, dtype=np.float32)
    else:
        w = None
    return w

def build_loss_fn(kind="mse", delta=0.5, grad_clip=None):
    base = nn.MSELoss(reduction="none") if kind == "mse" else nn.SmoothL1Loss(beta=delta, reduction="none")
    def loss_fn(pred, target, horizon, n_targets, step_w=None, day_mask=None, mask_mode="off", mask_weight=0.3):
        # pred, target: (B, horizon*n_targets) in normalized space
        L = base(pred, target)  # (B, horizon*n_targets)
        if day_mask is not None:
            dm = day_mask.reshape(pred.size(0), horizon*n_targets).float()
            if mask_mode == "drop":
                L = L * dm
                denom = dm.sum()
                return L.sum()/denom if denom > 0 else L.mean()
            elif mask_mode == "downweight":
                w = dm + (1.0 - dm) * mask_weight
                L = L * w
        if step_w is not None:
            sw = torch.from_numpy(step_w).to(pred.device).float()    # (horizon,)
            sw = sw.repeat_interleave(n_targets)                     # (horizon*n_targets,)
            L = L * sw
        return L.mean()
    return loss_fn


# ============================ Train / Predict ============================

def train_model(model, train_loader, val_loader, device, epochs, optimizer, scheduler,
                loss_fn, horizon, n_targets, step_w, day_thresh, mask_mode, mask_weight,
                model_path, patience=12, grad_clip=None):
    best_val = float("inf")
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            day_mask = (yb > day_thresh) if mask_mode != "off" else None
            loss = loss_fn(pred, yb, horizon, n_targets, step_w, day_mask, mask_mode, mask_weight)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tr += loss.item() * xb.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                day_mask = (yb > day_thresh) if mask_mode != "off" else None
                loss = loss_fn(pred, yb, horizon, n_targets, step_w, day_mask, mask_mode, mask_weight)
                vl += loss.item() * xb.size(0)
        vl /= len(val_loader.dataset)

        print(f"[EPOCH {epoch+1:03d}] train_loss={tr:.6f} | val_loss={vl:.6f}")

        improved = vl < best_val - 1e-6
        if improved:
            best_val = vl
            patience_left = patience
            torch.save(model.state_dict(), model_path)
            print(f"[SAVE] Mejor modelo actualizado: {model_path} | val_loss={best_val:.6f}")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("[EARLY] Sin mejora en validacion. Deteniendo entrenamiento.")
                break

        if scheduler is not None:
            # ReduceLROnPlateau expects val metric; else step by epoch
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl)
            else:
                scheduler.step()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_model(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            preds = model(xb)
            outputs.append(preds.detach().cpu().numpy())
    return np.vstack(outputs) if outputs else np.array([])


# ============================ Safe inverse transform ============================

def MinMax_inverse_like_safe(Y_flat_scaled, scaler_y, n_targets, horizon):
    if Y_flat_scaled.size == 0:
        return np.zeros((0, horizon, n_targets), dtype=np.float32)

    Z = Y_flat_scaled.reshape(-1, n_targets).astype(np.float64)
    Z[~np.isfinite(Z)] = 0.5
    Z = np.clip(Z, 0.0, 1.0)

    data_min = scaler_y.data_min_.astype(np.float64)
    data_max = scaler_y.data_max_.astype(np.float64)
    span = data_max - data_min

    Xrec = Z * span + data_min
    zero_span = (span == 0.0)
    if np.any(zero_span):
        Xrec[:, zero_span] = data_min[zero_span]

    return Xrec.reshape(-1, horizon, n_targets).astype(np.float32)


# ============================ Plotting ============================

def plot_sequence_example(Y_true_seq, Y_pred_seq, plants, title, save_path):
    for i, plant in enumerate(plants):
        plt.figure(figsize=(12, 4))
        plt.plot(Y_true_seq[:, i], label=f"Real {plant}")
        plt.plot(Y_pred_seq[:, i], label=f"Predicho {plant}")
        plt.title(f"{title} - {plant}")
        plt.xlabel("Paso de horizonte (horas)")
        plt.ylabel("Energia")
        plt.legend()
        base, ext = os.path.splitext(save_path)
        plt.savefig(f"{base}_{plant}{ext}", bbox_inches="tight", dpi=120)
        plt.close()

def plot_bar_metric_per_horizon(mae_per_h, rmse_per_h, title_prefix, save_dir):
    H = len(mae_per_h)
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(H), mae_per_h)
    plt.title(f"{title_prefix} - MAE por paso de horizonte")
    plt.xlabel("Paso (h)")
    plt.ylabel("MAE")
    plt.savefig(os.path.join(save_dir, f"{title_prefix}_MAE_por_h.png"), bbox_inches="tight", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(H), rmse_per_h)
    plt.title(f"{title_prefix} - RMSE por paso de horizonte")
    plt.xlabel("Paso (h)")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(save_dir, f"{title_prefix}_RMSE_por_h.png"), bbox_inches="tight", dpi=120)
    plt.close()


# ============================ Main pipeline ============================

def main(args):
    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = detect_device()

    # derive dirs
    tag, mdir, odir = derive_dirs(args.csv, args.models_dir, args.outputs_dir, args.look_back, args.horizon, args.tag)
    ensure_dir(mdir); ensure_dir(odir)
    print(f"[DIR] models_dir = {mdir}")
    print(f"[DIR] outputs_dir = {odir}")
    print(f"[INFO] dataset tag = {tag}")

    # read CSV
    print(f"[LOAD] Leyendo CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    # --- FILTRO: Solo horas diurnas con DNI > 0 ---
    if "DNI" not in df.columns:
        raise ValueError("No se encontro la columna DNI en el CSV.")
    n_before = len(df)
    df = df[df["DNI"] > 0].reset_index(drop=True)
    n_after = len(df)
    print(f"[FILTER] Filtrado por DNI>0: {n_before} -> {n_after} filas")
        
    if args.datetime_col not in df.columns:
        raise ValueError(f"No se encontro la columna datetime '{args.datetime_col}'.")
    plants = [p.strip() for p in args.plants.split(",")]
    for p in plants:
        if p not in df.columns:
            raise ValueError(f"No se encontro la columna objetivo '{p}' en el CSV.")

    # base columns + time features + lags
    base_cols = [args.datetime_col, "DNI", "Temperatura", "Viento", "Humedad"] + plants
    df = df[base_cols].copy()
    print(f"[INFO] Filas totales (raw): {len(df)} | Columnas: {list(df.columns)}")

    # time features
    df = add_time_features(df, dt_col=args.datetime_col, add_doy=args.add_doy, add_week=args.add_week)

    # lag features
    lag_list = []
    if args.use_lags and args.lags:
        lag_list = [int(x.strip()) for x in args.lags.split(",") if x.strip().isdigit()]
        if lag_list:
            df = add_lag_features(df, cols=["DNI", "Temperatura", "Humedad"], lags=lag_list)

    # drop rows with NaN introduced by lags
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[CLEAN] Drop por lags/NaN: {n_before - len(df)} filas | Final: {len(df)}")

    # feature list
    features = ["DNI", "Temperatura", "Viento", "Humedad",
                "hour_sin", "hour_cos"]
    if args.add_doy:
        features += ["doy_sin", "doy_cos"]
    if args.add_week:
        features += ["dow_sin", "dow_cos"]
    for L in lag_list:
        features += [f"DNI_lag_{L}", f"Temperatura_lag_{L}", f"Humedad_lag_{L}"]

    X = df[features].values
    Y = df[plants].values

    # split (time ordered)
    i_train_end, i_val_end = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    X_train_raw, Y_train_raw = X[:i_train_end], Y[:i_train_end]
    X_val_raw,   Y_val_raw   = X[i_train_end:i_val_end], Y[i_train_end:i_val_end]
    X_test_raw,  Y_test_raw  = X[i_val_end:], Y[i_val_end:]
    print(f"[SPLIT] train: {X_train_raw.shape[0]} | val: {X_val_raw.shape[0]} | test: {X_test_raw.shape[0]}")

    # scaling
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_X.fit_transform(X_train_raw)
    X_val_s   = scaler_X.transform(X_val_raw)
    X_test_s  = scaler_X.transform(X_test_raw)

    Y_train_s = scaler_y.fit_transform(Y_train_raw)
    Y_val_s   = scaler_y.transform(Y_val_raw)
    Y_test_s  = scaler_y.transform(Y_test_raw)

    sx_path = os.path.join(mdir, f"scaler_X_lb{args.look_back}_hz{args.horizon}.pkl")
    sy_path = os.path.join(mdir, f"scaler_y_lb{args.look_back}_hz{args.horizon}.pkl")
    joblib.dump(scaler_X, sx_path)
    joblib.dump(scaler_y, sy_path)
    with open(os.path.join(odir, "features.json"), "w", encoding="utf-8") as f:
        json.dump({"features": features, "plants": plants,
                   "look_back": args.look_back, "horizon": args.horizon,"rnn_type": args.rnn_type},
                  f, ensure_ascii=True, indent=2)
    print(f"[SAVE] Escaladores guardados: {sx_path} | {sy_path}")

    # windows (ya filtrado a solo diurno arriba)
    Xtr_mlp, Xtr_gru, Ytr = make_multistep_sequences(
        X_train_s, Y_train_s,
        args.look_back, args.horizon,
        drop_night_seqs=False, day_thresh_scaled=0.0
    )

    Xva_mlp, Xva_gru, Yva = make_multistep_sequences(
        X_val_s, Y_val_s,
        args.look_back, args.horizon,
        drop_night_seqs=False, day_thresh_scaled=0.0
    )

    Xte_mlp, Xte_gru, Yte = make_multistep_sequences(
        X_test_s, Y_test_s,
        args.look_back, args.horizon,
        drop_night_seqs=False, day_thresh_scaled=0.0
    )

    print(f"[SEQ] train seqs: {Ytr.shape[0]} | val seqs: {Yva.shape[0]} | test seqs: {Yte.shape[0]}")
    if Ytr.shape[0] == 0 or Yva.shape[0] == 0 or Yte.shape[0] == 0:
        print("[ERROR] No se generaron secuencias. Verifique look_back, horizon y tamanos de split.")
        return

    n_features = X.shape[1]
    n_targets  = Y.shape[1]
    out_dim    = args.horizon * n_targets

    # dataloaders
    train_loader_mlp = DataLoader(DatasetMLP(Xtr_mlp, Ytr), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader_mlp   = DataLoader(DatasetMLP(Xva_mlp, Yva), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader_mlp  = DataLoader(DatasetMLP(Xte_mlp, Yte), batch_size=args.batch_size, shuffle=False, drop_last=False)

    train_loader_gru = DataLoader(DatasetGRU(Xtr_gru, Ytr), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader_gru   = DataLoader(DatasetGRU(Xva_gru, Yva), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader_gru  = DataLoader(DatasetGRU(Xte_gru, Yte), batch_size=args.batch_size, shuffle=False, drop_last=False)

    # loss and weights
    step_w = make_step_weights(args.horizon, args.step_weights)
    loss_fn = build_loss_fn(args.loss, args.huber_delta)

    # --------- MLP ---------
    mlp_path = os.path.join(mdir, f"mlp_multistep_lb{args.look_back}_hz{args.horizon}.pt")
    mlp = MLP(input_dim=Xtr_mlp.shape[1], output_dim=out_dim, dropout=args.dropout).to(device)
    if os.path.exists(mlp_path) and not args.force_retrain:
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp.eval()
        print(f"[LOAD] Modelo MLP cargado: {mlp_path}")
    else:
        print(f"[TRAIN] Entrenando MLP -> {mlp_path}")
        opt = torch.optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(3, args.patience//3))
        else:
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)
        mlp = train_model(
            model=mlp,
            train_loader=train_loader_mlp,
            val_loader=val_loader_mlp,
            device=device,
            epochs=args.epochs,
            optimizer=opt,
            scheduler=sch,
            loss_fn=loss_fn,
            horizon=args.horizon,
            n_targets=n_targets,
            step_w=step_w,
            day_thresh=args.day_thresh,
            mask_mode=args.mask_mode,
            mask_weight=args.mask_weight,
            model_path=mlp_path,
            patience=args.patience,
            grad_clip=args.grad_clip
        )
    print("[PRED] Prediciendo con MLP en test...")
    Yte_pred_mlp_s = predict_model(mlp, test_loader_mlp, device)         # (n_seq_test, out_dim)
    Yte_pred_mlp   = MinMax_inverse_like_safe(Yte_pred_mlp_s, scaler_y, n_targets, args.horizon)

    # --------- RNN (GRU o LSTM) ---------
    rnn_path = os.path.join(mdir, f"{args.rnn_type}_multistep_lb{args.look_back}_hz{args.horizon}.pt")

    # Instancia según bandera
    if args.rnn_type == "gru":
        rnn = GRUModel(
            n_features=n_features, look_back=args.look_back, output_dim=out_dim,
            hidden_size=args.hidden, n_layers=args.gru_layers, dropout=args.dropout
        ).to(device)
    else:  # lstm
        rnn = LSTMModel(
            n_features=n_features, look_back=args.look_back, output_dim=out_dim,
            hidden_size=args.hidden, n_layers=args.gru_layers, dropout=args.dropout
        ).to(device)

    if os.path.exists(rnn_path) and not args.force_retrain:
        rnn.load_state_dict(torch.load(rnn_path, map_location=device))
        rnn.eval()
        print(f"[LOAD] Modelo {args.rnn_type.upper()} cargado: {rnn_path}")
    else:
        print(f"[TRAIN] Entrenando {args.rnn_type.upper()} -> {rnn_path}")
        opt_rnn = torch.optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch_rnn = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt_rnn, mode="min", factor=0.5, patience=max(3, args.patience//3))
                if args.scheduler == "plateau"
                else torch.optim.lr_scheduler.ExponentialLR(opt_rnn, gamma=args.lr_gamma))
        rnn = train_model(
            model=rnn,
            train_loader=train_loader_gru,   # el dataloader secuencial ya sirve para LSTM/GRU
            val_loader=val_loader_gru,
            device=device,
            epochs=args.epochs,
            optimizer=opt_rnn,
            scheduler=sch_rnn,
            loss_fn=loss_fn,
            horizon=args.horizon,
            n_targets=n_targets,
            step_w=step_w,
            day_thresh=args.day_thresh,
            mask_mode=args.mask_mode,
            mask_weight=args.mask_weight,
            model_path=rnn_path,
            patience=args.patience,
            grad_clip=args.grad_clip
        )

    print(f"[PRED] Prediciendo con {args.rnn_type.upper()} en test...")
    Yte_pred_rnn_s = predict_model(rnn, test_loader_gru, device)
    Yte_pred_rnn   = MinMax_inverse_like_safe(Yte_pred_rnn_s, scaler_y, n_targets, args.horizon)

    # true in original scale (inverse from scaled Yte)
    Yte_true = MinMax_inverse_like_safe(Yte, scaler_y, n_targets, args.horizon)

    # checks
    check_finite("Yte_true", Yte_true)
    check_finite("Yte_pred_mlp", Yte_pred_mlp)
    check_finite("Yte_pred_gru", Yte_pred_rnn)

    # ===== Metrics (global and per-plant), with sMAPE/MAPE too =====
    y_true_flat = Yte_true.reshape(-1, n_targets)
    y_mlp_flat  = Yte_pred_mlp.reshape(-1, n_targets)
    y_gru_flat  = Yte_pred_rnn.reshape(-1, n_targets)

    # day-only mask for evaluation, based on normalized Y
    Yte_scaled_flat = Yte.reshape(-1, n_targets)
    day_mask_eval = (Yte_scaled_flat > args.day_thresh)
    rnn_name = args.rnn_type.upper()

    results = []
    # global (all hours)
    results.append(metrics_row(y_true_flat, y_mlp_flat, label="MLP_global"))
    results.append(metrics_row(y_true_flat, Yte_pred_rnn.reshape(-1, n_targets), label=f"{rnn_name}_global"))
    # per-plant (all hours)
    for i, plant in enumerate(plants):
        results.append(metrics_row(y_true_flat[:, i], y_mlp_flat[:, i], label=f"MLP_{plant}"))
        results.append(metrics_row(y_true_flat[:, i], Yte_pred_rnn.reshape(-1, n_targets)[:, i], label=f"{rnn_name}_{plant}"))
    metrics_df = pd.DataFrame(results)
    metrics_csv = os.path.join(odir, f"metrics_multistep_lb{args.look_back}_hz{args.horizon}.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print("\n=== METRICAS (Test Multi-step, todas las horas) ===")
    print(metrics_df)
    print(f"[SAVE] Metricas guardadas en: {metrics_csv}")

    # day-only metrics
    results_day = []
    for i in range(n_targets):
        mask_i = day_mask_eval[:, i].astype(bool)
        if not np.any(mask_i):
            continue
        results_day.append(metrics_row(y_true_flat[mask_i, i], y_mlp_flat[mask_i, i], label=f"MLP_dayonly_{plants[i]}"))
        results_day.append(metrics_row(y_true_flat[mask_i, i], y_gru_flat[mask_i, i], label=f"GRU_dayonly_{plants[i]}"))
    if results_day:
        metrics_day_df = pd.DataFrame(results_day)
        metrics_day_csv = os.path.join(odir, f"metrics_multistep_dayonly_lb{args.look_back}_hz{args.horizon}.csv")
        metrics_day_df.to_csv(metrics_day_csv, index=False)
        print("\n=== METRICAS (Test solo diurno, segun day_thresh normalizado) ===")
        print(metrics_day_df)
        print(f"[SAVE] Metricas diurnas guardadas en: {metrics_day_csv}")

    # ===== Metrics per horizon step (avg across sequences and targets) =====
    H = args.horizon
    mae_mlp_per_h, rmse_mlp_per_h = [], []
    mae_gru_per_h, rmse_gru_per_h = [], []
    for h in range(H):
        y_true_h = Yte_true[:, h, :].reshape(-1, n_targets)
        y_mlp_h  = Yte_pred_mlp[:, h, :].reshape(-1, n_targets)
        y_gru_h  = Yte_pred_rnn[:, h, :].reshape(-1, n_targets)
        mae_mlp_per_h.append(mean_absolute_error(y_true_h, y_mlp_h))
        rmse_mlp_per_h.append(np.sqrt(mean_squared_error(y_true_h, y_mlp_h)))
        mae_gru_per_h.append(mean_absolute_error(y_true_h, y_gru_h))
        rmse_gru_per_h.append(np.sqrt(mean_squared_error(y_true_h, y_gru_h)))

    plot_bar_metric_per_horizon(mae_mlp_per_h, rmse_mlp_per_h, f"MLP_lb{args.look_back}_hz{args.horizon}", odir)
    plot_bar_metric_per_horizon(mae_gru_per_h, rmse_gru_per_h, f"GRU_lb{args.look_back}_hz{args.horizon}", odir)
    print("[PLOT] Barras de MAE y RMSE por paso generadas.")

    # example plots
    if len(Yte_true) > 0:
        seq_idx = 0
        plot_sequence_example(
            Y_true_seq=Yte_true[seq_idx],
            Y_pred_seq=Yte_pred_mlp[seq_idx],
            plants=plants,
            title=f"MLP Prediccion vs Real (ejemplo) - lb{args.look_back} hz{args.horizon}",
            save_path=os.path.join(odir, f"seq_example_mlp_lb{args.look_back}_hz{args.horizon}.png")
        )
        plot_sequence_example(
            Y_true_seq=Yte_true[seq_idx],
            Y_pred_seq=Yte_pred_rnn[seq_idx],
            plants=plants,
            title=f"GRU Prediccion vs Real (ejemplo) - lb{args.look_back} hz{args.horizon}",
            save_path=os.path.join(odir, f"seq_example_gru_lb{args.look_back}_hz{args.horizon}.png")
        )
        print("[PLOT] Graficos de secuencia ejemplo generados.")

    # save full predictions
    rows = []
    for s in range(Yte_true.shape[0]):
        for h in range(H):
            row = {"seq_idx": s, "h": h}
            for i, plant in enumerate(plants):
                row[f"real_{plant}"] = float(Yte_true[s, h, i])
                row[f"mlp_{plant}"]  = float(Yte_pred_mlp[s, h, i])
                row[f"{args.rnn_type}_{plant}"] = float(Yte_pred_rnn[s, h, i])  # ej: "gru_COLCA_SOLAR" o "lstm_COLCA_SOLAR"
            rows.append(row)
    preds_df = pd.DataFrame(rows)
    preds_csv = os.path.join(odir, f"preds_multistep_lb{args.look_back}_hz{args.horizon}.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"[SAVE] Predicciones completas guardadas en: {preds_csv}")

    print(f"\n[OK] Modelos guardados en: {mdir}")
    print(f"[OK] Metricas y graficos en: {odir}")
    print("[TIP] Para inferencia: cargar scaler_X/scaler_y y el modelo deseado, "
          "construir la ultima ventana (look_back) y obtener el vector (horizon*n_plants).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV con datetime y columnas de plantas")
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru",
                    help="Tipo de RNN para el modelo multistep")
    parser.add_argument("--plants", type=str, required=True, help="Columnas objetivo, separadas por coma")
    parser.add_argument("--datetime-col", type=str, default="datetime")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--look-back", type=int, default=168, help="Horas de historial para la entrada")
    parser.add_argument("--horizon", type=int, default=12, help="Horas a predecir (p. ej., 12h)")
    parser.add_argument("--force-retrain", action="store_true", help="Ignora modelos existentes y reentrena")
    parser.add_argument("--tag", type=str, default=None, help="Etiqueta opcional del dataset; por defecto usa el nombre del CSV")

    # improvements
    parser.add_argument("--loss", choices=["mse","huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--day-thresh", type=float, default=0.10, help="Umbral NORMALIZADO para considerar 'dia'")
    parser.add_argument("--mask-mode", choices=["off","downweight","drop"], default="drop")
    parser.add_argument("--mask-weight", type=float, default=0.3, help="Peso para noches si mask-mode=downweight")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--step-weights", choices=["none","frontloaded","linear"], default="frontloaded")
    parser.add_argument("--drop-night-seqs", action="store_true", help="Descarta secuencias cuyo futuro es todo noche")

    # feature flags
    parser.add_argument("--use-lags", action="store_true", help="Agrega lags 24,48,72 a DNI, Temperatura, Humedad")
    parser.add_argument("--lags", type=str, default="24,48,72")
    parser.add_argument("--add-doy", action="store_true", help="Agrega day-of-year sin/cos")
    parser.add_argument("--add-week", action="store_true", help="Agrega weekday sin/cos")

    # training controls
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-gamma", type=float, default=0.98, help="Usado si scheduler=exp")
    parser.add_argument("--scheduler", choices=["plateau","exp"], default="plateau")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

    '''
    python app.py --csv COLCA_SOLAR_MERGED.csv --plants "COLCA_SOLAR" --look-back 240 --horizon 6 --epochs 200 --tag COLCA_6H --mask-mode drop --drop-night-seqs --day-thresh 0.0 --hidden 128 --gru-layers 2 --dropout 0.2 --batch-size 64 --lr 0.001 --use-lags --lags "24,48,72" --add-doy --scheduler plateau --patience 20 --force-retrain

    python app.py \
    --csv COLCA_SOLAR_MERGED.csv \
    --plants "COLCA_SOLAR" \
    --look-back 240 \
    --horizon 6 \
    --epochs 200 \
    --tag COLCA_6H \
    --mask-mode drop \
    --drop-night-seqs \
    --day-thresh 0.0 \
    --hidden 128 \
    --gru-layers 2 \
    --dropout 0.2 \
    --batch-size 64 \
    --lr 0.001 \
    --use-lags \
    --lags "24,48,72" \
    --add-doy \
    --scheduler plateau \
    --patience 20 \
    --force-retrain

    '''