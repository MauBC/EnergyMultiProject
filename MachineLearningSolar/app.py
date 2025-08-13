
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# joblib to persist scalers
import joblib


# --------------------------
# Utils
# --------------------------

def detect_device():
    # Prefer CUDA on Windows with NVIDIA; fallback to CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # Optional, in case user installed it
        return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")


def smape(y_true, y_pred, eps=1e-6):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    diff = np.abs(y_true - y_pred)
    return 100.0 * np.mean(diff / denom)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def parse_datetime_column(df):
    # Try common datetime column names
    candidates = [c for c in df.columns if c.lower() in ("datetime","fechahora","timestamp","fecha_hora","date","time")]
    if not candidates:
        # If not found, try first column if it looks like datetime
        first = df.columns[0]
        candidates = [first]
    for c in candidates:
        try:
            dt = pd.to_datetime(df[c], utc=False, dayfirst=True, infer_datetime_format=True, errors="raise")
            return c, dt
        except Exception:
            continue
    # Fallback: try parsing the first column with coerce
    c = df.columns[0]
    dt = pd.to_datetime(df[c], utc=False, dayfirst=True, infer_datetime_format=True, errors="coerce")
    if dt.isna().all():
        raise ValueError("Could not parse any datetime column. Please ensure CSV has a datetime-like column.")
    return c, dt


def choose_target_column(df, exclude_cols):
    # Heuristic: pick numeric column most likely to be energy generation.
    # Exclude known met vars keywords.
    ndf = df.drop(columns=exclude_cols, errors="ignore")
    num_candidates = []
    meta_exclude = set([
        "dni","ghi","dhi","temp","temperatura","wspd","viento","humidity","humedad",
        "cloud","clouds","cloudcover","nubes","press","pressure","precip","rain"
    ])
    for c in ndf.columns:
        if c.lower() in meta_exclude:
            continue
        if pd.api.types.is_numeric_dtype(ndf[c]):
            num_candidates.append(c)
    if not num_candidates:
        # fallback: any numeric column in the df
        for c in df.columns:
            if c in exclude_cols: 
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                num_candidates.append(c)
    if not num_candidates:
        raise ValueError("No numeric columns found to use as target.")
    # choose column with largest variance (likely the plant output)
    best = max(num_candidates, key=lambda c: float(np.nanvar(pd.to_numeric(df[c], errors="coerce"))))
    return best


def make_time_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    # Cyclical encodings for hour of day and day of year
    hour = dt_index.hour.values
    doy = dt_index.dayofyear.values

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    doy_sin = np.sin(2 * np.pi * doy / 366.0)
    doy_cos = np.cos(2 * np.pi * doy / 366.0)

    return pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }, index=dt_index)


def resample_hourly(df, dt_col, how="mean"):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], utc=False, infer_datetime_format=True)
    df = df.set_index(dt_col).sort_index()
    # If data is already hourly, this will keep it; else aggregate to 1H
    if how == "sum":
        df = df.resample("1h").sum()
    else:
        df = df.resample("1h").mean()
    return df


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(data, target, lookback=168, horizon=6):
    # data: DataFrame with features + target column present
    # target: column name for the series to predict
    values = data.values
    target_idx = data.columns.get_loc(target)
    X_list, y_list = [], []
    for i in range(lookback, len(values) - horizon + 1):
        X_list.append(values[i - lookback:i, :])
        y_list.append(values[i:i + horizon, target_idx])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: [B, T, F]
        out, (hn, cn) = self.lstm(x)
        # use last time step
        last = out[:, -1, :]  # [B, H]
        yhat = self.head(last)  # [B, horizon]
        return yhat


def train_model(model, device, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}", flush=True)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def split_by_time(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])


def evaluate(model, device, data_loader, scaler_y=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)  # [B, horizon]
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # invert scaling only for target if scaler_y provided
    if scaler_y is not None:
        H = preds.shape[1]
        preds_inv = scaler_y.inverse_transform(preds)
        trues_inv = scaler_y.inverse_transform(trues)
    else:
        preds_inv = preds
        trues_inv = trues

    # metrics averaged across horizon steps
    mae = mean_absolute_error(trues_inv.reshape(-1), preds_inv.reshape(-1))
    rmse_val = rmse(trues_inv.reshape(-1), preds_inv.reshape(-1))
    smape_val = smape(trues_inv.reshape(-1), preds_inv.reshape(-1))
    return {"MAE": mae, "RMSE": rmse_val, "sMAPE_%": smape_val}, preds_inv, trues_inv


def main():
    parser = argparse.ArgumentParser(description="Train LSTM to predict next 6 hours of solar generation.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV (merged dataset).")
    parser.add_argument("--output_dir", type=str, default="./outputs_lstm6h", help="Where to save artifacts.")
    parser.add_argument("--lookback", type=int, default=168, help="Hours to look back (sequence length).")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon (hours).")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--target", type=str, default="", help="Explicit target column (optional).")
    parser.add_argument("--sum_target", action="store_true", help="Resample hourly with sum instead of mean (if energy is energy-per-hour).")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    print(f"Device: {device}", flush=True)

    # Load CSV robustly
    df = pd.read_csv(args.csv)

    # strip column names primero
    df.columns = [str(c).strip() for c in df.columns]

    # Filtrar horas sin irradiancia (DNI == 0)
    if "DNI" in df.columns:
        df = df[df["DNI"] > 0].copy()
        print(f"Filtrado DNI=0 -> quedan {len(df)} registros")
    else:
        print("Advertencia: no se encontro columna 'DNI', no se filtra.")

    # detect datetime col
    dt_col, dt_parsed = None, None
    try:
        name, dt = parse_datetime_column(df)
        dt_col, dt_parsed = name, dt
        df[dt_col] = dt
    except Exception as e:
        raise RuntimeError(f"Could not parse datetime column: {e}")

    # Resample to hourly
    how = "sum" if args.sum_target else "mean"
    df = resample_hourly(df, dt_col, how=how)

    # Keep only numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Interpolate missing
    df = df.interpolate(method="time").ffill().bfill()

    # Time features
    tf = make_time_features(df.index)
    df = pd.concat([df, tf], axis=1)

    # Choose target
    if args.target:
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Provided target '{target}' not found in columns.")
    else:
        target = choose_target_column(df, exclude_cols=[])
    print(f"Target column: {target}", flush=True)

    # Separate features and target
    features = [c for c in df.columns if c != target]
    Xdf = df[features].copy()
    ydf = df[[target]].copy()

    # Scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(Xdf.values)
    y_scaled = scaler_y.fit_transform(ydf.values).reshape(-1)

    # Recombine into a single DataFrame for sequence building
    scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=features)
    scaled_df[target] = y_scaled

    lookback = int(args.lookback)
    horizon = int(args.horizon)

    X, y = build_sequences(scaled_df[features + [target]], target=target, lookback=lookback, horizon=horizon)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_by_time(X, y, train_ratio=0.7, val_ratio=0.15)

    train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(SeqDataset(Xva, yva), batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(SeqDataset(Xte, yte), batch_size=args.batch, shuffle=False, drop_last=False)

    input_size = X.shape[-1]
    model = LSTMForecast(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=horizon).to(device)

    model, best_val = train_model(model, device, train_loader, val_loader, epochs=args.epochs, lr=args.lr, patience=args.patience)

    # Evaluate
    metrics_te, preds_te, trues_te = evaluate(model, device, test_loader, scaler_y=scaler_y)

    print("\n=== Test metrics (averaged over all 6 horizons) ===")
    for k, v in metrics_te.items():
        print(f"{k}: {v:.6f}")

        # Save artifacts
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outdir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # model
    torch.save(model.state_dict(), run_dir / "model.pt")
    # scalers
    joblib.dump(scaler_X, run_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, run_dir / "scaler_y.joblib")
    # columns
    with open(run_dir / "columns.json", "w", encoding="utf-8") as f:
        json.dump({"features": features, "target": target}, f, ensure_ascii=False, indent=2)
    # config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "lookback": lookback, "horizon": horizon, "batch": args.batch,
            "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
            "target": target, "resample": how
        }, f, ensure_ascii=False, indent=2)
    # metrics
    metrics_py = {k: float(v) for k, v in metrics_te.items()}
    with open(run_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics_py, f, indent=2)

    print(f"Run dir: {run_dir}")

    # Save a simple forecast on the last available window
    last_X = X[-1:]
    model.eval()
    with torch.no_grad():
        last_pred_scaled = model(torch.tensor(last_X, dtype=torch.float32, device=device)).cpu().numpy()[0]
    last_pred = scaler_y.inverse_transform(last_pred_scaled.reshape(1, -1)).reshape(-1)

    # Map forecast times
    last_end_time = df.index[-1]
    horizon_idx = pd.date_range(last_end_time + pd.Timedelta(hours=1), periods=horizon, freq="H")
    forecast_df = pd.DataFrame({"time": horizon_idx, "forecast": last_pred})
    forecast_df.to_csv(run_dir / "forecast_next_6h.csv", index=False)

    print(f"\nArtifacts saved to: {run_dir}")
    print(f"Example forecast saved to: {run_dir / 'forecast_next_6h.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", repr(e))
        sys.exit(1)
