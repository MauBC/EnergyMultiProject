# predict.py
# Generate next-48h forecast from the last window and drop night hours.
# Requires: app.py in same folder (imports models and utilities)

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

# import pieces from your app.py
from app import MLP, GRUModel, cyc_features, MinMax_inverse_like_safe

def load_scalers(models_dir, lb, hz):
    sx = joblib.load(os.path.join(models_dir, f"scaler_X_lb{lb}_hz{hz}.pkl"))
    sy = joblib.load(os.path.join(models_dir, f"scaler_y_lb{lb}_hz{hz}.pkl"))
    return sx, sy

def build_last_window(df, datetime_col, lb):
    df2 = cyc_features(df[[datetime_col, "DNI","Temperatura","Viento","Humedad"]].copy(), dt_col=datetime_col)
    feats = ["DNI","Temperatura","Viento","Humedad","hour_sin","hour_cos","month_sin","month_cos"]
    X = df2[feats].values
    if X.shape[0] < lb:
        raise ValueError("not enough rows for look_back")
    return X[-lb:, :], feats  # (lb, n_features)

def future_timestamps(last_dt, horizon):
    rng = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=horizon, freq="H")
    return rng

def predict_mlp(Xw_raw, scaler_X, scaler_y, models_dir, lb, hz, n_features, n_targets, device="cpu"):
    Xw = scaler_X.transform(Xw_raw)[None, :, :].astype(np.float32)            # (1, lb, F)
    Xw_mlp = Xw.reshape(1, lb * n_features)
    model = MLP(input_dim=lb*n_features, output_dim=hz*n_targets, dropout=0.0)
    model.load_state_dict(torch.load(os.path.join(models_dir, f"mlp_multistep_lb{lb}_hz{hz}.pt"), map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        y_scaled = model(torch.from_numpy(Xw_mlp).to(device)).cpu().numpy()   # (1, hz*n_targets)
    y = MinMax_inverse_like_safe(y_scaled, scaler_y, n_targets, hz)[0]        # (hz, n_targets)
    return y

def predict_gru(Xw_raw, scaler_X, scaler_y, models_dir, lb, hz, n_features, n_targets, device="cpu"):
    Xw = scaler_X.transform(Xw_raw)[None, :, :].astype(np.float32)            # (1, lb, F)
    model = GRUModel(n_features=n_features, look_back=lb, output_dim=hz*n_targets,
                     hidden_size=192, n_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(os.path.join(models_dir, f"gru_multistep_lb{lb}_hz{hz}.pt"), map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        y_scaled = model(torch.from_numpy(Xw).to(device)).cpu().numpy()       # (1, hz*n_targets)
    y = MinMax_inverse_like_safe(y_scaled, scaler_y, n_targets, hz)[0]        # (hz, n_targets)
    return y

def day_mask_from_predictions(y_pred, mode="frac_p95", thr=0.05):
    """
    Build a boolean mask for daytime based on predicted magnitude.
    mode:
      - abs: threshold is absolute value
      - frac_max: threshold = thr * max(pred)
      - frac_p95: threshold = thr * p95(pred)   (default)
    """
    y = np.asarray(y_pred, dtype=np.float64).copy()
    ref = np.maximum.reduce([y])  # single series (N, T). if multi-target, change accordingly
    ref = ref.squeeze()
    if mode == "abs":
        t = float(thr)
    elif mode == "frac_max":
        t = float(thr) * (np.nanmax(ref) if np.isfinite(ref).any() else 0.0)
    else:
        p95 = np.nanpercentile(ref, 95) if np.isfinite(ref).any() else 0.0
        t = float(thr) * p95
    return (ref > t), float(t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="input CSV with datetime,DNI,Temp,Viento,Humedad and plant")
    ap.add_argument("--plant", type=str, required=True, help="target column name")
    ap.add_argument("--tag", type=str, required=True, help="dataset tag used in app.py dirs (e.g. COLCA)")
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--outputs-dir", type=str, default="outputs")
    ap.add_argument("--look-back", type=int, default=168)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--model", choices=["mlp","gru","both"], default="both")
    ap.add_argument("--thr-mode", choices=["abs","frac_max","frac_p95"], default="frac_p95")
    ap.add_argument("--thr", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dirs
    mdir = os.path.join(args.models-dir, args.tag, f"lb{args.look_back}_hz{args.horizon}")
    odir = os.path.join(args.outputs_dir, args.tag, f"lb{args.look_back}_hz{args.horizon}")
    os.makedirs(odir, exist_ok=True)

    # load data
    df = pd.read_csv(args.csv)
    if "datetime" not in df.columns:
        raise ValueError("Missing datetime column")
    if args.plant not in df.columns:
        raise ValueError(f"Missing target column {args.plant}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # last window
    Xw_raw, feats = build_last_window(df, "datetime", args.look_back)
    n_features = Xw_raw.shape[1]
    n_targets = 1  # single plant typical

    # scalers
    sx, sy = load_scalers(mdir, args.look_back, args.horizon)

    # predict
    preds = {}
    if args.model in ("mlp", "both"):
        preds["mlp"] = predict_mlp(Xw_raw, sx, sy, mdir, args.look_back, args.horizon, n_features, n_targets, device=device).squeeze()
    if args.model in ("gru", "both"):
        preds["gru"] = predict_gru(Xw_raw, sx, sy, mdir, args.look_back, args.horizon, n_features, n_targets, device=device).squeeze()

    # timestamps for next 48h
    last_dt = df["datetime"].max()
    fut = future_timestamps(last_dt, args.horizon)
    hour = fut.hour

    # build dataframe
    out = pd.DataFrame({"datetime": fut, "hour": hour})
    for k, v in preds.items():
        out[f"{k}_pred_{args.plant}"] = v.astype(float)

    # ensemble column (optional)
    if len(preds) == 2:
        out[f"ens_pred_{args.plant}"] = (out[f"mlp_pred_{args.plant}"] + out[f"gru_pred_{args.plant}"]) / 2.0

    # day mask using predictions (default: frac_p95)
    ref_col = f"ens_pred_{args.plant}" if f"ens_pred_{args.plant}" in out.columns else list(preds.keys())[0] + f"_pred_{args.plant}"
    mask, thr_val = day_mask_from_predictions(out[ref_col].values, mode=args.thr_mode, thr=args.thr)
    out["is_day"] = mask.astype(int)
    print(f"[INFO] day threshold ({args.thr_mode}) = {thr_val:.6f} | day points: {mask.sum()} / {len(mask)}")

    # save all and day-only
    all_path = os.path.join(odir, f"forecast_next{args.horizon}_all.csv")
    day_path = os.path.join(odir, f"forecast_next{args.horizon}_dayonly.csv")
    out.to_csv(all_path, index=False)
    out[out["is_day"] == 1].to_csv(day_path, index=False)
    print(f"[SAVE] {all_path}")
    print(f"[SAVE] {day_path}")

    # simple plot (day-only)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        for c in out.columns:
            if c.endswith(args.plant) and c.startswith(("mlp_","gru_","ens_")):
                plt.plot(out.loc[out["is_day"]==1, "datetime"], out.loc[out["is_day"]==1, c], label=c)
        plt.title(f"Next {args.horizon}h forecast (day-only) - {args.plant}")
        plt.xlabel("datetime")
        plt.ylabel("energy")
        plt.legend()
        png = os.path.join(odir, f"forecast_next{args.horizon}_dayonly_{args.plant}.png")
        plt.savefig(png, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"[SAVE] {png}")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")

if __name__ == "__main__":
    main()
