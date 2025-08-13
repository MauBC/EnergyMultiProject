# appFuturo.py
# Reusar un modelo entrenado para exportar predicciones_vs_reales.csv
# y (opcional) graficar comparativa por semana.
# Comentarios sin acentos para evitar problemas.

import argparse
import json
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Importa utilidades del script de entrenamiento
from app import LSTMForecast, resample_hourly, make_time_features, build_sequences

def latest_run(base="outputs_lstm6h"):
    base = Path(base)
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        raise FileNotFoundError("No hay carpetas run_* en outputs_lstm6h")
    return runs[-1]

def smape(y_true, y_pred, eps=1e-6):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    diff = np.abs(y_true - y_pred)
    return 100.0 * float(np.mean(diff / denom))

def mape_safe(y_true, y_pred, eps=1e-6):
    return 100.0 * float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def main():
    ap = argparse.ArgumentParser(description="Reusar modelo entrenado para exportar predicciones_vs_reales.csv y graficar.")
    ap.add_argument("--csv", required=True, help="Ruta al CSV original (ej. COLCA_SOLAR_MERGED.csv)")
    ap.add_argument("--run_dir", default="", help="outputs_lstm6h\\run_YYYYmmdd_HHMMSS. Si vacio, usa el ultimo run")
    ap.add_argument("--out_name", default="predicciones_vs_reales.csv", help="Nombre del CSV a exportar en el run_dir")
    # Opcional: grafico y metricas en una ventana
    ap.add_argument("--start", default="", help="YYYY-MM-DD inicio para grafico/metricas (opcional)")
    ap.add_argument("--end", default="", help="YYYY-MM-DD fin exclusivo para grafico/metricas (opcional)")
    ap.add_argument("--h", type=int, default=1, help="Horizonte a graficar (1..6). Use 0 para promedio de horizontes.")
    ap.add_argument("--save_plot", default="", help="Archivo PNG de salida para grafico (opcional)")
    args = ap.parse_args()

    # 1) Seleccionar run
    run_dir = Path(args.run_dir) if args.run_dir else latest_run("outputs_lstm6h")
    print("Usando run_dir:", run_dir)

    # 2) Cargar artefactos del MISMO run
    cfg = json.load(open(run_dir / "config.json", "r", encoding="utf-8"))
    columns = json.load(open(run_dir / "columns.json", "r", encoding="utf-8"))
    features_expected = columns["features"]
    target = columns["target"]
    lookback = int(cfg["lookback"]); horizon = int(cfg["horizon"])

    scaler_X = joblib.load(run_dir / "scaler_X.joblib")
    scaler_y = joblib.load(run_dir / "scaler_y.joblib")

    model = LSTMForecast(
        input_size=len(features_expected) + 1,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        horizon=horizon
    )
    # Tus archivos son de confianza
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu",weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Preparar dataset como en entrenamiento
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Filtro DNI > 0 si existe (insensible a mayusculas)
    dni_col = next((c for c in df.columns if c.strip().lower() == "dni"), None)
    if dni_col is not None:
        before = len(df)
        df = df[df[dni_col] > 0].copy()
        print(f"Filtro DNI>0: {before} -> {len(df)} filas")

    # Detectar datetime
    dt_col = "datetime"
    if dt_col not in df.columns:
        raise ValueError("No se encontro columna 'datetime' en el CSV")
    df[dt_col] = pd.to_datetime(df[dt_col], infer_datetime_format=True)

    # Resampleo igual que en entrenamiento
    df = resample_hourly(df, dt_col, how=cfg.get("resample", "mean"))

    # Numericos + interpolacion temporal
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.interpolate(method="time").ffill().bfill()

    # Time features
    tf = make_time_features(df.index)
    df = pd.concat([df, tf], axis=1)

    # Verificar columnas esperadas
    missing = [c for c in features_expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas respecto al entrenamiento: {missing}")

    # Seleccionar features exactas y en el mismo orden
    Xdf = df[features_expected].copy()
    ydf = df[[target]].copy()

    # Escalado
    X_scaled = scaler_X.transform(Xdf.values)
    y_scaled = scaler_y.transform(ydf.values).reshape(-1)

    scaled_df = pd.DataFrame(X_scaled, index=df.index, columns=features_expected)
    scaled_df[target] = y_scaled

    # Secuencias
    X, y = build_sequences(
        scaled_df[features_expected + [target]],
        target=target,
        lookback=lookback,
        horizon=horizon
    )
    if len(X) == 0:
        raise RuntimeError("No hay ventanas suficientes para construir secuencias. Revisa lookback y longitud del dataset.")

    # 4) Inferencia sobre todas las ventanas
    preds, trues = [], []
    with torch.no_grad():
        for i in range(len(X)):
            out = model(torch.tensor(X[i:i+1], dtype=torch.float32)).cpu().numpy()
            preds.append(out); trues.append(y[i])

    preds = np.vstack(preds)            # [N, horizon]
    trues = np.vstack(trues)            # [N, horizon]

    # Invertir escalado del target
    preds_real = scaler_y.inverse_transform(preds)
    trues_real = scaler_y.inverse_transform(trues)

    # Mapear timestamps: cada ventana termina en un timestamp
    start_idx = lookback
    fechas = df.index[start_idx : start_idx + len(preds_real)]

    out_df = pd.DataFrame({
        "datetime": np.repeat(fechas, horizon),
        "horizon_h": np.tile(np.arange(1, horizon+1), len(fechas)),
        "real": trues_real.flatten(),
        "pred": preds_real.flatten()
    })

    out_csv = run_dir / args.out_name
    out_df.to_csv(out_csv, index=False)
    print("Guardado CSV:", out_csv)

    # 5) Opcional: metricas y grafico para una semana
    if args.start and args.end and args.save_plot:
        dfw = pd.read_csv(out_csv, parse_dates=["datetime"])
        m = (dfw["datetime"] >= pd.to_datetime(args.start)) & (dfw["datetime"] < pd.to_datetime(args.end))
        dfw = dfw.loc[m].copy()
        if dfw.empty:
            raise ValueError("No hay datos en el rango seleccionado para grafico/metricas")

        if args.h > 0:
            dfw = dfw[dfw["horizon_h"] == args.h].copy()
            title_h = f"h+{args.h}"
        else:
            dfw = dfw.groupby("datetime", as_index=False).agg({"real":"mean","pred":"mean"})
            dfw["horizon_h"] = 0
            title_h = "promedio_horizontes"

        dfw = dfw.sort_values("datetime").reset_index(drop=True)
        y_true = dfw["real"].to_numpy(dtype=float)
        y_pred = dfw["pred"].to_numpy(dtype=float)

        mae_v = mae(y_true, y_pred)
        rmse_v = rmse(y_true, y_pred)
        smape_v = smape(y_true, y_pred)
        mape_safe_v = mape_safe(y_true, y_pred)
        acc_pct = max(0.0, 100.0 - mape_safe_v)
        bias_pct = 100.0 * (float(np.mean(y_pred)) - float(np.mean(y_true))) / (float(np.mean(y_true)) + 1e-6)

        print("=== Metrics (semana seleccionada) ===")
        print(f"Rango: {args.start} a {args.end} | Horizonte: {title_h}")
        print(f"MAE: {mae_v:.6f}")
        print(f"RMSE: {rmse_v:.6f}")
        print(f"sMAPE_%: {smape_v:.2f}")
        print(f"MAPE_safe_%: {mape_safe_v:.2f}")
        print(f"Accuracy_% (100 - MAPE_safe): {acc_pct:.2f}")
        print(f"Bias_% (signo + sobrepredice): {bias_pct:.2f}")

        plt.figure(figsize=(12,5))
        plt.plot(dfw["datetime"], y_true, label="Real")
        plt.plot(dfw["datetime"], y_pred, label="Pred")
        plt.title(f"Real vs Pred ({title_h}) | {args.start} a {args.end}")
        plt.xlabel("datetime")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=140)
        print("Grafico guardado en:", args.save_plot)

if __name__ == "__main__":
    main()
