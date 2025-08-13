# comparar.py
# Visualizacion y metricas de predicciones vs reales (desde preds_multistep_...csv)
# Modo secuencia o global; soporta "solo diurno" con umbral configurable.
# Requisitos: pandas, numpy, matplotlib, scikit-learn

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --------- metricas en porcentaje ----------
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

def ape_per_step(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.abs(y_pred - y_true) / denom * 100.0

def summarize(y_true, y_mlp, y_gru, label_prefix=""):
    out = {}
    out[f"{label_prefix}MLP_MAPE%"]   = mape_safe(y_true, y_mlp)
    out[f"{label_prefix}MLP_sMAPE%"]  = smape(y_true, y_mlp)
    out[f"{label_prefix}MLP_R2"]      = r2_score(y_true, y_mlp)
    out[f"{label_prefix}GRU_MAPE%"]   = mape_safe(y_true, y_gru)
    out[f"{label_prefix}GRU_sMAPE%"]  = smape(y_true, y_gru)
    out[f"{label_prefix}GRU_R2"]      = r2_score(y_true, y_gru)
    return out

# --------- umbral de dia ----------
def day_mask_from_threshold(y_true, mode="frac_p95", thr=0.05):
    yt = np.asarray(y_true, dtype=np.float64)
    if mode == "abs":
        threshold = float(thr)
    elif mode == "frac_max":
        m = np.nanmax(yt) if np.isfinite(yt).any() else 0.0
        threshold = float(thr) * m
    else:  # "frac_p95" por defecto
        p95 = np.nanpercentile(yt, 95) if np.isfinite(yt).any() else 0.0
        threshold = float(thr) * p95
    return yt > threshold, threshold

def plot_seq(y_true, y_mlp, y_gru, plant, seq_idx, out_dir):
    plt.figure(figsize=(12,4))
    plt.plot(y_true, label=f"Real {plant}")
    plt.plot(y_mlp,  label="MLP")
    plt.plot(y_gru,  label="GRU")
    plt.title(f"Prediccion vs Real - {plant} - seq_idx={seq_idx}")
    plt.xlabel("Paso de horizonte (horas)")
    plt.ylabel("Energia")
    plt.legend()
    out_png = os.path.join(out_dir, f"viz_seq{seq_idx}_{plant}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[SAVE] Grafico linea: {out_png}")

def plot_ape(ape_vec, plant, seq_idx, model_tag, out_dir):
    H = len(ape_vec)
    plt.figure(figsize=(12,4))
    plt.bar(np.arange(H), ape_vec)
    plt.title(f"APE por paso ({model_tag}) - {plant} - seq_idx={seq_idx}")
    plt.xlabel("Paso (h)")
    plt.ylabel("APE (%)")
    out_png = os.path.join(out_dir, f"viz_seq{seq_idx}_{plant}_APE_{model_tag}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[SAVE] Barras APE {model_tag}: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True, help="Ruta al preds_multistep_lb...csv")
    ap.add_argument("--plant", type=str, required=True, help="Nombre de la columna objetivo (ej. COLCA_SOLAR)")
    ap.add_argument("--seq", type=int, default=0, help="Indice de secuencia para vista 'seq'")
    ap.add_argument("--mode", choices=["seq","global"], default="seq", help="seq=una secuencia, global=todas")
    ap.add_argument("--dayonly", action="store_true", help="Calcular tambien metricas 'solo diurno'")
    ap.add_argument("--thr-mode", choices=["abs","frac_max","frac_p95"], default="frac_p95",
                   help="Modo de umbral: abs valor fijo, frac_max=frac del max, frac_p95=frac del p95")
    ap.add_argument("--thr", type=float, default=0.05, help="Valor/Fraccion para el umbral de dia")
    args = ap.parse_args()

    preds_csv = args.preds
    plant = args.plant
    seq_idx = args.seq
    out_dir = os.path.dirname(preds_csv) if os.path.dirname(preds_csv) else "."

    df = pd.read_csv(preds_csv)
    # columnas esperadas: seq_idx, h, real_<plant>, mlp_<plant>, gru_<plant>
    rcol = f"real_{plant}"; mcol = f"mlp_{plant}"; gcol = f"gru_{plant}"
    for c in [rcol, mcol, gcol, "seq_idx", "h"]:
        if c not in df.columns:
            raise ValueError(f"Falta columna: {c}")

    if args.mode == "seq":
        sub = df[df["seq_idx"] == seq_idx].sort_values("h").reset_index(drop=True)
        y_true = sub[rcol].values
        y_mlp  = sub[mcol].values
        y_gru  = sub[gcol].values

        print(f"=== Metricas (seq_idx = {seq_idx}) ===")
        base = summarize(y_true, y_mlp, y_gru, "")
        for k,v in base.items(): print(f"{k}: {v:8.3f}")

        if args.dayonly:
            mask, thr_val = day_mask_from_threshold(y_true, args.thr_mode, args.thr)
            print(f"Dia: {mask.sum()} de {len(mask)} puntos | thr({args.thr_mode})={thr_val:.6f}")
            if mask.any():
                day = summarize(y_true[mask], y_mlp[mask], y_gru[mask], "DayOnly_")
                for k,v in day.items(): print(f"{k}: {v:8.3f}")

        # graficos
        plot_seq(y_true, y_mlp, y_gru, plant, seq_idx, out_dir)
        ape_mlp = ape_per_step(y_true, y_mlp)
        ape_gru = ape_per_step(y_true, y_gru)
        plot_ape(ape_mlp, plant, seq_idx, "MLP", out_dir)
        plot_ape(ape_gru, plant, seq_idx, "GRU", out_dir)

    else:  # global
        yt = df[rcol].values
        ym = df[mcol].values
        yg = df[gcol].values

        print("=== Metricas globales (todas las secuencias) ===")
        base = summarize(yt, ym, yg, "")
        for k,v in base.items(): print(f"{k}: {v:8.3f}")

        if args.dayonly:
            mask, thr_val = day_mask_from_threshold(yt, args.thr_mode, args.thr)
            print(f"Dia: {mask.sum()} de {len(mask)} puntos | thr({args.thr_mode})={thr_val:.6f}")
            if mask.any():
                day = summarize(yt[mask], ym[mask], yg[mask], "DayOnly_")
                for k,v in day.items(): print(f"{k}: {v:8.3f}")

        # grafico promedio por paso del horizonte
        H = int(df["h"].max() + 1)
        grp = df.groupby("h")[[rcol, mcol, gcol]].mean().reset_index()
        plt.figure(figsize=(12,4))
        plt.plot(grp["h"], grp[rcol], label=f"Real {plant} (prom)")
        plt.plot(grp["h"], grp[mcol], label="MLP (prom)")
        plt.plot(grp["h"], grp[gcol], label="GRU (prom)")
        plt.title(f"Promedio por paso del horizonte - {plant}")
        plt.xlabel("Paso (h)")
        plt.ylabel("Energia")
        plt.legend()
        out_png = os.path.join(out_dir, f"viz_global_avg_{plant}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"[SAVE] Grafico promedio global: {out_png}")

        # APE promedio por paso
        apes = []
        for h, sub in df.groupby("h"):
            apes.append({
                "h": h,
                "MLP_APE%": np.mean(ape_per_step(sub[rcol].values, sub[mcol].values)),
                "GRU_APE%": np.mean(ape_per_step(sub[rcol].values, sub[gcol].values)),
            })
        apedf = pd.DataFrame(apes).sort_values("h")
        plt.figure(figsize=(12,4))
        plt.plot(apedf["h"], apedf["MLP_APE%"], label="MLP APE% (prom)")
        plt.plot(apedf["h"], apedf["GRU_APE%"], label="GRU APE% (prom)")
        plt.title(f"APE promedio por paso - {plant}")
        plt.xlabel("Paso (h)")
        plt.ylabel("APE (%)")
        plt.legend()
        out_png2 = os.path.join(out_dir, f"viz_global_ape_{plant}.png")
        plt.savefig(out_png2, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"[SAVE] APE promedio por paso: {out_png2}")

if __name__ == "__main__":

    #COLCA python compararV2.py --preds outputs\COLCA\lb168_hz48\preds_multistep_lb168_hz48.csv --plant COLCA_SOLAR --mode global --dayonly --thr-mode frac_p95 --thr 0.05

    main()
