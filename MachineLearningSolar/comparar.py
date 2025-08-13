    # viz_preds_vs_real.py
# Visualizacion de predicciones vs reales y porcentaje de error
# Requisitos: pandas, numpy, matplotlib, scikit-learn (opcional para R2)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ====== CONFIG ======
# ajusta estas rutas y nombres a tu caso
'''COLCA
preds_csv = r"outputs\COLCA\lb168_hz48\preds_multistep_lb168_hz48.csv"
plant = "COLCA_SOLAR"
seq_idx = 0
'''
'''Arcus
preds_csv = r"outputs\ARCUS\lb168_hz48\preds_multistep_lb168_hz48.csv"
plant = "REPARTICION_ARCUS"
seq_idx = 0

'''
'''Tacna
preds_csv = r"outputs\TACNA\lb168_hz48\preds_multistep_lb168_hz48.csv"
plant = "TACNA_SOLAR"
seq_idx = 0

'''
preds_csv = r"outputs\COLCA\lb168_hz48\preds_multistep_lb168_hz48.csv"
plant = "COLCA_SOLAR"
seq_idx = 0
out_dir = os.path.dirname(preds_csv)

# ====== funciones de error en porcentaje ======
def mape_safe(y_true, y_pred, eps=1e-6):
    """
    MAPE seguro en %, ignora puntos donde y_true ~ 0 para evitar division por cero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100.0

def smape(y_true, y_pred, eps=1e-6):
    """
    sMAPE en %, robusto cuando hay ceros.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0

def ape_per_step(y_true, y_pred, eps=1e-6):
    """
    Absolute Percentage Error por paso en %, con proteccion ante ceros.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.abs(y_pred - y_true) / denom * 100.0

# ====== carga de datos ======
df = pd.read_csv(preds_csv)
sub = df[df["seq_idx"] == seq_idx].sort_values("h").reset_index(drop=True)

y_true = sub[f"real_{plant}"].values
y_mlp  = sub[f"mlp_{plant}"].values
y_gru  = sub[f"gru_{plant}"].values
H = len(sub)

# ====== metricas en porcentaje ======
mape_mlp = mape_safe(y_true, y_mlp)
mape_gru = mape_safe(y_true, y_gru)
smape_mlp = smape(y_true, y_mlp)
smape_gru = smape(y_true, y_gru)
r2_mlp = r2_score(y_true, y_mlp)
r2_gru = r2_score(y_true, y_gru)

print("=== Metricas porcentaje y R2 (seq_idx = {}) ===".format(seq_idx))
print("MLP  MAPE_safe: {:6.2f}% | sMAPE: {:6.2f}% | R2: {:6.3f}".format(mape_mlp, smape_mlp, r2_mlp))
print("GRU  MAPE_safe: {:6.2f}% | sMAPE: {:6.2f}% | R2: {:6.3f}".format(mape_gru, smape_gru, r2_gru))

# ====== grafico linea: real vs pred ======
plt.figure(figsize=(12, 4))
plt.plot(y_true, label=f"Real {plant}")
plt.plot(y_mlp, label="MLP")
plt.plot(y_gru, label="GRU")
plt.title(f"Prediccion vs Real - {plant} - seq_idx={seq_idx}")
plt.xlabel("Paso de horizonte (horas)")
plt.ylabel("Energia")
plt.legend()
out_png = os.path.join(out_dir, f"viz_seq{seq_idx}_{plant}.png")
plt.savefig(out_png, bbox_inches="tight", dpi=120)
plt.close()
print(f"[SAVE] Grafico linea: {out_png}")

# ====== barras de error por paso (APE %) ======
ape_mlp = ape_per_step(y_true, y_mlp)  # vector de largo H
ape_gru = ape_per_step(y_true, y_gru)

plt.figure(figsize=(12, 4))
plt.bar(np.arange(H), ape_mlp)
plt.title(f"APE por paso (MLP) - {plant} - seq_idx={seq_idx}")
plt.xlabel("Paso (h)")
plt.ylabel("APE (%)")
out_png2 = os.path.join(out_dir, f"viz_seq{seq_idx}_{plant}_APE_MLP.png")
plt.savefig(out_png2, bbox_inches="tight", dpi=120)
plt.close()
print(f"[SAVE] Barras APE MLP: {out_png2}")

plt.figure(figsize=(12, 4))
plt.bar(np.arange(H), ape_gru)
plt.title(f"APE por paso (GRU) - {plant} - seq_idx={seq_idx}")
plt.xlabel("Paso (h)")
plt.ylabel("APE (%)")
out_png3 = os.path.join(out_dir, f"viz_seq{seq_idx}_{plant}_APE_GRU.png")
plt.savefig(out_png3, bbox_inches="tight", dpi=120)
plt.close()
print(f"[SAVE] Barras APE GRU: {out_png3}")
