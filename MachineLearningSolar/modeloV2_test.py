# predict_12h_from_last_window.py
# Script para cargar tu modelo LSTM entrenado en PyTorch y predecir las proximas 12 horas
# usando solo las ultimas LOOKBACK horas del CSV.
# Comentarios sin tildes para evitar problemas de compatibilidad.
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ==============================
# CONFIG
# ==============================
CSV_PATH   = "COLCA_SOLAR_MERGED.csv"      # ruta del csv
MODEL_PATH = "modelo_lstm.pth"             # debe existir (entrenado con estos hiperparametros)
LOOKBACK   = 128                             # DEBE ser el que usaste al entrenar (por ejemplo 48 u 168)
HORIZON    = 12                             # DEBE ser el que usaste al entrenar (12)
BATCH_FIRST = True                          # el modelo fue definido con batch_first=True
USE_GPU = True                              # usa GPU si hay CUDA disponible

# Columnas usadas en el entrenamiento (orden y nombres deben coincidir)
# Ajusta estos nombres a los de tu csv real.
FEATURES = ["DNI", "Temperatura", "Viento", "Humedad", "COLCA_SOLAR"]

# ==============================
# MODELO (debe ser igual al usado al entrenar)
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, horizon=12):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # usamos la ultima salida
        return out

# ==============================
# UTILS
# ==============================
def pick_device():
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    # intentar detectar columna de tiempo
    dt_candidates = [c for c in df.columns if str(c).lower() in ("datetime", "fecha", "fechahora", "timestamp")]
    if not dt_candidates:
        raise ValueError("No se encontro columna de tiempo. Esperaba 'datetime' o 'fechahora'.")
    dt_col = dt_candidates[0]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)
    # validar features
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df

def fit_minmax_on_all(df, feature_names):
    scaler = MinMaxScaler()
    data = df[feature_names].astype(float).dropna().values
    data_scaled = scaler.fit_transform(data)
    return scaler, data_scaled

def inverse_minmax_column(scaled_vals, scaler, col_index):
    # inversa: X = (X_scaled - min_) / scale_
    return (scaled_vals - scaler.min_[col_index]) / scaler.scale_[col_index]

def build_model_and_load(device, input_size, model_path, horizon):
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, horizon=horizon).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo en {model_path}. Entrena primero o revisa la ruta.")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ==============================
# PREDICCION FUTURA DESDE ULTIMA VENTANA
# ==============================
def predict_next_12h_from_last_window(csv_path=CSV_PATH,
                                      model_path=MODEL_PATH,
                                      lookback=LOOKBACK,
                                      horizon=HORIZON,
                                      features=FEATURES,
                                      out_csv="pred_12h.csv"):
    device = pick_device()

    # 1) cargar datos y escalar
    df = load_dataframe(csv_path)
    scaler, data_scaled = fit_minmax_on_all(df, features)

    # 2) preparar ultima secuencia de longitud LOOKBACK
    if data_scaled.shape[0] < lookback:
        raise ValueError(f"No hay suficientes filas para LOOKBACK={lookback}. Filas={data_scaled.shape[0]}")
    last_seq = data_scaled[-lookback:, :]  # shape: (lookback, n_features)

    # 3) cargar modelo
    input_size = len(features)
    model = build_model_and_load(device, input_size, model_path, horizon)

    # 4) predecir
    x_input = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, lookback, n_features)
    with torch.no_grad():
        pred_scaled = model(x_input).cpu().numpy().reshape(-1)  # (horizon,)

    # 5) invertir escala SOLO de la columna energia (ultima columna de FEATURES)
    energy_col_idx = features.index("COLCA_SOLAR")
    pred_inv = inverse_minmax_column(pred_scaled, scaler, energy_col_idx)

    # 6) construir timestamps futuros
    last_time = df.index[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1),
                                 periods=horizon, freq="H")

    # 7) armar tabla
    out_df = pd.DataFrame({
        "datetime": future_times,
        "energia_predicha": pred_inv
    })

    # 8) guardar
    out_df.to_csv(out_csv, index=False)
    return out_df

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    table = predict_next_12h_from_last_window(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        lookback=LOOKBACK,
        horizon=HORIZON,
        features=FEATURES,
        out_csv="pred_12h.csv"
    )
    print("\n=== Prediccion proximas 12 horas ===")
    print(table.to_string(index=False))
    print("\nGuardado en: pred_12h.csv")

    # ==============================
    # GRAFICA DE 12 HORAS FUTURAS
    # ==============================
    plt.figure(figsize=(10,5))
    plt.plot(table["datetime"], table["energia_predicha"], marker="o", label="Prediccion")
    REAL_CSV_PATH = "COLCA_SOLAR_MERGED_Origen.csv"
    # --- Cargar CSV real y graficar ---
    if os.path.exists(REAL_CSV_PATH):
        df_real = pd.read_csv(REAL_CSV_PATH)
        df_real = df_real.tail(12)
        df_real["datetime"] = pd.to_datetime(df_real["datetime"])
        plt.plot(df_real["datetime"], df_real["COLCA_SOLAR"], marker="x", label="Real")

    plt.title("Prediccion vs Real - proximas 12 horas")
    plt.xlabel("Tiempo futuro")
    plt.ylabel("Energia (COLCA_SOLAR)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
