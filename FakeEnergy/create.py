import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Crear tiempo simulado (1 día, datos por minuto) ---
time_range = pd.date_range(start="2023-01-01", periods=24*60, freq="T")  # 1440 minutos

# --- 2. Generar energía base ---
np.random.seed(42)
energia_base = 5 + np.sin(np.linspace(0, 6*np.pi, len(time_range))) + np.random.normal(0, 0.3, len(time_range))

# --- 3. Insertar un pico artificial ---
energia_simulada = energia_base.copy()
pico_inicio = 600   # minuto donde empieza el pico
pico_fin = 660      # minuto donde termina el pico
energia_simulada[pico_inicio:pico_fin] += 10  # subir energía en ese rango

# --- 4. Crear DataFrame ---
df_fake = pd.DataFrame({
    "timestamp": time_range,
    "energia": energia_simulada*1000
})

# --- 5. Graficar ---
plt.figure(figsize=(12,6))
plt.plot(df_fake["timestamp"], df_fake["energia"], label="Energía simulada con pico")
plt.xlabel("Tiempo")
plt.ylabel("Energía (Wh)")
plt.legend()
plt.grid(True)
plt.show()

print(df_fake.head())
