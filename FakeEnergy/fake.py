import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('data.csv')
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()

if not df.empty:
    print("Información básica del DataFrame:")
    df.info()

    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head())

    print("\nEstadísticas descriptivas de las columnas numéricas:")
    print(df.describe())
else:
    print("El DataFrame está vacío, no se puede continuar con el análisis.")

# --- 1. Limpieza y Formateo ---
# Convertir el timestamp de Unix a datetime y ponerlo como índice
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)

# --- 2. Crear Métrica de Potencia Principal ---
# Calculamos la potencia activa promedio para cada fase en ese minuto
df['a_avg_act_power'] = (df['a_min_act_power'] + df['a_max_act_power']) / 2
df['b_avg_act_power'] = (df['b_min_act_power'] + df['b_max_act_power']) / 2
df['c_avg_act_power'] = (df['c_min_act_power'] + df['c_max_act_power']) / 2

# Sumamos las potencias promedio de cada fase para obtener la potencia total
#df['total_act_power'] = (df['a_avg_act_power'] + df['b_avg_act_power'] + df['c_avg_act_power'])*1.12
df['total_act_power'] = (df['a_avg_act_power'] + df['b_avg_act_power'] + df['c_avg_act_power'])*1.12 ### AGREGARLE ENERGIA


# --- 3. Ingeniería de Características Cíclicas ---
df['hora_del_dia'] = df.index.hour
df['dia_de_la_semana'] = df.index.dayofweek # Lunes=0, Domingo=6

# Transformación Seno/Coseno para la Hora
df['hora_sin'] = np.sin((df['hora_del_dia'] / 24) * 2 * np.pi)
df['hora_cos'] = np.cos((df['hora_del_dia'] / 24) * 2 * np.pi)

# Transformación Seno/Coseno para el Día
df['dia_sin'] = np.sin((df['dia_de_la_semana'] / 7) * 2 * np.pi)
df['dia_cos'] = np.cos((df['dia_de_la_semana'] / 7) * 2 * np.pi)
df_limpio = df[['total_act_power', 'hora_del_dia', 'dia_de_la_semana', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']].copy()


### MODIFICAR DATOS ENTRE 00:00 Y 06:00 PARA SIMULAR CONSUMO NOCTURNO
# Crear una máscara para las horas entre 0 y 5 am
mask = (df_limpio.index.hour >= 0) & (df_limpio.index.hour < 6)

# Reemplazar esos valores con aleatorios entre 300 y 500
df_limpio.loc[mask, 'total_act_power'] = np.random.randint(300, 501, mask.sum())

print("Valores modificados entre 00:00 y 05:00:")
print(df_limpio.loc[mask, 'total_act_power'].head(10))
######################

print("DataFrame Limpio y Preparado (primeras 5 filas):")
print(df_limpio.head())
print("----------------------------------------")
print(df_limpio['total_act_power'][0:10])
print("----------------------------------------")

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(df_limpio.index, df_limpio['total_act_power'])
plt.title('Consumo de Potencia Activa Total (6 Días)')
plt.xlabel('Fecha y Hora')
plt.ylabel('Potencia (W)')
plt.grid(True)
plt.savefig('consumo_6_dias.png')

print("\nSe ha generado un gráfico 'consumo_6_dias.png' con la visión general del consumo.")


# ---  Filtrar solo un día específico ---
# Por ejemplo: 2023-01-16
dia_especifico = "2023-01-17"

df_dia = df_limpio.loc[dia_especifico]

print(f"Mostrando datos de {dia_especifico}:")
print(df_dia.head())

# --- Graficar solo ese día ---
plt.figure(figsize=(15, 6))
plt.plot(df_dia.index, df_dia['total_act_power'])
plt.title(f'Consumo de Potencia Activa Total - {dia_especifico}')
plt.xlabel('Hora')
plt.ylabel('Potencia (W)')
plt.grid(True)
plt.savefig(f'consumo_{dia_especifico}.png')

print(f"\nSe ha generado un gráfico 'consumo_{dia_especifico}.png' con las 24 horas.")
