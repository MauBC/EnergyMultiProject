import pandas as pd
from pathlib import Path

# Ruta al CSV original
archivo_original = "COES.csv"

# Columnas de interés
col_fechahora = "fechahora"
plantas = [
    "COLCA_SOLAR",
    "REPARTICION_ARCUS",
    "TACNA_SOLAR"
]

# Carpeta de salida
out_dir = Path("salida_plantas")
out_dir.mkdir(parents=True, exist_ok=True)

# Cargar CSV completo
df = pd.read_csv(archivo_original)

# Generar un CSV por planta
for planta in plantas:
    if planta not in df.columns:
        print(f"⚠ La columna '{planta}' no existe en el CSV.")
        continue

    df_planta = df[[col_fechahora, planta]].copy()
    nombre_archivo = planta.replace(" ", "_").replace(".", "").replace("/", "_") + ".csv"
    out_path = out_dir / nombre_archivo

    df_planta.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Guardado: {out_path}")
