import pandas as pd
from pathlib import Path

def load_and_aggregate_coes(coes_csv="COES.csv"):
    df = pd.read_csv(coes_csv)
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)  # 'fechahora' -> 'datetime'
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
    plant_cols = [c for c in df.columns if c != "datetime"]
    for c in plant_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # promedio por hora
    dfh = df.set_index("datetime").resample("1H").mean().reset_index()
    return dfh

def load_nasa(nasa_csv="NASA.csv"):
    df = pd.read_csv(nasa_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

def run_merge(base_dir=".", out_name="MERGED.csv"):
    base = Path(base_dir)
    coes_h = load_and_aggregate_coes(base / "TACNA_SOLAR.csv")
    nasa_h = load_nasa(base / "NASA.csv")
    merged = pd.merge(nasa_h, coes_h, on="datetime", how="inner")
    cols_first = ["datetime", "DNI", "Temperatura", "Viento", "Humedad"]
    others = [c for c in merged.columns if c not in cols_first]
    merged = merged[cols_first + others]
    out_path = base / out_name
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Merged listo: {out_path}")
    return out_path
