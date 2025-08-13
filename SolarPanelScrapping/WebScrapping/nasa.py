# -*- coding: utf-8 -*-
import requests
import pandas as pd
from pathlib import Path

API_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# ids de parametros que pediras
DEFAULT_PARAMS = ["T2M", "ALLSKY_SFC_SW_DWN", "QV2M", "WS10M"]

def fetch_hourly(lat, lon, start_yyyymmdd, end_yyyymmdd, params=DEFAULT_PARAMS, community="RE", time_standard="UTC", timeout=60):
    """
    Descarga datos horarios de NASA POWER y retorna un dict {param: {timestamp: value}}
    """
    query = {
        "parameters": ",".join(params),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start_yyyymmdd,
        "end": end_yyyymmdd,
        "format": "JSON",
        "time-standard": time_standard,  # "UTC" o "LST"
    }
    # construye URL manualmente para evitar problemas con coma en parameters
    url = (f"{API_BASE}?parameters={query['parameters']}"
           f"&community={query['community']}&longitude={query['longitude']}&latitude={query['latitude']}"
           f"&start={query['start']}&end={query['end']}&format={query['format']}"
           f"&time-standard={query['time-standard']}")
    print("GET:", url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "properties" not in data or "parameter" not in data["properties"]:
        raise ValueError("Respuesta sin propiedades.parameter")
    return data["properties"]["parameter"]

def to_dataframe(parameter_block, params=DEFAULT_PARAMS):
    """
    Convierte el bloque 'parameter' a un dataframe wide por timestamp.
    parameter_block: dict {param: {timestamp: value}}
    """
    dfs = []
    for p in params:
        series_dict = parameter_block.get(p, {})
        # reemplaza -999 por NaN
        clean = {k: (pd.NA if v == -999 else v) for k, v in series_dict.items()}
        s = pd.Series(clean, name=p)
        dfs.append(s)
    df = pd.concat(dfs, axis=1)
    # timestamp horario suele venir como YYYYMMDDHH (ej: 2019010100)
    # conviertelo a datetime; si viniera con otro formato, ajusta aqui
    df.index.name = "timestamp"
    # intenta parsear con %Y%m%d%H; si falla, prueba %Y%m%d%H%M
    try:
        dt = pd.to_datetime(df.index.astype(str), format="%Y%m%d%H", errors="coerce")
    except Exception:
        dt = pd.to_datetime(df.index.astype(str), errors="coerce")
    df.insert(0, "datetime", dt)
    df = df.reset_index(drop=True).sort_values("datetime")
    return df

# Ajusta tu DEFAULT_PARAMS a esto en tu script:
DEFAULT_PARAMS = ["ALLSKY_SFC_SW_DNI", "T2M", "WS10M", "QV2M"]

def run(lat=-17.641, lon=-71.3425, start_yyyymmdd="20190101", end_yyyymmdd="20191231",
        params=DEFAULT_PARAMS, time_standard="UTC", out_dir=".", out_name=None):
    """
    Descarga datos horarios POWER y guarda CSV. Retorna ruta del CSV.
    - lat, lon: floats
    - start_yyyymmdd, end_yyyymmdd: strings YYYYMMDD
    - params: lista de IDs (por defecto: ALLSKY_SFC_SW_DNI, T2M, WS10M, QV2M)
    - time_standard: "UTC" o "LST"
    """
    # fuerza a pedir solo los 4 parametros que quieres, en el orden deseado
    wanted = ["ALLSKY_SFC_SW_DNI", "T2M", "WS10M", "QV2M"]
    params = wanted

    parameter_block = fetch_hourly(
        lat, lon, start_yyyymmdd, end_yyyymmdd,
        params=params, time_standard=time_standard
    )
    df = to_dataframe(parameter_block, params=params)

    # deja solo esas columnas y renombra a español
    keep_order = ["datetime", "ALLSKY_SFC_SW_DNI", "T2M", "WS10M", "QV2M"]
    df = df[keep_order].rename(columns={
        "ALLSKY_SFC_SW_DNI": "DNI",
        "T2M": "Temperatura",
        "WS10M": "Viento",
        "QV2M": "Humedad"
    })

    if out_name is None:
        out_name = f"power_hourly_{lat}_{lon}_{start_yyyymmdd}_{end_yyyymmdd}.csv"

    out_path = Path(out_dir) / out_name
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"CSV guardado en: {out_path}")
    return str(out_path)

if __name__ == "__main__":
    # ejemplo: 2019-01-01 a 2019-02-01
    run(
        lat=-11.038056,
        lon=-77.096389,
        start_yyyymmdd="20250531", #2025-05-31
        end_yyyymmdd="20250731", #2025-07-31
        # no pases DWN; ya lo forzamos a DNI arriba
        time_standard="LST",
        out_dir="data",
        out_name="NASA_COESFIX.csv"

    )
