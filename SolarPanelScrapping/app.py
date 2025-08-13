# app.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Asegura que podamos importar los módulos dentro de WebScrapping
BASE_DIR = Path(__file__).resolve().parent
SCR_DIR = BASE_DIR / "WebScrapping"
sys.path.append(str(SCR_DIR))

# Importa tus módulos
try:
    from WebScrapping.coes import run_coes_rango as coes_run
    from WebScrapping.nasa import run as nasa_run
    from WebScrapping.merge import run_merge
except Exception as e:
    raise RuntimeError(
        "No pude importar los módulos. Asegúrate de que existen WebScrapping/coes.py, WebScrapping/nasa.py y WebScrapping/merge.py"
    ) from e


def ymd(s: str) -> str:
    """Valida/normaliza fecha YYYY-MM-DD a YYYY-MM-DD, retorna str."""
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def yyyymmdd(s: str) -> str:
    """Convierte YYYY-MM-DD -> YYYYMMDD (para NASA)."""
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")


def main():
    p = argparse.ArgumentParser(
        description="Pipeline: COES + NASA POWER (horario) + merge en ./data"
    )
    p.add_argument("--start", required=True, help="Fecha inicio (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="Fecha fin (YYYY-MM-DD) (exclusiva o inclusiva según tu scraping)")
    p.add_argument("--lat", type=float, default=-17.641, help="Latitud (NASA)")
    p.add_argument("--lon", type=float, default=-71.3425, help="Longitud (NASA)")
    p.add_argument("--tz", default="LST", choices=["UTC", "LST"], help="time-standard para NASA (recomiendo LST)")
    args = p.parse_args()

    start_coes = ymd(args.start)     # COES usa YYYY-MM-DD
    end_coes   = ymd(args.end)
    start_nasa = yyyymmdd(args.start)  # NASA usa YYYYMMDD
    end_nasa   = yyyymmdd(args.end)

    out_dir = BASE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    '''
    # 1) COES -> COES.csv en ./data
    print("\n=== Paso 1/3: Descargando COES y generando COES.csv ===")
    coes_csv_path = coes_run(
    fecha_ini=start_coes,
    fecha_fin=end_coes,
    out_temp_dir="data/temp",
    out_final_dir=str(out_dir),
    out_final="COES.csv",
    )
    print(f"OK -> {coes_csv_path}")
    '''
    # 2) NASA -> NASA.csv en ./data
    print("\n=== Paso 2/3: Descargando NASA POWER (hourly) y generando NASA.csv ===")
    nasa_csv_path = nasa_run(
        lat=args.lat,
        lon=args.lon,
        start_yyyymmdd=start_nasa,
        end_yyyymmdd=end_nasa,
        time_standard=args.tz,   # LST recomendado para alinear con hora local
        out_dir=str(out_dir),
        out_name="NASA.csv",
    )
    print(f"OK -> {nasa_csv_path}")

    # 3) Merge -> MERGED.csv en ./data
    print("\n=== Paso 3/3: Agregando COES 15-min -> 1H (promedio) y merge con NASA ===")
    merged_path = run_merge(base_dir=str(out_dir), out_name="MERGED.csv")
    print(f"\n✅ Pipeline completo. Archivo combinado: {merged_path}")


if __name__ == "__main__":
    main()
