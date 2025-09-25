import pandas as pd
import numpy as np

# === config ===
CSV_IN = "data.csv"
CSV_OUT = "data2.csv"  # opcional

def parse_timestamp(col):
    # si es numerico, asumimos epoch en segundos
    if np.issubdtype(col.dtype, np.number):
        return pd.to_datetime(col, unit="s", errors="coerce")
    # si ya viene string/obj, parse directo
    return pd.to_datetime(col, errors="coerce")

def compute_total_active_power(df):
    """
    retorna serie total_act_power:
    - si ya existe la columna, la usa
    - si no existe, calcula con promedio por fase: (min+max)/2 y suma fases
    """
    if "total_act_power" in df.columns:
        return pd.to_numeric(df["total_act_power"], errors="coerce")

    needed = [
        "a_min_act_power","a_max_act_power",
        "b_min_act_power","b_max_act_power",
        "c_min_act_power","c_max_act_power",
    ]
    if all(c in df.columns for c in needed):
        a_avg = (pd.to_numeric(df["a_min_act_power"], errors="coerce") +
                 pd.to_numeric(df["a_max_act_power"], errors="coerce")) / 2.0
        b_avg = (pd.to_numeric(df["b_min_act_power"], errors="coerce") +
                 pd.to_numeric(df["b_max_act_power"], errors="coerce")) / 2.0
        c_avg = (pd.to_numeric(df["c_min_act_power"], errors="coerce") +
                 pd.to_numeric(df["c_max_act_power"], errors="coerce")) / 2.0
        return a_avg.add(b_avg, fill_value=np.nan).add(c_avg, fill_value=np.nan)

    # fallback si no hay forma de calcular
    return pd.Series(np.nan, index=df.index, dtype="float64")

def compute_total_apparent_power(df):
    """
    calcula potencia aparente total S a partir de *_aprt_power si existen.
    usa promedio (min+max)/2 por fase y suma.
    """
    needed = [
        "a_min_aprt_power","a_max_aprt_power",
        "b_min_aprt_power","b_max_aprt_power",
        "c_min_aprt_power","c_max_aprt_power",
    ]
    if all(c in df.columns for c in needed):
        a_s = (pd.to_numeric(df["a_min_aprt_power"], errors="coerce") +
               pd.to_numeric(df["a_max_aprt_power"], errors="coerce")) / 2.0
        b_s = (pd.to_numeric(df["b_min_aprt_power"], errors="coerce") +
               pd.to_numeric(df["b_max_aprt_power"], errors="coerce")) / 2.0
        c_s = (pd.to_numeric(df["c_min_aprt_power"], errors="coerce") +
               pd.to_numeric(df["c_max_aprt_power"], errors="coerce")) / 2.0
        return a_s.add(b_s, fill_value=np.nan).add(c_s, fill_value=np.nan)
    return pd.Series(np.nan, index=df.index, dtype="float64")

def compute_fp(df, total_act_power):
    """
    calcula factor de potencia:
    1) si hay S (apparent) -> FP = P/S con clamp [0,1]
    2) si no hay S, intenta con energias reactivas: Q = lag+lead (por fases)
       y usa FP = P / sqrt(P^2 + Q^2). Aqui P y Q deben ser consistentes
       por registro (si tus energias son acumuladas, tomalo como proxy).
    """
    total_aprt_power = compute_total_apparent_power(df)

    # metodo por potencias
    fp_power = np.where(
        (total_aprt_power.notna()) & (total_aprt_power > 0),
        np.clip(total_act_power / total_aprt_power, 0.0, 1.0),
        np.nan
    )

    # si no se puede por potencias, intentar con energias
    cols_reac = [
        "a_lag_react_energy","a_lead_react_energy",
        "b_lag_react_energy","b_lead_react_energy",
        "c_lag_react_energy","c_lead_react_energy"
    ]
    cols_act = [
        "a_total_act_energy","b_total_act_energy","c_total_act_energy"
    ]

    fp_energy = np.full(len(df), np.nan, dtype="float64")
    if all(c in df.columns for c in cols_reac) and all(c in df.columns for c in cols_act):
        q = (
            pd.to_numeric(df["a_lag_react_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["a_lead_react_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["b_lag_react_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["b_lead_react_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["c_lag_react_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["c_lead_react_energy"], errors="coerce").fillna(0)
        )
        p_e = (
            pd.to_numeric(df["a_total_act_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["b_total_act_energy"], errors="coerce").fillna(0) +
            pd.to_numeric(df["c_total_act_energy"], errors="coerce").fillna(0)
        )

        denom = np.sqrt(p_e**2 + q**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp_energy = np.where(denom > 0, np.clip(p_e / denom, 0.0, 1.0), np.nan)

    # combinar: prioriza metodo por potencias, si NaN usa energia
    fp = pd.Series(fp_power, index=df.index)
    fp = fp.fillna(pd.Series(fp_energy, index=df.index))
    return fp

# === main ===
df = pd.read_csv(CSV_IN)

# timestamp limpio
if "timestamp" not in df.columns:
    raise ValueError("CSV no tiene columna 'timestamp'")

ts = parse_timestamp(df["timestamp"])

# total_act_power (usar existente o calcular)
total_p = compute_total_active_power(df)

# features temporales
hora_del_dia = ts.dt.hour
dia_de_la_semana = ts.dt.dayofweek  # lunes=0

hora_sin = np.sin((hora_del_dia / 24.0) * 2.0 * np.pi)
hora_cos = np.cos((hora_del_dia / 24.0) * 2.0 * np.pi)
dia_sin = np.sin((dia_de_la_semana / 7.0) * 2.0 * np.pi)
dia_cos = np.cos((dia_de_la_semana / 7.0) * 2.0 * np.pi)

# factor de potencia
fp = compute_fp(df, total_p)

# armar salida exacta
out = pd.DataFrame({
    "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
    "total_act_power": total_p,
    "hora_del_dia": hora_del_dia,
    "dia_de_la_semana": dia_de_la_semana,
    "hora_sin": hora_sin,
    "hora_cos": hora_cos,
    "dia_sin": dia_sin,
    "dia_cos": dia_cos,
    "factor_potencia": fp
})

# ordenar por timestamp y limpiar imposibles
out = out.sort_values("timestamp").reset_index(drop=True)

print(out.head(10))
out.to_csv(CSV_OUT, index=False)
