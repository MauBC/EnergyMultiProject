import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

# ==========================
# CONFIGURACIONES GLOBALES
# ==========================

# Historial de alertas (máximo 5)
alert_history = []
MAX_ALERTS = 5
def push_alert(title: str, text: str, type_: str = "warning"):
    """Agrega una alerta al historial (máximo 5 visibles)."""
    global alert_history
    alert_history.append({"title": title, "text": text, "type": type_})
    if len(alert_history) > MAX_ALERTS:
        alert_history = alert_history[-MAX_ALERTS:]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Acumulador de energía del día (kWh)
consumo_hoy_kwh = 0.0
ultimo_dia = None

# ==========================
# MODELO DE ANOMALÍAS
# ==========================
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "Gestion", "modelo_anomalias.pkl")
try:
    anomaly_model = joblib.load(MODEL_PATH)
    print("✅ Modelo de anomalías cargado en gestion.py")
except Exception as e:
    anomaly_model = None
    print(f"⚠️ No se pudo cargar el modelo de anomalías: {e}")

# ==========================
# DATOS CSV
# ==========================
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "data2.csv")
df_realtime = pd.read_csv(CSV_PATH)

# Asegurar que columnas clave existan y sean numéricas/fechas
df_realtime['total_act_power'] = pd.to_numeric(df_realtime['total_act_power'], errors='coerce')
if 'timestamp' in df_realtime.columns:
    df_realtime['timestamp'] = pd.to_datetime(df_realtime['timestamp'])
else:
    df_realtime['timestamp'] = pd.date_range(
        datetime.now() - timedelta(minutes=len(df_realtime)),
        periods=len(df_realtime),
        freq="T"
    )

# Empezar simulación un poco después del inicio
START_OFFSET_FRAC = 0.17
start_offset = int(len(df_realtime) * START_OFFSET_FRAC)
current_index = max(0, min(start_offset, max(0, len(df_realtime) - 1)))

# Series para gráfico en tiempo real
time_series = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))
power_series = pd.Series(dtype='float64')

# ==========================
# DASH APP
# ==========================
app = dash.Dash(
    __name__,
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    requests_pathname_prefix="/dashboard/gestion/"
)
server = app.server
pio.templates.default = "plotly_dark"

# ==========================
# LAYOUT
# ==========================
app.layout = html.Div(children=[

    html.H1('Dashboard de Gestión Energética - EnergIA', className='header'),

    # KPIs y alertas
    html.Div(className='row', children=[
        html.Div(id='kpi-container', className='eight columns'),
        html.Div(className='four columns', children=[html.Div(id='alerts-container', className='alert-container')]),
    ]),

    html.Hr(),

    # ROI section
    html.Div([
        html.H2("Retorno de tu Inversión (ROI)", className='section-title'),
        html.Div(id='roi-metrics-container', className='row roi-metrics'),
        dcc.Graph(id='roi-bar-chart')
    ], className='roi-section'),

    html.Hr(),

    # Gráfico en tiempo real + recomendaciones
    html.Div(className='row content-section', children=[
        html.Div(className='eight columns', children=[dcc.Graph(id='real-time-power-graph')]),
        html.Div(className='four columns', children=[html.Div(id='recommendations-table-container')]),
    ]),

    # Intervalo de actualización cada 5 segundos
    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0)
], className='dashboard-container')

# ==========================
# CALLBACK 1: KPIs + Gráfico tiempo real
# ==========================
@app.callback(
    [Output('kpi-container', 'children'),
     Output('real-time-power-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_metrics(n):
    global current_index, df_realtime, time_series, power_series
    global consumo_hoy_kwh, ultimo_dia

    # Reiniciar cuando se llega al final del CSV
    if current_index >= len(df_realtime):
        current_index = 0

    # Leer fila actual
    row = df_realtime.iloc[current_index]
    current_index += 1
    new_time = row['timestamp']
    new_power = float(row['total_act_power'])  # Potencia en W

    # Actualizar series de tiempo
    time_series = pd.concat([time_series, pd.Series([new_time])], ignore_index=True)
    power_series = pd.concat([power_series, pd.Series([new_power])], ignore_index=True)

    # Potencia actual en kW
    potencia_actual_kw = new_power / 1000

    # Resetear acumulador si cambia de día
    if ultimo_dia is None or new_time.date() != ultimo_dia:
        consumo_hoy_kwh = 0.0
        ultimo_dia = new_time.date()

    # Acumular consumo en kWh: P[kW] * (5/3600)h
    consumo_hoy_kwh += potencia_actual_kw * (5 / 3600)

    # Estimación de costo
    costo_estimado_dia = consumo_hoy_kwh * 0.75

    # Factor de potencia (si está en CSV)
    try:
        factor_potencia = float(row["factor_potencia"])
    except Exception:
        factor_potencia = np.nan

    # Alertas de FP
    if not np.isnan(factor_potencia):
        if factor_potencia <= 0.94:
            push_alert("FP bajo", f"FP={factor_potencia:.3f} en {new_time}", "warning")
        if factor_potencia > 1.0:
            push_alert("FP > 1", f"FP={factor_potencia:.3f} en {new_time} (revisar medicion)", "danger")

    # KPI cards
    kpi_layout = html.Div([
        html.Div([html.H3(f"{consumo_hoy_kwh:.2f} kWh"), html.P("Consumo Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"S/ {costo_estimado_dia:.2f}"), html.P("Costo Estimado Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"{potencia_actual_kw:.2f} kW"), html.P("Potencia Actual")], className='three columns kpi-card'),
        html.Div([html.H3(f"{factor_potencia:.2f}" if not np.isnan(factor_potencia) else "N/A"),
                  html.P("Factor Potencia")], className='three columns kpi-card'),
        html.Div([html.H3("XXXX"), html.P("Indicador Energetico Dia Anterior")], className='three columns kpi-card'),
    ], className='row')

    # Gráfico en tiempo real (últimos 10 minutos)
    real_time_fig = go.Figure(go.Scatter(
        x=time_series, y=power_series,
        mode='lines', name='Potencia', line=dict(color='cyan')
    ))

    if len(time_series) > 0:
        xmax = time_series.iloc[-1]
        xmin = xmax - timedelta(minutes=10)
        real_time_fig.update_xaxes(range=[xmin, xmax])

    real_time_fig.update_layout(
        title_text='Potencia Activa (Últimos 10 Minutos)',
        margin=dict(t=50, b=50, l=30, r=30),
        xaxis_title="Tiempo",
        yaxis_title="Potencia (W)"
    )

    return kpi_layout, real_time_fig

# ==========================
# CALLBACK 2: ROI + Recomendaciones + Alertas
# ==========================
@app.callback(
    [Output('roi-metrics-container', 'children'),
     Output('roi-bar-chart', 'figure'),
     Output('recommendations-table-container', 'children'),
     Output('alerts-container', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_roi_and_alerts(n):
    global consumo_hoy_kwh

    # ROI base
    if 'ahorro_soles' not in df_realtime.columns:
        df_realtime['ahorro_soles'] = 0.0
    ahorro_mes_actual = df_realtime['ahorro_soles'].sum()
    costo_servicio = 150.00
    ahorro_neto = ahorro_mes_actual - costo_servicio
    roi_porcentaje = (ahorro_neto / costo_servicio) * 100 if costo_servicio > 0 else 0

    roi_metrics_layout = [
        html.Div([html.H3(f"S/ {ahorro_mes_actual:.2f}"), html.P("Ahorro Total del Mes")], className='three columns roi-metric'),
        html.Div([html.H3(f"S/ {costo_servicio:.2f}"), html.P("Costo del Servicio")], className='three columns roi-metric'),
        html.Div([html.H3(f"S/ {ahorro_neto:.2f}", style={'color': '#2ECC71'}), html.P("Ahorro Neto")], className='three columns roi-metric'),
        html.Div([html.H3(f"{roi_porcentaje:.0f}%"), html.P("ROI Mensual")], className='three columns roi-metric'),
    ]

    # =======================
    # Consumo por día (kWh)
    # =======================
    df_daily = df_realtime.copy()
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"], errors="coerce")
    df_daily["total_act_power"] = pd.to_numeric(df_daily["total_act_power"], errors="coerce")
    df_daily = df_daily.dropna(subset=["timestamp", "total_act_power"])
    df_daily["fecha"] = df_daily["timestamp"].dt.date

    # Energía por fila en kWh (1 minuto de intervalo)
    df_daily["energia_kwh"] = df_daily["total_act_power"] / 1000 * (1/60)

    # Sumar energía por día
    daily_sum = df_daily.groupby("fecha", as_index=False)["energia_kwh"].sum().sort_values("fecha")

    # Detectar última fecha del CSV
    ultima_fecha = df_daily["fecha"].max()

    # Tomar últimos 6 días excluyendo la última fecha (para no duplicar)
    last_days = daily_sum[daily_sum["fecha"] < ultima_fecha].tail(6)


    # Energía histórica del último día en el CSV
    energia_hist_dia = daily_sum.loc[daily_sum["fecha"] == ultima_fecha, "energia_kwh"].sum()

    # Energía en tiempo real del acumulador
    energia_actual_dia = float(consumo_hoy_kwh)

    # Total del día actual
    energia_total_hoy = energia_hist_dia + energia_actual_dia

    # Crear dataframe con el día actual
    hoy = pd.DataFrame([{
        "fecha": ultima_fecha,
        "energia_kwh": energia_total_hoy
    }])



    # Concatenar
    daily_tail = pd.concat([last_days, hoy], ignore_index=True)

    # Construcción del gráfico
    roi_fig = go.Figure()
    x_labels = daily_tail["fecha"].astype(str).tolist()
    y_total = daily_tail["energia_kwh"].values.astype(float)
    y_obj = y_total * 0.8

    roi_fig.add_trace(go.Bar(name="Consumo total del dia", x=x_labels, y=y_total))
    roi_fig.add_trace(go.Bar(name="Meta 80% del dia", x=x_labels, y=y_obj))

    roi_fig.update_layout(
        title_text="Consumo total por dia (ultimos dias disponibles)",
        barmode="group",
        margin=dict(t=50, b=50, l=30, r=30),
        yaxis_title="Energía (kWh)"
    )

    # =======================
    # Tabla de recomendaciones
    # =======================
    recommendations_table = html.Div([
        html.H5("Impacto de Recomendaciones"),
        html.Table([
            html.Thead(html.Tr([html.Th("Recomendación"), html.Th("Ahorro Estimado")])),
            html.Tbody([
                html.Tr([html.Td("Apagar equipos fuera de horario"), html.Td("S/ 125.50")]),
                html.Tr([html.Td("Optimizar 'Horno 2' a horario valle"), html.Td("S/ 210.00")]),
                html.Tr([html.Td("Corregir Factor de Potencia"), html.Td("S/ 115.20")]),
            ])
        ])
    ], className='recommendations-table')

    # =======================
    # Alertas
    # =======================
    alerts_layout = html.Div([
        html.H4("Centro de Alertas"),
        *[
            html.Div([
                html.H5(alert["title"]),
                html.P(alert["text"])
            ], className=f'alert-card {alert["type"]}')
            for alert in reversed(alert_history)
        ]
    ])

    return roi_metrics_layout, roi_fig, recommendations_table, alerts_layout

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run_server(debug=True)
