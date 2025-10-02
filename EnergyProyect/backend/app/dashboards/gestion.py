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

from app.database import db

# ==========================
# CONFIGURACIONES GLOBALES
# ==========================
alert_history = []
MAX_ALERTS = 5
def push_alert(title: str, text: str, type_: str = "warning"):
    global alert_history
    alert_history.append({"title": title, "text": text, "type": type_})
    if len(alert_history) > MAX_ALERTS:
        alert_history = alert_history[-MAX_ALERTS:]

consumo_hoy_kwh = 0.0
ultimo_dia = None

# Modelo de anomalías (opcional)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "Gestion", "modelo_anomalias.pkl")
try:
    anomaly_model = joblib.load(MODEL_PATH)
    print("✅ Modelo de anomalías cargado en gestion.py")
except Exception as e:
    anomaly_model = None
    print(f"⚠️ No se pudo cargar el modelo de anomalías: {e}")

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

    html.Div(className='row', children=[
        html.Div(id='kpi-container', className='eight columns'),
        html.Div(className='four columns', children=[html.Div(id='alerts-container', className='alert-container')]),
    ]),

    html.Hr(),

    html.Div([
        html.H2("Retorno de tu Inversión (ROI)", className='section-title'),
        html.Div(id='roi-metrics-container', className='row roi-metrics'),
        dcc.Graph(id='roi-bar-chart')
    ], className='roi-section'),

    html.Hr(),

    html.Div(className='row content-section', children=[
        html.Div(className='eight columns', children=[dcc.Graph(id='real-time-power-graph')]),
        html.Div(className='four columns', children=[html.Div(id='recommendations-table-container')]),
    ]),

    dcc.Interval(id='interval-component', interval=10 * 1000, n_intervals=0)  # cada 10s
], className='dashboard-container')

# ==========================
# CALLBACK 1: KPIs + Real Time Graph
# ==========================
@app.callback(
    [Output('kpi-container', 'children'),
     Output('real-time-power-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_metrics(n):
    global consumo_hoy_kwh, ultimo_dia

    # Traer último documento
    doc = list(db["energy_data"].find().sort("timestamp", -1).limit(1))
    if not doc:
        return html.Div("No hay datos aún"), go.Figure()

    row = doc[0]
    new_time = pd.to_datetime(row["timestamp"])
    new_power_kw = float(row.get("total_act_power", 0))  # ya está en kW

    # Resetear acumulador si cambia de día
    if ultimo_dia is None or new_time.date() != ultimo_dia:
        consumo_hoy_kwh = 0.0
        ultimo_dia = new_time.date()

    # Cada doc = 1 minuto en kW → convertir a kWh
    consumo_hoy_kwh += new_power_kw * (1/60)

    # Estimación de costo
    costo_estimado_dia = consumo_hoy_kwh * 0.75

    # Factor de potencia
    factor_potencia = row.get("factor_potencia", np.nan)
    if not np.isnan(factor_potencia) and factor_potencia <= 0.94:
        push_alert("FP bajo", f"FP={factor_potencia:.3f} en {new_time}", "warning")

    # KPIs
    kpi_layout = html.Div([
        html.Div([html.H3(f"{consumo_hoy_kwh:.2f} kWh"), html.P("Consumo Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"S/ {costo_estimado_dia:.2f}"), html.P("Costo Estimado Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"{new_power_kw:.2f} kW"), html.P("Potencia Actual")], className='three columns kpi-card'),
        html.Div([html.H3(f"{factor_potencia:.2f}" if not np.isnan(factor_potencia) else "N/A"),
                  html.P("Factor Potencia")], className='three columns kpi-card'),
    ], className='row')

    # ======================
    # Datos últimos 10 min
    # ======================
    ten_min_ago = datetime.now() - timedelta(minutes=10)  # ⚡ hora local
    cursor = db["energy_data"].find({"timestamp": {"$gte": ten_min_ago}}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))

    if df.empty or "timestamp" not in df.columns:
        return kpi_layout, go.Figure()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    real_time_fig = go.Figure(go.Scatter(
        x=df["timestamp"],
        y=df["total_act_power"].astype(float),
        mode='lines',
        name='Potencia',
        line=dict(color='cyan')
    ))

    real_time_fig.update_layout(
        title_text='Potencia Activa (Últimos 10 Minutos)',
        margin=dict(t=50, b=50, l=30, r=30),
        xaxis_title="Tiempo",
        yaxis_title="Potencia (kW)"
    )

    return kpi_layout, real_time_fig

# ==========================
# CALLBACK 2: ROI + Histórico + Alertas
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

    seven_days_ago = datetime.now() - timedelta(days=7)
    cursor = db["energy_data"].find({"timestamp": {"$gte": seven_days_ago}})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return [], go.Figure(), html.Div("No hay recomendaciones"), html.Div("No hay alertas")

    df["fecha"] = pd.to_datetime(df["timestamp"]).dt.date
    df["energia_kwh"] = df["total_act_power"].astype(float) * (1/60)

    daily_sum = df.groupby("fecha", as_index=False)["energia_kwh"].sum()

    roi_fig = go.Figure()
    roi_fig.add_trace(go.Bar(name="Consumo total del día", x=daily_sum["fecha"].astype(str), y=daily_sum["energia_kwh"]))
    roi_fig.add_trace(go.Bar(name="Meta 80%", x=daily_sum["fecha"].astype(str), y=daily_sum["energia_kwh"]*0.8))
    roi_fig.update_layout(barmode="group", title_text="Consumo total últimos 7 días")

    ahorro_mes_actual = daily_sum["energia_kwh"].sum() * 0.75
    costo_servicio = 150.00
    ahorro_neto = ahorro_mes_actual - costo_servicio
    roi_porcentaje = (ahorro_neto / costo_servicio) * 100 if costo_servicio > 0 else 0

    roi_metrics_layout = [
        html.Div([html.H3(f"S/ {ahorro_mes_actual:.2f}"), html.P("Ahorro Total")], className='three columns roi-metric'),
        html.Div([html.H3(f"S/ {costo_servicio:.2f}"), html.P("Costo del Servicio")], className='three columns roi-metric'),
        html.Div([html.H3(f"S/ {ahorro_neto:.2f}"), html.P("Ahorro Neto")], className='three columns roi-metric'),
        html.Div([html.H3(f"{roi_porcentaje:.0f}%"), html.P("ROI Mensual")], className='three columns roi-metric'),
    ]

    recommendations_table = html.Div([
        html.H5("Impacto de Recomendaciones"),
        html.Table([
            html.Thead(html.Tr([html.Th("Recomendación"), html.Th("Ahorro Estimado")])),
            html.Tbody([
                html.Tr([html.Td("Apagar equipos fuera de horario"), html.Td("S/ 125.50")]),
                html.Tr([html.Td("Optimizar uso en horario valle"), html.Td("S/ 210.00")]),
            ])
        ])
    ])

    alerts_layout = html.Div([
        html.H4("Centro de Alertas"),
        *[html.Div([html.H5(a["title"]), html.P(a["text"])], className=f'alert-card {a["type"]}') for a in reversed(alert_history)]
    ])

    return roi_metrics_layout, roi_fig, recommendations_table, alerts_layout

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run_server(debug=True)
