import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio
import os
import joblib



# Historial de alertas (máximo 10)
alert_history = []
MAX_ALERTS = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Acumulador de energía del día
consumo_hoy_kwh = 0.0
ultimo_dia = None



# --- CARGAR MODELO DE ANOMALÍAS ---
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "Gestion", "modelo_anomalias.pkl")
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_anomalias.pkl")
try:
    anomaly_model = joblib.load(MODEL_PATH)
    print("✅ Modelo de anomalías cargado en gestion.py")
except Exception as e:
    anomaly_model = None
    print(f"⚠️ No se pudo cargar el modelo de anomalías: {e}")

# --- CARGA DE DATOS REALES DESDE CSV ---
CSV_PATH    = os.path.join(BASE_DIR, "..", "data", "data_modificada_15.csv")

#CSV_PATH = os.path.join(os.path.dirname(__file__), "data_modificada_15.csv")
df_realtime = pd.read_csv(CSV_PATH)
df_realtime['total_act_power'] = pd.to_numeric(df_realtime['total_act_power'], errors='coerce')

# Asegurar que exista columna fecha
if 'timestamp' in df_realtime.columns:
    df_realtime['timestamp'] = pd.to_datetime(df_realtime['timestamp'])
else:
    # Si no tiene columna fecha, se genera una serie temporal ficticia
    df_realtime['timestamp'] = pd.date_range(
        datetime.now() - timedelta(minutes=len(df_realtime)),
        periods=len(df_realtime),
        freq="T"
    )

# Índice global para recorrer el CSV
current_index = 0

# Series para la gráfica
time_series = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))
power_series = pd.Series(dtype='float64')

# --- DASH APP ---
app = dash.Dash(
    __name__,
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    requests_pathname_prefix="/dashboard/gestion/"
)
server = app.server

pio.templates.default = "plotly_dark"

# --- LAYOUT ---
app.layout = html.Div(children=[
    html.H1(children='Dashboard de Gestión Energética - EnergIA', className='header'),

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

    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0)
], className='dashboard-container')

# --- CALLBACK PARA KPIs Y GRAFICO ---
@app.callback(
    [Output('kpi-container', 'children'), Output('real-time-power-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_metrics(n):
    global current_index, df_realtime, time_series, power_series, anomaly_model
    global consumo_hoy_kwh, ultimo_dia   # 👈 agrega esto



    if current_index >= len(df_realtime):
        current_index = 0

    row = df_realtime.iloc[current_index]
    current_index += 1

    new_time = row['timestamp']
    new_power = float(row['total_act_power'])  # asegurar numérico

    # --- actualizar series ---
    time_series = pd.concat([time_series, pd.Series([new_time])], ignore_index=True)
    power_series = pd.concat([power_series, pd.Series([new_power])], ignore_index=True)

    # Potencia actual en kW
    potencia_actual_kw = new_power / 1000
    # Energía activa total (kWh)
    #df['activa_total'] = df['a_total_act_energy'] + df['b_total_act_energy'] + df['c_total_act_energy']

    # Energía reactiva total (kvarh) = suma de lag + lead por fase
    ''' FALTA LLENAR ESTO DEL CSV ORIGINAL
    df['reactiva_total'] = (
        df['a_lag_react_energy'] + df['a_lead_react_energy'] +
        df['b_lag_react_energy'] + df['b_lead_react_energy'] +
        df['c_lag_react_energy'] + df['c_lead_react_energy']
    )

    # Factor de potencia instantáneo (por cada registro)
    df['fp'] = df['activa_total'] / ( (df['activa_total']**2 + df['reactiva_total']**2) ** 0.5 )
    
    # Promedio de FP en todo el dataset
    fp_promedio = df['fp'].mean()
    
    print("Factor de potencia promedio:", fp_promedio)
    '''
    # Si es un nuevo día, resetear acumulador
    if ultimo_dia is None or new_time.date() != ultimo_dia:
        consumo_hoy_kwh = 0.0
        ultimo_dia = new_time.date()

    # Acumular consumo (intervalo de 5s → 5/3600 horas)
    consumo_hoy_kwh += potencia_actual_kw * (5 / 3600)

    # Calcular costo y factor
    costo_estimado_dia = consumo_hoy_kwh * 0.75
    factor_potencia = 0.92 + np.random.rand() * 0.07

    kpi_layout = html.Div([
        html.Div([html.H3(f"{consumo_hoy_kwh:.2f} kWh"), html.P("Consumo Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"S/ {costo_estimado_dia:.2f}"), html.P("Costo Estimado Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"{potencia_actual_kw:.2f} kW"), html.P("Potencia Actual")], className='three columns kpi-card'),
        html.Div([html.H3(f"{factor_potencia:.2f} PF"), html.P("Factor Potencia")], className='three columns kpi-card'),
        ## Falta llenar una fila
        ## total_act_power/kgHarinaUsada

    ], className='row')

    # --- Gráfico tiempo real ---
    real_time_fig = go.Figure(go.Scatter(
        x=time_series, y=power_series, mode='lines', name='Potencia', line=dict(color='cyan')
    ))

    # Configurar ventana móvil de 10 minutos
    if len(time_series) > 0:
        xmax = time_series.iloc[-1]
        xmin = xmax - timedelta(minutes=10)
        real_time_fig.update_xaxes(range=[xmin, xmax])

    # Si hay anomalía, agregar punto rojo
    # (tu código original de anomaly_flag aquí)

    real_time_fig.update_layout(
        title_text='Potencia Activa (Últimos 10 Minutos)',
        margin=dict(t=50, b=50, l=30, r=30),
        xaxis_title="Tiempo",
        yaxis_title="Potencia (W)"
    )

    return kpi_layout, real_time_fig

# --- CALLBACK PARA ROI Y ALERTAS ---
@app.callback(
    [Output('roi-metrics-container', 'children'),
     Output('roi-bar-chart', 'figure'),
     Output('recommendations-table-container', 'children'),
     Output('alerts-container', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_roi_and_alerts(n):
    # Métricas ROI usando todo el CSV
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

    # Grafico ROI últimos 7 días
    df_chart = df_realtime.tail(7)
    roi_fig = go.Figure()
    if 'consumo_kwh_base' in df_realtime.columns and 'consumo_kwh_real' in df_realtime.columns:
        roi_fig.add_trace(go.Bar(x=df_chart['timestamp'], y=df_chart['consumo_kwh_base'], name='Consumo Base (Estimado)'))
        roi_fig.add_trace(go.Bar(x=df_chart['timestamp'], y=df_chart['consumo_kwh_real'], name='Consumo Real'))
    roi_fig.update_layout(title_text='Consumo Estimado vs. Real (Últimos 7 Días)', barmode='group',
                          margin=dict(t=50, b=50, l=30, r=30))

    # Tabla recomendaciones fija (puedes ligarla al CSV si quieres)
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

    # Alertas ejemplo
    alerts_layout = html.Div([
        html.H4("Centro de Alertas"),
        *[
            html.Div([
                html.H5(alert["title"]),
                html.P(alert["text"])
            ], className=f'alert-card {alert["type"]}')
            for alert in reversed(alert_history)  # Mostrar la más reciente arriba
        ]
    ])

    return roi_metrics_layout, roi_fig, recommendations_table, alerts_layout
