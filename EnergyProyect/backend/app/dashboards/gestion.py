import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio
import os

# --- SIMULACIÓN DE DATOS ---
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
historical_dates = pd.date_range(start_date, end_date, freq='D')
historical_data = {
    'fecha': historical_dates,
    'consumo_kwh_base': np.random.randint(100, 150, size=len(historical_dates)),
    'consumo_kwh_real': np.random.randint(80, 120, size=len(historical_dates))
}
df_historical = pd.DataFrame(historical_data)
df_historical['ahorro_kwh'] = df_historical['consumo_kwh_base'] - df_historical['consumo_kwh_real']
df_historical['ahorro_soles'] = df_historical['ahorro_kwh'] * 0.75

time_series = pd.to_datetime(pd.Series(dtype='datetime64[ns]'))
power_series = pd.Series(dtype='float64')

# --- INICIALIZACIÓN ---
app = dash.Dash(
    __name__,
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    requests_pathname_prefix="/dashboard/gestion/"
)
server = app.server

pio.templates.default = "plotly_dark"

# --- DISEÑO ---
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

# --- CALLBACKS ---
@app.callback(
    [Output('kpi-container', 'children'), Output('real-time-power-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_metrics(n):
    global time_series, power_series
    new_time = datetime.now()
    new_power = 500 + np.random.randint(-100, 100)
    if np.random.rand() > 0.9:
        new_power = 3000 + np.random.randint(-500, 500)

    time_series = pd.concat([time_series, pd.Series([new_time])], ignore_index=True)
    power_series = pd.concat([power_series, pd.Series([new_power])], ignore_index=True)
    mask = time_series > (new_time - timedelta(minutes=10))
    time_series, power_series = time_series[mask], power_series[mask]

    potencia_actual_kw = new_power / 1000
    consumo_hoy_kwh = df_historical['consumo_kwh_real'].iloc[-1]
    costo_estimado_dia = consumo_hoy_kwh * 0.75
    factor_potencia = 0.92 + np.random.rand() * 0.07

    kpi_layout = html.Div([
        html.Div([html.H3(f"{consumo_hoy_kwh:.2f} kWh"), html.P("Consumo Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"S/ {costo_estimado_dia:.2f}"), html.P("Costo Estimado Hoy")], className='three columns kpi-card'),
        html.Div([html.H3(f"{potencia_actual_kw:.2f} kW"), html.P("Potencia Actual")], className='three columns kpi-card'),
        html.Div([html.H3(f"{factor_potencia:.2f} PF"), html.P("Factor Potencia")], className='three columns kpi-card'),
    ], className='row')

    real_time_fig = go.Figure(go.Scatter(x=time_series, y=power_series, mode='lines', name='Potencia', line=dict(color='cyan')))
    real_time_fig.update_layout(title_text='Potencia Activa (Últimos 10 Minutos)', margin=dict(t=50, b=50, l=30, r=30))

    return kpi_layout, real_time_fig

@app.callback(
    [Output('roi-metrics-container', 'children'), Output('roi-bar-chart', 'figure'),
     Output('recommendations-table-container', 'children'), Output('alerts-container', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_roi_and_alerts(n):
    ahorro_mes_actual = df_historical['ahorro_soles'].sum()
    costo_servicio = 150.00
    ahorro_neto = ahorro_mes_actual - costo_servicio
    roi_porcentaje = (ahorro_neto / costo_servicio) * 100 if costo_servicio > 0 else 0

    roi_metrics_layout = [
        html.Div([html.H3(f"S/ {ahorro_mes_actual:.2f}"), html.P("Ahorro Total del Mes")], className='three columns roi-metric'),
        html.Div([html.H3(f"S/ {costo_servicio:.2f}"), html.P("Costo del Servicio")], className='three columns roi-metric'),
        html.Div([html.H3(f"S/ {ahorro_neto:.2f}", style={'color': '#2ECC71'}), html.P("Ahorro Neto")], className='three columns roi-metric'),
        html.Div([html.H3(f"{roi_porcentaje:.0f}%"), html.P("ROI Mensual")], className='three columns roi-metric'),
    ]

    df_chart = df_historical.tail(7)
    roi_fig = go.Figure()
    roi_fig.add_trace(go.Bar(x=df_chart['fecha'], y=df_chart['consumo_kwh_base'], name='Consumo Base (Estimado)'))
    roi_fig.add_trace(go.Bar(x=df_chart['fecha'], y=df_chart['consumo_kwh_real'], name='Consumo Real'))
    roi_fig.update_layout(title_text='Consumo Estimado vs. Real (Últimos 7 Días)', barmode='group',
                          margin=dict(t=50, b=50, l=30, r=30))

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

    alerts_layout = html.Div([
        html.H4("Centro de Alertas"),
        html.Div([
            html.H5("⚠️ Bajo Factor de Potencia"),
            html.P("Se detectó un PF de 0.88 ayer a las 4 PM.")
        ], className='alert-card alert-warning'),
        html.Div([
            html.H5("📈 Aumento de Consumo (Amasadora)"),
            html.P("El consumo ha aumentado un 12%. Se recomienda revisión.")
        ], className='alert-card alert-info'),
        html.Div([
            html.H5("💡 Oportunidad de Ahorro"),
            html.P("Operar el horno principal después de las 10 PM podría ahorrar S/ 200 al mes.")
        ], className='alert-card alert-success'),
    ])

    return roi_metrics_layout, roi_fig, recommendations_table, alerts_layout
