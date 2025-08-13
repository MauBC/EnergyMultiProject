import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================

def simular_datos():
    base_time = datetime.now()
    time_range = pd.to_datetime(pd.date_range(
        start=base_time - timedelta(hours=24),
        end=base_time + timedelta(hours=48),
        freq='H'
    ))
    df = pd.DataFrame({'timestamp': time_range})

    max_capacity = 50
    hours = df['timestamp'].dt.hour
    solar_profile = np.sin(np.pi * (hours - 5) / 14)
    solar_profile[(hours < 5) | (hours > 19)] = 0

    noise = np.random.normal(0, 0.03, len(df))
    generation = max_capacity * solar_profile * (1 - np.abs(noise))
    generation[generation < 0] = 0

    df['predicted'] = np.round(generation, 2)

    error_factor = np.random.normal(0, 0.02, len(df))
    multiplier = np.clip(1 + error_factor, 0.95, 1.05)
    df['actual'] = df['predicted'] * multiplier
    df.loc[df['timestamp'] > base_time, 'actual'] = np.nan
    df['actual'] = np.round(df['actual'], 2)

    df['confidence_upper'] = df['predicted'] * 1.10
    df['confidence_lower'] = df['predicted'] * 0.90

    df['ghi'] = 1000 * solar_profile * (1 - np.abs(noise))
    df['cloud_cover'] = np.round(np.abs(noise) * 100, 1)

    return df

# ==============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    requests_pathname_prefix="/dashboard/prediccion/"
)
server = app.server


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard de Predicción Fotovoltaica", className="text-center text-primary mb-4"), width=12),
        dbc.Col(html.P(id='live-update-time', className="text-end text-muted"), width=12)
    ]),
    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Generación Actual", className="card-title"),
            html.H2(id='kpi-actual', className="card-text")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Predicción Actual", className="card-title"),
            html.H2(id='kpi-predicted', className="card-text")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Desviación", className="card-title"),
            html.H2(id='kpi-deviation', className="card-text")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Energía Generada Hoy", className="card-title"),
            html.H2(id='kpi-total-today', className="card-text")
        ])), width=3),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([dcc.Graph(id='intraday-chart', style={'height': '400px'})], width=9),
        dbc.Col([
            dcc.Graph(id='power-gauge', style={'height': '300px'}),
            dbc.Card(dbc.CardBody([
                html.H6("Irradiancia (GHI)", className="card-title text-center"),
                html.H4(id='kpi-ghi', className="card-text text-center text-warning")
            ]), className="mt-3"),
            dbc.Card(dbc.CardBody([
                html.H6("Cobertura de Nubes", className="card-title text-center"),
                html.H4(id='kpi-clouds', className="card-text text-center text-info")
            ]), className="mt-3")
        ], width=3)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([dcc.Graph(id='forecast-chart', style={'height': '400px'})], width=8),
        dbc.Col([
            html.H4("Rendimiento del Modelo", className="text-center"),
            dcc.Graph(id='error-scatter-plot', style={'height': '300px'}),
            dbc.Card(dbc.CardBody([
                html.H6("MAE de Ayer (Simulado)", className="card-title text-center"),
                html.H4("0.85 MWh", className="card-text text-center text-danger")
            ]))
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Detalle del Pronóstico por Hora", className="mt-4"),
            dash_table.DataTable(
                id='forecast-table',
                style_cell={'textAlign': 'center', 'backgroundColor': '#222', 'color': 'white'},
                style_header={'fontWeight': 'bold'},
            )
        ], width=12)
    ])
], fluid=True)

# ==============================================================================

@app.callback(
    [Output('live-update-time', 'children'),
     Output('kpi-actual', 'children'),
     Output('kpi-predicted', 'children'),
     Output('kpi-deviation', 'children'),
     Output('kpi-total-today', 'children'),
     Output('intraday-chart', 'figure'),
     Output('power-gauge', 'figure'),
     Output('kpi-ghi', 'children'),
     Output('kpi-clouds', 'children'),
     Output('forecast-chart', 'figure'),
     Output('error-scatter-plot', 'figure'),
     Output('forecast-table', 'data'),
     Output('forecast-table', 'columns')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    data = simular_datos()
    now = datetime.now()
    current_data = data[data['timestamp'] <= now].iloc[-1]
    actual_power = current_data['actual']
    predicted_power = current_data['predicted']
    deviation = actual_power - predicted_power
    ghi_now = current_data['ghi']
    clouds_now = current_data['cloud_cover']
    today_data = data[data['timestamp'].dt.date == now.date()]
    total_today_energy = today_data['actual'].sum()

    today_chart_data = data[data['timestamp'].dt.date == now.date()]
    intraday_fig = go.Figure()
    intraday_fig.add_trace(go.Scatter(x=today_chart_data['timestamp'], y=today_chart_data['actual'], mode='lines', name='Generación Real', line=dict(color='cyan', width=3)))
    intraday_fig.add_trace(go.Scatter(x=today_chart_data['timestamp'], y=today_chart_data['predicted'], mode='lines', name='Pronóstico', line=dict(color='red', dash='dash')))
    intraday_fig.update_layout(title='Generación Intradía: Real vs. Pronóstico', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_range=[-5, 60], height=400)

    gauge_fig = go.Figure(go.Indicator(mode="gauge+number", value=actual_power, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Potencia Actual (MW)"}, gauge={'axis': {'range': [None, 50]}, 'bar': {'color': "cyan"}}))
    gauge_fig.update_layout(template='plotly_dark', height=300, margin=dict(l=20, r=20, t=50, b=20))

    forecast_data = data[data['timestamp'] >= now]
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamp'], y=forecast_data['confidence_upper'], fill=None, mode='lines', line_color='rgba(255,255,255,0.2)', name='Superior'))
    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamp'], y=forecast_data['confidence_lower'], fill='tonexty', mode='lines', line_color='rgba(255,255,255,0.2)', name='Intervalo Conf.'))
    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamp'], y=forecast_data['predicted'], mode='lines', name='Pronóstico a 48h', line=dict(color='orange', width=3)))
    forecast_fig.update_layout(title='Pronóstico de Generación a 48 Horas', template='plotly_dark', yaxis_range=[-5, 60], height=400)

    scatter_data = data.dropna(subset=['actual', 'predicted'])
    scatter_fig = go.Figure(data=go.Scatter(x=scatter_data['actual'], y=scatter_data['predicted'], mode='markers', marker=dict(color='cyan', opacity=0.6)))
    scatter_fig.add_trace(go.Scatter(x=[0, 50], y=[0, 50], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
    scatter_fig.update_layout(title='Real vs. Predicho (Histórico)', template='plotly_dark', xaxis_title='Real (MW)', yaxis_title='Predicho (MW)', height=300, margin=dict(l=20, r=20, t=40, b=20))

    table_data = data[data['timestamp'] > now].head(24).copy()
    table_data['timestamp'] = table_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    table_data = table_data.rename(columns={'timestamp': 'Fecha', 'predicted': 'Predicción (MW)', 'confidence_lower': 'Límite Inferior (MW)', 'confidence_upper': 'Límite Superior (MW)'})
    table_cols = [{"name": i, "id": i} for i in table_data.columns if i in ['Fecha', 'Predicción (MW)', 'Límite Inferior (MW)', 'Límite Superior (MW)']]

    return (
        f"Última actualización: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{actual_power:.2f} MW", f"{predicted_power:.2f} MW", f"{deviation:+.2f} MW",
        f"{total_today_energy:.2f} MWh", intraday_fig, gauge_fig, f"{ghi_now:.1f} W/m²",
        f"{clouds_now:.1f} %", forecast_fig, scatter_fig, table_data.to_dict('records'), table_cols
    )
