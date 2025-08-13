# prediccion.py
# -*- coding: utf-8 -*-

# ========== ML inference: cargar modelo y utilidades ==========
import os
import joblib
import torch
from torch import nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- IMPORTS DASH (dashboard) ---
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go

# -------------------- Config segun tu training --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_BASE = os.path.join(BASE_DIR, "..", "models", "COLCA", "lb168_hz48")
CSV_PATH    = os.path.join(BASE_DIR, "..", "data", "COLCA_SOLAR_MERGED.csv")
PLANT = "COLCA_SOLAR"

LOOK_BACK = 168
HORIZON   = 48
USE_LAGS  = True
LAGS      = [24, 48, 72]
ADD_DOY   = True
ADD_WEEK  = False

SCALER_X_PATH = os.path.join(MODELS_BASE, f"scaler_X_lb{LOOK_BACK}_hz{HORIZON}.pkl")
SCALER_Y_PATH = os.path.join(MODELS_BASE, f"scaler_y_lb{LOOK_BACK}_hz{HORIZON}.pkl")
GRU_PATH      = os.path.join(MODELS_BASE, f"gru_multistep_lb{LOOK_BACK}_hz{HORIZON}.pt")
LOAD_MODEL_KIND = os.getenv("MODEL_KIND", "gru")  # "gru" o "mlp"

# -------------------- Modelos como en training --------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.01), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.LeakyReLU(0.01), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.LeakyReLU(0.01),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.net(x)

class GRUModel(nn.Module):
    def __init__(self, n_features, look_back, output_dim, hidden_size=192, n_layers=2, dropout=0.10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, dropout=(dropout if n_layers > 1 else 0.0)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.LeakyReLU(0.01), nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)

# -------------------- Utils de features --------------------
def add_time_features(df, dt_col="datetime", add_doy=True, add_week=False):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    df["hour"] = df[dt_col].dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    if add_doy:
        df["doy"] = df[dt_col].dt.dayofyear
        df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365.0)
        df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365.0)
    if add_week:
        df["dow"] = df[dt_col].dt.dayofweek
        df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7.0)
        df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7.0)
    return df

def add_lag_features(df, cols, lags):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        for L in lags:
            df[f"{c}_lag_{L}"] = df[c].shift(L)
    return df

def feature_names(add_doy=True, add_week=False, lags=LAGS):
    feats = ["DNI","Temperatura","Viento","Humedad","hour_sin","hour_cos"]
    if add_doy:
        feats += ["doy_sin","doy_cos"]
    if add_week:
        feats += ["dow_sin","dow_cos"]
    for L in lags:
        feats += [f"DNI_lag_{L}", f"Temperatura_lag_{L}", f"Humedad_lag_{L}"]
    return feats

def minmax_inverse_like_safe(Y_flat_scaled, scaler_y, n_targets, horizon):
    if Y_flat_scaled.size == 0:
        return np.zeros((0, horizon, n_targets), dtype=np.float32)
    Z = Y_flat_scaled.reshape(-1, n_targets).astype(np.float64)
    Z[~np.isfinite(Z)] = 0.5
    Z = np.clip(Z, 0.0, 1.0)
    data_min = scaler_y.data_min_.astype(np.float64)
    data_max = scaler_y.data_max_.astype(np.float64)
    span = data_max - data_min
    Xrec = Z * span + data_min
    zero_span = (span == 0.0)
    if np.any(zero_span):
        Xrec[:, zero_span] = data_min[zero_span]
    return Xrec.reshape(-1, horizon, n_targets).astype(np.float32)

# -------------------- Cargar scalers y modelo --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando scalers: {e}")

FEATS = feature_names(ADD_DOY, ADD_WEEK, LAGS)
N_FEATURES = len(FEATS)
N_TARGETS  = 1

try:
    if LOAD_MODEL_KIND == "gru":
        model = GRUModel(n_features=N_FEATURES, look_back=LOOK_BACK, output_dim=HORIZON*N_TARGETS).to(device)
        model.load_state_dict(torch.load(GRU_PATH, map_location=device))
    else:
        mlp_path = os.path.join(MODELS_BASE, f"mlp_multistep_lb{LOOK_BACK}_hz{HORIZON}.pt")
        model = MLP(input_dim=LOOK_BACK*N_FEATURES, output_dim=HORIZON*N_TARGETS, dropout=0.10).to(device)
        model.load_state_dict(torch.load(mlp_path, map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error cargando modelo: {e}")

# -------------------- Helpers de lectura y prediccion --------------------
def get_history_df(csv_path, now_ts):
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("no se encontro la columna 'datetime' en el csv")
    need_raw = LOOK_BACK + (max(LAGS) if (USE_LAGS and LAGS) else 0)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # Filtra hasta 'ahora' (si el CSV llega hasta el pasado, igual entra todo)
    df = df[df["datetime"] <= now_ts]

    # Toma un buffer razonable (si el CSV es corto, entra lo que haya)
    return df.tail(max(need_raw + 24, 300)).reset_index(drop=True)

def _fill_lag_nans_inplace(df):
    """Rellena NaNs en columnas de lags para no perder filas por dropna."""
    lag_cols = [c for c in df.columns if "_lag_" in c]
    if not lag_cols:
        return
    # primero bfill/ffill y finalmente 0 si quedo algo
    df[lag_cols] = df[lag_cols].bfill().ffill().fillna(0.0)

def _pad_to_lookback(df):
    """Si tras features quedan < LOOK_BACK filas, repite la primera fila hasta completar."""
    if len(df) >= LOOK_BACK:
        return df
    need = LOOK_BACK - len(df)
    if len(df) == 0:
        raise ValueError("no hay filas en el CSV tras preprocesar")
    pad_block = pd.concat([df.iloc[[0]].copy() for _ in range(need)], ignore_index=True)
    df_padded = pd.concat([pad_block, df], ignore_index=True)
    return df_padded

def infer_48h_from_df(df_raw):
    # 1) features base
    df = add_time_features(df_raw, dt_col="datetime", add_doy=ADD_DOY, add_week=ADD_WEEK)
    # 2) lags
    if USE_LAGS and LAGS:
        df = add_lag_features(df, ["DNI","Temperatura","Humedad"], LAGS)
        _fill_lag_nans_inplace(df)  # evitar dropna por lags
    # 3) garantizar suficientes filas para la ventana
    df = df.reset_index(drop=True)
    df = _pad_to_lookback(df)

    # 4) construir ventana
    df_last = df.tail(LOOK_BACK).copy()
    X = df_last[FEATS].values
    Xs = scaler_X.transform(X)

    # 5) inferencia
    with torch.no_grad():
        if LOAD_MODEL_KIND == "gru":
            xb = torch.from_numpy(Xs.astype(np.float32)).unsqueeze(0).to(device)  # (1, L, F)
        else:
            xb = torch.from_numpy(Xs.reshape(1, -1).astype(np.float32)).to(device)  # (1, L*F)
        pred_s = model(xb).cpu().numpy()

    # 6) inversa escala
    y_pred = minmax_inverse_like_safe(pred_s, scaler_y, n_targets=N_TARGETS, horizon=HORIZON)  # (1,H,1)
    vec = y_pred[0, :, 0].astype(float)

    # 7) timestamps futuros desde el ultimo datetime real disponible en df_raw (si no, usa df_last)
    if len(df_raw) > 0:
        last_dt = pd.to_datetime(df_raw["datetime"].iloc[-1])
    else:
        last_dt = pd.to_datetime(df_last["datetime"].iloc[-1])
    future_ts = pd.date_range(last_dt + timedelta(hours=1), periods=HORIZON, freq="h")
    return future_ts, vec

def build_dashboard_df(now_ts):
    # 1) leer historia real desde CSV (tolerante)
    raw = get_history_df(CSV_PATH, now_ts)

    # 2) inferir 48h futuras (tolerante a historia corta)
    fut_ts, pred = infer_48h_from_df(raw)

    # 3) armar dataframe completo: 24h pasadas + 48h futuras
    past_24 = pd.date_range(now_ts - timedelta(hours=24), now_ts, freq="h")
    past = pd.DataFrame(columns=["timestamp","actual","dni"])
    if len(raw):
        cols_ok = ["datetime", PLANT, "DNI"]
        missing = [c for c in cols_ok if c not in raw.columns]
        if not missing:
            past = raw[raw["datetime"].isin(past_24)][["datetime", PLANT, "DNI"]].rename(
                columns={"datetime":"timestamp", PLANT:"actual", "DNI":"dni"}
            )
            past = past.set_index("timestamp").reindex(past_24).reset_index().rename(columns={"index":"timestamp"})
        else:
            # si faltan columnas, crea past vacio con index de 24h
            past = pd.DataFrame({"timestamp": past_24, "actual": np.nan, "dni": 0.0})
    else:
        past = pd.DataFrame({"timestamp": past_24, "actual": np.nan, "dni": 0.0})

    future = pd.DataFrame({"timestamp": fut_ts, "predicted": pred})
    df = pd.merge(past, future, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)

    # bandas simple (10%)
    df["confidence_upper"] = df["predicted"] * 1.10
    df["confidence_lower"] = df["predicted"] * 0.90

    # GHI proxy
    df["ghi"] = df["dni"].fillna(0.0)
    df["cloud_cover"] = (np.abs(df["ghi"] - df["ghi"].rolling(6, min_periods=1).mean()) /
                         (df["ghi"].rolling(6, min_periods=1).mean() + 1e-6) * 100.0).clip(0, 100).fillna(0.0)
    return df, fut_ts[0] - pd.Timedelta(hours=1)

# ========== DASH APP (expone `server`) ==========
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    requests_pathname_prefix="/dashboard/prediccion/"
)
server = app.server  # <-- necesario para app.main: app.mount("/dashboard/prediccion", WSGIMiddleware(prediccion.server))

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard de Predicción Fotovoltaica", className="text-center text-primary mb-4"), width=12),
        dbc.Col(html.P(id='live-update-time', className="text-end text-muted"), width=12)
    ]),
    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),  # cada 5 s
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
                html.H6("MAE de Ayer (Demo)", className="card-title text-center"),
                html.H4("—", className="card-text text-center text-danger")
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
    now = datetime.now()
    df,ref_ts  = build_dashboard_df(now)

    # valores actuales respecto a ref_ts
    current_row = df[df['timestamp'] <= ref_ts].iloc[-1] if (df['timestamp'] <= ref_ts).any() else df.iloc[0]
    actual_power = current_row['actual'] if pd.notna(current_row.get('actual', np.nan)) else np.nan
    predicted_power = current_row['predicted'] if pd.notna(current_row.get('predicted', np.nan)) else 0.0
    deviation = (actual_power - predicted_power) if pd.notna(actual_power) else np.nan
    ghi_now = current_row.get('ghi', 0.0)
    clouds_now = current_row.get('cloud_cover', 0.0)

    # intradía usando el día de ref_ts
    today_data = df[df['timestamp'].dt.date == ref_ts.date()]
    total_today_energy = today_data['actual'].sum(skipna=True) if not today_data.empty else 0.0

    # intraday chart
    today_chart_data = today_data
    intraday_fig = go.Figure()
    intraday_fig.add_trace(go.Scatter(x=today_chart_data['timestamp'], y=today_chart_data['actual'], mode='lines',
                                      name='Generación Real', line=dict(color='cyan', width=3)))
    intraday_fig.add_trace(go.Scatter(x=today_chart_data['timestamp'], y=today_chart_data['predicted'], mode='lines',
                                      name='Pronóstico', line=dict(color='red', dash='dash')))
    intraday_fig.update_layout(
        title='Generación Intradía: Real vs. Pronóstico', template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_range=[-5, max(60, float(np.nanmax(df[['actual','predicted']].values)) + 10)],
        height=400
    )

    # Gauge
    gauge_fig = go.Figure(go.Indicator(mode="gauge+number",
                                       value=float(predicted_power) if pd.notna(predicted_power) else 0.0,
                                       domain={'x': [0, 1], 'y': [0, 1]},
                                       title={'text': "Potencia (MW)"},
                                       gauge={'axis': {'range': [None, max(50, float(np.nanmax(df['predicted'].values)) + 10)]},
                                              'bar': {'color': "cyan"}}))
    gauge_fig.update_layout(template='plotly_dark', height=300, margin=dict(l=20, r=20, t=50, b=20))

    # forecast: >= ref_ts
    forecast_data = df[df['timestamp'] >= ref_ts]
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamp'], y=forecast_data['confidence_upper'], fill=None,
                                      mode='lines', line_color='rgba(255,255,255,0.2)', name='Superior'))
    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamp'], y=forecast_data['confidence_lower'], fill='tonexty',
                                      mode='lines', line_color='rgba(255,255,255,0.2)', name='Intervalo Conf.'))
    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamp'], y=forecast_data['predicted'], mode='lines',
                                      name='Pronóstico a 48h', line=dict(color='orange', width=3)))
    forecast_fig.update_layout(
        title='Pronóstico de Generación a 48 Horas', template='plotly_dark',
        yaxis_range=[-5, max(60, float(np.nanmax(df['predicted'].values)) + 10)],
        height=400
    )

    # Scatter real vs predicho (histórico del día)
    scatter_data = df.dropna(subset=['actual', 'predicted'])
    scatter_fig = go.Figure(data=go.Scatter(x=scatter_data['actual'], y=scatter_data['predicted'],
                                            mode='markers', marker=dict(color='cyan', opacity=0.6)))
    scatter_fig.add_trace(go.Scatter(x=[0, 50], y=[0, 50], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
    scatter_fig.update_layout(title='Real vs. Predicho (Histórico)', template='plotly_dark',
                              xaxis_title='Real (MW)', yaxis_title='Predicho (MW)', height=300,
                              margin=dict(l=20, r=20, t=40, b=20))

    # tabla próximas 24h respecto a ref_ts
    table_data = df[df['timestamp'] > ref_ts].head(24).copy()
    table_data['timestamp'] = table_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    table_data = table_data.rename(columns={'timestamp': 'Fecha', 'predicted': 'Predicción (MW)',
                                            'confidence_lower': 'Límite Inferior (MW)',
                                            'confidence_upper': 'Límite Superior (MW)'})
    table_cols = [{"name": i, "id": i} for i in ['Fecha', 'Predicción (MW)', 'Límite Inferior (MW)', 'Límite Superior (MW)']]

    return (
        f"Última actualización: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{actual_power:.2f} MW" if pd.notna(actual_power) else "—",
        f"{predicted_power:.2f} MW" if pd.notna(predicted_power) else "—",
        f"{deviation:+.2f} MW" if pd.notna(deviation) else "—",
        f"{total_today_energy:.2f} MWh",
        intraday_fig, gauge_fig,
        f"{float(ghi_now):.1f} W/m²",
        f"{float(clouds_now):.1f} %",
        forecast_fig, scatter_fig,
        table_data.to_dict('records'), table_cols
    )
