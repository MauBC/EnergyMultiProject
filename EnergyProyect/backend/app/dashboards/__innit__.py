import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# tu código de funciones aquí (simular_datos, layout, callbacks...)

# ⬅ Importante: NO ejecutamos app.run(), solo dejamos la variable `app`
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # Esto es necesario para WSGIMiddleware