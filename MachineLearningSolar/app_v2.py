import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIGURACION
# ==============================
CSV_PATH = "COLCA_SOLAR_MERGED.csv"
MODEL_PATH = "modelo_lstm.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK = 128   # horas pasadas usadas como input
HORIZON = 12   # prediccion a 12 horas adelante
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

# ==============================
# CARGA DE DATOS
# ==============================
df = pd.read_csv(CSV_PATH)

# Asegurar columna datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Seleccionamos features climaticos + energia
features = ["DNI", "Temperatura", "Viento", "Humedad", "COLCA_SOLAR"]
data = df[features].dropna().values

# Escalado
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ==============================
# DATASET SECUENCIAL
# ==============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback, :]
        y = self.data[idx+self.lookback:idx+self.lookback+self.horizon, -1]  # solo energia
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Split train/test
split = int(len(data_scaled)*0.8)
train_data, test_data = data_scaled[:split], data_scaled[split:]

train_dataset = TimeSeriesDataset(train_data, LOOKBACK, HORIZON)
test_dataset = TimeSeriesDataset(test_data, LOOKBACK, HORIZON)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# MODELO LSTM
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # usamos la ultima salida
        return out

input_size = len(features)
model = LSTMModel(input_size).to(DEVICE)

# ==============================
# ENTRENAMIENTO
# ==============================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

if os.path.exists(MODEL_PATH):
    print("Cargando modelo entrenado...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("Entrenando modelo...")
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(losses):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

# ==============================
# EVALUACION
# ==============================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Invertir escalado (solo energia)
scaler_energy = scaler.scale_[-1]
min_energy = scaler.min_[-1]

y_true_inv = y_true / scaler_energy - min_energy/scaler_energy
y_pred_inv = y_pred / scaler_energy - min_energy/scaler_energy

# ==============================
# METRICAS
# ==============================
mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
r2 = r2_score(y_true_inv, y_pred_inv)

print("\n=== METRICAS ===")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# ==============================
# GRAFICAS
# ==============================
plt.figure(figsize=(12,6))
plt.plot(y_true_inv[:200,0], label="Real")
plt.plot(y_pred_inv[:200,0], label="Predicho")
plt.legend()
plt.title("Prediccion de energia solar (12h adelante)")
plt.show()
