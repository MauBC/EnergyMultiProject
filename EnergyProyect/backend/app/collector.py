import os
import time
import requests
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# ======================
# CONFIGURACIÓN
# ======================
load_dotenv()

SERVER = "https://shelly-193-eu.shelly.cloud"
AUTH_KEY = os.getenv("SHELLY_AUTH_KEY")  # ponlo en .env
DEVICE_ID = "2cbcbba6337c"               # tu medidor actual

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["energy_proyect"]
collection = db["energy_data"]

print("✅ Collector iniciado")

# ======================
# FUNCIONES
# ======================
def fetch_realtime_data():
    """Obtiene snapshot actual del medidor desde Shelly Cloud."""
    url = f"{SERVER}/device/status"
    params = {"id": DEVICE_ID, "auth_key": AUTH_KEY}
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        status = data["data"]["device_status"]
        em = status.get("em:0", {})
        ts = datetime.now()

        return {
            "timestamp": ts,
            "total_act_power": float(em.get("total_act_power", 0)),
            "a_voltage": float(em.get("a_voltage", 0)),
            "b_voltage": float(em.get("b_voltage", 0)),
            "c_voltage": float(em.get("c_voltage", 0)),
            "total_current": float(em.get("total_current", 0))
        }
    except Exception as e:
        print("❌ Error obteniendo datos Shelly:", e)
        return None

def save_minute_average(buffer, minute_start):
    """Guarda en Mongo el promedio de un minuto."""
    if not buffer:
        return
    avg_power = np.mean([doc["total_act_power"] for doc in buffer])
    avg_voltage_a = np.mean([doc["a_voltage"] for doc in buffer])
    avg_voltage_b = np.mean([doc["b_voltage"] for doc in buffer])
    avg_voltage_c = np.mean([doc["c_voltage"] for doc in buffer])
    avg_current = np.mean([doc["total_current"] for doc in buffer])

    doc = {
        "timestamp": minute_start,
        "total_act_power": float(avg_power),
        "a_voltage": float(avg_voltage_a),
        "b_voltage": float(avg_voltage_b),
        "c_voltage": float(avg_voltage_c),
        "total_current": float(avg_current)
    }
    collection.insert_one(doc)
    print(f"💾 Guardado en Mongo: {doc}")

# ======================
# LOOP PRINCIPAL
# ======================
def run_collector():
    buffer = []
    current_minute = datetime.now().replace(second=0, microsecond=0)

    while True:
        doc = fetch_realtime_data()
        if doc:
            buffer.append(doc)

        now_minute = datetime.now().replace(second=0, microsecond=0)

        # Si cambió el minuto, guardamos promedio
        if now_minute > current_minute:
            save_minute_average(buffer, current_minute)
            buffer = []
            current_minute = now_minute

        time.sleep(1)  # consultar cada 1s

if __name__ == "__main__":
    run_collector()
