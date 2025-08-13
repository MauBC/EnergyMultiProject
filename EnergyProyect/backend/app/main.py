from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from datetime import datetime

# Rutas API
from app.routes import auth_routes, dashboard_routes

# Dashboards (al importarse, prediccion carga modelo/escalers)
from app.dashboards import prediccion, gestion
from app.dashboards.prediccion import build_dashboard_df  # para warmup

app = FastAPI(title="EnergyProyect Backend")

# ==== CORS ====
origins = [
    "http://localhost:3000",  # React local
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Rutas API ====
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])
app.include_router(dashboard_routes.router, tags=["Dashboard"])

# ==== Montar dashboards de Dash ====
app.mount("/dashboard/prediccion", WSGIMiddleware(prediccion.server))
app.mount("/dashboard/gestion", WSGIMiddleware(gestion.server))

# ==== Warm-up ML al iniciar (opcional, recomendado) ====
@app.on_event("startup")
async def warmup_ml():
    try:
        _ = build_dashboard_df(datetime.now())
        print("[ML] warmup OK: modelo y scalers cargados, prediccion lista.")
    except Exception as e:
        # No detiene el server; solo avisa en logs
        print(f"[ML] warmup FAILED: {e}")

# ==== Ruta raíz ====
@app.get("/")
def root():
    return {"msg": "API funcionando"}
