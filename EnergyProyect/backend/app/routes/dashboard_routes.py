from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from app.auth import decode_access_token

router = APIRouter()
security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    payload = decode_access_token(token.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    return payload

@router.get("/dashboard")
def dashboard(user: dict = Depends(get_current_user)):
    return {
        "msg": f"Bienvenido {user['sub']}",
        "data": {
            "produccion_kwh": 153.4,
            "empresa": "Mi Empresa"
        }
    }
