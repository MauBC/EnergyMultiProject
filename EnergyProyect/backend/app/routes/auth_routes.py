from fastapi import APIRouter, HTTPException
from app.models import UserRegister, UserLogin
from app.database import db
from app.auth import hash_password, verify_password, create_access_token

router = APIRouter()

@router.post("/register")
def register(user: UserRegister):
    if db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Usuario ya existe")
    
    hashed_pw = hash_password(user.password)
    db.users.insert_one({
        "email": user.email,
        "password": hashed_pw,
        "company": user.company,
        "role": user.role
    })
    return {"msg": "Usuario registrado con éxito"}

@router.post("/login")
def login(user: UserLogin):
    db_user = db.users.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}
