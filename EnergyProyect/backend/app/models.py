from pydantic import BaseModel, EmailStr

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    company: str
    role: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str
