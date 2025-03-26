from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from ..services.auth_service import AuthService

router = APIRouter()
auth_service = AuthService()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = auth_service.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다."
        )
    
    access_token = auth_service.create_access_token(user)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user["role"]
    }

@router.get("/verify")
async def verify_token(token: str):
    try:
        payload = auth_service.verify_token(token)
        return {"valid": True, "role": payload.get("role")}
    except HTTPException:
        return {"valid": False} 