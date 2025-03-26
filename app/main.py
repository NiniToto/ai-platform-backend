from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import rag, auth, health
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="TNLabs 백엔드 API 서버",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
app.include_router(health.router, prefix="/api/health", tags=["health"])

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {"message": "TNLabs Backend API"} 
