from pydantic_settings import BaseSettings
from typing import Optional, List
import os

# 현재 환경 확인
ENV = os.getenv("ENV", "development")

class Settings(BaseSettings):
    # 환경 설정
    ENV: str = ENV
    
    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TNLabs Backend"
    
    # 보안 설정
    SECRET_KEY: str = "your-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # RAG 설정
    RAG_MODEL: str = "Llama-3.1"
    UPLOAD_DIR: str = "uploads"
    
    # HWP 설정
    HWP_TEMP_DIR: str = "temp"
    
    # CORS 설정
    FRONTEND_LOCAL_URL: str = "http://localhost:3000"
    FRONTEND_PROD_URL: str = "https://astonishing-crepe-6c2532.netlify.app"
    
    # CORS 허용 도메인 목록
    @property
    def CORS_ORIGINS(self) -> List[str]:
        origins = [self.FRONTEND_LOCAL_URL, "http://localhost:8000"]
        if self.ENV == "production":
            origins.append(self.FRONTEND_PROD_URL)
        return origins
    
    class Config:
        case_sensitive = True

settings = Settings() 