from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
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
    
    class Config:
        case_sensitive = True

settings = Settings() 