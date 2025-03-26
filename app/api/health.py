from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

@router.get("/", response_model=Dict[str, str])
async def health_check():
    """
    서버 상태를 확인하는 헬스체크 엔드포인트입니다.
    """
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"헬스체크 요청: {current_time}")
        
        return {
            "status": "healthy",
            "message": "TNLabs Backend API가 정상적으로 실행 중입니다.",
            "timestamp": current_time
        }
    except Exception as e:
        logger.error(f"헬스체크 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 