from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    model: str = "Llama-3.1"
    context_type: Optional[str] = None  # 선택적 필드로 변경
    additional_info: Optional[str] = None  # 추가 정보 (예시, 단계, 주의사항 등)

class ChatResponse(BaseModel):
    answer: str
    model: str 