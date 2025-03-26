from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    model: str = "gemma-3-12b"

class ChatResponse(BaseModel):
    answer: str
    model: str 