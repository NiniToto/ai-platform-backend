from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class FileInfo(BaseModel):
    name: str
    uploaded_at: str
    size: int
    chunk_count: Optional[int] = None
    llm_model: str
    embedding_model: str

class FileList(BaseModel):
    files: list[FileInfo]

class FileUploadResponse(BaseModel):
    message: str
    files: list[str]

class FileDeleteResponse(BaseModel):
    message: str 