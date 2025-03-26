from pydantic import BaseModel

class DocumentInfo(BaseModel):
    name: str
    size: int
