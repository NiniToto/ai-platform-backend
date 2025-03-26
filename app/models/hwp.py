from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class DocumentSpec(BaseModel):
    title: str
    content: str
    format_content: Optional[bool] = False
    save_filename: Optional[str] = None
    preserve_linebreaks: Optional[bool] = True

class BatchOperation(BaseModel):
    operation: str
    params: Dict[str, Any]

class BatchRequest(BaseModel):
    operations: List[BatchOperation] 