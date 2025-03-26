from . import rag
from fastapi import APIRouter
from app.api import auth, rag

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(rag.router, prefix="/rag", tags=["rag"])
