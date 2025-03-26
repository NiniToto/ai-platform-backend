from . import rag, health
from fastapi import APIRouter
from app.api import auth, rag, health

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(rag.router, prefix="/rag", tags=["rag"])
router.include_router(health.router, prefix="/health", tags=["health"])
