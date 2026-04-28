"""Health check router"""
import time
from fastapi import APIRouter
from models.schemas import HealthResponse
from services.model_manager import ModelManager
from config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=ModelManager.status(),
        inference_mode=settings.inference_mode,
        version="3.0.0",
        uptime_seconds=round(ModelManager.uptime(), 1),
    )
