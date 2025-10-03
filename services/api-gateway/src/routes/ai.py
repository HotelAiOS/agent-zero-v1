from fastapi import APIRouter, HTTPException
import httpx
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["AI"])

AI_ROUTER_URL = os.getenv("AI_ROUTER_URL", "http://ai-router-service:8000")

@router.post("/generate")
async def generate(request: Dict[str, Any]):
    """Proxy do AI Router - generacja"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{AI_ROUTER_URL}/generate",
                json=request
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """Lista dostępnych modeli AI"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{AI_ROUTER_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
