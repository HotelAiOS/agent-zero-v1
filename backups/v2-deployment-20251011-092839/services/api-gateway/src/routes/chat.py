from fastapi import APIRouter, HTTPException
import httpx
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://chat-service:8001")

@router.post("/")
async def send_message(request: Dict[str, Any]):
    """Wyślij wiadomość w czacie"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{CHAT_SERVICE_URL}/chat",
                json=request
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{user_id}")
async def get_sessions(user_id: str):
    """Lista sesji użytkownika"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{CHAT_SERVICE_URL}/sessions/{user_id}"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Historia sesji"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{CHAT_SERVICE_URL}/sessions/{session_id}/history"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
