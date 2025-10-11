from fastapi import APIRouter, HTTPException
import httpx
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/docs-api", tags=["Documentation"])

DOCS_SERVICE_URL = os.getenv("DOCS_SERVICE_URL", "http://docs-service:8003")

@router.post("/parse")
async def parse_file(request: Dict[str, Any]):
    """Parsuj plik Python"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{DOCS_SERVICE_URL}/parse",
                json=request
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_docs(request: Dict[str, Any]):
    """Generuj dokumentację dla katalogu"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{DOCS_SERVICE_URL}/generate",
                json=request
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Generate docs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/watch/start")
async def start_watching(directories: list[str]):
    """Rozpocznij obserwowanie"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{DOCS_SERVICE_URL}/watch/start",
                params={"directories": directories}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Start watching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/watch/stop")
async def stop_watching():
    """Zatrzymaj watchera"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{DOCS_SERVICE_URL}/watch/stop")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Stop watching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/watch/status")
async def watch_status():
    """Status watchera"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DOCS_SERVICE_URL}/watch/status")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Watch status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
