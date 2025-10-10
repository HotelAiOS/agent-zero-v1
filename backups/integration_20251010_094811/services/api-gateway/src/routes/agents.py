from fastapi import APIRouter, HTTPException
import httpx
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["Agents"])

ORCHESTRATOR_URL = os.getenv("AGENT_ORCHESTRATOR_URL", "http://agent-orchestrator-service:8002")

@router.get("/")
async def list_agents():
    """Lista wszystkich agentów"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ORCHESTRATOR_URL}/agents/")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"List agents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/task")
async def create_task(task_data: Dict[str, Any]):
    """Wykonaj zadanie przez agenta"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/task",
                json=task_data
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Create task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflow/{workflow_name}")
async def execute_workflow(workflow_name: str, description: str, language: str = "python"):
    """Wykonaj workflow"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/workflow/{workflow_name}",
                params={
                    "description": description,
                    "language": language
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Execute workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_agent(agent_data: Dict[str, Any]):
    """Utwórz custom agenta"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/agents/create",
                json=agent_data
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Create agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/custom")
async def list_custom_agents():
    """Lista custom agentów"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{ORCHESTRATOR_URL}/agents/custom")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"List custom agents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/custom/{agent_name}")
async def delete_agent(agent_name: str):
    """Usuń custom agenta"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{ORCHESTRATOR_URL}/agents/custom/{agent_name}"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Delete agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
