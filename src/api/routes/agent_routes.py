"""Agent API Routes"""
from fastapi import APIRouter

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/")
def list_agents():
    return {"agents": []}

@router.post("/{agent_id}/execute")
def execute_agent(agent_id: str, task: dict):
    return {"status": "success", "agent_id": agent_id, "task": task}
