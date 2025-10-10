from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskType(str, Enum):
    """Typy zadań"""
    CODE = "code_generation"
    TEST = "testing"
    DOCS = "documentation"

class TaskRequest(BaseModel):
    """Żądanie wykonania zadania"""
    type: TaskType
    description: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

class TaskResponse(BaseModel):
    """Odpowiedź zadania"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_used: str
    execution_time_ms: float

class WorkflowRequest(BaseModel):
    """Żądanie workflow"""
    tasks: List[TaskRequest]
    description: str = "Multi-agent workflow"

class WorkflowResponse(BaseModel):
    """Odpowiedź workflow"""
    workflow_id: str
    tasks_completed: int
    tasks_failed: int
    results: List[TaskResponse]

class AgentStatus(BaseModel):
    """Status agenta"""
    id: str
    name: str
    capabilities: List[str]
    completed_tasks: int
