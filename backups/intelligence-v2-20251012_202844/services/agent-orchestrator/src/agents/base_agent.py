from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import uuid
from datetime import datetime

class AgentCapability(str, Enum):
    """Możliwości agenta"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    SECURITY_AUDIT = "security_audit"

class TaskStatus(str, Enum):
    """Status zadania"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentTask:
    """Zadanie dla agenta"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: AgentCapability = AgentCapability.CODE_GENERATION
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class AgentMetadata:
    """Metadane agenta"""
    id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    version: str = "1.0.0"

class BaseAgent(ABC):
    """Bazowa klasa dla wszystkich agentów"""
    
    def __init__(self, metadata: AgentMetadata, ai_router_url: str):
        self.metadata = metadata
        self.ai_router_url = ai_router_url
        self.tasks_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
    
    @abstractmethod
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Wykonaj zadanie"""
        pass
    
    def can_handle(self, capability: AgentCapability) -> bool:
        """Sprawdź czy agent posiada daną możliwość"""
        return capability in self.metadata.capabilities
    
    async def process_task(self, task: AgentTask) -> AgentTask:
        """Przetwórz zadanie (wrapper z error handling)"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        try:
            result = await self.execute(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        finally:
            task.completed_at = datetime.utcnow()
            self.completed_tasks.append(task)
        
        return task
