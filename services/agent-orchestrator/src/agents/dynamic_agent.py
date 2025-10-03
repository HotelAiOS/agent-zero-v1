import logging
from typing import Dict, Any
from .base_agent import BaseAgent, AgentTask, AgentCapability, TaskStatus

logger = logging.getLogger(__name__)

class DynamicAgent(BaseAgent):
    """Agent dynamiczny tworzony w runtime z custom prompt"""
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        capabilities: list[str],
        ai_router_url: str,
        model_preference: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        # Konwertuj string capabilities na AgentCapability enum
        capability_enums = []
        for cap in capabilities:
            try:
                capability_enums.append(AgentCapability(cap))
            except ValueError:
                logger.warning(f"Unknown capability: {cap}")
        
        super().__init__(
            name=name,
            capabilities=capability_enums,
            ai_router_url=ai_router_url
        )
        
        self.system_prompt = system_prompt
        self.model_preference = model_preference
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def execute(self, task: AgentTask) -> AgentTask:
        """Wykonaj zadanie używając custom system prompt"""
        try:
            logger.info(f"{self.name} executing task: {task.id}")
            
            # Zbuduj prompt z system prompt + task
            full_prompt = f"""
{self.system_prompt}

Task: {task.description}

Input Data:
{task.input_data}

Provide a detailed, high-quality response.
"""
            
            # Wywołaj AI
            result = await self._call_ai(
                prompt=full_prompt,
                task_type=task.type,
                model=self.model_preference if self.model_preference != "auto" else None,
                max_tokens=self.max_tokens
            )
            
            task.result = {"output": result}
            task.status = TaskStatus.COMPLETED
            
            logger.info(f"{self.name} completed task: {task.id}")
            return task
            
        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return task
