import logging
from typing import List, Dict, Any, Optional
from ..agents.base_agent import BaseAgent, AgentTask, AgentCapability, TaskStatus
from ..agents.code_agent import CodeAgent
from ..agents.test_agent import TestAgent
from ..agents.docs_agent import DocsAgent

logger = logging.getLogger(__name__)

class SwarmCoordinator:
    """Koordynator roju agentów"""
    
    def __init__(self, ai_router_url: str):
        self.agents: List[BaseAgent] = []
        self.ai_router_url = ai_router_url
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Inicjalizuj wszystkich agentów"""
        self.agents = [
            CodeAgent(self.ai_router_url),
            TestAgent(self.ai_router_url),
            DocsAgent(self.ai_router_url)
        ]
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def find_agent(self, capability: AgentCapability) -> Optional[BaseAgent]:
        """Znajdź agenta z daną możliwością"""
        for agent in self.agents:
            if agent.can_handle(capability):
                return agent
        return None
    
    async def execute_task(self, task: AgentTask) -> AgentTask:
        """Wykonaj zadanie przez odpowiedniego agenta"""
        agent = self.find_agent(task.type)
        
        if not agent:
            logger.error(f"No agent found for capability: {task.type}")
            task.status = TaskStatus.FAILED
            task.error = f"No agent with capability {task.type}"
            return task
        
        logger.info(f"Assigning task {task.id} to {agent.metadata.name}")
        result = await agent.process_task(task)
        return result
    
    async def execute_workflow(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Wykonaj workflow z wielu zadań"""
        results = []
        
        for task in tasks:
            logger.info(f"Executing task {task.id}: {task.description}")
            result = await self.execute_task(task)
            results.append(result)
            
            # Jeśli zadanie się nie powiodło, przerwij workflow
            if result.status == TaskStatus.FAILED:
                logger.error(f"Workflow failed at task {task.id}")
                break
        
        return results
    
    def get_agents_status(self) -> List[Dict[str, Any]]:
        """Pobierz status wszystkich agentów"""
        return [
            {
                "id": agent.metadata.id,
                "name": agent.metadata.name,
                "capabilities": [cap.value for cap in agent.metadata.capabilities],
                "completed_tasks": len(agent.completed_tasks)
            }
            for agent in self.agents
        ]
