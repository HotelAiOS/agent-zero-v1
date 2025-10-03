import httpx
import logging
from typing import Dict, Any
from .base_agent import BaseAgent, AgentMetadata, AgentCapability, AgentTask

logger = logging.getLogger(__name__)

class TestAgent(BaseAgent):
    """Agent specjalizujący się w tworzeniu testów"""
    
    def __init__(self, ai_router_url: str):
        metadata = AgentMetadata(
            id="test-agent",
            name="Test Agent",
            description="Creates comprehensive test suites",
            capabilities=[AgentCapability.TESTING]
        )
        super().__init__(metadata, ai_router_url)
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Generuj testy"""
        logger.info(f"TestAgent executing: {task.description}")
        
        code_to_test = task.input_data.get("code", "")
        test_framework = task.input_data.get("test_framework", "pytest")
        
        prompt = f"""You are an expert test engineer.

Task: Write comprehensive tests for the following code.

Code to test:


Requirements:
- Use {test_framework}
- Include unit tests
- Include edge cases
- Add docstrings

Generate the test code:
"""
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ai_router_url}/generate",
                json={
                    "prompt": prompt,
                    "task_type": "code",
                    "max_tokens": 2048,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            data = response.json()
        
        return {
            "tests": data["response"],
            "framework": test_framework,
            "model": data["model"],
            "agent": self.metadata.name
        }
