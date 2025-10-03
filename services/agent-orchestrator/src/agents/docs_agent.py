import httpx
import logging
from typing import Dict, Any
from .base_agent import BaseAgent, AgentMetadata, AgentCapability, AgentTask

logger = logging.getLogger(__name__)

class DocsAgent(BaseAgent):
    """Agent specjalizujący się w dokumentacji"""
    
    def __init__(self, ai_router_url: str):
        metadata = AgentMetadata(
            id="docs-agent",
            name="Documentation Agent",
            description="Creates comprehensive documentation",
            capabilities=[AgentCapability.DOCUMENTATION]
        )
        super().__init__(metadata, ai_router_url)
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Generuj dokumentację"""
        logger.info(f"DocsAgent executing: {task.description}")
        
        code_to_document = task.input_data.get("code", "")
        doc_format = task.input_data.get("format", "markdown")
        
        prompt = f"""You are a technical documentation expert.

Task: Create comprehensive documentation for the following code.

Code:

Requirements:
- Format: {doc_format}
- Include overview
- Document all functions/classes
- Add usage examples
- Include API reference if applicable

Generate the documentation:
"""
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ai_router_url}/generate",
                json={
                    "prompt": prompt,
                    "task_type": "documentation",
                    "max_tokens": 3000,
                    "temperature": 0.4
                }
            )
            response.raise_for_status()
            data = response.json()
        
        return {
            "documentation": data["response"],
            "format": doc_format,
            "model": data["model"],
            "agent": self.metadata.name
        }
