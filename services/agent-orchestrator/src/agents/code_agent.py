import httpx
import logging
from typing import Dict, Any
from .base_agent import BaseAgent, AgentMetadata, AgentCapability, AgentTask

logger = logging.getLogger(__name__)

class CodeAgent(BaseAgent):
    """Agent specjalizujący się w generacji kodu"""
    
    def __init__(self, ai_router_url: str):
        metadata = AgentMetadata(
            id="code-agent",
            name="Code Agent",
            description="Generates high-quality code based on specifications",
            capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.CODE_REVIEW]
        )
        super().__init__(metadata, ai_router_url)
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Generuj kod"""
        logger.info(f"CodeAgent executing: {task.description}")
        
        prompt = self._build_prompt(task)
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ai_router_url}/generate",
                json={
                    "prompt": prompt,
                    "task_type": "code",
                    "max_tokens": 2048,
                    "temperature": 0.3  # Niska temperatura dla dokładnego kodu
                }
            )
            response.raise_for_status()
            data = response.json()
        
        return {
            "code": data["response"],
            "model": data["model"],
            "tokens": data["tokens"],
            "agent": self.metadata.name
        }
    
    def _build_prompt(self, task: AgentTask) -> str:
        """Zbuduj prompt dla generacji kodu"""
        language = task.input_data.get("language", "python")
        framework = task.input_data.get("framework", "")
        
        prompt = f"""You are an expert {language} developer.

Task: {task.description}

Requirements:
{task.input_data.get('requirements', 'Write clean, well-documented code.')}

Language: {language}
"""
        if framework:
            prompt += f"Framework: {framework}\n"
        
        if task.context:
            prompt += f"\nContext:\n{task.context}\n"
        
        prompt += "\nGenerate the code:"
        
        return prompt
