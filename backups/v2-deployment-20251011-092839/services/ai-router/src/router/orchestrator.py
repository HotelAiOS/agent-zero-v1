import logging
import time
from typing import Optional
from ..models.schemas import Provider, TaskType, GenerateRequest, GenerateResponse
from ..clients.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """Orkiestrator wyboru najlepszego modelu AI"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        # Tutaj dodasz więcej klientów (Claude, OpenAI, Gemini)
        
        # Mapowanie task_type -> preferowany model
        self.task_model_map = {
            TaskType.CODE: "qwen2.5-coder:7b",
            TaskType.CHAT: "llama3.2:3b",
            TaskType.ANALYSIS: "llama3.2:3b",
            TaskType.DOCUMENTATION: "qwen2.5-coder:7b",
        }
    
    async def route(self, request: GenerateRequest) -> GenerateResponse:
        """Routuj żądanie do najlepszego modelu"""
        start_time = time.time()
        
        # Wybierz dostawcę
        provider = request.provider or Provider.OLLAMA
        
        # Wybierz model
        if request.model:
            model = request.model
        else:
            model = self.task_model_map.get(request.task_type, "llama3.2:3b")
        
        logger.info(f"Routing to {provider.value}/{model} for task: {request.task_type.value}")
        
        # Generuj odpowiedź
        if provider == Provider.OLLAMA:
            result = await self.ollama_client.generate(
                prompt=request.prompt,
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream
            )
        else:
            raise NotImplementedError(f"Provider {provider} not yet implemented")
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerateResponse(
            response=result["response"],
            provider=provider,
            model=result["model"],
            tokens=result["tokens"],
            latency_ms=latency_ms
        )
    
    async def health_check(self) -> dict[str, bool]:
        """Sprawdź status wszystkich dostawców"""
        return {
            "ollama": await self.ollama_client.health_check(),
            "claude": False,  # Placeholder
            "openai": False,  # Placeholder
            "gemini": False   # Placeholder
        }
