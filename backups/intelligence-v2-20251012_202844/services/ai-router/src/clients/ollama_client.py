import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    """Klient dla Ollama"""
    
    def __init__(self, base_url: str = "http://ollama-service:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(
        self, 
        prompt: str, 
        model: str = "llama3.2:3b",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False
    ) -> dict:
        """Generuj odpowiedź przez Ollama"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "response": data.get("response", ""),
                "model": model,
                "tokens": data.get("eval_count", 0)
            }
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise
    
    async def list_models(self) -> list[str]:
        """Lista dostępnych modeli"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Sprawdź czy Ollama działa"""
        try:
            response = await self.client.get(self.base_url, timeout=5.0)
            return response.status_code == 200
        except:
            return False
