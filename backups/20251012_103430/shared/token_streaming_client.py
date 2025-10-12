import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TokenStreamingOllama:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    async def chat_stream(self, model: str, messages: list, timeout: int = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream AI response token by token"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True  # Enable streaming
        }
        
        try:
            timeout_config = httpx.Timeout(timeout) if timeout else httpx.Timeout(None)
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                logger.info(f"🚀 Starting token stream for {model}")
                
                async with client.stream(
                    "POST", 
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    
                    if response.status_code != 200:
                        yield {
                            "error": f"HTTP {response.status_code}",
                            "detail": response.text
                        }
                        return
                    
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        # Process complete JSON lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                                
                            try:
                                data = json.loads(line)
                                yield data
                                
                                # Check if stream is complete
                                if data.get("done", False):
                                    return
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error: {e}")
                                continue
                        
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield {
                "error": str(e),
                "detail": "Connection or parsing failed"
            }

# Global instance
token_streaming_ollama = TokenStreamingOllama()
