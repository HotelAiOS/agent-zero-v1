import subprocess
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SystemOllamaClient:
    """System ollama client - avoids pip dependency conflicts"""
    
    def __init__(self):
        self.available_models = self._get_available_models()
        logger.info(f"🤖 Ollama client initialized with {len(self.available_models)} models")
    
    def _get_available_models(self) -> List[str]:
        """Get available ollama models from system"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]  # First column
                        models.append(model_name)
                logger.info(f"📋 Found {len(models)} ollama models")
                return models
        except Exception as e:
            logger.error(f"❌ Failed to get ollama models: {e}")
        return []
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
             timeout: int = 60) -> Dict[str, Any]:
        """Chat with ollama model via system call"""
        
        if model not in self.available_models:
            logger.error(f"❌ Model {model} not available. Available: {self.available_models}")
            return {'error': f'Model {model} not found'}
        
        try:
            # Build prompt from messages
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    prompt_parts.append(f"<SYSTEM>\n{content}\n</SYSTEM>")
                elif role == 'user':
                    prompt_parts.append(f"<USER>\n{content}\n</USER>")
                elif role == 'assistant':
                    prompt_parts.append(f"<ASSISTANT>\n{content}\n</ASSISTANT>")
            
            full_prompt = "\n\n".join(prompt_parts)
            full_prompt += "\n\n<ASSISTANT>\n"  # Prompt for response
            
            logger.info(f"🧠 Calling {model} with prompt length: {len(full_prompt)} chars")
            
            # Call ollama via subprocess
            result = subprocess.run([
                'ollama', 'run', model, full_prompt
            ], capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                response_text = result.stdout.strip()
                logger.info(f"✅ {model} responded with {len(response_text)} chars")
                
                return {
                    'message': {'content': response_text},
                    'model': model,
                    'done': True
                }
            else:
                error_msg = result.stderr.strip() or "Unknown ollama error"
                logger.error(f"❌ Ollama error: {error_msg}")
                return {'error': error_msg}
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Ollama timeout ({timeout}s) for model {model}")
            return {'error': f'timeout after {timeout}s'}
        except Exception as e:
            logger.error(f"❌ Ollama system error: {e}")
            return {'error': str(e)}
    
    def is_model_available(self, model: str) -> bool:
        """Check if model is available"""
        return model in self.available_models

# Global instance
ollama_client = SystemOllamaClient()

# Compatibility wrapper (like original ollama.chat)
def chat(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Compatibility wrapper for ollama.chat API"""
    return ollama_client.chat(model, messages, **kwargs)

# Create ollama module-like interface
class OllamaModule:
    def chat(self, model: str, messages: List[Dict[str, str]], **kwargs):
        return chat(model, messages, **kwargs)

# For import compatibility
ollama = OllamaModule()
