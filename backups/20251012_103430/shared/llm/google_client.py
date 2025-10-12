"""
Google Client - Gemini provider
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .base_client import BaseLLMClient, LLMResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleClient(BaseLLMClient):
    """Google Gemini Client"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google Gemini client
        
        Args:
            config: Provider configuration from config.yaml
        """
        super().__init__(config)
        
        # Get API key
        self.api_key = config.get('api_key', os.getenv('GOOGLE_API_KEY'))
        if not self.api_key or self.api_key.startswith('${'):
            logger.warning("Google API key not configured")
            self.api_key = None
        
        # Import Google library only if needed
        if self.api_key:
            try:
                import google.generativeai as genai
                self.genai = genai
                genai.configure(api_key=self.api_key)
                logger.info("Google Gemini client initialized successfully")
            except ImportError:
                logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
                raise
        else:
            self.genai = None
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat with Google Gemini model
        
        Args:
            messages: Chat messages
            agent_type: Agent type (auto-select model)
            model: Explicit model override
            temperature: Temperature override
            max_tokens: Max tokens override
        
        Returns:
            LLMResponse with unified format
        """
        if not self.api_key or not self.genai:
            raise RuntimeError("Google client not properly initialized (missing API key)")
        
        # Determine model
        model_name = model or self.get_model_for_agent(agent_type or 'default')
        temp = self.get_temperature(temperature)
        max_tok = self.get_max_tokens(max_tokens)
        
        logger.info(f"Google call: {model_name} (temp={temp})")
        
        try:
            # Create model instance
            gemini_model = self.genai.GenerativeModel(model_name)
            
            # Convert messages to Gemini format
            # Gemini uses "user" and "model" roles
            gemini_messages = []
            for msg in messages:
                role = "user" if msg['role'] in ['user', 'system'] else "model"
                gemini_messages.append({
                    'role': role,
                    'parts': [msg['content']]
                })
            
            # Generate response
            response = gemini_model.generate_content(
                gemini_messages,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=max_tok
                )
            )
            
            # Extract content
            content = response.text
            
            # Token usage (Gemini provides this)
            prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
            completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
            
            logger.info(f"✓ {model_name}: {completion_tokens} output, {prompt_tokens} input tokens")
            
            return LLMResponse(
                content=content,
                model=model_name,
                provider="google",
                tokens_used=prompt_tokens + completion_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Google error ({model_name}): {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Google Gemini API is available"""
        if not self.api_key or not self.genai:
            return False
        
        try:
            # Simple test
            model = self.genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Hi")
            return response is not None
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """List available Gemini models"""
        if not self.api_key or not self.genai:
            return []
        
        try:
            models = self.genai.list_models()
            return [m.name.split('/')[-1] for m in models if 'gemini' in m.name]
        except Exception as e:
            logger.error(f"Error listing Google models: {e}")
            return ["gemini-1.5-pro", "gemini-1.5-flash"]  # Fallback
