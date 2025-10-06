"""
OpenAI Client - GPT-4, GPT-4o provider
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .base_client import BaseLLMClient, LLMResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT Client"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI client
        
        Args:
            config: Provider configuration from config.yaml
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.get('api_key', os.getenv('OPENAI_API_KEY'))
        if not self.api_key or self.api_key.startswith('${'):
            logger.warning("OpenAI API key not configured")
            self.api_key = None
        
        self.organization = config.get('organization', os.getenv('OPENAI_ORG_ID'))
        
        # Import OpenAI library only if needed
        if self.api_key:
            try:
                import openai
                self.openai = openai
                self.openai.api_key = self.api_key
                if self.organization:
                    self.openai.organization = self.organization
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                raise
        else:
            self.openai = None
    
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
        Chat with OpenAI GPT model
        
        Args:
            messages: Chat messages
            agent_type: Agent type (auto-select model)
            model: Explicit model override
            temperature: Temperature override
            max_tokens: Max tokens override
        
        Returns:
            LLMResponse with unified format
        """
        if not self.api_key or not self.openai:
            raise RuntimeError("OpenAI client not properly initialized (missing API key)")
        
        # Determine model
        model_name = model or self.get_model_for_agent(agent_type or 'default')
        temp = self.get_temperature(temperature)
        max_tok = self.get_max_tokens(max_tokens)
        
        logger.info(f"OpenAI call: {model_name} (temp={temp})")
        
        try:
            response = self.openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )
            
            # Extract data
            content = response.choices[0].message.content
            usage = response.usage
            
            logger.info(f"✓ {model_name}: {usage.completion_tokens} output, {usage.prompt_tokens} input tokens")
            
            return LLMResponse(
                content=content,
                model=model_name,
                provider="openai",
                tokens_used=usage.total_tokens,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"OpenAI error ({model_name}): {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is available"""
        if not self.api_key or not self.openai:
            return False
        
        try:
            # Try to list models
            models = self.openai.Model.list()
            return len(models.data) > 0
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """List all available OpenAI models"""
        if not self.api_key or not self.openai:
            return []
        
        try:
            models = self.openai.Model.list()
            return [m.id for m in models.data if 'gpt' in m.id]
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            return []
