"""
Anthropic Client - Claude provider
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .base_client import BaseLLMClient, LLMResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude Client"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic client
        
        Args:
            config: Provider configuration from config.yaml
        """
        super().__init__(config)
        
        # Get API key
        self.api_key = config.get('api_key', os.getenv('ANTHROPIC_API_KEY'))
        if not self.api_key or self.api_key.startswith('${'):
            logger.warning("Anthropic API key not configured")
            self.api_key = None
        
        # Import Anthropic library only if needed
        if self.api_key:
            try:
                import anthropic
                self.anthropic = anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized successfully")
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                raise
        else:
            self.anthropic = None
            self.client = None
    
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
        Chat with Anthropic Claude model
        
        Args:
            messages: Chat messages
            agent_type: Agent type (auto-select model)
            model: Explicit model override
            temperature: Temperature override
            max_tokens: Max tokens override
        
        Returns:
            LLMResponse with unified format
        """
        if not self.api_key or not self.client:
            raise RuntimeError("Anthropic client not properly initialized (missing API key)")
        
        # Determine model
        model_name = model or self.get_model_for_agent(agent_type or 'default')
        temp = self.get_temperature(temperature)
        max_tok = self.get_max_tokens(max_tokens)
        
        logger.info(f"Anthropic call: {model_name} (temp={temp})")
        
        try:
            # Convert messages format (Claude requires system message separate)
            system_msg = None
            claude_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg['content']
                else:
                    claude_messages.append(msg)
            
            # Call Claude API
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tok,
                temperature=temp,
                system=system_msg,
                messages=claude_messages,
                **kwargs
            )
            
            # Extract content
            content = response.content[0].text if response.content else ''
            
            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            logger.info(f"✓ {model_name}: {output_tokens} output, {input_tokens} input tokens")
            
            return LLMResponse(
                content=content,
                model=model_name,
                provider="anthropic",
                tokens_used=input_tokens + output_tokens,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Anthropic error ({model_name}): {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Anthropic API is available"""
        if not self.api_key or not self.client:
            return False
        
        try:
            # Simple test message
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return response is not None
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """List available Anthropic models (hardcoded as API doesn't provide list)"""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
