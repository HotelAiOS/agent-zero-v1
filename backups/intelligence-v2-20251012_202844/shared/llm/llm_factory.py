"""
LLM Factory - Multi-provider orchestration with automatic fallback
"""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base_client import BaseLLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM clients with automatic fallback
    Supports: Ollama, OpenAI, Anthropic, Google
    """
    
    _config: Dict[str, Any] = {}
    _clients_cache: Dict[str, BaseLLMClient] = {}
    _config_path: Optional[Path] = None
    
    @classmethod
    def load_config(cls, config_path: str = "shared/llm/config.yaml"):
        """
        Load multi-provider configuration
        
        Args:
            config_path: Path to config.yaml
        """
        path = Path(config_path)
        cls._config_path = path
        
        if not path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            cls._config = cls._default_config()
            return
        
        with open(path, 'r') as f:
            cls._config = yaml.safe_load(f)
        
        logger.info(f"✅ Loaded LLM config from {config_path}")
        logger.info(f"   Default provider: {cls._config.get('default_provider', 'ollama')}")
    
    @classmethod
    def _default_config(cls) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'default_provider': 'ollama',
            'providers': {
                'ollama': {
                    'enabled': True,
                    'base_url': 'http://localhost:11434',
                    'models': {'default': 'deepseek-coder:6.7b'}
                }
            },
            'fallback_chain': ['ollama']
        }
    
    @classmethod
    def create(cls, provider: Optional[str] = None) -> BaseLLMClient:
        """
        Create LLM client for specified provider
        
        Args:
            provider: Provider name (ollama, openai, anthropic, google)
                     If None, uses default_provider from config
        
        Returns:
            BaseLLMClient instance
        """
        if not cls._config:
            cls.load_config()
        
        # Determine provider
        if provider is None:
            provider = cls._config.get('default_provider', 'ollama')
        
        # Return cached client if available
        if provider in cls._clients_cache:
            return cls._clients_cache[provider]
        
        # Get provider config
        providers = cls._config.get('providers', {})
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
        
        provider_config = providers[provider]
        
        # Check if enabled
        if not provider_config.get('enabled', True):
            raise RuntimeError(f"Provider {provider} is disabled in config")
        
        # Create client
        client_class = cls._get_client_class(provider)
        
        try:
            if provider == 'ollama':
                # Pass full config path for Ollama
                client = client_class(
                    provider_config,
                    config_path=str(cls._config_path) if cls._config_path else None
                )
            else:
                client = client_class(provider_config)
            
            cls._clients_cache[provider] = client
            logger.info(f"✅ Created {provider} client")
            return client
        
        except Exception as e:
            logger.error(f"❌ Failed to create {provider} client: {e}")
            raise
    
    @classmethod
    def _get_client_class(cls, provider: str) -> type:
        """Get client class for provider"""
        mapping = {
            'ollama': OllamaClient,
            'openai': OpenAIClient,
            'anthropic': AnthropicClient,
            'google': GoogleClient
        }
        
        if provider not in mapping:
            raise ValueError(f"Unknown provider: {provider}")
        
        return mapping[provider]
    
    @classmethod
    async def create_with_fallback(cls, preferred: Optional[str] = None) -> BaseLLMClient:
        """
        Create LLM client with automatic fallback
        
        Args:
            preferred: Preferred provider (None = use default)
        
        Returns:
            BaseLLMClient instance (first available from fallback chain)
        """
        if not cls._config:
            cls.load_config()
        
        # Determine preferred provider
        if preferred is None:
            preferred = cls._config.get('default_provider', 'ollama')
        
        # Try preferred first
        try:
            client = cls.create(preferred)
            if await client.health_check():
                logger.info(f"✅ Using {preferred} (preferred)")
                return client
            else:
                logger.warning(f"⚠️ {preferred} health check failed")
        except Exception as e:
            logger.warning(f"⚠️ {preferred} unavailable: {e}")
        
        # Try fallback chain
        fallback_chain = cls._config.get('fallback_chain', ['ollama'])
        
        # Remove preferred from chain to avoid retry
        fallback_chain = [p for p in fallback_chain if p != preferred]
        
        for fallback in fallback_chain:
            try:
                providers = cls._config.get('providers', {})
                if fallback not in providers:
                    continue
                
                if not providers[fallback].get('enabled', True):
                    continue
                
                client = cls.create(fallback)
                if await client.health_check():
                    logger.info(f"✅ Using {fallback} (fallback from {preferred})")
                    return client
            except Exception as e:
                logger.warning(f"⚠️ {fallback} fallback failed: {e}")
                continue
        
        raise RuntimeError(f"No LLM provider available. Tried: {preferred}, {fallback_chain}")
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all configured providers"""
        if not cls._config:
            cls.load_config()
        
        return list(cls._config.get('providers', {}).keys())
    
    @classmethod
    def get_default_provider(cls) -> str:
        """Get default provider name"""
        if not cls._config:
            cls.load_config()
        
        return cls._config.get('default_provider', 'ollama')
