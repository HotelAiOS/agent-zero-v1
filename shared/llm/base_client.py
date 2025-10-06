"""
Base LLM Client - Abstract interface for all LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    num_ctx: Optional[int] = None  # For Ollama
    num_predict: Optional[int] = None  # For Ollama


@dataclass
class LLMResponse:
    """Unified LLM response format"""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM providers
    Unified interface: Ollama, OpenAI, Anthropic, Google
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.provider_name = self.__class__.__name__.replace('Client', '').lower()
        self.model_mapping = config.get("models", {})
        self.default_params = config.get("parameters", {})
        
        logger.info(f"{self.__class__.__name__} initializing...")
    
    @abstractmethod
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
        Send chat request to LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            agent_type: Agent type for model selection
            model: Explicit model override
            temperature: Temperature override
            max_tokens: Max tokens override
            **kwargs: Provider-specific parameters
        
        Returns:
            LLMResponse with unified format
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is available and healthy
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    def get_model_for_agent(self, agent_type: str) -> str:
        """
        Get model name for specific agent type
        
        Args:
            agent_type: Type of agent (backend, frontend, etc.)
        
        Returns:
            Model name
        """
        return self.model_mapping.get(
            agent_type,
            self.model_mapping.get("default", "default-model")
        )
    
    def get_temperature(self, override: Optional[float] = None) -> float:
        """Get temperature with override support"""
        if override is not None:
            return override
        return self.default_params.get("temperature", 0.7)
    
    def get_max_tokens(self, override: Optional[int] = None) -> int:
        """Get max_tokens with override support"""
        if override is not None:
            return override
        return self.default_params.get("max_tokens", 4096)
    
    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List all available models from provider
        
        Returns:
            List of model names
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name}>"
