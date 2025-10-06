"""
Ollama Client - Local LLM provider
Multi-model orchestration for heterogeneous AI team
"""

import ollama
import yaml
import logging
import httpx
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_client import BaseLLMClient, LLMResponse, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """
    Ollama Multi-Model Client
    Heterogeneous model assignment for specialized agents
    """
    
    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        """
        Initialize Ollama client
        
        Args:
            config: Provider configuration from config.yaml
            config_path: Optional path to full config.yaml
        """
        super().__init__(config)
        
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.client = ollama.Client(
            host=self.base_url,
            timeout=httpx.Timeout(None)  # No timeout - unlimited wait for quality
        )
        
        # Model assignment cache
        self.agent_models: Dict[str, ModelConfig] = {}
        self.protocol_models: Dict[str, ModelConfig] = {}
        
        # Load full config if provided
        if config_path:
            self._load_full_config(config_path)
        else:
            self._initialize_from_config(config)
        
        logger.info(f"OllamaClient initialized with {len(self.agent_models)} agent models")
    
    def _load_full_config(self, config_path: str):
        """Load full YAML configuration"""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found: {config_path}, using provider config only")
            return
        
        with open(path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        self._initialize_from_config(full_config.get('llm', {}))
    
    def _initialize_from_config(self, llm_config: Dict[str, Any]):
        """Initialize model assignments from config"""
        agents = llm_config.get('agents', {})
        default = llm_config.get('default', {})
        
        # Agent models
        for agent_type, cfg in agents.items():
            self.agent_models[agent_type] = ModelConfig(
                model=cfg.get('model', 'deepseek-coder:6.7b'),
                temperature=cfg.get('temperature', default.get('temperature', 0.7)),
                num_ctx=cfg.get('num_ctx', default.get('num_ctx', 8192)),
                num_predict=cfg.get('num_predict', default.get('num_predict', 4096))
            )
        
        # Protocol models
        protocols = llm_config.get('protocols', {})
        for protocol_type, cfg in protocols.items():
            self.protocol_models[protocol_type] = ModelConfig(
                model=cfg.get('model', 'deepseek-coder:6.7b'),
                temperature=cfg.get('temperature', default.get('temperature', 0.7)),
                num_ctx=cfg.get('num_ctx', default.get('num_ctx', 8192)),
                num_predict=cfg.get('num_predict', default.get('num_predict', 4096))
            )
        
        logger.info(f"Loaded {len(self.agent_models)} agent models, {len(self.protocol_models)} protocol models")
    
    def get_model_for_agent(self, agent_type: str) -> ModelConfig:
        """Get model config for agent type"""
        if agent_type in self.agent_models:
            return self.agent_models[agent_type]
        
        # Fallback to default
        return ModelConfig(
            model=self.model_mapping.get('default', 'deepseek-coder:6.7b'),
            temperature=0.7
        )
    
    def get_model_for_protocol(self, protocol_type: str) -> ModelConfig:
        """Get model config for protocol type"""
        if protocol_type in self.protocol_models:
            return self.protocol_models[protocol_type]
        
        return ModelConfig(model='deepseek-coder:6.7b', temperature=0.7)
    
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
        Chat with Ollama model
        
        Args:
            messages: Chat messages
            agent_type: Agent type (auto-select model)
            model: Explicit model override
            temperature: Temperature override
            max_tokens: Max tokens override (num_predict in Ollama)
        
        Returns:
            LLMResponse with unified format
        """
        # Determine model config
        if model:
            model_config = ModelConfig(model=model, temperature=temperature or 0.7)
        elif agent_type:
            model_config = self.get_model_for_agent(agent_type)
            if temperature is not None:
                model_config.temperature = temperature
        else:
            model_config = ModelConfig(
                model=self.model_mapping.get('default', 'deepseek-coder:6.7b'),
                temperature=temperature or 0.7
            )
        
        # Override max_tokens if provided
        if max_tokens:
            model_config.num_predict = max_tokens
        
        logger.info(f"Ollama call: {model_config.model} (temp={model_config.temperature})")
        
        try:
            response = self.client.chat(
                model=model_config.model,
                messages=messages,
                options={
                    'temperature': model_config.temperature,
                    'num_ctx': model_config.num_ctx or 8192,
                    'num_predict': model_config.num_predict or 4096
                }
            )
            
            # Extract content
            content = response.get('message', {}).get('content', '')
            
            # Extract token counts
            eval_count = response.get('eval_count', 0)
            prompt_eval_count = response.get('prompt_eval_count', 0)
            
            logger.info(f"✓ {model_config.model}: {eval_count} output tokens, {prompt_eval_count} input tokens")
            
            return LLMResponse(
                content=content,
                model=model_config.model,
                provider="ollama",
                tokens_used=eval_count + prompt_eval_count,
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                raw_response=response
            )
        
        except Exception as e:
            logger.error(f"Ollama error ({model_config.model}): {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Ollama is available"""
        try:
            models = self.list_available_models()
            return len(models) > 0
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """List all available Ollama models"""
        try:
            response = self.client.list()
            
            if hasattr(response, 'models'):
                models = response.models
                names = []
                for m in models:
                    if hasattr(m, 'model'):
                        names.append(m.model)
                return names
            
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get info about specific model"""
        try:
            return self.client.show(model)
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def verify_models(self) -> Dict[str, bool]:
        """
        Verify all configured models are available
        
        Returns:
            Dict of model -> available status
        """
        available = self.list_available_models()
        results = {}
        
        # Check agent models
        for agent_type, cfg in self.agent_models.items():
            results[f"{agent_type}:{cfg.model}"] = cfg.model in available
        
        # Check protocol models
        for protocol_type, cfg in self.protocol_models.items():
            results[f"{protocol_type}:{cfg.model}"] = cfg.model in available
        
        missing = [k for k, v in results.items() if not v]
        if missing:
            logger.warning(f"Missing models: {', '.join(missing)}")
        else:
            logger.info("✓ All configured models available")
        
        return results
