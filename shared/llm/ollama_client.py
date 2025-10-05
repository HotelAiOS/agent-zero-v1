"""
Ollama Client
Multi-model orchestration for heterogeneous AI team
"""

import ollama
import yaml
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model: str
    temperature: float = 0.7
    num_ctx: int = 8192
    num_predict: int = 4096


class OllamaClient:
    """
    Ollama Multi-Model Client
    Heterogeneous model assignment for specialized agents
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Ollama client with multi-model config
        
        Args:
            config_path: Path to config YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_url = self.config['llm']['base_url']
        self.client = ollama.Client(host=self.base_url)
        
        # Model assignment cache
        self.agent_models: Dict[str, ModelConfig] = {}
        self.protocol_models: Dict[str, ModelConfig] = {}
        
        self._initialize_models()
        
        logger.info(f"OllamaClient initialized with {len(self.agent_models)} agent models")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if YAML not found"""
        return {
            'llm': {
                'base_url': 'http://localhost:11434',
                'default': {
                    'temperature': 0.7,
                    'num_ctx': 8192,
                    'num_predict': 4096
                },
                'agents': {
                    'backend': {'model': 'deepseek-coder:33b'}
                }
            }
        }
    
    def _initialize_models(self):
        """Initialize model assignments from config"""
        agents = self.config['llm'].get('agents', {})
        default = self.config['llm'].get('default', {})
        
        # Agent models
        for agent_type, cfg in agents.items():
            self.agent_models[agent_type] = ModelConfig(
                model=cfg.get('model'),
                temperature=cfg.get('temperature', default.get('temperature', 0.7)),
                num_ctx=cfg.get('num_ctx', default.get('num_ctx', 8192)),
                num_predict=cfg.get('num_predict', default.get('num_predict', 4096))
            )
        
        # Protocol models
        protocols = self.config['llm'].get('protocols', {})
        for protocol_type, cfg in protocols.items():
            self.protocol_models[protocol_type] = ModelConfig(
                model=cfg.get('model'),
                temperature=cfg.get('temperature', default.get('temperature', 0.7)),
                num_ctx=cfg.get('num_ctx', default.get('num_ctx', 8192)),
                num_predict=cfg.get('num_predict', default.get('num_predict', 4096))
            )
        
        logger.info(f"Loaded {len(self.agent_models)} agent models, {len(self.protocol_models)} protocol models")
    
    def get_model_for_agent(self, agent_type: str) -> ModelConfig:
        """
        Get optimal model for agent type
        
        Args:
            agent_type: Agent type (architect, backend, etc.)
        
        Returns:
            ModelConfig for this agent
        """
        if agent_type not in self.agent_models:
            logger.warning(f"No model config for {agent_type}, using default")
            return ModelConfig(model='deepseek-coder:33b')
        
        return self.agent_models[agent_type]
    
    def get_model_for_protocol(self, protocol_type: str) -> ModelConfig:
        """Get optimal model for protocol"""
        if protocol_type not in self.protocol_models:
            logger.warning(f"No model config for protocol {protocol_type}")
            return ModelConfig(model='mixtral:8x7b')
        
        return self.protocol_models[protocol_type]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model_config: Optional[ModelConfig] = None,
        agent_type: Optional[str] = None,
        protocol_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chat with Ollama model
        
        Args:
            messages: Chat messages
            model_config: Explicit model config
            agent_type: Agent type (auto-select model)
            protocol_type: Protocol type (auto-select model)
        
        Returns:
            Response dict
        """
        # Determine model config
        if model_config is None:
            if agent_type:
                model_config = self.get_model_for_agent(agent_type)
            elif protocol_type:
                model_config = self.get_model_for_protocol(protocol_type)
            else:
                model_config = ModelConfig(model='deepseek-coder:33b')
        
        logger.info(f"Ollama call: {model_config.model} (temp={model_config.temperature})")
        
        try:
            response = self.client.chat(
                model=model_config.model,
                messages=messages,
                options={
                    'temperature': model_config.temperature,
                    'num_ctx': model_config.num_ctx,
                    'num_predict': model_config.num_predict
                }
            )
            
            # Log token usage
            eval_count = response.get('eval_count', 0)
            prompt_eval_count = response.get('prompt_eval_count', 0)
            logger.info(f"✓ {model_config.model}: {eval_count} output tokens, {prompt_eval_count} input tokens")
            
            return response
        
        except Exception as e:
            logger.error(f"Ollama error ({model_config.model}): {e}")
            raise
    
    def list_available_models(self) -> List[str]:
        """List all available Ollama models"""
        try:
            # POPRAWKA: response to ListResponse object z atrybutem 'models'
            response = self.client.list()
            
            # response.models to lista obiektów Model
            if hasattr(response, 'models'):
                models = response.models
                # Każdy Model ma atrybut 'model' (nie 'name'!)
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
