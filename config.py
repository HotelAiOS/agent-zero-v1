# Agent Zero V1 - Configuration Management
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # Neo4j Settings
    neo4j_uri: str = field(default_factory=lambda: os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    neo4j_username: str = field(default_factory=lambda: os.getenv('NEO4J_USERNAME', 'neo4j'))
    neo4j_password: str = field(default_factory=lambda: os.getenv('NEO4J_PASSWORD', 'agentzerov1'))
    neo4j_database: str = field(default_factory=lambda: os.getenv('NEO4J_DATABASE', 'neo4j'))
    neo4j_max_connection_lifetime: int = field(default_factory=lambda: int(os.getenv('NEO4J_MAX_CONNECTION_LIFETIME', '3600')))
    neo4j_max_connection_pool_size: int = field(default_factory=lambda: int(os.getenv('NEO4J_MAX_CONNECTION_POOL_SIZE', '50')))
    
    # Redis Settings
    redis_host: str = field(default_factory=lambda: os.getenv('REDIS_HOST', 'localhost'))
    redis_port: int = field(default_factory=lambda: int(os.getenv('REDIS_PORT', '6379')))
    redis_db: int = field(default_factory=lambda: int(os.getenv('REDIS_DB', '0')))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv('REDIS_PASSWORD'))
    redis_timeout: int = field(default_factory=lambda: int(os.getenv('REDIS_TIMEOUT', '5')))

@dataclass
class MessageBrokerConfig:
    """Message broker configuration settings"""
    rabbitmq_host: str = field(default_factory=lambda: os.getenv('RABBITMQ_HOST', 'localhost'))
    rabbitmq_port: int = field(default_factory=lambda: int(os.getenv('RABBITMQ_PORT', '5672')))
    rabbitmq_username: str = field(default_factory=lambda: os.getenv('RABBITMQ_USERNAME', 'agentzerov1'))
    rabbitmq_password: str = field(default_factory=lambda: os.getenv('RABBITMQ_PASSWORD', 'agentzerov1'))
    rabbitmq_virtual_host: str = field(default_factory=lambda: os.getenv('RABBITMQ_VIRTUAL_HOST', 'agent_zero'))
    
    # Queue Configuration
    task_queue: str = field(default_factory=lambda: os.getenv('RABBITMQ_TASK_QUEUE', 'agent_tasks'))
    result_queue: str = field(default_factory=lambda: os.getenv('RABBITMQ_RESULT_QUEUE', 'agent_results'))
    notification_queue: str = field(default_factory=lambda: os.getenv('RABBITMQ_NOTIFICATION_QUEUE', 'notifications'))
    prefetch_count: int = field(default_factory=lambda: int(os.getenv('RABBITMQ_PREFETCH_COUNT', '10')))

@dataclass
class LLMConfig:
    """LLM integration configuration"""
    # OpenAI
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    openai_model: str = field(default_factory=lambda: os.getenv('OPENAI_MODEL', 'gpt-4'))
    openai_max_tokens: int = field(default_factory=lambda: int(os.getenv('OPENAI_MAX_TOKENS', '4000')))
    openai_temperature: float = field(default_factory=lambda: float(os.getenv('OPENAI_TEMPERATURE', '0.7')))
    
    # Anthropic
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY'))
    anthropic_model: str = field(default_factory=lambda: os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'))
    
    # Ollama (Local)
    ollama_host: str = field(default_factory=lambda: os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
    ollama_model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'deepseek-coder:33b'))
    ollama_timeout: int = field(default_factory=lambda: int(os.getenv('OLLAMA_TIMEOUT', '120')))

@dataclass
class WebSocketConfig:
    """WebSocket configuration settings"""
    host: str = field(default_factory=lambda: os.getenv('WEBSOCKET_HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_PORT', '8000')))
    path: str = field(default_factory=lambda: os.getenv('WEBSOCKET_PATH', '/ws'))
    origins: str = field(default_factory=lambda: os.getenv('WEBSOCKET_ORIGINS', '*'))
    ping_interval: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_PING_INTERVAL', '30')))
    ping_timeout: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_PING_TIMEOUT', '10')))
    max_connections: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_MAX_CONNECTIONS', '100')))

@dataclass
class AgentConfig:
    """Agent system configuration"""
    max_concurrent_agents: int = field(default_factory=lambda: int(os.getenv('MAX_CONCURRENT_AGENTS', '8')))
    agent_timeout: int = field(default_factory=lambda: int(os.getenv('AGENT_TIMEOUT', '300')))
    agent_retry_attempts: int = field(default_factory=lambda: int(os.getenv('AGENT_RETRY_ATTEMPTS', '3')))
    heartbeat_interval: int = field(default_factory=lambda: int(os.getenv('AGENT_HEARTBEAT_INTERVAL', '30')))
    
    # Task Decomposer
    task_decomposer_max_depth: int = field(default_factory=lambda: int(os.getenv('TASK_DECOMPOSER_MAX_DEPTH', '5')))
    task_decomposer_max_subtasks: int = field(default_factory=lambda: int(os.getenv('TASK_DECOMPOSER_MAX_SUBTASKS', '20')))
    task_decomposer_timeout: int = field(default_factory=lambda: int(os.getenv('TASK_DECOMPOSER_TIMEOUT', '60')))
    
    # Code Generator
    code_generator_max_files: int = field(default_factory=lambda: int(os.getenv('CODE_GENERATOR_MAX_FILES', '50')))
    code_generator_max_size_mb: int = field(default_factory=lambda: int(os.getenv('CODE_GENERATOR_MAX_SIZE_MB', '10')))
    code_generator_timeout: int = field(default_factory=lambda: int(os.getenv('CODE_GENERATOR_TIMEOUT', '120')))

@dataclass
class MonitoringConfig:
    """Monitoring and performance configuration"""
    prometheus_enabled: bool = field(default_factory=lambda: os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true')
    prometheus_port: int = field(default_factory=lambda: int(os.getenv('PROMETHEUS_PORT', '9091')))
    prometheus_metrics_path: str = field(default_factory=lambda: os.getenv('PROMETHEUS_METRICS_PATH', '/metrics'))
    
    # Health Checks
    health_check_interval: int = field(default_factory=lambda: int(os.getenv('HEALTH_CHECK_INTERVAL', '30')))
    health_check_timeout: int = field(default_factory=lambda: int(os.getenv('HEALTH_CHECK_TIMEOUT', '10')))
    health_check_retries: int = field(default_factory=lambda: int(os.getenv('HEALTH_CHECK_RETRIES', '3')))

@dataclass
class FileSystemConfig:
    """File system and storage configuration"""
    base_dir: Path = field(default_factory=lambda: Path(os.getenv('BASE_DIR', '/app')))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv('LOGS_DIR', '/app/logs')))
    temp_dir: Path = field(default_factory=lambda: Path(os.getenv('TEMP_DIR', '/app/temp')))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv('OUTPUT_DIR', '/app/output')))
    upload_dir: Path = field(default_factory=lambda: Path(os.getenv('UPLOAD_DIR', '/app/uploads')))
    
    # File Limits
    max_upload_size_mb: int = field(default_factory=lambda: int(os.getenv('MAX_UPLOAD_SIZE_MB', '100')))
    max_output_files: int = field(default_factory=lambda: int(os.getenv('MAX_OUTPUT_FILES', '1000')))
    temp_file_retention_hours: int = field(default_factory=lambda: int(os.getenv('TEMP_FILE_RETENTION_HOURS', '24')))

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'change-this-in-production'))
    jwt_secret_key: str = field(default_factory=lambda: os.getenv('JWT_SECRET_KEY', 'change-this-too'))
    jwt_algorithm: str = field(default_factory=lambda: os.getenv('JWT_ALGORITHM', 'HS256'))
    jwt_expiration_hours: int = field(default_factory=lambda: int(os.getenv('JWT_EXPIRATION_HOURS', '24')))
    
    # CORS
    cors_origins: list = field(default_factory=lambda: os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8000').split(','))
    cors_methods: list = field(default_factory=lambda: os.getenv('CORS_METHODS', 'GET,POST,PUT,DELETE,OPTIONS').split(','))
    
    # Rate Limiting
    rate_limit_enabled: bool = field(default_factory=lambda: os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true')
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_REQUESTS', '100')))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_WINDOW', '60')))

class Config:
    """Main configuration class that combines all settings"""
    
    def __init__(self):
        # Core Application Settings
        self.app_name = os.getenv('APP_NAME', 'Agent Zero V1')
        self.app_version = os.getenv('APP_VERSION', '1.0.0')
        self.app_environment = os.getenv('APP_ENVIRONMENT', 'development')
        self.app_debug = os.getenv('APP_DEBUG', 'false').lower() == 'true'
        self.app_host = os.getenv('APP_HOST', '0.0.0.0')
        self.app_port = int(os.getenv('APP_PORT', '8000'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_format = os.getenv('LOG_FORMAT', 'json')
        self.log_file = os.getenv('LOG_FILE', '/app/logs/agent_zero.log')
        
        # Component Configurations
        self.database = DatabaseConfig()
        self.message_broker = MessageBrokerConfig()
        self.llm = LLMConfig()
        self.websocket = WebSocketConfig()
        self.agent = AgentConfig()
        self.monitoring = MonitoringConfig()
        self.filesystem = FileSystemConfig()
        self.security = SecurityConfig()
        
        # Feature Flags
        self.feature_advanced_monitoring = os.getenv('FEATURE_ADVANCED_MONITORING', 'true').lower() == 'true'
        self.feature_auto_scaling = os.getenv('FEATURE_AUTO_SCALING', 'false').lower() == 'true'
        self.feature_experimental_agents = os.getenv('FEATURE_EXPERIMENTAL_AGENTS', 'false').lower() == 'true'
        
        # Ensure directories exist
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.filesystem.logs_dir,
            self.filesystem.temp_dir,
            self.filesystem.output_dir,
            self.filesystem.upload_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging based on settings"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        if self.log_format.lower() == 'json':
            import json_logging
            json_logging.init_fastapi(enable_json=True)
            json_logging.init_request_instrument()
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_database_url(self) -> str:
        """Get Neo4j connection URL"""
        return f"{self.database.neo4j_uri}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.database.redis_password:
            return f"redis://:{self.database.redis_password}@{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
        return f"redis://{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    def get_rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL"""
        return f"amqp://{self.message_broker.rabbitmq_username}:{self.message_broker.rabbitmq_password}@{self.message_broker.rabbitmq_host}:{self.message_broker.rabbitmq_port}/{self.message_broker.rabbitmq_virtual_host}"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.app_environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.app_environment.lower() == 'development'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for debugging)"""
        config_dict = {}
        
        # Basic settings
        config_dict.update({
            'app_name': self.app_name,
            'app_version': self.app_version,
            'app_environment': self.app_environment,
            'app_debug': self.app_debug,
            'app_host': self.app_host,
            'app_port': self.app_port,
        })
        
        # Component configurations
        for component_name in ['database', 'message_broker', 'llm', 'websocket', 'agent', 'monitoring', 'filesystem', 'security']:
            component = getattr(self, component_name)
            config_dict[component_name] = {
                field.name: getattr(component, field.name)
                for field in component.__dataclass_fields__.values()
                if not field.name.endswith('password') and not field.name.endswith('key')  # Hide sensitive data
            }
        
        return config_dict
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required settings
        if not self.security.secret_key or self.security.secret_key == 'change-this-in-production':
            if self.is_production():
                errors.append("SECRET_KEY must be set in production")
        
        if self.llm.openai_api_key and not self.llm.openai_api_key.startswith('sk-'):
            errors.append("Invalid OpenAI API key format")
        
        # Check file permissions
        try:
            self.filesystem.logs_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            errors.append(f"Cannot create logs directory: {self.filesystem.logs_dir}")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        return True

# Global configuration instance
config = Config()

# Validate configuration on import
if not config.validate():
    logging.warning("Configuration validation failed. Some features may not work correctly.")

# Export commonly used configurations
__all__ = [
    'config',
    'Config',
    'DatabaseConfig',
    'MessageBrokerConfig',
    'LLMConfig',
    'WebSocketConfig',
    'AgentConfig',
    'MonitoringConfig',
    'FileSystemConfig',
    'SecurityConfig'
]
    # Add PORT alias for backward compatibility
    PORT = property(lambda self: self.app_port)
