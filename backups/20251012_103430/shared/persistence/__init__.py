"""
Persistence Module
Warstwa trwałości danych - Database, Storage, Cache
"""

from .database import (
    DatabaseManager, 
    get_db_session, 
    init_database, 
    get_database,
    Base
)
from .models import (
    ProjectModel,
    AgentModel,
    TaskModel,
    ProtocolModel,
    AnalysisModel,
    PatternModel,
    AntiPatternModel,
    RecommendationModel
)
from .repositories import (
    ProjectRepository,
    AgentRepository,
    TaskRepository,
    ProtocolRepository,
    AnalysisRepository,
    PatternRepository,
    create_repositories
)
from .cache import CacheManager, get_cache

__all__ = [
    # Database
    'DatabaseManager',
    'get_db_session',
    'init_database',
    'get_database',
    'Base',
    
    # Models
    'ProjectModel',
    'AgentModel',
    'TaskModel',
    'ProtocolModel',
    'AnalysisModel',
    'PatternModel',
    'AntiPatternModel',
    'RecommendationModel',
    
    # Repositories
    'ProjectRepository',
    'AgentRepository',
    'TaskRepository',
    'ProtocolRepository',
    'AnalysisRepository',
    'PatternRepository',
    'create_repositories',
    
    # Cache
    'CacheManager',
    'get_cache'
]

__version__ = '1.0.0'
