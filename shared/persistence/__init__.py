"""
Persistence Module
Warstwa trwałości danych - Database, Storage, Cache
"""

from .database import DatabaseManager, get_db_session
from .models import (
    ProjectModel,
    AgentModel,
    TaskModel,
    ProtocolModel,
    AnalysisModel
)
from .repositories import (
    ProjectRepository,
    AgentRepository,
    TaskRepository,
    ProtocolRepository,
    AnalysisRepository
)
from .cache import CacheManager

__all__ = [
    'DatabaseManager',
    'get_db_session',
    'ProjectModel',
    'AgentModel',
    'TaskModel',
    'ProtocolModel',
    'AnalysisModel',
    'ProjectRepository',
    'AgentRepository',
    'TaskRepository',
    'ProtocolRepository',
    'AnalysisRepository',
    'CacheManager'
]

__version__ = '1.0.0'
