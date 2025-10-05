"""
Core Module
Główny silnik Agent Zero - integracja wszystkich modułów
"""

from .engine import AgentZeroCore, ProjectExecution
from .project_manager import ProjectManager, ProjectPhase

__all__ = [
    'AgentZeroCore',
    'ProjectExecution',
    'ProjectManager',
    'ProjectPhase'
]

__version__ = '1.0.0'
