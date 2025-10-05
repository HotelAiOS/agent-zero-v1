"""
Agent Factory Module
Dynamiczne tworzenie i zarządzanie wyspecjalizowanymi agentami AI
"""

from .factory import AgentFactory
from .capabilities import CapabilityMatcher, AgentCapability
from .lifecycle import AgentLifecycleManager

__all__ = [
    'AgentFactory',
    'CapabilityMatcher', 
    'AgentCapability',
    'AgentLifecycleManager'
]

__version__ = '1.0.0'
