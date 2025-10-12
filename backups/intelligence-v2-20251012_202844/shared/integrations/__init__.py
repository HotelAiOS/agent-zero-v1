"""
Integrations Module
External integrations - GitHub, GitLab, Cloud providers
"""

from .base import BaseIntegration, IntegrationType
from .github import GitHubIntegration
from .workflows import WorkflowGenerator

__all__ = [
    'BaseIntegration',
    'IntegrationType',
    'GitHubIntegration',
    'WorkflowGenerator'
]

__version__ = '1.0.0'
