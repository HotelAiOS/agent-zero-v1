"""
Base Integration
Klasa bazowa dla wszystkich integracji zewnętrznych
"""

from enum import Enum
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Typy integracji"""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    SLACK = "slack"
    DISCORD = "discord"
    JIRA = "jira"


class BaseIntegration(ABC):
    """
    Base Integration
    Abstrakcyjna klasa bazowa dla wszystkich integracji
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Konfiguracja integracji
                - api_token: Token API
                - base_url: Base URL (opcjonalne)
                - other integration-specific config
        """
        self.config = config
        self.api_token = config.get('api_token')
        self.base_url = config.get('base_url')
        self._validate_config()
        logger.info(f"{self.__class__.__name__} zainicjalizowany")
    
    @abstractmethod
    def _validate_config(self):
        """Walidacja konfiguracji"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test połączenia z zewnętrznym serwisem
        
        Returns:
            True jeśli połączenie działa
        """
        pass
    
    @abstractmethod
    def get_integration_type(self) -> IntegrationType:
        """Pobierz typ integracji"""
        pass
    
    def is_connected(self) -> bool:
        """Sprawdź czy integracja jest połączona"""
        return self.test_connection()
    
    def get_config(self) -> Dict[str, Any]:
        """Pobierz konfigurację (bez tokenów)"""
        safe_config = self.config.copy()
        if 'api_token' in safe_config:
            safe_config['api_token'] = '***'
        return safe_config
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.get_integration_type().value}>"
