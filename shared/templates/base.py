"""
Base Project Template
Klasa bazowa dla wszystkich szablonów projektów
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class TemplateCategory(Enum):
    """Kategorie szablonów"""
    WEB_APPLICATION = "web_application"
    MOBILE_BACKEND = "mobile_backend"
    MICROSERVICES = "microservices"
    ECOMMERCE = "ecommerce"
    CMS = "cms"
    SAAS = "saas"
    API_BACKEND = "api_backend"
    CUSTOM = "custom"


@dataclass
class TemplateRequirement:
    """Pojedyncze wymaganie"""
    name: str
    description: str
    priority: int  # 1-5
    complexity: int  # 1-10
    estimated_hours: float


@dataclass
class TemplateConfig:
    """Konfiguracja szablonu"""
    template_id: str
    template_name: str
    category: TemplateCategory
    description: str
    
    # Tech stack
    tech_stack: List[str]
    frameworks: List[str]
    databases: List[str]
    
    # Requirements
    requirements: List[TemplateRequirement]
    
    # Team
    required_agents: List[str]
    team_size: int
    
    # Estimates
    estimated_duration_days: float
    estimated_cost: float
    
    # Quality
    min_test_coverage: float = 0.8
    code_review_required: bool = True
    
    # Additional
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None


class ProjectTemplate(ABC):
    """
    Base Project Template
    Szablon projektu z pre-configured settings
    """
    
    def __init__(self):
        self.config: Optional[TemplateConfig] = None
    
    @abstractmethod
    def get_config(self) -> TemplateConfig:
        """Pobierz konfigurację szablonu"""
        pass
    
    @abstractmethod
    def get_business_requirements(self) -> List[str]:
        """Pobierz wymagania biznesowe"""
        pass
    
    def get_template_id(self) -> str:
        """Pobierz ID szablonu"""
        return self.get_config().template_id
    
    def get_template_name(self) -> str:
        """Pobierz nazwę szablonu"""
        return self.get_config().template_name
    
    def get_category(self) -> TemplateCategory:
        """Pobierz kategorię"""
        return self.get_config().category
    
    def get_estimated_duration(self) -> float:
        """Pobierz szacowany czas (dni)"""
        return self.get_config().estimated_duration_days
    
    def get_estimated_cost(self) -> float:
        """Pobierz szacowany koszt"""
        return self.get_config().estimated_cost
    
    def get_tech_stack(self) -> List[str]:
        """Pobierz tech stack"""
        return self.get_config().tech_stack
    
    def get_required_agents(self) -> List[str]:
        """Pobierz wymagane typy agentów"""
        return self.get_config().required_agents
    
    def customize(self, customizations: Dict[str, Any]) -> 'ProjectTemplate':
        """
        Dostosuj szablon
        
        Args:
            customizations: Dict z customizacjami
                - name: Nowa nazwa
                - tech_stack: Zmień tech stack
                - team_size: Zmień rozmiar zespołu
                - etc.
        
        Returns:
            Self (dla chaining)
        """
        config = self.get_config()
        
        if 'name' in customizations:
            config.template_name = customizations['name']
        
        if 'tech_stack' in customizations:
            config.tech_stack = customizations['tech_stack']
        
        if 'team_size' in customizations:
            config.team_size = customizations['team_size']
        
        if 'estimated_duration_days' in customizations:
            config.estimated_duration_days = customizations['estimated_duration_days']
        
        if 'estimated_cost' in customizations:
            config.estimated_cost = customizations['estimated_cost']
        
        self.config = config
        return self
    
    def to_project_data(self) -> Dict[str, Any]:
        """
        Konwertuj do danych projektu (dla API)
        
        Returns:
            Dict gotowy do POST /api/v1/projects/
        """
        config = self.get_config()
        
        return {
            'project_name': config.template_name,
            'project_type': config.category.value,
            'business_requirements': self.get_business_requirements(),
            'schedule_strategy': 'load_balanced',
            'tech_stack': config.tech_stack,
            'team_size': config.team_size,
            'estimated_duration_days': config.estimated_duration_days,
            'estimated_cost': config.estimated_cost,
            'min_test_coverage': config.min_test_coverage,
            'code_review_required': config.code_review_required
        }
    
    def __repr__(self):
        config = self.get_config()
        return f"<{self.__class__.__name__}: {config.template_name} ({config.category.value})>"
