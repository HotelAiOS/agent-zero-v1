"""
Template Registry
Centralny rejestr wszystkich szablonów
"""

from typing import Dict, List, Optional
from .base import ProjectTemplate, TemplateCategory
from .saas import SaaSTemplate
from .ecommerce import EcommerceTemplate
from .cms import CMSTemplate
from .microservices import MicroservicesTemplate
from .mobile_backend import MobileBackendTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Template Registry
    Centralne zarządzanie szablonami projektów
    """
    
    def __init__(self):
        self.templates: Dict[str, ProjectTemplate] = {}
        self._register_default_templates()
        logger.info(f"TemplateRegistry zainicjalizowany z {len(self.templates)} szablonami")
    
    def _register_default_templates(self):
        """Zarejestruj domyślne szablony"""
        self.register(SaaSTemplate())
        self.register(EcommerceTemplate())
        self.register(CMSTemplate())
        self.register(MicroservicesTemplate())
        self.register(MobileBackendTemplate())
    
    def register(self, template: ProjectTemplate):
        """
        Zarejestruj szablon
        
        Args:
            template: Instancja szablonu
        """
        template_id = template.get_template_id()
        self.templates[template_id] = template
        logger.info(f"Zarejestrowano szablon: {template_id}")
    
    def get(self, template_id: str) -> Optional[ProjectTemplate]:
        """
        Pobierz szablon po ID
        
        Args:
            template_id: ID szablonu
        
        Returns:
            Template lub None
        """
        return self.templates.get(template_id)
    
    def get_all(self) -> List[ProjectTemplate]:
        """Pobierz wszystkie szablony"""
        return list(self.templates.values())
    
    def get_by_category(self, category: TemplateCategory) -> List[ProjectTemplate]:
        """
        Pobierz szablony po kategorii
        
        Args:
            category: Kategoria
        
        Returns:
            Lista szablonów
        """
        return [
            template for template in self.templates.values()
            if template.get_category() == category
        ]
    
    def search(self, query: str) -> List[ProjectTemplate]:
        """
        Wyszukaj szablony
        
        Args:
            query: Zapytanie (nazwa, tagi, tech stack)
        
        Returns:
            Lista pasujących szablonów
        """
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            config = template.get_config()
            
            # Search in name
            if query_lower in config.template_name.lower():
                results.append(template)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in config.tags):
                results.append(template)
                continue
            
            # Search in tech stack
            if any(query_lower in tech.lower() for tech in config.tech_stack):
                results.append(template)
                continue
        
        return results
    
    def list_categories(self) -> List[TemplateCategory]:
        """Lista wszystkich kategorii"""
        return list(set(t.get_category() for t in self.templates.values()))
    
    def get_summary(self) -> Dict[str, int]:
        """Pobierz podsumowanie registry"""
        return {
            'total_templates': len(self.templates),
            'categories': len(self.list_categories()),
            'templates_by_category': {
                category.value: len(self.get_by_category(category))
                for category in self.list_categories()
            }
        }


# Global registry instance
_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Pobierz globalny template registry"""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry
