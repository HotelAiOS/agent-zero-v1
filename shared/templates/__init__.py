"""
Templates Module
Gotowe szablony projektów - instant project creation
"""

from .base import ProjectTemplate, TemplateCategory
from .saas import SaaSTemplate
from .ecommerce import EcommerceTemplate
from .cms import CMSTemplate
from .microservices import MicroservicesTemplate
from .mobile_backend import MobileBackendTemplate
from .registry import TemplateRegistry, get_template_registry

__all__ = [
    'ProjectTemplate',
    'TemplateCategory',
    'SaaSTemplate',
    'EcommerceTemplate',
    'CMSTemplate',
    'MicroservicesTemplate',
    'MobileBackendTemplate',
    'TemplateRegistry',
    'get_template_registry'
]

__version__ = '1.0.0'
