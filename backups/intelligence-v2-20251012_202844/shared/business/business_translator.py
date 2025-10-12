"""Agent Zero V1 - Business to Technical Translator

Translacja wymagań biznesowych na specyfikację techniczną
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from .context_enricher import EnrichedContext
    from .constraint_analyzer import ConstraintAnalysis


@dataclass
class TechnicalSpecification:
    """Specyfikacja techniczna"""

    # Komponenty architektury
    architecture_components: List[Dict[str, Any]] = field(default_factory=list)

    # Zadania implementacyjne
    implementation_tasks: List[Dict[str, Any]] = field(default_factory=list)

    # Stack technologiczny
    technology_stack: Dict[str, str] = field(default_factory=dict)

    # Oszacowania
    estimated_effort: str = ""
    estimated_cost: float = 0.0

    # Metadane
    generation_confidence: float = 0.0


class BusinessToTechnicalTranslator:
    """Translator wymagań biznesowych na techniczne"""

    def __init__(self):
        self.component_templates = {
            'user_accounts': {
                'type': 'backend',
                'services': ['authentication', 'user_management', 'session_management']
            },
            'payment': {
                'type': 'integration',
                'services': ['payment_gateway', 'transaction_processing', 'receipt_generation']
            },
            'api': {
                'type': 'backend',
                'services': ['rest_api', 'api_documentation', 'rate_limiting']
            }
        }

    async def translate(self, 
                       context: "EnrichedContext", 
                       constraints: "ConstraintAnalysis") -> TechnicalSpecification:
        """Translacja na specyfikację techniczną"""
        spec = TechnicalSpecification()

        if context.original_intent:
            intent = context.original_intent

            # Generuj komponenty architektury
            spec.architecture_components = self._generate_architecture(intent, context)

            # Generuj zadania implementacyjne
            spec.implementation_tasks = self._generate_tasks(intent, context)

            # Określ stack technologiczny
            spec.technology_stack = intent.technology_preferences or {}

            # Oszacuj wysiłek i koszt
            spec.estimated_effort = self._estimate_effort(intent, context)
            spec.estimated_cost = self._estimate_cost(spec.estimated_effort)

            # Oblicz confidence
            spec.generation_confidence = 0.85  # Placeholder

        return spec

    def _generate_architecture(self, intent, context) -> List[Dict[str, Any]]:
        """Generuj komponenty architektury"""
        components = []

        # Na podstawie celów projektu
        for goal in intent.primary_goals[:3]:
            goal_lower = goal.lower()

            if 'api' in goal_lower:
                components.append({
                    'name': 'REST API',
                    'type': 'backend',
                    'description': 'RESTful API endpoints',
                    'priority': 'high'
                })

            if 'user' in goal_lower or 'account' in goal_lower:
                components.append({
                    'name': 'User Management',
                    'type': 'backend',
                    'description': 'User authentication and profiles',
                    'priority': 'high'
                })

            if 'payment' in goal_lower or 'checkout' in goal_lower:
                components.append({
                    'name': 'Payment Processing',
                    'type': 'integration',
                    'description': 'Payment gateway integration',
                    'priority': 'high'
                })

        # Dodaj komponenty z missing requirements
        for req in context.missing_requirements[:2]:
            components.append({
                'name': req.replace('_', ' ').title(),
                'type': 'feature',
                'description': f'Implementation of {req}',
                'priority': 'medium'
            })

        return components

    def _generate_tasks(self, intent, context) -> List[Dict[str, Any]]:
        """Generuj zadania implementacyjne"""
        tasks = []

        # Setup phase
        tasks.append({
            'phase': 'setup',
            'name': 'Project initialization',
            'description': 'Setup development environment and repositories',
            'estimated_hours': 8
        })

        # Implementation phase
        for idx, component in enumerate(self._generate_architecture(intent, context)[:5]):
            tasks.append({
                'phase': 'implementation',
                'name': f"Implement {component['name']}",
                'description': component['description'],
                'estimated_hours': 40 if component['priority'] == 'high' else 24
            })

        # Testing phase
        tasks.append({
            'phase': 'testing',
            'name': 'Integration testing',
            'description': 'End-to-end testing and QA',
            'estimated_hours': 24
        })

        return tasks

    def _estimate_effort(self, intent, context) -> str:
        """Oszacuj wysiłek (story points lub czas)"""
        base_effort = len(intent.primary_goals) * 40  # 40h per major goal

        if context.missing_requirements:
            base_effort += len(context.missing_requirements) * 16

        if intent.urgency.value == 'critical':
            base_effort *= 1.2  # Buffer for rush

        weeks = base_effort / 40

        return f"{int(weeks)} weeks ({int(base_effort)} hours)"

    def _estimate_cost(self, effort_string: str) -> float:
        """Oszacuj koszt (USD)"""
        # Extract hours from effort string
        import re
        match = re.search(r'(\d+)\s*hours', effort_string)
        if match:
            hours = int(match.group(1))
            hourly_rate = 100  # $100/hour
            return hours * hourly_rate
        return 0.0
