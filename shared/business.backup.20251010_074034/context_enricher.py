"""Agent Zero V1 - Context Enricher

Wzbogacanie kontekstu o wiedzę domenową
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import asyncio

if TYPE_CHECKING:
    from .intent_extractor import ExtractedIntent


class Severity(Enum):
    """Poziom wagi problemu"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Problem walidacyjny"""
    severity: Severity
    message: str
    field: str = ""
    suggestion: str = ""


@dataclass
class EnrichedContext:
    """Wzbogacony kontekst biznesowy"""

    # Brakujące wymagania
    missing_requirements: List[str] = field(default_factory=list)

    # Sugerowane funkcjonalności
    suggested_features: List[str] = field(default_factory=list)

    # Best practices
    best_practices: List[str] = field(default_factory=list)

    # Problemy walidacyjne
    validation_issues: List[ValidationIssue] = field(default_factory=list)

    # Wzbogacone dane
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)

    # Metadane
    enrichment_confidence: float = 0.0
    original_intent: Optional["ExtractedIntent"] = None


class ContextEnricher:
    """Wzbogacacz kontekstu"""

    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path

        # Baza wiedzy domenowej
        self.domain_patterns = {
            'ecommerce': {
                'required': ['product_catalog', 'shopping_cart', 'payment', 'user_accounts'],
                'suggested': ['reviews', 'wishlist', 'search', 'recommendations'],
                'best_practices': ['SSL/TLS', 'PCI compliance', 'responsive design']
            },
            'api': {
                'required': ['authentication', 'rate_limiting', 'documentation'],
                'suggested': ['versioning', 'caching', 'monitoring'],
                'best_practices': ['RESTful design', 'proper error handling', 'API keys']
            }
        }

    async def enrich_with_domain_knowledge(self, intent: "ExtractedIntent") -> EnrichedContext:
        """Wzbogać kontekst o wiedzę domenową"""
        context = EnrichedContext(original_intent=intent)

        project_type = intent.project_type.value

        # Znajdź matching domain pattern
        domain_key = self._find_domain_match(project_type)

        if domain_key and domain_key in self.domain_patterns:
            pattern = self.domain_patterns[domain_key]

            # Sprawdź brakujące wymagania
            context.missing_requirements = self._find_missing_requirements(
                intent, pattern['required']
            )

            # Dodaj sugestie
            context.suggested_features = pattern['suggested']

            # Dodaj best practices
            context.best_practices = pattern['best_practices']

        # Walidacja
        context.validation_issues = self._validate_intent(intent)

        # Oblicz confidence
        context.enrichment_confidence = self._calculate_enrichment_confidence(context)

        return context

    def _find_domain_match(self, project_type: str) -> Optional[str]:
        """Znajdź matching domain pattern"""
        if 'ecommerce' in project_type or 'shop' in project_type:
            return 'ecommerce'
        if 'api' in project_type:
            return 'api'
        return None

    def _find_missing_requirements(self, intent: "ExtractedIntent", required: List[str]) -> List[str]:
        """Znajdź brakujące wymagania"""
        missing = []

        all_text = ' '.join(intent.primary_goals + intent.secondary_goals).lower()

        for req in required:
            req_keywords = req.replace('_', ' ').split()
            if not any(keyword in all_text for keyword in req_keywords):
                missing.append(req)

        return missing

    def _validate_intent(self, intent: "ExtractedIntent") -> List[ValidationIssue]:
        """Walidacja intencji"""
        issues = []

        if not intent.primary_goals:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                message="Brak głównych celów projektu",
                field="primary_goals",
                suggestion="Określ jasne cele biznesowe"
            ))

        if not intent.stakeholders:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                message="Nie określono stakeholders",
                field="stakeholders",
                suggestion="Wskaż kluczowych interesariuszy"
            ))

        if intent.urgency.value == 'critical' and len(intent.primary_goals) > 5:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                message="Krytyczny deadline z dużym zakresem",
                field="urgency",
                suggestion="Rozważ podział na fazy"
            ))

        return issues

    def _calculate_enrichment_confidence(self, context: EnrichedContext) -> float:
        """Oblicz confidence wzbogacenia"""
        score = 0.0

        if context.suggested_features:
            score += 0.3
        if context.best_practices:
            score += 0.3
        if context.missing_requirements:
            score += 0.2
        if len([i for i in context.validation_issues if i.severity == Severity.ERROR]) == 0:
            score += 0.2

        return min(score, 1.0)
