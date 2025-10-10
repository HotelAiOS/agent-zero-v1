"""Agent Zero V1 - Constraint Analyzer

Analiza ograniczeń i walidacja logiki biznesowej
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from .context_enricher import EnrichedContext, ValidationIssue


@dataclass
class ConstraintAnalysis:
    """Analiza ograniczeń"""

    # Zidentyfikowane ograniczenia
    technical_constraints: List[str] = field(default_factory=list)
    business_constraints: List[str] = field(default_factory=list)
    resource_constraints: List[str] = field(default_factory=list)

    # Ryzyka
    identified_risks: List[Dict[str, Any]] = field(default_factory=list)

    # Problemy walidacyjne
    validation_issues: List["ValidationIssue"] = field(default_factory=list)

    # Metadane
    analysis_confidence: float = 0.0
    feasibility_score: float = 0.0


class ConstraintAnalyzer:
    """Analizator ograniczeń"""

    def __init__(self):
        self.risk_patterns = {
            'scope_creep': ['many features', 'extensive', 'comprehensive'],
            'tight_deadline': ['urgent', 'critical', 'asap'],
            'tech_complexity': ['distributed', 'microservices', 'advanced']
        }

    async def analyze(self, context: "EnrichedContext") -> ConstraintAnalysis:
        """Analiza ograniczeń i ryzyk"""
        analysis = ConstraintAnalysis()

        if context.original_intent:
            intent = context.original_intent

            # Analiza ograniczeń technicznych
            if intent.technology_preferences:
                analysis.technical_constraints = [
                    f"Technology stack: {', '.join(intent.technology_preferences.values())}"
                ]

            # Analiza ograniczeń biznesowych
            if intent.urgency.value in ['critical', 'high']:
                analysis.business_constraints.append(
                    f"Tight deadline: {intent.urgency.value}"
                )

            # Analiza zasobów
            analysis.resource_constraints = self._analyze_resources(intent)

            # Identyfikacja ryzyk
            analysis.identified_risks = self._identify_risks(intent, context)

            # Kopiuj validation issues z context
            analysis.validation_issues = context.validation_issues.copy()

        # Oblicz scores
        analysis.analysis_confidence = 0.8  # Placeholder
        analysis.feasibility_score = self._calculate_feasibility(analysis)

        return analysis

    def _analyze_resources(self, intent) -> List[str]:
        """Analiza wymaganych zasobów"""
        resources = []

        if len(intent.primary_goals) > 3:
            resources.append("Wymaga większego zespołu")

        if intent.technology_preferences:
            resources.append(f"Wymaga ekspertów: {', '.join(intent.technology_preferences.keys())}")

        return resources

    def _identify_risks(self, intent, context) -> List[Dict[str, Any]]:
        """Identyfikacja ryzyk projektu"""
        risks = []

        # Ryzyko scope creep
        if len(intent.primary_goals) > 5:
            risks.append({
                'type': 'scope_creep',
                'severity': 'high',
                'description': 'Duży zakres funkcjonalności',
                'mitigation': 'Podziel na fazy/MVP'
            })

        # Ryzyko deadline
        if intent.urgency.value == 'critical' and len(intent.primary_goals) > 3:
            risks.append({
                'type': 'tight_deadline',
                'severity': 'high',
                'description': 'Krytyczny termin z dużym zakresem',
                'mitigation': 'Zredukuj zakres lub przedłuż termin'
            })

        # Ryzyko techniczne
        if len(context.missing_requirements) > 3:
            risks.append({
                'type': 'incomplete_requirements',
                'severity': 'medium',
                'description': 'Brakujące wymagania',
                'mitigation': 'Uzupełnij specyfikację'
            })

        return risks

    def _calculate_feasibility(self, analysis: ConstraintAnalysis) -> float:
        """Oblicz wykonalność projektu"""
        score = 1.0

        # Odejmij za ryzyka
        high_risk_count = sum(1 for r in analysis.identified_risks if r.get('severity') == 'high')
        score -= (high_risk_count * 0.15)

        # Odejmuj za ograniczenia
        score -= (len(analysis.business_constraints) * 0.05)

        return max(score, 0.0)
