"""Agent Zero V1 - Intent Extractor

Ekstrakcja intencji biznesowych z języka naturalnego
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import re
import asyncio


class ProjectType(Enum):
    """Typ projektu biznesowego"""
    WEB_APP = "web_application"
    MOBILE_APP = "mobile_application"
    API = "api_service"
    ECOMMERCE = "ecommerce"
    DASHBOARD = "dashboard"
    INTEGRATION = "integration"
    DATA_PIPELINE = "data_pipeline"
    AUTOMATION = "automation"
    OTHER = "other"


class Urgency(Enum):
    """Poziom pilności projektu"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExtractedIntent:
    """Wyekstrahowana intencja biznesowa"""

    # Główne cele
    primary_goals: List[str] = field(default_factory=list)
    secondary_goals: List[str] = field(default_factory=list)

    # Klasyfikacja
    project_type: ProjectType = ProjectType.OTHER
    urgency: Urgency = Urgency.NORMAL

    # Stakeholders i wymagania
    stakeholders: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    technology_preferences: Dict[str, str] = field(default_factory=dict)

    # Metadane
    extraction_confidence: float = 0.0
    raw_input: str = ""


class IntentExtractor:
    """Ekstraktor intencji biznesowych"""

    def __init__(self):
        self.keywords_mapping = {
            'create': ['create', 'build', 'develop', 'make', 'implement'],
            'update': ['update', 'modify', 'change', 'enhance', 'improve'],
            'integrate': ['integrate', 'connect', 'link', 'sync'],
            'analyze': ['analyze', 'report', 'dashboard', 'metrics']
        }

        self.project_type_keywords = {
            ProjectType.WEB_APP: ['website', 'web app', 'portal', 'platform'],
            ProjectType.MOBILE_APP: ['mobile', 'ios', 'android', 'app'],
            ProjectType.API: ['api', 'endpoint', 'rest', 'graphql'],
            ProjectType.ECOMMERCE: ['ecommerce', 'shop', 'store', 'cart', 'payment'],
            ProjectType.DASHBOARD: ['dashboard', 'analytics', 'reporting'],
        }

        self.urgency_keywords = {
            Urgency.CRITICAL: ['urgent', 'critical', 'asap', 'immediately'],
            Urgency.HIGH: ['soon', 'quickly', 'priority', 'important'],
            Urgency.NORMAL: ['normal', 'standard'],
        }

    async def extract_complete_intent(self, text: str) -> ExtractedIntent:
        """Ekstrakcja pełnej intencji z tekstu"""
        intent = ExtractedIntent(raw_input=text)

        text_lower = text.lower()

        # Ekstrakcja celów
        intent.primary_goals = self._extract_goals(text)

        # Określenie typu projektu
        intent.project_type = self._detect_project_type(text_lower)

        # Określenie pilności
        intent.urgency = self._detect_urgency(text_lower)

        # Ekstrakcja stakeholders
        intent.stakeholders = self._extract_stakeholders(text)

        # Ekstrakcja technologii
        intent.technology_preferences = self._extract_tech_preferences(text)

        # Kryteria sukcesu
        intent.success_criteria = self._extract_success_criteria(text)

        # Oblicz confidence
        intent.extraction_confidence = self._calculate_confidence(intent)

        return intent

    def _extract_goals(self, text: str) -> List[str]:
        """Ekstrakcja głównych celów"""
        goals = []

        # Proste zdania z czasownikami akcji
        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for action_type, keywords in self.keywords_mapping.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    if len(sentence.strip()) > 10:
                        goals.append(sentence.strip())
                        break

        return goals[:5]  # Max 5 głównych celów

    def _detect_project_type(self, text: str) -> ProjectType:
        """Wykrycie typu projektu"""
        scores = {}

        for proj_type, keywords in self.project_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[proj_type] = score

        if scores:
            return max(scores, key=scores.get)
        return ProjectType.OTHER

    def _detect_urgency(self, text: str) -> Urgency:
        """Wykrycie poziomu pilności"""
        for urgency, keywords in self.urgency_keywords.items():
            if any(keyword in text for keyword in keywords):
                return urgency
        return Urgency.NORMAL

    def _extract_stakeholders(self, text: str) -> List[str]:
        """Ekstrakcja stakeholders"""
        stakeholders = []

        patterns = [
            r'(CEO|CTO|CFO|manager|director|team|department|customer|user)',
            r'(stakeholder|client|partner|vendor)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stakeholders.extend(matches)

        return list(set(stakeholders))[:5]

    def _extract_tech_preferences(self, text: str) -> Dict[str, str]:
        """Ekstrakcja preferencji technologicznych"""
        tech_map = {
            'frontend': ['react', 'vue', 'angular', 'svelte'],
            'backend': ['python', 'node', 'java', 'go', 'ruby'],
            'database': ['postgresql', 'mysql', 'mongodb', 'redis'],
            'cloud': ['aws', 'azure', 'gcp', 'heroku']
        }

        preferences = {}
        text_lower = text.lower()

        for category, technologies in tech_map.items():
            for tech in technologies:
                if tech in text_lower:
                    preferences[category] = tech
                    break

        return preferences

    def _extract_success_criteria(self, text: str) -> List[str]:
        """Ekstrakcja kryteriów sukcesu"""
        criteria = []

        # Szukaj fraz z metrykami
        metric_patterns = [
            r'measured by ([^.]+)',
            r'success.*?([^.]+rate|[^.]+value|[^.]+time)',
            r'kpi.*?([^.]+)'
        ]

        for pattern in metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend(matches)

        return criteria[:3]

    def _calculate_confidence(self, intent: ExtractedIntent) -> float:
        """Oblicz confidence score"""
        score = 0.0

        if intent.primary_goals:
            score += 0.3
        if intent.project_type != ProjectType.OTHER:
            score += 0.2
        if intent.stakeholders:
            score += 0.2
        if intent.technology_preferences:
            score += 0.15
        if intent.success_criteria:
            score += 0.15

        return min(score, 1.0)
