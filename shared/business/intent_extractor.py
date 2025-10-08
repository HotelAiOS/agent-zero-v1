"""Agent Zero V1 - Business Intent Extraction Engine

🎯 Intent Extractor - Core Business Intelligence Component
======================================================

Developer A Implementation - Week 42 (9 października 2025)
Priority 1: Business Requirements Parser - Intent Extraction

Capabilities:
- Extract business goals from natural language
- Identify key stakeholders and roles
- Classify project types and domains
- Parse success criteria and KPIs
- Detect urgency and priority levels

Integration: Multi-agent project orchestration enhancement
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Agent Zero core imports
try:
    from shared.llm.ollama_client import OllamaClient
    from shared.models.business_models import BusinessIntent, Stakeholder, ProjectType
except ImportError:
    # Fallback for development
    logging.warning("Core dependencies not available - using mock implementations")


class ProjectDomain(Enum):
    """Business project domain classification"""
    ECOMMERCE = "e-commerce"
    FINTECH = "financial_technology"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERPRISE = "enterprise_software"
    CONSUMER_APP = "consumer_application"
    B2B_SAAS = "b2b_saas"
    API_SERVICE = "api_service"
    DATA_ANALYTICS = "data_analytics"
    AI_ML = "artificial_intelligence"
    UNKNOWN = "unknown"


class UrgencyLevel(Enum):
    """Project urgency classification"""
    CRITICAL = "critical"  # Emergency, immediate action
    HIGH = "high"         # Important, within days
    MEDIUM = "medium"     # Standard, within weeks
    LOW = "low"          # Nice to have, flexible timing


@dataclass
class ExtractedIntent:
    """Structured business intent extraction result"""
    
    # Core business goals
    primary_goals: List[str] = field(default_factory=list)
    secondary_goals: List[str] = field(default_factory=list)
    
    # Stakeholder information
    stakeholders: List[Dict[str, Any]] = field(default_factory=list)
    decision_makers: List[str] = field(default_factory=list)
    end_users: List[str] = field(default_factory=list)
    
    # Project classification
    project_type: ProjectDomain = ProjectDomain.UNKNOWN
    project_scale: str = "medium"  # small, medium, large, enterprise
    
    # Success criteria
    success_metrics: List[str] = field(default_factory=list)
    kpis: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timeline and urgency
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    timeline_hints: List[str] = field(default_factory=list)
    
    # Technical hints
    technology_preferences: List[str] = field(default_factory=list)
    integration_requirements: List[str] = field(default_factory=list)
    
    # Confidence scores
    extraction_confidence: float = 0.0
    classification_confidence: float = 0.0


class IntentExtractor:
    """Business Intent Extraction Engine
    
    Extracts structured business intelligence from natural language requirements.
    Core component of V2.0 Intelligence Layer for multi-agent orchestration.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client or self._initialize_llm()
        
        # Business domain patterns
        self.domain_patterns = self._initialize_domain_patterns()
        
        # Stakeholder identification patterns
        self.stakeholder_patterns = self._initialize_stakeholder_patterns()
        
        # Urgency detection patterns
        self.urgency_patterns = self._initialize_urgency_patterns()
    
    def _initialize_llm(self) -> Optional[Any]:
        """Initialize LLM client for advanced intent extraction"""
        try:
            return OllamaClient(model="llama3.2:3b")
        except Exception as e:
            self.logger.warning(f"LLM client initialization failed: {e}")
            return None
    
    def _initialize_domain_patterns(self) -> Dict[ProjectDomain, List[str]]:
        """Initialize project domain classification patterns"""
        return {
            ProjectDomain.ECOMMERCE: [
                r"\b(shop|store|ecommerce|e-commerce|marketplace|cart|checkout|payment)\b",
                r"\b(product catalog|inventory|order management|shipping)\b",
                r"\b(customer|buyer|seller|merchant|transaction)\b"
            ],
            ProjectDomain.FINTECH: [
                r"\b(financial|fintech|banking|payment|wallet|trading)\b",
                r"\b(investment|loan|credit|debit|crypto|blockchain)\b",
                r"\b(compliance|regulation|kyc|aml|security)\b"
            ],
            ProjectDomain.ENTERPRISE: [
                r"\b(enterprise|corporate|business|crm|erp|hr)\b",
                r"\b(workflow|approval|dashboard|reporting|analytics)\b",
                r"\b(employee|department|organization|management)\b"
            ],
            ProjectDomain.API_SERVICE: [
                r"\b(api|service|endpoint|microservice|integration)\b",
                r"\b(rest|graphql|webhook|authentication|authorization)\b",
                r"\b(data|sync|communication|third.?party)\b"
            ],
            ProjectDomain.AI_ML: [
                r"\b(ai|artificial intelligence|machine learning|ml|neural)\b",
                r"\b(model|algorithm|prediction|classification|nlp)\b",
                r"\b(training|inference|optimization|automation)\b"
            ]
        }
    
    def _initialize_stakeholder_patterns(self) -> Dict[str, List[str]]:
        """Initialize stakeholder identification patterns"""
        return {
            "decision_makers": [
                r"\b(ceo|cto|manager|director|lead|head|owner|founder)\b",
                r"\b(decision maker|stakeholder|sponsor|client)\b"
            ],
            "end_users": [
                r"\b(user|customer|client|consumer|visitor|member)\b",
                r"\b(employee|staff|team member|operator)\b"
            ],
            "technical_team": [
                r"\b(developer|engineer|architect|devops|qa|tester)\b",
                r"\b(programmer|coder|tech team|development team)\b"
            ]
        }
    
    def _initialize_urgency_patterns(self) -> Dict[UrgencyLevel, List[str]]:
        """Initialize urgency level detection patterns"""
        return {
            UrgencyLevel.CRITICAL: [
                r"\b(urgent|critical|emergency|asap|immediately|now)\b",
                r"\b(deadline|crisis|broken|down|failing)\b"
            ],
            UrgencyLevel.HIGH: [
                r"\b(important|priority|soon|quickly|fast)\b",
                r"\b(this week|few days|urgent need)\b"
            ],
            UrgencyLevel.MEDIUM: [
                r"\b(standard|normal|regular|planned|scheduled)\b",
                r"\b(next month|few weeks|standard timeline)\b"
            ],
            UrgencyLevel.LOW: [
                r"\b(future|eventually|when possible|low priority)\b",
                r"\b(nice to have|optional|flexible|sometime)\b"
            ]
        }
    
    async def extract_business_goals(self, natural_language_input: str) -> List[str]:
        """Extract primary and secondary business goals
        
        Args:
            natural_language_input: Raw business requirements text
            
        Returns:
            List of extracted business goals
        """
        try:
            # Enhanced goal extraction with LLM if available
            if self.llm_client:
                return await self._extract_goals_with_llm(natural_language_input)
            
            # Fallback to pattern-based extraction
            return self._extract_goals_with_patterns(natural_language_input)
            
        except Exception as e:
            self.logger.error(f"Goal extraction failed: {e}")
            return []
    
    async def _extract_goals_with_llm(self, text: str) -> List[str]:
        """Extract goals using LLM for enhanced understanding"""
        prompt = f"""
        Extract the main business goals from this requirement:
        
        {text}
        
        Return only the core business objectives, one per line.
        Focus on WHAT the business wants to achieve, not HOW.
        """
        
        try:
            response = await self.llm_client.generate_async(prompt)
            goals = [line.strip() for line in response.split('\n') if line.strip()]
            return goals[:5]  # Limit to top 5 goals
        except Exception as e:
            self.logger.warning(f"LLM goal extraction failed: {e}")
            return self._extract_goals_with_patterns(text)
    
    def _extract_goals_with_patterns(self, text: str) -> List[str]:
        """Extract goals using pattern matching (fallback method)"""
        goal_patterns = [
            r"\b(?:want to|need to|should|must|goal is to|objective is to|aim to)\s+([^.!?]+)",
            r"\b(?:requirement|feature|functionality)\s*:?\s*([^.!?]+)",
            r"\b(?:user should be able to|system should|application should)\s+([^.!?]+)"
        ]
        
        goals = []
        text_lower = text.lower()
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            goals.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return list(set(goals))[:5]  # Remove duplicates, limit to 5
    
    async def identify_stakeholders(self, requirements_text: str) -> List[Dict[str, Any]]:
        """Identify and classify stakeholders from requirements
        
        Args:
            requirements_text: Business requirements text
            
        Returns:
            List of identified stakeholders with roles and attributes
        """
        stakeholders = []
        text_lower = requirements_text.lower()
        
        for role_type, patterns in self.stakeholder_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    stakeholder = {
                        "type": role_type,
                        "title": match,
                        "influence": self._assess_stakeholder_influence(role_type),
                        "involvement": self._assess_stakeholder_involvement(match)
                    }
                    stakeholders.append(stakeholder)
        
        return stakeholders
    
    def _assess_stakeholder_influence(self, role_type: str) -> str:
        """Assess stakeholder influence level"""
        influence_map = {
            "decision_makers": "high",
            "end_users": "medium", 
            "technical_team": "medium"
        }
        return influence_map.get(role_type, "low")
    
    def _assess_stakeholder_involvement(self, title: str) -> str:
        """Assess stakeholder involvement level"""
        high_involvement = ["owner", "manager", "lead", "user", "customer"]
        if any(term in title.lower() for term in high_involvement):
            return "high"
        return "medium"
    
    async def classify_project_type(self, description: str) -> Tuple[ProjectDomain, float]:
        """Classify project type and domain
        
        Args:
            description: Project description text
            
        Returns:
            Tuple of (project_domain, confidence_score)
        """
        description_lower = description.lower()
        domain_scores = {}
        
        # Score each domain based on pattern matches
        for domain, patterns in self.domain_patterns.items():
            score = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                matches = len(re.findall(pattern, description_lower))
                score += matches
            
            # Normalize score
            domain_scores[domain] = score / total_patterns if total_patterns > 0 else 0
        
        # Find best match
        if not domain_scores or max(domain_scores.values()) == 0:
            return ProjectDomain.UNKNOWN, 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = min(domain_scores[best_domain], 1.0)
        
        return best_domain, confidence
    
    def _detect_urgency(self, text: str) -> Tuple[UrgencyLevel, float]:
        """Detect urgency level from text
        
        Args:
            text: Requirements text
            
        Returns:
            Tuple of (urgency_level, confidence_score)
        """
        text_lower = text.lower()
        urgency_scores = {}
        
        for urgency, patterns in self.urgency_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            urgency_scores[urgency] = score
        
        if not urgency_scores or max(urgency_scores.values()) == 0:
            return UrgencyLevel.MEDIUM, 0.5
        
        best_urgency = max(urgency_scores, key=urgency_scores.get)
        confidence = min(urgency_scores[best_urgency] / 3.0, 1.0)  # Normalize
        
        return best_urgency, confidence
    
    async def extract_complete_intent(self, natural_language_input: str) -> ExtractedIntent:
        """Complete intent extraction pipeline
        
        Args:
            natural_language_input: Raw business requirements
            
        Returns:
            Structured ExtractedIntent object
        """
        self.logger.info("Starting complete intent extraction")
        
        try:
            # Extract business goals
            goals = await self.extract_business_goals(natural_language_input)
            primary_goals = goals[:2] if goals else []
            secondary_goals = goals[2:] if len(goals) > 2 else []
            
            # Identify stakeholders
            stakeholders = await self.identify_stakeholders(natural_language_input)
            
            # Classify project
            project_type, type_confidence = await self.classify_project_type(natural_language_input)
            
            # Detect urgency
            urgency, urgency_confidence = self._detect_urgency(natural_language_input)
            
            # Extract technical hints
            tech_preferences = self._extract_technical_preferences(natural_language_input)
            integrations = self._extract_integration_requirements(natural_language_input)
            
            # Calculate overall confidence
            extraction_confidence = (type_confidence + urgency_confidence) / 2
            
            return ExtractedIntent(
                primary_goals=primary_goals,
                secondary_goals=secondary_goals,
                stakeholders=stakeholders,
                decision_makers=[s["title"] for s in stakeholders if s["type"] == "decision_makers"],
                end_users=[s["title"] for s in stakeholders if s["type"] == "end_users"],
                project_type=project_type,
                urgency=urgency,
                technology_preferences=tech_preferences,
                integration_requirements=integrations,
                extraction_confidence=extraction_confidence,
                classification_confidence=type_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Complete intent extraction failed: {e}")
            return ExtractedIntent()  # Return empty intent
    
    def _extract_technical_preferences(self, text: str) -> List[str]:
        """Extract technology preferences from requirements"""
        tech_patterns = [
            r"\b(python|java|javascript|react|vue|angular|node\.?js)\b",
            r"\b(postgresql|mysql|mongodb|redis|elasticsearch)\b",
            r"\b(aws|azure|gcp|docker|kubernetes|microservices)\b",
            r"\b(rest api|graphql|websocket|grpc)\b"
        ]
        
        preferences = []
        text_lower = text.lower()
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            preferences.extend(matches)
        
        return list(set(preferences))
    
    def _extract_integration_requirements(self, text: str) -> List[str]:
        """Extract integration requirements"""
        integration_patterns = [
            r"\bintegrat(?:e|ion) with ([a-zA-Z0-9\s]+?)(?:\s|\.|,|$)",
            r"\bconnect to ([a-zA-Z0-9\s]+?)(?:\s|\.|,|$)",
            r"\buse ([a-zA-Z0-9\s]+?) (?:api|service)\b"
        ]
        
        integrations = []
        
        for pattern in integration_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            integrations.extend([match.strip() for match in matches])
        
        return list(set(integrations))[:5]  # Limit to 5 most common


# Development and testing utilities
async def test_intent_extractor():
    """Test the intent extractor with sample business requirements"""
    extractor = IntentExtractor()
    
    sample_requirements = """
    We need to build an e-commerce platform for our startup. 
    The CEO wants customers to be able to browse products, add them to cart, and checkout securely.
    This is urgent as we have a deadline next month for investor demo.
    The system should integrate with Stripe for payments and use React for the frontend.
    Success will be measured by conversion rate and user engagement.
    """
    
    intent = await extractor.extract_complete_intent(sample_requirements)
    
    print("\n🎯 Intent Extraction Results:")
    print(f"Primary Goals: {intent.primary_goals}")
    print(f"Project Type: {intent.project_type.value}")
    print(f"Urgency: {intent.urgency.value}")
    print(f"Stakeholders: {len(intent.stakeholders)}")
    print(f"Tech Preferences: {intent.technology_preferences}")
    print(f"Confidence: {intent.extraction_confidence:.2f}")


if __name__ == "__main__":
    asyncio.run(test_intent_extractor())
