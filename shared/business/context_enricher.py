"""Agent Zero V1 - Business Context Enrichment Engine

🔧 Context Enricher - Domain Knowledge Integration
==============================================

Developer A Implementation - Week 42 (9 października 2025)  
Priority 2: Context Enrichment Engine

Capabilities:
- Enrich business context with domain expertise
- Suggest missing requirements based on project type
- Validate business logic consistency
- Add industry best practices and patterns
- Cross-reference with business glossary

Integration: Business Requirements Parser Pipeline
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Agent Zero imports
try:
    from .intent_extractor import ExtractedIntent, ProjectDomain, UrgencyLevel
    from shared.knowledge.domain_knowledge import DomainKnowledgeBase
    from shared.models.business_models import EnrichedContext, BusinessGlossary
except ImportError:
    logging.warning("Some dependencies not available - using fallback implementations")


class EnrichmentLevel(Enum):
    """Context enrichment depth levels"""
    BASIC = "basic"           # Essential missing requirements only
    STANDARD = "standard"     # Standard + best practices
    COMPREHENSIVE = "comprehensive"  # Full domain expertise
    EXPERT = "expert"         # Industry-specific advanced patterns


class ValidationSeverity(Enum):
    """Business logic validation severity"""
    ERROR = "error"           # Critical logic conflicts
    WARNING = "warning"       # Potential issues
    INFO = "info"            # Suggestions for improvement
    OPTIMIZATION = "optimization"  # Performance/efficiency hints


@dataclass
class ValidationIssue:
    """Business logic validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    confidence: float = 0.0
    line_reference: Optional[str] = None


@dataclass
class EnrichedContext:
    """Enriched business context with domain knowledge"""
    
    # Original intent (preserved)
    original_intent: Optional[ExtractedIntent] = None
    
    # Enriched information
    domain_expertise: Dict[str, Any] = field(default_factory=dict)
    missing_requirements: List[str] = field(default_factory=list)
    suggested_features: List[Dict[str, Any]] = field(default_factory=list)
    
    # Industry best practices
    best_practices: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    security_considerations: List[str] = field(default_factory=list)
    
    # Validation results
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    consistency_score: float = 0.0
    
    # Enhanced metadata
    similar_projects: List[Dict[str, Any]] = field(default_factory=list)
    complexity_assessment: str = "medium"
    risk_factors: List[str] = field(default_factory=list)
    
    # Business glossary terms
    domain_terms: Dict[str, str] = field(default_factory=dict)
    
    # Enrichment metadata
    enrichment_level: EnrichmentLevel = EnrichmentLevel.STANDARD
    enrichment_confidence: float = 0.0
    processing_time: float = 0.0


class ContextEnricher:
    """Business Context Enrichment Engine
    
    Enhances parsed business requirements with domain knowledge,
    industry best practices, and validation insights.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize domain knowledge
        self.domain_knowledge = self._load_domain_knowledge(knowledge_base_path)
        self.business_glossary = self._load_business_glossary()
        
        # Domain-specific requirement templates
        self.requirement_templates = self._initialize_requirement_templates()
        
        # Best practices database
        self.best_practices = self._initialize_best_practices()
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
    
    def _load_domain_knowledge(self, knowledge_path: Optional[str]) -> Dict[str, Any]:
        """Load domain knowledge base"""
        try:
            if knowledge_path and Path(knowledge_path).exists():
                with open(knowledge_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load domain knowledge: {e}")
        
        # Fallback to built-in knowledge
        return self._create_default_domain_knowledge()
    
    def _create_default_domain_knowledge(self) -> Dict[str, Any]:
        """Create default domain knowledge base"""
        return {
            ProjectDomain.ECOMMERCE.value: {
                "essential_features": [
                    "Product catalog management",
                    "Shopping cart functionality", 
                    "Secure payment processing",
                    "User authentication",
                    "Order management system",
                    "Inventory tracking"
                ],
                "optional_features": [
                    "Product recommendations",
                    "Customer reviews and ratings",
                    "Wishlist functionality",
                    "Multi-currency support",
                    "Loyalty program",
                    "Advanced search and filtering"
                ],
                "technical_requirements": [
                    "SSL/TLS encryption",
                    "PCI DSS compliance",
                    "Database backup strategy",
                    "Performance monitoring",
                    "Mobile responsiveness"
                ],
                "integrations": [
                    "Payment gateways (Stripe, PayPal)",
                    "Shipping providers",
                    "Email marketing platforms",
                    "Analytics tools",
                    "Customer support systems"
                ]
            },
            ProjectDomain.FINTECH.value: {
                "essential_features": [
                    "User identity verification (KYC)",
                    "Transaction processing",
                    "Account management",
                    "Security authentication (2FA)",
                    "Audit logging",
                    "Regulatory compliance"
                ],
                "compliance_requirements": [
                    "PCI DSS compliance",
                    "SOX compliance",
                    "GDPR compliance",
                    "AML procedures",
                    "Data encryption"
                ],
                "security_measures": [
                    "End-to-end encryption",
                    "Fraud detection",
                    "Rate limiting",
                    "Session management",
                    "Penetration testing"
                ]
            },
            ProjectDomain.ENTERPRISE.value: {
                "essential_features": [
                    "User role management",
                    "Workflow automation",
                    "Reporting and analytics",
                    "Data export/import",
                    "System integration capabilities"
                ],
                "scalability_requirements": [
                    "Load balancing",
                    "Database clustering",
                    "Caching strategy",
                    "Horizontal scaling",
                    "Performance optimization"
                ]
            }
        }
    
    def _load_business_glossary(self) -> Dict[str, str]:
        """Load business terminology glossary"""
        return {
            # E-commerce terms
            "conversion_rate": "Percentage of visitors who complete a purchase",
            "cart_abandonment": "When users add items to cart but don't complete purchase",
            "clv": "Customer Lifetime Value - total revenue from a customer",
            "aov": "Average Order Value - mean transaction amount",
            
            # Technical terms
            "api": "Application Programming Interface for system communication",
            "microservices": "Architectural approach using small, independent services",
            "scalability": "System's ability to handle increased load",
            "sla": "Service Level Agreement defining performance standards",
            
            # Business terms
            "mvp": "Minimum Viable Product - basic version for initial launch",
            "kpi": "Key Performance Indicator - metric measuring success",
            "roi": "Return on Investment - profitability measure",
            "stakeholder": "Person or group with interest in project outcome"
        }
    
    def _initialize_requirement_templates(self) -> Dict[ProjectDomain, Dict[str, List[str]]]:
        """Initialize domain-specific requirement templates"""
        return {
            ProjectDomain.ECOMMERCE: {
                "security": [
                    "Implement secure payment processing with PCI DSS compliance",
                    "Add user authentication with password strength requirements",
                    "Include data encryption for sensitive customer information",
                    "Set up fraud detection and prevention measures"
                ],
                "user_experience": [
                    "Design mobile-responsive interface for all devices",
                    "Implement intuitive product search and filtering",
                    "Add product recommendations based on user behavior",
                    "Include customer review and rating system"
                ],
                "business_logic": [
                    "Set up inventory management with low-stock alerts",
                    "Implement order tracking and status updates",
                    "Add support for discount codes and promotions",
                    "Include abandoned cart recovery system"
                ]
            },
            ProjectDomain.API_SERVICE: {
                "architecture": [
                    "Design RESTful API with proper HTTP methods",
                    "Implement API versioning strategy",
                    "Add comprehensive API documentation",
                    "Include rate limiting and throttling"
                ],
                "monitoring": [
                    "Set up API monitoring and alerting",
                    "Implement logging for all API requests",
                    "Add performance metrics and analytics",
                    "Include health check endpoints"
                ]
            }
        }
    
    def _initialize_best_practices(self) -> Dict[str, List[str]]:
        """Initialize best practices database"""
        return {
            "security": [
                "Implement principle of least privilege",
                "Use HTTPS/TLS for all communications",
                "Regular security audits and penetration testing",
                "Keep all dependencies updated",
                "Implement proper error handling without exposing internals"
            ],
            "performance": [
                "Implement caching at multiple layers",
                "Optimize database queries and indexes",
                "Use CDN for static assets",
                "Implement proper connection pooling",
                "Monitor and optimize critical paths"
            ],
            "maintainability": [
                "Follow consistent coding standards",
                "Write comprehensive unit tests",
                "Implement CI/CD pipelines",
                "Document API endpoints and business logic",
                "Use version control with meaningful commit messages"
            ],
            "scalability": [
                "Design stateless services",
                "Implement horizontal scaling capabilities",
                "Use message queues for async processing",
                "Plan for database partitioning",
                "Implement proper load balancing"
            ]
        }
    
    def _initialize_validation_rules(self) -> List[Dict[str, Any]]:
        """Initialize business logic validation rules"""
        return [
            {
                "rule": "missing_payment_method",
                "condition": lambda intent: (
                    intent.project_type == ProjectDomain.ECOMMERCE and
                    not any("payment" in goal.lower() for goal in intent.primary_goals + intent.secondary_goals)
                ),
                "severity": ValidationSeverity.ERROR,
                "message": "E-commerce project missing payment processing requirements",
                "suggestion": "Add secure payment processing with multiple payment methods"
            },
            {
                "rule": "missing_authentication",
                "condition": lambda intent: (
                    any(term in intent.project_type.value for term in ["enterprise", "fintech"]) and
                    not any("auth" in goal.lower() or "login" in goal.lower() for goal in intent.primary_goals + intent.secondary_goals)
                ),
                "severity": ValidationSeverity.WARNING,
                "message": "Enterprise/Fintech project should include user authentication",
                "suggestion": "Consider adding multi-factor authentication for enhanced security"
            },
            {
                "rule": "unrealistic_timeline",
                "condition": lambda intent: (
                    intent.urgency == UrgencyLevel.CRITICAL and
                    len(intent.primary_goals) > 3
                ),
                "severity": ValidationSeverity.WARNING,
                "message": "Critical timeline with extensive feature set may be unrealistic",
                "suggestion": "Consider MVP approach focusing on core features first"
            }
        ]
    
    async def enrich_with_domain_knowledge(self, parsed_requirements: ExtractedIntent) -> EnrichedContext:
        """Enrich parsed requirements with domain-specific knowledge
        
        Args:
            parsed_requirements: Intent extracted from business requirements
            
        Returns:
            EnrichedContext with domain knowledge and suggestions
        """
        self.logger.info(f"Enriching context for {parsed_requirements.project_type.value} project")
        
        import time
        start_time = time.time()
        
        # Get domain-specific knowledge
        domain_info = self.domain_knowledge.get(
            parsed_requirements.project_type.value, 
            {}
        )
        
        # Create enriched context
        enriched = EnrichedContext(
            original_intent=parsed_requirements,
            domain_expertise=domain_info,
            enrichment_level=EnrichmentLevel.STANDARD
        )
        
        # Add missing requirements
        enriched.missing_requirements = await self._identify_missing_requirements(
            parsed_requirements, domain_info
        )
        
        # Suggest additional features
        enriched.suggested_features = await self._suggest_features(
            parsed_requirements, domain_info
        )
        
        # Add best practices
        enriched.best_practices = self._get_relevant_best_practices(
            parsed_requirements.project_type
        )
        
        # Add domain terminology
        enriched.domain_terms = self._extract_relevant_terms(
            parsed_requirements
        )
        
        # Calculate enrichment confidence
        enriched.enrichment_confidence = self._calculate_enrichment_confidence(
            parsed_requirements, domain_info
        )
        
        enriched.processing_time = time.time() - start_time
        
        return enriched
    
    async def _identify_missing_requirements(self, intent: ExtractedIntent, domain_info: Dict[str, Any]) -> List[str]:
        """Identify missing essential requirements"""
        missing = []
        
        # Get essential features for this domain
        essential_features = domain_info.get("essential_features", [])
        
        # Check which essential features are missing
        current_goals_text = " ".join(intent.primary_goals + intent.secondary_goals).lower()
        
        for feature in essential_features:
            # Simple keyword matching for missing features
            feature_keywords = feature.lower().split()
            if not any(keyword in current_goals_text for keyword in feature_keywords[:2]):
                missing.append(feature)
        
        return missing[:5]  # Limit to top 5 most critical
    
    async def _suggest_features(self, intent: ExtractedIntent, domain_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest additional features based on domain knowledge"""
        suggestions = []
        
        optional_features = domain_info.get("optional_features", [])
        current_goals_text = " ".join(intent.primary_goals + intent.secondary_goals).lower()
        
        for feature in optional_features:
            feature_keywords = feature.lower().split()
            if not any(keyword in current_goals_text for keyword in feature_keywords[:2]):
                suggestions.append({
                    "feature": feature,
                    "priority": "medium",
                    "business_value": self._assess_business_value(feature, intent),
                    "implementation_effort": self._assess_implementation_effort(feature)
                })
        
        # Sort by business value
        suggestions.sort(key=lambda x: x["business_value"], reverse=True)
        return suggestions[:5]
    
    def _assess_business_value(self, feature: str, intent: ExtractedIntent) -> float:
        """Assess business value of a suggested feature"""
        # Simple heuristic based on project type and urgency
        base_value = 0.5
        
        # Higher value for e-commerce revenue-driving features
        if intent.project_type == ProjectDomain.ECOMMERCE:
            if any(term in feature.lower() for term in ["recommendation", "personalization", "conversion"]):
                base_value += 0.3
        
        # Adjust for urgency
        if intent.urgency == UrgencyLevel.HIGH:
            base_value += 0.1
        elif intent.urgency == UrgencyLevel.CRITICAL:
            base_value += 0.2
        
        return min(base_value, 1.0)
    
    def _assess_implementation_effort(self, feature: str) -> str:
        """Assess implementation effort for a feature"""
        complex_features = ["recommendation", "machine learning", "ai", "analytics"]
        medium_features = ["search", "filtering", "notification", "integration"]
        
        feature_lower = feature.lower()
        
        if any(term in feature_lower for term in complex_features):
            return "high"
        elif any(term in feature_lower for term in medium_features):
            return "medium"
        else:
            return "low"
    
    def _get_relevant_best_practices(self, project_type: ProjectDomain) -> List[str]:
        """Get relevant best practices for project type"""
        practices = []
        
        # Always include security for sensitive domains
        if project_type in [ProjectDomain.ECOMMERCE, ProjectDomain.FINTECH, ProjectDomain.ENTERPRISE]:
            practices.extend(self.best_practices["security"][:3])
        
        # Always include performance and maintainability
        practices.extend(self.best_practices["performance"][:2])
        practices.extend(self.best_practices["maintainability"][:2])
        
        return practices
    
    def _extract_relevant_terms(self, intent: ExtractedIntent) -> Dict[str, str]:
        """Extract relevant business glossary terms"""
        relevant_terms = {}
        
        # Extract terms mentioned in goals
        all_text = " ".join(intent.primary_goals + intent.secondary_goals).lower()
        
        for term, definition in self.business_glossary.items():
            if term in all_text:
                relevant_terms[term] = definition
        
        return relevant_terms
    
    def _calculate_enrichment_confidence(self, intent: ExtractedIntent, domain_info: Dict[str, Any]) -> float:
        """Calculate confidence in enrichment quality"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for known domains
        if domain_info:
            confidence += 0.3
        
        # Higher confidence for detailed intents
        if len(intent.primary_goals) >= 2:
            confidence += 0.1
        
        if intent.stakeholders:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def suggest_missing_requirements(self, context: EnrichedContext) -> List[str]:
        """Suggest missing requirements based on enriched context
        
        Args:
            context: Enriched business context
            
        Returns:
            List of suggested missing requirements
        """
        suggestions = []
        
        # Add missing requirements from domain knowledge
        suggestions.extend(context.missing_requirements)
        
        # Add compliance requirements for specific domains
        if context.original_intent and context.original_intent.project_type == ProjectDomain.FINTECH:
            compliance_reqs = context.domain_expertise.get("compliance_requirements", [])
            suggestions.extend(compliance_reqs)
        
        # Add security requirements for sensitive domains
        if context.original_intent and context.original_intent.project_type in [
            ProjectDomain.ECOMMERCE, ProjectDomain.FINTECH, ProjectDomain.ENTERPRISE
        ]:
            suggestions.append("Implement comprehensive security audit logging")
            suggestions.append("Add data backup and disaster recovery procedures")
        
        return list(set(suggestions))  # Remove duplicates
    
    async def validate_business_logic(self, requirements: EnrichedContext) -> List[ValidationIssue]:
        """Validate business logic consistency and completeness
        
        Args:
            requirements: Enriched business requirements context
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        if not requirements.original_intent:
            return issues
        
        # Run validation rules
        for rule in self.validation_rules:
            try:
                if rule["condition"](requirements.original_intent):
                    issue = ValidationIssue(
                        severity=rule["severity"],
                        category=rule["rule"],
                        message=rule["message"],
                        suggestion=rule.get("suggestion"),
                        confidence=0.8  # Rule-based validation has high confidence
                    )
                    issues.append(issue)
            except Exception as e:
                self.logger.warning(f"Validation rule {rule['rule']} failed: {e}")
        
        # Additional custom validations
        issues.extend(await self._validate_technical_feasibility(requirements))
        issues.extend(await self._validate_timeline_realism(requirements))
        
        return issues
    
    async def _validate_technical_feasibility(self, context: EnrichedContext) -> List[ValidationIssue]:
        """Validate technical feasibility of requirements"""
        issues = []
        
        if not context.original_intent:
            return issues
        
        # Check for conflicting technology preferences
        tech_prefs = context.original_intent.technology_preferences
        if "php" in tech_prefs and "node.js" in tech_prefs:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical_conflict",
                message="Multiple backend technologies specified (PHP and Node.js)",
                suggestion="Choose primary backend technology for consistency",
                confidence=0.9
            ))
        
        return issues
    
    async def _validate_timeline_realism(self, context: EnrichedContext) -> List[ValidationIssue]:
        """Validate timeline realism based on scope"""
        issues = []
        
        if not context.original_intent:
            return issues
        
        intent = context.original_intent
        total_features = len(intent.primary_goals) + len(intent.secondary_goals) + len(context.missing_requirements)
        
        # Warn about unrealistic timelines
        if intent.urgency == UrgencyLevel.CRITICAL and total_features > 5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="timeline_risk",
                message=f"Critical timeline with {total_features} features may be unrealistic",
                suggestion="Consider phased delivery approach with MVP first",
                confidence=0.7
            ))
        
        return issues


# Development and testing utilities
async def test_context_enricher():
    """Test the context enricher with sample data"""
    from .intent_extractor import IntentExtractor
    
    # Create test intent
    extractor = IntentExtractor()
    sample_text = """
    Build an e-commerce platform for selling electronics.
    Users should browse products and make purchases.
    The CEO needs this urgently for Q4 launch.
    """
    
    intent = await extractor.extract_complete_intent(sample_text)
    
    # Enrich the context
    enricher = ContextEnricher()
    enriched = await enricher.enrich_with_domain_knowledge(intent)
    
    print("\n🔧 Context Enrichment Results:")
    print(f"Missing Requirements: {len(enriched.missing_requirements)}")
    print(f"Suggested Features: {len(enriched.suggested_features)}")
    print(f"Best Practices: {len(enriched.best_practices)}")
    print(f"Domain Terms: {len(enriched.domain_terms)}")
    print(f"Enrichment Confidence: {enriched.enrichment_confidence:.2f}")
    
    # Validate business logic
    validation_issues = await enricher.validate_business_logic(enriched)
    print(f"Validation Issues: {len(validation_issues)}")
    for issue in validation_issues:
        print(f"  - {issue.severity.value}: {issue.message}")


if __name__ == "__main__":
    asyncio.run(test_context_enricher())
