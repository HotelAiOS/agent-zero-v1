"""
Business Requirements Parser - Complete Implementation
Natural language to technical specifications converter with validation
Version: 1.0.0 - Production Ready
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# === PYDANTIC MODELS ===
class BusinessRequest(BaseModel):
    """Input model for business requirement parsing"""
    request: str
    context: Optional[Dict[str, Any]] = {}
    priority: Optional[str] = "medium"  # low, medium, high, critical
    deadline: Optional[str] = None
    
    @validator('request')
    def validate_request(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Request must be at least 10 characters long')
        if len(v) > 5000:
            raise ValueError('Request too long (max 5000 characters)')
        return v.strip()
    
    @validator('priority')
    def validate_priority(cls, v):
        if v and v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError('Priority must be: low, medium, high, or critical')
        return v

class TechnicalSpec(BaseModel):
    """Output model for technical specifications"""
    intent: str
    entities: List[str]
    complexity: str  # Simple, Moderate, Complex, Enterprise
    agents_needed: List[str]
    estimated_cost: float
    estimated_time_minutes: int
    technical_requirements: Dict[str, Any]
    constraints: List[str]
    confidence_score: float

class ValidationResponse(BaseModel):
    """Validation result model"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []
    confidence: float = 0.0
    sanitized_request: Optional[str] = None

# === CORE BUSINESS REQUIREMENTS PARSER ===
class BusinessRequirementsParser:
    """
    Complete Business Requirements Parser
    Converts natural language business requirements to technical specifications
    """
    
    def __init__(self):
        """Initialize parser with pattern mappings"""
        self.intent_patterns = {
            'CREATE': ['create', 'build', 'develop', 'generate', 'make', 'implement', 'design'],
            'UPDATE': ['update', 'modify', 'change', 'edit', 'improve', 'enhance', 'refactor'],
            'ANALYZE': ['analyze', 'review', 'examine', 'evaluate', 'assess', 'audit'],
            'PROCESS': ['process', 'transform', 'convert', 'migrate', 'import', 'export'],
            'SEARCH': ['search', 'find', 'lookup', 'query', 'retrieve', 'filter'],
            'DELETE': ['delete', 'remove', 'clean', 'purge', 'archive']
        }
        
        self.entity_patterns = {
            'api': ['api', 'endpoint', 'service', 'rest', 'graphql', 'webhook'],
            'database': ['database', 'db', 'table', 'collection', 'data', 'storage'],
            'user': ['user', 'customer', 'client', 'person', 'account', 'profile'],
            'file': ['file', 'document', 'csv', 'json', 'xml', 'image', 'pdf'],
            'code': ['code', 'function', 'class', 'method', 'script', 'algorithm'],
            'report': ['report', 'dashboard', 'chart', 'graph', 'analytics', 'metrics'],
            'ui': ['interface', 'ui', 'frontend', 'form', 'page', 'screen'],
            'auth': ['authentication', 'authorization', 'login', 'security', 'token']
        }
        
        self.complexity_factors = {
            'simple_keywords': ['simple', 'basic', 'quick', 'small', 'minimal'],
            'complex_keywords': ['complex', 'advanced', 'enterprise', 'scalable', 'distributed'],
            'integration_keywords': ['integrate', 'connect', 'sync', 'migration', 'workflow'],
            'security_keywords': ['secure', 'authentication', 'authorization', 'encrypt', 'compliance']
        }

    def parse_intent(self, business_request: str) -> str:
        """
        Extract primary intent from business request
        Returns: Intent string (CREATE, UPDATE, ANALYZE, etc.)
        """
        request_lower = business_request.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score, or default to PROCESS
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return 'PROCESS'

    def extract_entities(self, business_request: str) -> List[str]:
        """
        Extract business entities from request
        Returns: List of entity types found
        """
        request_lower = business_request.lower()
        found_entities = []
        
        for entity, keywords in self.entity_patterns.items():
            if any(keyword in request_lower for keyword in keywords):
                found_entities.append(entity)
        
        return list(set(found_entities))  # Remove duplicates

    def assess_complexity(self, business_request: str, entities: List[str]) -> str:
        """
        Assess task complexity using multiple factors
        Returns: Complexity level (Simple, Moderate, Complex, Enterprise)
        """
        request_lower = business_request.lower()
        complexity_score = 0
        
        # Factor 1: Keyword analysis
        if any(keyword in request_lower for keyword in self.complexity_factors['simple_keywords']):
            complexity_score -= 1
        if any(keyword in request_lower for keyword in self.complexity_factors['complex_keywords']):
            complexity_score += 2
        if any(keyword in request_lower for keyword in self.complexity_factors['integration_keywords']):
            complexity_score += 1
        if any(keyword in request_lower for keyword in self.complexity_factors['security_keywords']):
            complexity_score += 1
            
        # Factor 2: Number of entities
        entity_count = len(entities)
        if entity_count >= 4:
            complexity_score += 2
        elif entity_count >= 2:
            complexity_score += 1
            
        # Factor 3: Request detail level
        word_count = len(business_request.split())
        if word_count > 50:
            complexity_score += 1
        elif word_count > 100:
            complexity_score += 2
            
        # Factor 4: Technical depth indicators
        technical_terms = ['architecture', 'microservice', 'distributed', 'real-time', 'concurrent']
        if any(term in request_lower for term in technical_terms):
            complexity_score += 1
            
        # Classify complexity
        if complexity_score <= 0:
            return "Simple"
        elif complexity_score <= 2:
            return "Moderate"
        elif complexity_score <= 4:
            return "Complex"
        else:
            return "Enterprise"

    def select_agents(self, intent: str, entities: List[str], complexity: str) -> List[str]:
        """
        Determine required agents for task execution
        Returns: List of agent types needed
        """
        agents = ['orchestrator']  # Always required
        
        # Intent-based agent selection
        if intent in ['CREATE', 'UPDATE'] and 'code' in entities:
            agents.append('code_generator')
        if intent == 'ANALYZE':
            agents.append('data_analyst')
        if 'api' in entities:
            agents.append('api_specialist')
        if 'database' in entities:
            agents.append('database_specialist')
        if 'ui' in entities:
            agents.append('frontend_specialist')
        if 'auth' in entities or 'user' in entities:
            agents.append('security_specialist')
        if 'report' in entities:
            agents.append('reporting_specialist')
            
        # Complexity-based additions
        if complexity in ['Complex', 'Enterprise']:
            agents.append('solution_architect')
        if complexity == 'Enterprise':
            agents.extend(['performance_specialist', 'devops_specialist'])
            
        return list(set(agents))

    def estimate_cost_and_time(self, complexity: str, agents: List[str], entities: List[str]) -> tuple:
        """
        Estimate implementation cost and time
        Returns: (estimated_cost_usd, estimated_time_minutes)
        """
        # Base estimates by complexity level
        base_estimates = {
            'Simple': {'cost': 0.02, 'time': 10},
            'Moderate': {'cost': 0.08, 'time': 30}, 
            'Complex': {'cost': 0.25, 'time': 90},
            'Enterprise': {'cost': 0.75, 'time': 240}
        }
        
        base = base_estimates.get(complexity, base_estimates['Moderate'])
        
        # Agent multiplier (more specialists = higher cost)
        agent_count = len(agents)
        agent_multiplier = 1.0 + (agent_count - 1) * 0.15
        
        # Entity multiplier (more integrations = more complexity)
        entity_count = len(entities)
        entity_multiplier = 1.0 + entity_count * 0.08
        
        # Calculate final estimates
        estimated_cost = base['cost'] * agent_multiplier * entity_multiplier
        estimated_time = int(base['time'] * agent_multiplier * entity_multiplier)
        
        return estimated_cost, estimated_time

    def generate_technical_spec(self, intent: str, entities: List[str], complexity: str, 
                              business_request: str, context: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive technical specification
        Returns: Complete technical specification dictionary
        """
        agents = self.select_agents(intent, entities, complexity)
        cost, time = self.estimate_cost_and_time(complexity, agents, entities)
        
        # Generate technical requirements
        tech_requirements = {}
        
        if 'api' in entities:
            tech_requirements['api'] = {
                'type': 'REST',
                'authentication': 'JWT' if complexity in ['Complex', 'Enterprise'] else 'API Key',
                'documentation': 'OpenAPI/Swagger',
                'rate_limiting': complexity in ['Complex', 'Enterprise'],
                'versioning': complexity == 'Enterprise'
            }
            
        if 'database' in entities:
            tech_requirements['database'] = {
                'type': 'PostgreSQL' if complexity in ['Complex', 'Enterprise'] else 'SQLite',
                'migrations': True,
                'indexing': complexity in ['Complex', 'Enterprise'],
                'backup_strategy': complexity == 'Enterprise',
                'replication': complexity == 'Enterprise'
            }
            
        if 'ui' in entities:
            tech_requirements['frontend'] = {
                'framework': 'React' if complexity in ['Complex', 'Enterprise'] else 'HTML/JS',
                'responsive': True,
                'accessibility': complexity in ['Complex', 'Enterprise'],
                'internationalization': complexity == 'Enterprise'
            }
            
        if 'auth' in entities:
            tech_requirements['security'] = {
                'authentication': 'JWT',
                'authorization': 'RBAC' if complexity in ['Complex', 'Enterprise'] else 'Basic',
                'encryption': 'AES-256',
                'audit_logging': complexity in ['Complex', 'Enterprise']
            }
            
        # Extract constraints from request
        constraints = []
        request_lower = business_request.lower()
        
        if any(term in request_lower for term in ['secure', 'security', 'safe']):
            constraints.append('Security compliance required')
        if any(term in request_lower for term in ['fast', 'quick', 'performance']):
            constraints.append('Performance optimization priority')
        if any(term in request_lower for term in ['scalable', 'scale', 'growth']):
            constraints.append('Horizontal scaling capability required')
        if any(term in request_lower for term in ['real-time', 'live', 'instant']):
            constraints.append('Real-time processing required')
        if any(term in request_lower for term in ['mobile', 'responsive']):
            constraints.append('Mobile-friendly interface required')
            
        # Calculate confidence score
        confidence = 0.75  # Base confidence
        
        # Boost confidence for clear requests
        if len(entities) >= 2:
            confidence += 0.10
        if intent != 'PROCESS':  # Clear intent identified
            confidence += 0.05
        if complexity in ['Simple', 'Moderate']:  # Lower uncertainty
            confidence += 0.05
        if context and len(context) > 0:  # Additional context provided
            confidence += 0.05
            
        # Reduce confidence for very complex or vague requests
        if complexity == 'Enterprise':
            confidence -= 0.05
        if len(entities) == 0:
            confidence -= 0.15
            
        return {
            'intent': intent,
            'entities': entities,
            'complexity': complexity,
            'agents_needed': agents,
            'estimated_cost': round(cost, 4),
            'estimated_time_minutes': time,
            'technical_requirements': tech_requirements,
            'constraints': constraints,
            'confidence_score': max(0.0, min(confidence, 1.0))
        }

    def validate_request(self, request: str) -> Dict[str, Any]:
        """
        Comprehensive validation of business request
        Returns: Validation result with errors, warnings, suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Sanitize input first
        sanitized_request = self.sanitize_input(request)
        
        # Length validation
        if len(sanitized_request.strip()) < 10:
            errors.append("Request too short. Please provide more details (minimum 10 characters).")
        if len(sanitized_request) > 5000:
            errors.append("Request too long. Please be more concise (maximum 5000 characters).")
            
        # Content validation
        if not re.search(r'[a-zA-Z]', sanitized_request):
            errors.append("Request must contain alphabetic characters.")
            
        # Analyze completeness
        intent = self.parse_intent(sanitized_request)
        entities = self.extract_entities(sanitized_request)
        
        # Check for vague requests
        if intent == 'PROCESS' and len(entities) == 0:
            warnings.append("Request appears vague. Consider specifying what you want to create, update, or analyze.")
            
        # Missing entities suggestions
        if len(entities) == 0:
            suggestions.append("Include specific components: API, database, user interface, reports, etc.")
        elif len(entities) == 1:
            suggestions.append("Consider mentioning additional components that might be needed.")
            
        # Missing action verb
        action_words = ['create', 'build', 'develop', 'update', 'modify', 'analyze', 'generate']
        if not any(word in sanitized_request.lower() for word in action_words):
            suggestions.append("Consider starting with a clear action: 'Create...', 'Update...', 'Analyze...', etc.")
            
        # Security considerations
        auth_terms = ['password', 'token', 'auth', 'login', 'user', 'account']
        security_terms = ['secure', 'security', 'encrypt', 'protection', 'safe']
        
        has_auth_terms = any(term in sanitized_request.lower() for term in auth_terms)
        has_security_terms = any(term in sanitized_request.lower() for term in security_terms)
        
        if has_auth_terms and not has_security_terms:
            suggestions.append("Consider adding security requirements for user/authentication features.")
            
        # Data handling suggestions
        data_terms = ['data', 'information', 'database', 'store', 'save']
        privacy_terms = ['privacy', 'gdpr', 'compliance', 'consent', 'protection']
        
        has_data_terms = any(term in sanitized_request.lower() for term in data_terms)
        has_privacy_terms = any(term in sanitized_request.lower() for term in privacy_terms)
        
        if has_data_terms and not has_privacy_terms:
            suggestions.append("For data handling, consider privacy requirements (GDPR, data protection).")
            
        # Calculate confidence
        confidence = self._calculate_validation_confidence(errors, warnings, sanitized_request)
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "confidence": confidence,
            "sanitized_request": sanitized_request
        }

    def sanitize_input(self, request: str) -> str:
        """
        Clean and sanitize user input
        Returns: Sanitized string
        """
        if not request:
            return ""
            
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]*>', '', request)
        
        # Remove potentially dangerous characters
        cleaned = re.sub(r'[<>"\']', '', cleaned)
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Truncate if too long
        if len(cleaned) > 5000:
            cleaned = cleaned[:4970] + "... (truncated)"
            
        return cleaned.strip()

    def _calculate_validation_confidence(self, errors: List[str], warnings: List[str], request: str) -> float:
        """
        Calculate confidence score for validation
        Returns: Confidence score between 0.0 and 1.0
        """
        # Start with base confidence
        confidence = 0.8
        
        # Penalize errors heavily
        confidence -= len(errors) * 0.25
        
        # Penalize warnings moderately
        confidence -= len(warnings) * 0.08
        
        # Reward detailed requests
        word_count = len(request.split())
        if word_count > 15:
            confidence += 0.05
        if word_count > 30:
            confidence += 0.10
            
        # Ensure valid range
        return max(0.0, min(1.0, confidence))

# === API ROUTER SETUP ===
router = APIRouter(prefix="/api/business", tags=["business"])
parser = BusinessRequirementsParser()

@router.post("/parse", response_model=TechnicalSpec)
async def parse_business_request(request: BusinessRequest):
    """
    Parse business requirements into technical specifications
    
    Converts natural language business requirements into structured
    technical specifications with cost estimates and agent assignments.
    """
    try:
        # Validate request first
        validation = parser.validate_request(request.request)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Invalid business request",
                    "errors": validation['errors'],
                    "warnings": validation['warnings'],
                    "suggestions": validation['suggestions']
                }
            )
        
        # Generate technical specification
        intent = parser.parse_intent(request.request)
        entities = parser.extract_entities(request.request)
        complexity = parser.assess_complexity(request.request, entities)
        
        spec = parser.generate_technical_spec(
            intent=intent,
            entities=entities,
            complexity=complexity,
            business_request=request.request,
            context=request.context
        )
        
        logger.info(f"Successfully parsed: {intent} | {complexity} | {len(entities)} entities")
        
        return TechnicalSpec(**spec)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal parsing error: {str(e)}")

@router.post("/validate", response_model=ValidationResponse)
async def validate_business_request(request: BusinessRequest):
    """
    Validate business request quality and completeness
    
    Checks request validity and provides suggestions for improvement
    before processing.
    """
    try:
        validation_result = parser.validate_request(request.request)
        
        return ValidationResponse(
            is_valid=validation_result['is_valid'],
            errors=validation_result['errors'],
            warnings=validation_result['warnings'],
            suggestions=validation_result['suggestions'],
            confidence=validation_result['confidence'],
            sanitized_request=validation_result['sanitized_request']
        )
        
    except Exception as e:
        logger.error(f"Error validating request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@router.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "service": "business_requirements_parser",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "parser": "operational",
            "validator": "operational",
            "api": "operational"
        }
    }

@router.get("/capabilities")
async def get_capabilities():
    """Get parser capabilities and supported features"""
    return {
        "supported_intents": list(parser.intent_patterns.keys()),
        "supported_entities": list(parser.entity_patterns.keys()),
        "complexity_levels": ["Simple", "Moderate", "Complex", "Enterprise"],
        "features": {
            "intent_detection": True,
            "entity_extraction": True,
            "complexity_assessment": True,
            "agent_selection": True,
            "cost_estimation": True,
            "time_estimation": True,
            "validation": True,
            "security_suggestions": True
        }
    }

# === INTEGRATION INSTRUCTIONS ===
"""
To integrate with your FastAPI application:

1. Add to main app:
   from business.requirements_parser import router as business_router
   app.include_router(business_router)

2. Ensure dependencies are installed:
   pip install fastapi pydantic

3. Test endpoints:
   POST /api/business/parse
   POST /api/business/validate
   GET /api/business/health
   GET /api/business/capabilities
"""
