# NASZ Agent Zero V1 - Business Requirements Parser Implementation
**Developer A Action Plan - 9 października 2025, 12:00 CEST**

## 🎯 NASZE Projektu - HotelAiOS Agent Zero V1

**Zrozumiałem!** To jest zupełnie NASZ WŁASNY projekt "Agent Zero V1" w repozytorium HotelAiOS, nie związany z oficjalnym Agent Zero framework. 

### Kontekst NASZEGO Projektu:
- **Repository:** `https://github.com/HotelAiOS/agent-zero-v1` 
- **Nasze własne multi-agentowe rozwiązanie**
- **Tech Stack:** Python 3.11, Neo4j, RabbitMQ, Ollama, FastAPI, Docker
- **Current Status:** Phase 1 (75% done), Phase 2 (25% - Web Interface) 
- **Team:** 2 developers (Backend + Frontend)
- **Environment:** Arch Linux + Fish Shell

## 🏗️ DIRECTORY STRUCTURE - Bazując na NASZYM Kodzie

Ponieważ to jest NASZ projekt, struktura może być inna. Bez dostępu do repo zakładam typową strukturę multi-agentowego systemu:

```
agent-zero-v1/                    # NASZE główne repo
├── src/                          # Core system code
│   ├── agents/                   # Nasze agent implementations
│   ├── core/                     # Agent execution engine
│   │   ├── agent_executor.py     # ✅ Naprawione signatures
│   │   └── task_decomposer.py    # ✅ JSON parsing fixed  
│   ├── api/                      # FastAPI endpoints
│   └── services/                 # Backend services
├── infrastructure/               # Docker, configs
├── neo4j/                       # ✅ Knowledge graph - operational
├── rabbitmq/                    # ✅ Message queue - active  
├── web/                         # Frontend interface (Phase 2)
├── business_intelligence/        # 🆕 NASZ V2.0 Intelligence Layer
│   ├── __init__.py
│   ├── business/                # Core NLP logic
│   │   ├── __init__.py
│   │   ├── requirements_parser.py # Main orchestrator
│   │   ├── intent_extractor.py   # NLP processing
│   │   ├── context_enricher.py   # Context understanding
│   │   ├── business_translator.py # Business → Technical
│   │   └── validators.py         # Input validation
│   ├── api/                     # API integration
│   │   ├── __init__.py
│   │   ├── business_endpoints.py # REST endpoints
│   │   └── streaming_endpoints.py # WebSocket real-time
│   ├── models/                  # Data models
│   │   ├── __init__.py
│   │   ├── business_models.py   # Input schemas
│   │   └── technical_specs.py  # Output specifications
│   └── tests/                   # Testing framework
├── tests/                       # ✅ Integration tests PASSED (5/5)
├── requirements.txt
└── docker-compose.yml           # ✅ All services operational
```

---

## 📦 SETUP PLAN DLA NASZEGO PROJEKTU

### 1. Repository Setup
```bash
# Navigate to OUR project
cd /home/ianua/projects/agent-zero-v1

# Create feature branch for our Business Requirements Parser
git checkout -b feature/business-requirements-parser-v2

# Create our new Intelligence Layer module
mkdir -p business_intelligence/{business,api,models,tests}

# Create Python module structure
touch business_intelligence/__init__.py
touch business_intelligence/{business,api,models,tests}/__init__.py
```

### 2. Dependencies dla NASZEGO System
Do NASZEGO requirements.txt dodajemy:
```txt
# V2.0 Intelligence Layer dla NASZEGO Agent Zero V1
spacy>=3.6.0
transformers>=4.30.0
torch>=2.0.0
pydantic>=2.0.0
nltk>=3.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
aiofiles>=23.1.0

# Integration z NASZYM existing stack
asyncio-mqtt>=0.16.0  # Jeśli używamy MQTT z RabbitMQ
neo4j-driver>=5.0.0   # NASZE Neo4j integration
python-multipart>=0.0.6
```

---

## 📄 COMPLETE FILE IMPLEMENTATIONS dla NASZEGO Systemu

### 1. business_intelligence/__init__.py
```python
"""
NASZE Agent Zero V1 - V2.0 Intelligence Layer
Business Requirements Parser Module

Rozszerza NASZ istniejący Agent Zero V1 multi-agentowy system
o możliwości przetwarzania naturalnego języka biznesowego
na specyfikacje techniczne.

Integration Points z NASZYM systemem:
- Neo4j Knowledge Graph (NASZE existing)
- RabbitMQ Message Queue (NASZE existing)
- Agent Executor (NASZE core system)
- Task Decomposer (NASZE orchestration)
"""

__version__ = "2.0.0"
__author__ = "HotelAiOS Agent Zero V1 Team"
__project__ = "HotelAiOS Agent Zero V1 Intelligence Layer"

from .business.requirements_parser import BusinessRequirementsParser
from .models.business_models import BusinessRequirement, TechnicalSpecification

__all__ = [
    'BusinessRequirementsParser',
    'BusinessRequirement', 
    'TechnicalSpecification'
]
```

### 2. business_intelligence/models/business_models.py
```python
"""
Business Requirements Data Models
Designed specifically for NASZ Agent Zero V1 architecture
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from enum import Enum
import uuid
from datetime import datetime

class IndustryType(str, Enum):
    """Business domains dla NASZEGO Agent Zero V1"""
    HOSPITALITY = "hospitality"      # Primary for HotelAiOS
    TRAVEL = "travel"               
    ECOMMERCE = "ecommerce"
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERPRISE = "enterprise"
    SAAS = "saas"
    LOGISTICS = "logistics"
    MANUFACTURING = "manufacturing"

class ProjectComplexity(str, Enum):
    """Complexity levels dla NASZEGO orchestration system"""
    SIMPLE = "simple"        # Single agent, straightforward
    STANDARD = "standard"    # Multi-agent, typical workflow  
    COMPLEX = "complex"      # Advanced orchestration, multiple systems
    ENTERPRISE = "enterprise" # Full platform integration

class OurBusinessContext(BaseModel):
    """Context model dla NASZEGO Agent Zero V1 system"""
    industry: Optional[IndustryType] = None
    complexity: Optional[ProjectComplexity] = None
    timeline: Optional[str] = Field(None, description="Expected delivery timeline")
    budget_indication: Optional[str] = Field(None, description="Budget range")
    existing_integrations: List[str] = Field(default_factory=list)
    target_user_base: Optional[str] = None
    business_objectives: List[str] = Field(default_factory=list)
    
    # NASZE Agent Zero V1 specific requirements
    agent_orchestration_needed: bool = Field(default=True, description="Multi-agent coordination required")
    neo4j_knowledge_required: bool = Field(default=False, description="Knowledge graph integration")
    rabbitmq_messaging_required: bool = Field(default=False, description="Real-time messaging needed") 
    web_interface_required: bool = Field(default=True, description="Frontend interface needed")
    
    # Performance requirements dla NASZEGO system
    expected_load: Optional[str] = Field(None, description="Expected system load")
    scalability_requirements: List[str] = Field(default_factory=list)

class OurBusinessRequirement(BaseModel):
    """Main input model dla NASZEGO Agent Zero V1 Intelligence Layer"""
    
    # Core requirement
    business_requirement: str = Field(..., min_length=10, description="Natural language requirement")
    
    # Context and metadata
    context: Optional[OurBusinessContext] = None
    constraints: List[str] = Field(default_factory=list, description="Business/technical constraints")
    priority: int = Field(default=3, ge=1, le=5, description="Priority level (1=highest)")
    
    # NASZE system metadata
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requested_by: Optional[str] = Field(None, description="Stakeholder name")
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # NASZE Agent Zero V1 orchestration hints
    multi_agent_coordination: bool = Field(default=True, description="Requires multiple agents")
    real_time_processing: bool = Field(default=False, description="Real-time requirements")
    data_heavy_operations: bool = Field(default=False, description="Heavy data processing")
    external_integrations: List[str] = Field(default_factory=list, description="Required external systems")

class OurIntentClassification(BaseModel):
    """NLP analysis result dla NASZEGO Agent Zero V1 orchestration"""
    
    # Classification results
    primary_intent: str = Field(..., description="Main classified business intent")
    secondary_intents: List[str] = Field(default_factory=list, description="Supporting intents")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    
    # Extracted business information
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    detected_industry: Optional[IndustryType] = None
    technical_concepts: List[str] = Field(default_factory=list)
    
    # NASZE Agent Zero V1 orchestration recommendations
    suggested_agent_roles: List[str] = Field(default_factory=list, description="Recommended agent types")
    orchestration_strategy: Optional[str] = Field(None, description="coordination|pipeline|parallel|hierarchical")
    system_complexity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Processing metadata
    processing_duration_ms: Optional[float] = None
    nlp_model_version: str = Field(default="hotel-aios-agent-zero-v1-nlp")

# NASZE Agent Zero V1 specific intent categories
OUR_INTENT_CATEGORIES = [
    "multi_agent_orchestration",     # NASZE core capability
    "knowledge_management_system",   # Neo4j integration
    "real_time_data_processing",     # RabbitMQ integration
    "web_application_development",   # Phase 2 web interface
    "api_service_architecture",     # FastAPI integration
    "data_pipeline_automation",     # NASZE data processing
    "user_experience_optimization", # Frontend focus
    "system_integration_project",   # External system integration
    "automated_workflow_system",    # NASZE orchestration strength
    "hospitality_domain_solution"   # HotelAiOS specialization
]

# NASZE Agent roles/specializations
OUR_AGENT_ROLES = {
    "orchestration": [
        "TaskCoordinator",
        "WorkflowManager", 
        "ResourceAllocator",
        "SystemSupervisor"
    ],
    "data_processing": [
        "DataAnalyzer", 
        "DataValidator",
        "DataTransformer",
        "KnowledgeExtractor"
    ],
    "integration": [
        "APIConnector",
        "SystemBridge", 
        "DatabaseConnector",
        "ExternalServiceIntegrator"
    ],
    "user_interface": [
        "UIComponentGenerator",
        "UXOptimizer",
        "ResponseFormatter",
        "FrontendCoordinator"
    ],
    "monitoring": [
        "PerformanceMonitor",
        "SystemHealthChecker", 
        "AlertManager",
        "QualityAssurance"
    ]
}

# NASZE system integration patterns
OUR_INTEGRATION_PATTERNS = {
    "neo4j_operations": [
        "knowledge_storage", "graph_analysis", "relationship_mapping", 
        "semantic_search", "data_modeling"
    ],
    "rabbitmq_messaging": [
        "real_time_updates", "event_driven_processing", "async_communication",
        "message_queuing", "workflow_coordination"  
    ],
    "web_interface": [
        "dashboard", "user_portal", "admin_interface", 
        "real_time_display", "interactive_components"
    ],
    "api_services": [
        "rest_endpoints", "graphql_api", "websocket_communication",
        "external_integration", "service_mesh"
    ]
}

# Industry-specific keyword patterns (focus on hospitality dla HotelAiOS)
INDUSTRY_KEYWORDS = {
    IndustryType.HOSPITALITY: [
        "hotel", "booking", "reservation", "guest", "room", "accommodation",
        "check-in", "check-out", "concierge", "hospitality", "front-desk",
        "housekeeping", "amenities", "occupancy", "revenue management"
    ],
    IndustryType.TRAVEL: [
        "travel", "trip", "flight", "airline", "tourism", "vacation",
        "itinerary", "destination", "transportation", "booking engine"
    ],
    IndustryType.ECOMMERCE: [
        "shop", "store", "product", "inventory", "cart", "payment",
        "checkout", "order", "customer", "catalog", "e-commerce"
    ],
    IndustryType.FINTECH: [
        "payment", "transaction", "banking", "finance", "wallet",
        "billing", "invoicing", "financial", "accounting"
    ]
}
```

### 3. business_intelligence/business/intent_extractor.py  
```python
"""
Intent Extraction Engine dla NASZEGO Agent Zero V1
NLP processing optimized dla NASZEJ multi-agent architecture
"""

import spacy
from transformers import pipeline
from typing import Dict, List, Optional, Tuple
import re
import time
import logging
import asyncio

from ..models.business_models import (
    OurIntentClassification, 
    IndustryType, 
    OUR_INTENT_CATEGORIES,
    OUR_AGENT_ROLES,
    INDUSTRY_KEYWORDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OurIntentExtractor:
    """
    Intent extractor dostosowany do NASZEGO Agent Zero V1 system
    Koncentruje się na multi-agent orchestration i hospitality domain
    """
    
    def __init__(self):
        """Initialize NLP models dla NASZEGO Agent Zero V1"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize classifier dla NASZYCH intent categories
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self._gpu_available() else -1
            )
            
            # NASZE Agent Zero V1 specific patterns
            self.orchestration_strategies = {
                "coordination": ["coordinate", "manage", "orchestrate", "supervise"],
                "pipeline": ["sequence", "step-by-step", "workflow", "process"],
                "parallel": ["simultaneously", "concurrent", "parallel", "at-once"],
                "hierarchical": ["levels", "hierarchy", "structured", "organized"]
            }
            
            # Complexity indicators dla NASZEGO system
            self.complexity_indicators = {
                "simple": ["basic", "simple", "straightforward", "minimal", "quick"],
                "standard": ["typical", "normal", "standard", "regular", "common"],
                "complex": ["complex", "advanced", "sophisticated", "comprehensive", "detailed"],
                "enterprise": ["enterprise", "large-scale", "distributed", "scalable", "mission-critical"]
            }
            
            # HotelAiOS specific domain indicators  
            self.hotel_domain_terms = [
                "hotel", "hospitality", "guest", "booking", "reservation",
                "accommodation", "room", "check-in", "concierge", "front-desk"
            ]
            
            logger.info("OurIntentExtractor initialized dla NASZEGO Agent Zero V1")
            
        except Exception as e:
            logger.error(f"Failed to initialize OurIntentExtractor: {e}")
            raise
    
    async def extract_intent(self, requirement_text: str) -> OurIntentClassification:
        """
        Extract business intent optimized dla NASZEGO Agent Zero V1 system
        
        Args:
            requirement_text: Natural language business requirement
            
        Returns:
            OurIntentClassification z NASZYMI system-specific recommendations
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            clean_text = self._preprocess_text(requirement_text)
            
            # Extract entities using spaCy
            doc = self.nlp(clean_text)
            entities = self._extract_entities(doc)
            
            # Classify intent dla NASZEGO system
            classification = self.classifier(clean_text, OUR_INTENT_CATEGORIES)
            
            # Determine orchestration strategy dla NASZEGO multi-agent system
            orchestration_strategy = self._determine_orchestration_strategy(clean_text)
            
            # Suggest NASZE agent roles
            suggested_agents = self._suggest_our_agent_roles(clean_text, classification['labels'][0])
            
            # Calculate system complexity dla NASZEGO orchestration
            complexity_score = self._calculate_system_complexity(clean_text, entities)
            
            # Detect industry (focus on hospitality dla HotelAiOS)
            detected_industry = self._detect_industry(clean_text)
            
            # Extract technical concepts relevant to NASZEGO stack
            tech_concepts = self._extract_our_technical_concepts(clean_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return OurIntentClassification(
                primary_intent=classification['labels'][0],
                secondary_intents=classification['labels'][1:3],
                confidence_score=classification['scores'][0],
                extracted_entities=entities,
                detected_industry=detected_industry,
                technical_concepts=tech_concepts,
                suggested_agent_roles=suggested_agents,
                orchestration_strategy=orchestration_strategy,
                system_complexity_score=complexity_score,
                processing_duration_ms=processing_time,
                nlp_model_version="hotel-aios-agent-zero-v1-nlp-1.0"
            )
            
        except Exception as e:
            logger.error(f"Intent extraction failed dla NASZEGO system: {e}")
            # Fallback dla NASZEGO Agent Zero V1
            return OurIntentClassification(
                primary_intent="multi_agent_orchestration",
                secondary_intents=["system_integration_project"],
                confidence_score=0.1,
                extracted_entities={},
                suggested_agent_roles=["TaskCoordinator", "SystemSupervisor"],
                orchestration_strategy="coordination",
                system_complexity_score=0.5,
                processing_duration_ms=(time.time() - start_time) * 1000
            )
    
    def _determine_orchestration_strategy(self, text: str) -> Optional[str]:
        """Determine best orchestration strategy dla NASZEGO Agent Zero V1"""
        text_lower = text.lower()
        strategy_scores = {}
        
        for strategy, keywords in self.orchestration_strategies.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                strategy_scores[strategy] = score
        
        # Default dla NASZEGO system
        if not strategy_scores:
            return "coordination"  # NASZE default multi-agent approach
        
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _suggest_our_agent_roles(self, text: str, primary_intent: str) -> List[str]:
        """Suggest agent roles dla NASZEGO Agent Zero V1 system"""
        text_lower = text.lower()
        suggested_roles = []
        
        # Map intent to NASZE agent roles
        intent_to_roles = {
            "multi_agent_orchestration": ["TaskCoordinator", "WorkflowManager"],
            "knowledge_management_system": ["KnowledgeExtractor", "DataAnalyzer"],
            "real_time_data_processing": ["DataAnalyzer", "SystemHealthChecker"],
            "web_application_development": ["UIComponentGenerator", "FrontendCoordinator"],
            "api_service_architecture": ["APIConnector", "SystemBridge"],
            "hospitality_domain_solution": ["TaskCoordinator", "UIComponentGenerator", "DataAnalyzer"]
        }
        
        # Get roles for primary intent
        if primary_intent in intent_to_roles:
            suggested_roles.extend(intent_to_roles[primary_intent])
        
        # Add roles based on content analysis
        if any(term in text_lower for term in ["data", "analysis", "process", "analytics"]):
            suggested_roles.extend(["DataAnalyzer", "DataValidator"])
        
        if any(term in text_lower for term in ["ui", "interface", "frontend", "web", "dashboard"]):
            suggested_roles.extend(["UIComponentGenerator", "UXOptimizer"])
        
        if any(term in text_lower for term in ["api", "service", "endpoint", "integration"]):
            suggested_roles.extend(["APIConnector", "SystemBridge"])
        
        if any(term in text_lower for term in ["monitor", "health", "performance", "alert"]):
            suggested_roles.extend(["PerformanceMonitor", "SystemHealthChecker"])
        
        # HotelAiOS hospitality-specific agents
        if any(term in text_lower for term in self.hotel_domain_terms):
            suggested_roles.extend(["TaskCoordinator", "KnowledgeExtractor"])
        
        # Remove duplicates and limit dla NASZEGO system performance
        suggested_roles = list(set(suggested_roles))[:5]
        
        # Ensure we have basic coordination
        if not suggested_roles:
            suggested_roles = ["TaskCoordinator", "SystemSupervisor"]
        
        return suggested_roles
    
    def _calculate_system_complexity(self, text: str, entities: Dict[str, List[str]]) -> float:
        """Calculate complexity score dla NASZEGO Agent Zero V1 orchestration"""
        complexity_score = 0.5  # Base dla NASZEGO system
        text_lower = text.lower()
        
        # Adjust based on complexity indicators
        for level, indicators in self.complexity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                level_scores = {"simple": 0.25, "standard": 0.5, "complex": 0.75, "enterprise": 1.0}
                complexity_score = max(complexity_score, level_scores[level])
        
        # Adjust based on NASZE system requirements
        if any(term in text_lower for term in ["neo4j", "knowledge", "graph"]):
            complexity_score += 0.1
        
        if any(term in text_lower for term in ["rabbitmq", "messaging", "real-time", "queue"]):
            complexity_score += 0.1
        
        if any(term in text_lower for term in ["multi-agent", "orchestration", "coordination"]):
            complexity_score += 0.15
        
        # Integration complexity dla NASZEGO system
        integration_terms = ["integrate", "connect", "sync", "bridge", "interface"]
        if any(term in text_lower for term in integration_terms):
            complexity_score += 0.1
        
        # Scale indicators
        if any(term in text_lower for term in ["enterprise", "scalable", "distributed", "high-volume"]):
            complexity_score += 0.15
        
        return min(complexity_score, 1.0)
    
    def _extract_our_technical_concepts(self, text: str) -> List[str]:
        """Extract technical concepts relevant to NASZEGO Agent Zero V1 stack"""
        text_lower = text.lower()
        
        # NASZE technology stack keywords
        our_tech_concepts = [
            # NASZE core stack
            "python", "neo4j", "rabbitmq", "fastapi", "docker", "ollama",
            # Multi-agent concepts (NASZE specialization)
            "orchestration", "coordination", "multi-agent", "workflow", "automation",
            # AI/ML (NASZE intelligence layer)
            "artificial intelligence", "machine learning", "nlp", "natural language",
            # Architecture patterns (NASZE system design)
            "microservices", "api", "rest", "websocket", "real-time",
            # Data concepts (NASZE data processing)
            "knowledge graph", "data pipeline", "analytics", "processing",
            # Hospitality domain (HotelAiOS focus)
            "hospitality", "hotel management", "booking system", "guest services",
            # Infrastructure (NASZE deployment)
            "containerization", "scaling", "monitoring", "performance"
        ]
        
        found_concepts = []
        for concept in our_tech_concepts:
            if concept in text_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _detect_industry(self, text: str) -> Optional[IndustryType]:
        """Detect industry z focus na hospitality dla HotelAiOS"""
        text_lower = text.lower()
        
        industry_scores = {}
        for industry, keywords in INDUSTRY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                industry_scores[industry] = score
        
        if not industry_scores:
            return None
        
        # Return industry with highest score
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        if best_industry[1] >= 2:  # Require confidence
            return best_industry[0]
        
        return None
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text dla NASZEGO system"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize terms dla NASZEGO Agent Zero V1
        replacements = {
            'multi-agent': 'multi agent',
            'real-time': 'realtime', 
            'hotel ai': 'hotel artificial intelligence',
            'ai os': 'artificial intelligence operating system'
        }
        
        for old, new in replacements.items():
            cleaned = re.sub(rf'\b{old}\b', new, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract entities relevant to NASZEGO Agent Zero V1"""
        entities = {}
        
        # Standard NER entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "MONEY", "PERCENT", "DATE", "TIME"]:
                key = ent.label_.lower()
                if key not in entities:
                    entities[key] = []
                entities[key].append(ent.text)
        
        # NASZE Agent Zero V1 specific terms
        our_system_terms = [
            "agent zero", "multi-agent", "orchestration", "coordination",
            "neo4j", "rabbitmq", "fastapi", "hotel aios"
        ]
        
        found_system_terms = []
        text_lower = doc.text.lower()
        for term in our_system_terms:
            if term in text_lower:
                found_system_terms.append(term)
        
        if found_system_terms:
            entities["our_system_terms"] = found_system_terms
        
        # Business processes dla NASZEGO domain
        business_processes = [
            "booking", "reservation", "check-in", "guest services",
            "room management", "revenue optimization", "customer service"
        ]
        
        found_processes = []
        for process in business_processes:
            if process in text_lower:
                found_processes.append(process)
        
        if found_processes:
            entities["business_processes"] = found_processes
        
        return entities
    
    def _gpu_available(self) -> bool:
        """Check GPU availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

# Example usage dla NASZEGO Agent Zero V1
if __name__ == "__main__":
    extractor = OurIntentExtractor()
    
    # Test z HotelAiOS hospitality requirements
    test_requirements = [
        "Create a multi-agent hotel booking orchestration system with real-time room availability using Neo4j knowledge graph",
        "Build automated guest check-in workflow with RabbitMQ messaging and FastAPI endpoints",
        "Develop hospitality CRM integration with multi-agent coordination for guest services optimization",
        "Design intelligent concierge system using Agent Zero V1 orchestration with web interface"
    ]
    
    async def test_extraction():
        for req in test_requirements:
            result = await extractor.extract_intent(req)
            print(f"\nRequirement: {req}")
            print(f"Primary Intent: {result.primary_intent}")
            print(f"Orchestration: {result.orchestration_strategy}")
            print(f"Suggested Agents: {result.suggested_agent_roles}")
            print(f"Complexity: {result.system_complexity_score:.2f}")
            print(f"Industry: {result.detected_industry}")
    
    asyncio.run(test_extraction())
```

---

## 🚀 IMMEDIATE ACTION PLAN dla NASZEGO Systemu

### TODAY (12:00-17:00) - Phase 1 Implementation
```bash
# 1. Setup NASZEGO Business Intelligence Layer
cd /home/ianua/projects/agent-zero-v1  # NASZE repo
git checkout -b feature/business-requirements-parser-v2

# 2. Create module structure
mkdir -p business_intelligence/{business,api,models,tests}
touch business_intelligence/__init__.py
touch business_intelligence/{business,api,models,tests}/__init__.py

# 3. Install dependencies dla NASZEGO stack
pip install spacy transformers torch pydantic nltk scikit-learn
python -m spacy download en_core_web_sm

# 4. Implement files (using code above)
# - business_intelligence/models/business_models.py
# - business_intelligence/business/intent_extractor.py  
# - business_intelligence/__init__.py
```

### TOMORROW (10 października) - Integration z NASZYM System
1. **Business Translator** - Convert intents to technical specs dla NASZEGO stack
2. **Requirements Parser** - Main orchestrator integrating z NASZYMI components  
3. **API Endpoints** - FastAPI integration z NASZYM existing API layer
4. **Neo4j Integration** - Store requirements w NASZEJ knowledge graph
5. **RabbitMQ Integration** - Real-time updates przez NASZE message queue
6. **Testing** - Integration tests z NASZYM existing system

### Key Integration Points z NASZYM Agent Zero V1:
- **Neo4j Knowledge Graph** - NASZE existing infrastructure
- **RabbitMQ Messaging** - NASZE existing message system  
- **Agent Executor** - NASZE core agent execution engine
- **FastAPI** - NASZE existing API layer
- **Docker** - NASZE containerization setup

Czy chcesz żebym kontynuował z implementacją kolejnych komponentów dla NASZEGO Agent Zero V1 systemu?