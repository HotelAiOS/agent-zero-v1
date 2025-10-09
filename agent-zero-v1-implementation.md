# Agent Zero V1 - Business Requirements Parser Implementation Guide
**Developer A Action Plan - 9 października 2025, 12:00 CEST**

## 🎯 Projekt Overview - NASZ Agent Zero V1

Przepraszam za pomyłkę! Teraz wiem że pracujemy nad projektem **HotelAiOS/agent-zero-v1** - to jest Twój własny multi-agentowy projekt, nie oficjalny Agent Zero framework.

### Kontekst Naszego Projektu:
- **Repository:** `https://github.com/HotelAiOS/agent-zero-v1`
- **Tech Stack:** Python 3.11, Neo4j, RabbitMQ, Ollama, FastAPI, Docker
- **Current Phase:** Infrastructure 100% operational, V2.0 Intelligence Layer development started
- **Your Role:** Developer A (Backend) - Business Requirements Parser implementation

### Status Obecny:
✅ **Wszystkie infrastructure blokery resolved** (wczoraj 8 października)
✅ **Neo4j** - w pełni operacyjna
✅ **RabbitMQ** - message queue aktywny
✅ **AgentExecutor** - signatures naprawione
✅ **WebSocket** - real-time monitoring działa

## 🏗️ DIRECTORY STRUCTURE - Bazując na Istniejącym Kodzie

Zakładając standardową strukturę Python projektu multi-agentowego:

```
agent-zero-v1/
├── src/                           # Main source code
│   ├── core/                      # Core agent system
│   │   ├── agent_executor.py      # ✅ Fixed and operational
│   │   └── ...
│   ├── agents/                    # Agent implementations
│   ├── api/                       # FastAPI endpoints
│   └── utils/
├── shared/                        # Shared components
│   ├── knowledge/                 # Neo4j integration
│   │   └── neo4j_client.py       # ✅ Working
│   ├── orchestration/            # System orchestration
│   │   ├── project_orchestrator.py
│   │   └── task_decomposer.py    # ✅ JSON parsing fixed
│   └── messaging/                 # RabbitMQ integration
├── business_intelligence/         # 🆕 NASZ NOWY MODUŁ V2.0
│   ├── __init__.py
│   ├── business/                  # Core business logic
│   │   ├── __init__.py
│   │   ├── requirements_parser.py # Main orchestrator
│   │   ├── intent_extractor.py   # NLP processing
│   │   ├── context_enricher.py   # Context understanding
│   │   ├── business_translator.py # Business → Technical
│   │   └── validators.py         # Input validation
│   ├── api/                      # API integration
│   │   ├── __init__.py
│   │   ├── business_endpoints.py # REST endpoints
│   │   └── streaming_endpoints.py # WebSocket integration
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   ├── business_models.py    # Input schemas
│   │   └── technical_specs.py   # Output specifications
│   └── tests/                    # Testing framework
│       ├── __init__.py
│       ├── test_requirements_parser.py
│       ├── test_intent_extractor.py
│       └── test_business_translator.py
├── tests/                        # Global tests
│   └── test_full_integration.py  # ✅ 5/5 tests PASSED
├── docker-compose.yml            # ✅ All services operational
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📦 SETUP & DEPENDENCIES

### 1. Repository Setup
```bash
# Navigate to your project
cd /home/ianua/projects/agent-zero-v1

# Create feature branch
git checkout -b feature/business-requirements-parser

# Create our new module directory
mkdir -p business_intelligence/{business,api,models,tests}

# Create __init__.py files
touch business_intelligence/__init__.py
touch business_intelligence/{business,api,models,tests}/__init__.py
```

### 2. Update requirements.txt
Dodaj do istniejącego requirements.txt:
```txt
# V2.0 Intelligence Layer Dependencies
spacy>=3.6.0
transformers>=4.30.0
torch>=2.0.0
pydantic>=2.0.0
nltk>=3.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
aiofiles>=23.1.0
```

### 3. Install Dependencies
```bash
# Activate your environment (assuming you have one)
# conda activate AgentZero  # or your environment name

# Install new dependencies
pip install spacy transformers torch pydantic nltk scikit-learn numpy pandas aiofiles

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 📄 COMPLETE FILE IMPLEMENTATIONS

### 1. business_intelligence/__init__.py
```python
"""
Agent Zero V2.0 Intelligence Layer
Business Requirements Parser Module

This module extends the existing Agent Zero V1 multi-agent platform
with natural language processing capabilities to convert business
requirements into structured technical specifications.

Integration Points:
- Neo4j Knowledge Graph (shared/knowledge/neo4j_client.py)
- RabbitMQ Messaging (shared/messaging/)  
- AgentExecutor (src/core/agent_executor.py)
- ProjectOrchestrator (shared/orchestration/project_orchestrator.py)
"""

__version__ = "2.0.0"
__author__ = "HotelAiOS Agent Zero V1 Team"

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
Input structures for natural language business requirements
Compatible with Agent Zero V1 architecture
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from enum import Enum
import uuid
from datetime import datetime

class IndustryType(str, Enum):
    """Business industry categories for Agent Zero V1 context"""
    ECOMMERCE = "e-commerce"
    FINTECH = "fintech" 
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERPRISE = "enterprise"
    SAAS = "saas"
    HOSPITALITY = "hospitality"  # For HotelAiOS context
    TRAVEL = "travel"
    LOGISTICS = "logistics"
    MANUFACTURING = "manufacturing"

class ProjectScale(str, Enum):
    """Project complexity for Agent Zero V1 orchestration"""
    SMALL = "small"          # 1-3 months, single agent
    MEDIUM = "medium"        # 3-6 months, multi-agent
    LARGE = "large"          # 6-12 months, complex orchestration
    ENTERPRISE = "enterprise" # 12+ months, full platform

class BusinessContext(BaseModel):
    """Enhanced context for Agent Zero V1 integration"""
    industry: Optional[IndustryType] = None
    scale: Optional[ProjectScale] = None
    timeline: Optional[str] = Field(None, description="Expected timeline")
    budget_range: Optional[str] = Field(None, description="Budget indication")
    existing_systems: List[str] = Field(default_factory=list)
    target_users: Optional[str] = None
    business_goals: List[str] = Field(default_factory=list)
    
    # Agent Zero V1 specific fields
    agent_count_estimate: Optional[int] = Field(None, description="Estimated number of agents needed")
    orchestration_complexity: Optional[str] = Field(None, description="simple|medium|complex")
    neo4j_integration: bool = Field(default=False, description="Requires knowledge graph")
    realtime_processing: bool = Field(default=False, description="Requires RabbitMQ messaging")

class BusinessRequirement(BaseModel):
    """Main business requirement input - Agent Zero V1 compatible"""
    
    # Core requirement
    business_requirement: str = Field(..., min_length=10, description="Natural language business requirement")
    
    # Context and metadata
    context: Optional[BusinessContext] = None
    constraints: List[str] = Field(default_factory=list)
    priority_level: int = Field(default=3, ge=1, le=5)  # 1=highest
    
    # Agent Zero V1 integration metadata
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submitted_by: Optional[str] = None
    submission_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Multi-agent orchestration hints
    requires_coordination: bool = Field(default=False, description="Multiple agents needed")
    data_intensive: bool = Field(default=False, description="Heavy data processing")
    user_facing: bool = Field(default=True, description="User interface required")

class IntentClassification(BaseModel):
    """NLP analysis result for Agent Zero V1 orchestration"""
    
    # Classification results
    primary_intent: str = Field(..., description="Main classified intent")
    secondary_intents: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Extracted information
    extracted_entities: Dict[str, str] = Field(default_factory=dict)
    business_domain: Optional[IndustryType] = None
    technical_keywords: List[str] = Field(default_factory=list)
    
    # Agent Zero V1 orchestration hints
    suggested_agent_types: List[str] = Field(default_factory=list, description="Recommended agent specializations")
    orchestration_pattern: Optional[str] = Field(None, description="workflow|pipeline|broadcast|hierarchical")
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="System complexity estimate")
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = "agent-zero-v1-nlp"

# Agent Zero V1 specific intent categories
AGENT_ZERO_INTENT_CATEGORIES = [
    "multi_agent_workflow",
    "data_pipeline_orchestration", 
    "real_time_processing",
    "knowledge_graph_integration",
    "user_interface_system",
    "api_service_mesh",
    "automated_decision_system",
    "monitoring_dashboard",
    "integration_platform",
    "orchestration_system"
]

# Agent specialization mapping for our system
AGENT_SPECIALIZATIONS = {
    "data_processing": ["DataAnalyzer", "DataValidator", "DataTransformer"],
    "user_interface": ["UIGenerator", "UXOptimizer", "ResponseHandler"],
    "integration": ["APIConnector", "DataBridge", "SystemIntegrator"],
    "orchestration": ["TaskCoordinator", "WorkflowManager", "ResourceAllocator"],
    "monitoring": ["PerformanceMonitor", "HealthChecker", "AlertManager"]
}

# Neo4j integration patterns
NEO4J_PATTERNS = {
    "knowledge_management": ["document", "search", "recommendation"],
    "relationship_analysis": ["connection", "network", "dependency"],
    "data_modeling": ["entity", "classification", "hierarchy"]
}
```

### 3. business_intelligence/business/requirements_parser.py
```python
"""
Business Requirements Parser - Main Orchestrator
Integration with Agent Zero V1 multi-agent architecture
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Agent Zero V1 integrations
try:
    from src.core.agent_executor import AgentExecutor
    from shared.knowledge.neo4j_client import Neo4jClient
    from shared.orchestration.project_orchestrator import ProjectOrchestrator
except ImportError as e:
    logging.warning(f"Agent Zero V1 components not available: {e}")
    # Fallback for development/testing
    AgentExecutor = None
    Neo4jClient = None
    ProjectOrchestrator = None

# Our new V2.0 components
from .intent_extractor import IntentExtractor
from .context_enricher import ContextEnricher
from .business_translator import BusinessTranslator
from .validators import RequirementValidator

from ..models.business_models import BusinessRequirement, IntentClassification
from ..models.technical_specs import TechnicalSpecification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessRequirementsParser:
    """
    Main orchestrator for Business Requirements Parser
    Integrates with Agent Zero V1 multi-agent architecture
    """
    
    def __init__(self, 
                 neo4j_client: Optional[Neo4jClient] = None,
                 agent_executor: Optional[AgentExecutor] = None,
                 project_orchestrator: Optional[ProjectOrchestrator] = None):
        """
        Initialize Business Requirements Parser
        
        Args:
            neo4j_client: Agent Zero V1 Neo4j knowledge graph client
            agent_executor: Agent Zero V1 agent execution system
            project_orchestrator: Agent Zero V1 project coordination system
        """
        
        # Agent Zero V1 integrations
        self.neo4j_client = neo4j_client
        self.agent_executor = agent_executor
        self.project_orchestrator = project_orchestrator
        
        # V2.0 Intelligence Layer components
        self.intent_extractor = IntentExtractor()
        self.context_enricher = ContextEnricher()
        self.business_translator = BusinessTranslator()
        self.validator = RequirementValidator()
        
        # Processing metrics
        self.processing_stats = {
            "total_processed": 0,
            "success_rate": 0.0,
            "avg_processing_time": 0.0
        }
        
        logger.info("BusinessRequirementsParser initialized with Agent Zero V1 integration")
    
    async def process_requirement(self, requirement: BusinessRequirement) -> Dict[str, Any]:
        """
        Main processing pipeline for business requirements
        Integrates with Agent Zero V1 orchestration system
        
        Args:
            requirement: Business requirement to process
            
        Returns:
            Complete processing result with technical specification
        """
        start_time = datetime.utcnow()
        processing_id = requirement.request_id
        
        try:
            logger.info(f"Processing requirement {processing_id}")
            
            # Step 1: Validate input requirement
            validation_result = await self._validate_requirement(requirement)
            if not validation_result.is_valid:
                return self._create_error_response(
                    processing_id, 
                    "Validation failed", 
                    validation_result.issues
                )
            
            # Step 2: Enrich context
            enriched_requirement = self.context_enricher.enrich_context(requirement)
            
            # Step 3: Extract intent and classify
            intent_result = self.intent_extractor.extract_intent(
                enriched_requirement.business_requirement
            )
            
            # Step 4: Store in Neo4j knowledge graph (if available)
            if self.neo4j_client:
                await self._store_requirement_knowledge(enriched_requirement, intent_result)
            
            # Step 5: Translate to technical specification
            tech_spec = await self.business_translator.translate_to_technical(
                enriched_requirement, 
                intent_result
            )
            
            # Step 6: Agent Zero V1 orchestration planning
            orchestration_plan = await self._create_orchestration_plan(
                enriched_requirement, 
                intent_result, 
                tech_spec
            )
            
            # Step 7: Create execution plan for Agent Zero V1
            execution_plan = await self._create_execution_plan(
                tech_spec,
                orchestration_plan
            )
            
            # Calculate processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update stats
            self._update_processing_stats(processing_time, True)
            
            # Return complete result
            return {
                "processing_id": processing_id,
                "status": "success",
                "processing_time_ms": processing_time,
                "business_requirement": enriched_requirement,
                "intent_classification": intent_result,
                "technical_specification": tech_spec,
                "orchestration_plan": orchestration_plan,
                "execution_plan": execution_plan,
                "agent_zero_v1_ready": True
            }
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_processing_stats(processing_time, False)
            
            logger.error(f"Processing failed for {processing_id}: {e}")
            return self._create_error_response(
                processing_id, 
                "Processing failed", 
                [str(e)]
            )
    
    async def _validate_requirement(self, requirement: BusinessRequirement):
        """Validate business requirement input"""
        return self.validator.validate(requirement)
    
    async def _store_requirement_knowledge(self, 
                                         requirement: BusinessRequirement, 
                                         intent: IntentClassification):
        """Store requirement and intent in Neo4j knowledge graph"""
        if not self.neo4j_client:
            return
        
        try:
            # Create requirement node
            req_query = """
            CREATE (req:BusinessRequirement {
                id: $req_id,
                text: $requirement_text,
                primary_intent: $primary_intent,
                confidence: $confidence_score,
                industry: $industry,
                scale: $scale,
                timestamp: datetime()
            })
            RETURN req.id
            """
            
            await self.neo4j_client.run(req_query, {
                "req_id": requirement.request_id,
                "requirement_text": requirement.business_requirement,
                "primary_intent": intent.primary_intent,
                "confidence_score": intent.confidence_score,
                "industry": requirement.context.industry.value if requirement.context and requirement.context.industry else None,
                "scale": requirement.context.scale.value if requirement.context and requirement.context.scale else None
            })
            
            logger.info(f"Stored requirement {requirement.request_id} in Neo4j")
            
        except Exception as e:
            logger.warning(f"Failed to store in Neo4j: {e}")
    
    async def _create_orchestration_plan(self, 
                                       requirement: BusinessRequirement,
                                       intent: IntentClassification,
                                       tech_spec: TechnicalSpecification) -> Dict[str, Any]:
        """Create Agent Zero V1 orchestration plan"""
        
        orchestration_plan = {
            "plan_id": f"orch_{requirement.request_id[:8]}",
            "orchestration_type": intent.orchestration_pattern or "hierarchical",
            "estimated_agents": len(intent.suggested_agent_types) or 3,
            "coordination_pattern": "supervisor" if requirement.requires_coordination else "simple",
            "data_flow": "streaming" if requirement.data_intensive else "batch",
            "ui_integration": requirement.user_facing,
            "neo4j_operations": requirement.context.neo4j_integration if requirement.context else False,
            "realtime_messaging": requirement.context.realtime_processing if requirement.context else False,
            "agent_specializations": intent.suggested_agent_types,
            "execution_stages": [
                {
                    "stage": "preparation",
                    "agents": ["ResourceAllocator", "ConfigurationManager"],
                    "duration_estimate": "5-10 minutes"
                },
                {
                    "stage": "implementation", 
                    "agents": intent.suggested_agent_types,
                    "duration_estimate": tech_spec.resource_estimate.estimated_timeline
                },
                {
                    "stage": "integration",
                    "agents": ["SystemIntegrator", "TestCoordinator"],
                    "duration_estimate": "20-30% of implementation time"
                }
            ]
        }
        
        return orchestration_plan
    
    async def _create_execution_plan(self, 
                                   tech_spec: TechnicalSpecification,
                                   orchestration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan for Agent Zero V1"""
        
        execution_plan = {
            "plan_id": f"exec_{tech_spec.specification_id[:8]}",
            "total_estimated_hours": tech_spec.resource_estimate.total_estimated_hours,
            "parallel_execution": orchestration_plan["estimated_agents"] > 1,
            "critical_path": [
                {
                    "task": "Architecture Setup",
                    "agents": ["ArchitectureAgent"],
                    "dependencies": [],
                    "estimated_hours": max(4, tech_spec.resource_estimate.total_estimated_hours * 0.1)
                },
                {
                    "task": "Core Implementation",
                    "agents": orchestration_plan["agent_specializations"], 
                    "dependencies": ["Architecture Setup"],
                    "estimated_hours": tech_spec.resource_estimate.total_estimated_hours * 0.6
                },
                {
                    "task": "Integration & Testing",
                    "agents": ["IntegrationAgent", "TestAgent"],
                    "dependencies": ["Core Implementation"],
                    "estimated_hours": tech_spec.resource_estimate.total_estimated_hours * 0.3
                }
            ],
            "resource_requirements": {
                "neo4j_access": orchestration_plan["neo4j_operations"],
                "rabbitmq_channels": orchestration_plan["realtime_messaging"],
                "api_endpoints": len(tech_spec.features) * 2,  # Rough estimate
                "database_tables": len([f for f in tech_spec.features if "data" in f.name.lower()])
            },
            "success_criteria": tech_spec.compliance_requirements,
            "rollback_plan": {
                "checkpoints": ["Architecture", "Core Features", "Integration"],
                "automated_rollback": True,
                "recovery_time": "< 30 minutes"
            }
        }
        
        return execution_plan
    
    def _create_error_response(self, processing_id: str, error: str, details: List[str]) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "processing_id": processing_id,
            "status": "error",
            "error": error,
            "details": details,
            "agent_zero_v1_ready": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _update_processing_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        self.processing_stats["total_processed"] += 1
        
        if success:
            # Update success rate
            total = self.processing_stats["total_processed"]
            current_successes = self.processing_stats["success_rate"] * (total - 1)
            self.processing_stats["success_rate"] = (current_successes + 1) / total
        
        # Update average processing time
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["avg_processing_time"]
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()

# Example usage for testing
async def main():
    """Test the Business Requirements Parser"""
    parser = BusinessRequirementsParser()
    
    test_requirement = BusinessRequirement(
        business_requirement="Create a hotel booking system that integrates with our existing CRM and provides real-time availability updates to customers",
        context=BusinessContext(
            industry=IndustryType.HOSPITALITY,
            scale=ProjectScale.MEDIUM,
            timeline="4 months",
            existing_systems=["CRM", "Payment Gateway"],
            realtime_processing=True,
            neo4j_integration=True
        )
    )
    
    result = await parser.process_requirement(test_requirement)
    print(f"Processing result: {result['status']}")
    print(f"Orchestration plan: {result.get('orchestration_plan', {}).get('orchestration_type')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. business_intelligence/business/intent_extractor.py
```python
"""
Intent Extraction Engine for Agent Zero V1
NLP processing optimized for multi-agent orchestration
"""

import spacy
from transformers import pipeline
from typing import Dict, List, Optional, Tuple
import re
import time
import logging

from ..models.business_models import (
    IntentClassification, 
    IndustryType, 
    AGENT_ZERO_INTENT_CATEGORIES,
    AGENT_SPECIALIZATIONS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentExtractor:
    """
    NLP-powered intent extraction optimized for Agent Zero V1 multi-agent system
    """
    
    def __init__(self):
        """Initialize NLP models for Agent Zero V1 integration"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize classifier for Agent Zero V1 specific intents
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self._gpu_available() else -1
            )
            
            # Agent Zero V1 specific patterns
            self.orchestration_patterns = {
                "workflow": ["sequence", "step", "process", "pipeline", "flow"],
                "hierarchical": ["manage", "coordinate", "oversee", "supervise"],
                "broadcast": ["notify", "alert", "broadcast", "announce"],
                "pipeline": ["data", "transform", "process", "analyze"]
            }
            
            self.complexity_indicators = {
                "low": ["simple", "basic", "straightforward", "minimal"],
                "medium": ["standard", "typical", "moderate", "normal"],
                "high": ["complex", "advanced", "sophisticated", "comprehensive"],
                "very_high": ["enterprise", "massive", "distributed", "scalable"]
            }
            
            logger.info("IntentExtractor initialized for Agent Zero V1")
            
        except Exception as e:
            logger.error(f"Failed to initialize IntentExtractor: {e}")
            raise
    
    def extract_intent(self, requirement_text: str) -> IntentClassification:
        """
        Extract business intent optimized for Agent Zero V1 orchestration
        
        Args:
            requirement_text: Natural language business requirement
            
        Returns:
            IntentClassification with Agent Zero V1 specific recommendations
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            clean_text = self._preprocess_text(requirement_text)
            
            # Extract entities using spaCy
            doc = self.nlp(clean_text)
            entities = self._extract_entities(doc)
            
            # Classify intent for Agent Zero V1
            classification = self.classifier(clean_text, AGENT_ZERO_INTENT_CATEGORIES)
            
            # Determine orchestration pattern
            orchestration_pattern = self._determine_orchestration_pattern(clean_text)
            
            # Suggest agent specializations
            suggested_agents = self._suggest_agent_types(clean_text, classification['labels'][0])
            
            # Calculate complexity
            complexity_score = self._calculate_complexity(clean_text, entities)
            
            # Detect industry/domain
            detected_industry = self._detect_industry(clean_text)
            
            # Extract technical keywords
            tech_keywords = self._extract_technical_keywords(clean_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return IntentClassification(
                primary_intent=classification['labels'][0],
                secondary_intents=classification['labels'][1:3],
                confidence_score=classification['scores'][0],
                extracted_entities=entities,
                business_domain=detected_industry,
                technical_keywords=tech_keywords,
                suggested_agent_types=suggested_agents,
                orchestration_pattern=orchestration_pattern,
                complexity_score=complexity_score,
                processing_time_ms=processing_time,
                model_version="agent-zero-v1-nlp-1.0"
            )
            
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            # Return fallback for Agent Zero V1
            return IntentClassification(
                primary_intent="multi_agent_workflow",
                secondary_intents=["orchestration_system"],
                confidence_score=0.1,
                extracted_entities={},
                suggested_agent_types=["TaskCoordinator", "WorkflowManager"],
                orchestration_pattern="hierarchical",
                complexity_score=0.5,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _determine_orchestration_pattern(self, text: str) -> Optional[str]:
        """Determine best orchestration pattern for Agent Zero V1"""
        text_lower = text.lower()
        pattern_scores = {}
        
        for pattern, keywords in self.orchestration_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                pattern_scores[pattern] = score
        
        if not pattern_scores:
            return "hierarchical"  # Default for Agent Zero V1
        
        return max(pattern_scores.items(), key=lambda x: x[1])[0]
    
    def _suggest_agent_types(self, text: str, primary_intent: str) -> List[str]:
        """Suggest Agent Zero V1 agent specializations"""
        text_lower = text.lower()
        suggested = []
        
        # Map intent to agent categories
        intent_to_agents = {
            "multi_agent_workflow": ["TaskCoordinator", "WorkflowManager"],
            "data_pipeline_orchestration": ["DataAnalyzer", "DataTransformer"],
            "real_time_processing": ["StreamProcessor", "EventHandler"],
            "knowledge_graph_integration": ["KnowledgeManager", "GraphAnalyzer"],
            "user_interface_system": ["UIGenerator", "UXOptimizer"],
            "api_service_mesh": ["APIConnector", "ServiceMesh"],
            "orchestration_system": ["SystemOrchestrator", "ResourceManager"]
        }
        
        # Get agents for primary intent
        if primary_intent in intent_to_agents:
            suggested.extend(intent_to_agents[primary_intent])
        
        # Add agents based on text content
        if any(word in text_lower for word in ["data", "analysis", "processing"]):
            suggested.extend(["DataAnalyzer", "DataValidator"])
        
        if any(word in text_lower for word in ["ui", "interface", "frontend", "dashboard"]):
            suggested.extend(["UIGenerator", "ResponseHandler"])
        
        if any(word in text_lower for word in ["integrate", "connect", "api"]):
            suggested.extend(["APIConnector", "SystemIntegrator"])
        
        if any(word in text_lower for word in ["monitor", "track", "alert"]):
            suggested.extend(["PerformanceMonitor", "AlertManager"])
        
        # Remove duplicates and limit to reasonable number
        suggested = list(set(suggested))[:6]
        
        # Ensure we have at least basic agents
        if not suggested:
            suggested = ["TaskCoordinator", "WorkflowManager", "SystemIntegrator"]
        
        return suggested
    
    def _calculate_complexity(self, text: str, entities: Dict[str, str]) -> float:
        """Calculate system complexity score for Agent Zero V1 planning"""
        complexity_score = 0.5  # Base complexity
        text_lower = text.lower()
        
        # Adjust based on complexity indicators
        for level, indicators in self.complexity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                level_scores = {"low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 1.0}
                complexity_score = max(complexity_score, level_scores[level])
        
        # Adjust based on entities and technical terms
        tech_terms = len(entities.get("technology", []))
        if tech_terms > 3:
            complexity_score += 0.1
        
        # Adjust based on integrations
        if "integration" in entities or any(word in text_lower for word in ["integrate", "connect", "sync"]):
            complexity_score += 0.15
        
        # Adjust based on real-time requirements
        if any(word in text_lower for word in ["real-time", "live", "instant", "streaming"]):
            complexity_score += 0.1
        
        # Adjust based on scale indicators
        if any(word in text_lower for word in ["enterprise", "scalable", "distributed", "multi-tenant"]):
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)  # Cap at 1.0
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for Agent Zero V1 processing"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize Agent Zero V1 specific terms
        replacements = {
            'multi-agent': 'multi agent',
            'real-time': 'realtime',
            'end-to-end': 'end to end',
            'ai': 'artificial intelligence',
            'ml': 'machine learning'
        }
        
        for old, new in replacements.items():
            cleaned = re.sub(rf'\b{old}\b', new, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _extract_entities(self, doc) -> Dict[str, str]:
        """Extract entities relevant to Agent Zero V1 orchestration"""
        entities = {}
        
        # Standard NER entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "MONEY", "PERCENT", "DATE", "TIME"]:
                key = ent.label_.lower()
                if key not in entities:
                    entities[key] = []
                entities[key].append(ent.text)
        
        # Agent Zero V1 specific technical terms
        agent_zero_terms = [
            "neo4j", "rabbitmq", "fastapi", "docker", "orchestration",
            "multi-agent", "knowledge graph", "message queue", "microservice"
        ]
        
        found_terms = []
        text_lower = doc.text.lower()
        for term in agent_zero_terms:
            if term in text_lower:
                found_terms.append(term)
        
        if found_terms:
            entities["agent_zero_tech"] = found_terms
        
        # Business processes relevant to multi-agent systems
        process_terms = [
            "orchestration", "coordination", "automation", "integration", 
            "workflow", "pipeline", "processing", "monitoring"
        ]
        
        found_processes = []
        for term in process_terms:
            if term in text_lower:
                found_processes.append(term)
        
        if found_processes:
            entities["processes"] = found_processes
        
        return entities
    
    def _detect_industry(self, text: str) -> Optional[IndustryType]:
        """Detect industry with focus on HotelAiOS use cases"""
        text_lower = text.lower()
        
        # Enhanced industry patterns for HotelAiOS context
        industry_patterns = {
            IndustryType.HOSPITALITY: [
                "hotel", "booking", "reservation", "guest", "room", "hospitality",
                "check-in", "check-out", "concierge", "accommodation"
            ],
            IndustryType.TRAVEL: [
                "travel", "trip", "flight", "airline", "tourism", "vacation",
                "itinerary", "destination", "transport"
            ],
            IndustryType.ECOMMERCE: [
                "shop", "store", "product", "inventory", "cart", "payment",
                "checkout", "order", "customer", "catalog"
            ],
            IndustryType.FINTECH: [
                "finance", "payment", "banking", "transaction", "money",
                "wallet", "trading", "investment", "fintech"
            ]
        }
        
        industry_scores = {}
        
        for industry, keywords in industry_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                industry_scores[industry] = score
        
        if not industry_scores:
            return None
        
        # Return industry with highest score if confident
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        if best_industry[1] >= 2:  # Require at least 2 matches
            return best_industry[0]
        
        return None
    
    def _extract_technical_keywords(self, text: str) -> List[str]:
        """Extract technical keywords relevant to Agent Zero V1"""
        text_lower = text.lower()
        
        # Agent Zero V1 relevant technologies
        tech_keywords = [
            # Core stack
            "python", "neo4j", "rabbitmq", "fastapi", "docker", "ollama",
            # Multi-agent concepts
            "orchestration", "coordination", "multi-agent", "workflow",
            # AI/ML
            "artificial intelligence", "machine learning", "nlp", "llm",
            # Architecture patterns
            "microservices", "api", "rest", "graphql", "websocket",
            # Data processing
            "pipeline", "streaming", "batch", "realtime", "event-driven",
            # Infrastructure
            "kubernetes", "containerization", "cloud", "scaling"
        ]
        
        found_keywords = []
        for keyword in tech_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _gpu_available(self) -> bool:
        """Check GPU availability for transformers"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

# Example usage for Agent Zero V1
if __name__ == "__main__":
    extractor = IntentExtractor()
    
    # Test with hotel/hospitality focused requirements
    test_requirements = [
        "Create a multi-agent hotel booking system with real-time room availability and Neo4j integration",
        "Build an orchestrated workflow for processing guest check-ins using RabbitMQ messaging",
        "Develop API-based integration between hotel management system and payment processing",
        "Design automated guest experience system with AI-powered concierge services"
    ]
    
    for req in test_requirements:
        result = extractor.extract_intent(req)
        print(f"\nRequirement: {req}")
        print(f"Primary Intent: {result.primary_intent}")
        print(f"Orchestration: {result.orchestration_pattern}")
        print(f"Suggested Agents: {result.suggested_agent_types}")
        print(f"Complexity: {result.complexity_score:.2f}")
```

---

## 🚀 IMPLEMENTATION ACTION PLAN

### Phase 1: TODAY (12:00-17:00)
1. **Setup & Structure (30 min)**
   ```bash
   cd /home/ianua/projects/agent-zero-v1
   git checkout -b feature/business-requirements-parser
   mkdir -p business_intelligence/{business,api,models,tests}
   ```

2. **Dependencies & Models (45 min)**
   - Update requirements.txt
   - Install NLP dependencies
   - Create data models (`business_models.py`)

3. **Core Components (3h)**
   - Implement `IntentExtractor` (1.5h)
   - Implement `ContextEnricher` (1h)
   - Start `BusinessRequirementsParser` (30min)

4. **Basic Testing (30min)**
   - Test intent extraction with sample requirements
   - Verify NLP models working

### Phase 2: TOMORROW (10 października)
1. **Business Translation (2h)**
   - Complete `BusinessTranslator` 
   - Technical specification generation
   - Agent Zero V1 orchestration planning

2. **API Integration (2h)**
   - REST endpoints
   - WebSocket integration
   - FastAPI integration with existing system

3. **Agent Zero V1 Integration (2h)**
   - Neo4j knowledge storage
   - AgentExecutor communication
   - ProjectOrchestrator coordination

4. **Testing & Validation (2h)**
   - End-to-end integration tests
   - Performance validation (<500ms)
   - Agent Zero V1 compatibility testing

### Integration Points with Your Existing System:
1. **Neo4j Integration** - Store business requirements and technical specs
2. **AgentExecutor** - Execute generated technical specifications
3. **RabbitMQ** - Real-time progress updates
4. **ProjectOrchestrator** - Coordinate multi-agent execution
5. **FastAPI** - New endpoints for Business Requirements Parser

Czy chcesz żebym kontynuował z kolejnymi plikami (`business_translator.py`, `validators.py`, API endpoints) czy masz pytania o obecną implementację?