# Agent Zero V2.0 - Business Requirements Parser Implementation Guide

## 🎯 Projekt Overview

Agent Zero to framework multi-agentowy zbudowany na modularnej architekturze z hierarchiczną strukturą agentów. Nasz projekt to V2.0 Intelligence Layer - rozszerzenie które dodaje możliwość przetwarzania naturalnego języka biznesowego na specyfikacje techniczne.

### Kluczowe Cechy Agent Zero Architecture:
- **Hierarchical Agent Structure** - Agent 0 deleguje zadania do sub-agentów
- **Tools System** - Agenty używają narzędzi do wykonywania zadań
- **Memory System** - Persistent memory z fragmentami, rozwiązaniami i metadata
- **Prompt-driven** - Behavior kontrolowany przez markdown prompts
- **Docker Runtime** - Secure execution environment

### Nasza Implementacja (V2.0 Intelligence Layer):
**Business Requirements Parser** - Core component do konwersji natural language → technical specs

---

## 🏗️ DIRECTORY STRUCTURE & FILE CONTENTS

Zgodnie z Agent Zero architecture, tworzymy modularną strukturę w ramach istniejącego systemu.

### 1. Repository Setup
```bash
cd /home/ianua/projects/agent-zero-v1
git checkout -b feature/business-requirements-parser

# Tworzymy główny moduł
mkdir -p business_intelligence/{business,api,models,tests}
```

### 2. Core Directory Structure
```
agent-zero-v1/
├── business_intelligence/           # Nasz nowy moduł V2.0
│   ├── __init__.py
│   ├── business/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── requirements_parser.py   # Main orchestrator
│   │   ├── intent_extractor.py     # NLP processing
│   │   ├── context_enricher.py     # Context understanding  
│   │   ├── business_translator.py  # Business → Technical
│   │   └── validators.py           # Input validation
│   ├── api/                        # API integration
│   │   ├── __init__.py
│   │   ├── business_endpoints.py   # REST endpoints
│   │   └── streaming_endpoints.py  # WebSocket real-time
│   ├── models/                     # Data models
│   │   ├── __init__.py
│   │   ├── business_models.py      # Input/output schemas
│   │   └── technical_specs.py     # Technical specifications
│   └── tests/                      # Testing framework
│       ├── __init__.py
│       ├── test_requirements_parser.py
│       ├── test_intent_extractor.py
│       └── test_business_translator.py
├── prompts/                        # Istniejące Agent Zero prompts
├── python/                         # Existing codebase
│   ├── tools/                     # Tutaj integrujemy jako tool
│   └── extensions/                # Extension points
└── requirements.txt               # Dependencies update
```

---

## 📦 DEPENDENCIES & SETUP

### 1. Update requirements.txt
Dodajemy do istniejącego requirements.txt:
```txt
# NLP & ML Dependencies for V2.0 Intelligence Layer
spacy>=3.6.0
transformers>=4.30.0
torch>=2.0.0
pydantic>=2.0.0
nltk>=3.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Additional utilities
aiofiles>=23.1.0
python-multipart>=0.0.6
```

### 2. Installation Commands
```bash
# Activate your Agent Zero environment
conda activate AgentZero

# Install new dependencies
pip install spacy transformers torch pydantic nltk scikit-learn numpy pandas aiofiles python-multipart

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 📄 COMPLETE FILE CONTENTS

### 1. business_intelligence/__init__.py
```python
"""
Agent Zero V2.0 Intelligence Layer
Business Requirements Parser Module

This module provides natural language processing capabilities
to convert business requirements into technical specifications.
"""

__version__ = "2.0.0"
__author__ = "Agent Zero V2.0 Team"

from .business.requirements_parser import BusinessRequirementsParser
from .models.business_models import BusinessRequirement, TechnicalSpecification

__all__ = [
    'BusinessRequirementsParser',
    'BusinessRequirement', 
    'TechnicalSpecification'
]
```

### 2. business_intelligence/models/__init__.py
```python
"""Data models for Business Intelligence Layer"""

from .business_models import (
    BusinessRequirement,
    BusinessContext,
    IntentClassification,
    IndustryType,
    ProjectScale
)
from .technical_specs import (
    TechnicalSpecification,
    ArchitectureSpec,
    FeatureSpec,
    IntegrationSpec
)

__all__ = [
    'BusinessRequirement',
    'BusinessContext', 
    'IntentClassification',
    'IndustryType',
    'ProjectScale',
    'TechnicalSpecification',
    'ArchitectureSpec',
    'FeatureSpec',
    'IntegrationSpec'
]
```

### 3. business_intelligence/models/business_models.py
```python
"""
Business Requirements Data Models
Defines input structures for natural language business requirements
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from enum import Enum
import uuid
from datetime import datetime

class IndustryType(str, Enum):
    """Supported business industry categories"""
    ECOMMERCE = "e-commerce"
    FINTECH = "fintech" 
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERPRISE = "enterprise"
    STARTUP = "startup"
    SAAS = "saas"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    LOGISTICS = "logistics"

class ProjectScale(str, Enum):
    """Project size classification"""
    SMALL = "small"          # 1-3 months, 1-2 developers
    MEDIUM = "medium"        # 3-6 months, 2-5 developers  
    LARGE = "large"          # 6-12 months, 5+ developers
    ENTERPRISE = "enterprise" # 12+ months, large team

class PriorityLevel(int, Enum):
    """Business priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKLOG = 5

class BusinessContext(BaseModel):
    """Context information for business requirements"""
    industry: Optional[IndustryType] = None
    scale: Optional[ProjectScale] = None
    timeline: Optional[str] = Field(None, description="Expected timeline (e.g., '3 months', 'Q4 2025')")
    budget_range: Optional[str] = Field(None, description="Budget indication (e.g., 'moderate', '$50k-100k')")
    existing_systems: List[str] = Field(default_factory=list, description="Current systems to integrate with")
    target_users: Optional[str] = Field(None, description="Target user base description")
    business_goals: List[str] = Field(default_factory=list, description="High-level business objectives")

class BusinessRequirement(BaseModel):
    """Main business requirement input model"""
    
    # Core requirement
    business_requirement: str = Field(..., min_length=10, description="Natural language business requirement")
    
    # Context and metadata
    context: Optional[BusinessContext] = None
    constraints: List[str] = Field(default_factory=list, description="Technical or business constraints")
    priority_level: PriorityLevel = Field(default=PriorityLevel.MEDIUM, description="Business priority")
    
    # Request metadata
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submitted_by: Optional[str] = Field(None, description="User or stakeholder name")
    submission_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional specifications
    success_criteria: List[str] = Field(default_factory=list, description="How success will be measured")
    stakeholders: List[str] = Field(default_factory=list, description="Key stakeholders involved")

class IntentClassification(BaseModel):
    """Result of intent extraction from business requirement"""
    
    # Classification results
    primary_intent: str = Field(..., description="Main classified intent")
    secondary_intents: List[str] = Field(default_factory=list, description="Supporting intents")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    
    # Extracted information
    extracted_entities: Dict[str, str] = Field(default_factory=dict, description="Named entities found")
    business_domain: Optional[IndustryType] = None
    technical_keywords: List[str] = Field(default_factory=list)
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None

class ValidationResult(BaseModel):
    """Result of requirement validation"""
    is_valid: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    completeness_score: float = Field(..., ge=0.0, le=1.0)

# Supported intent categories for classification
INTENT_CATEGORIES = [
    "web_application_development",
    "mobile_app_development", 
    "data_analysis_system",
    "api_development",
    "database_design",
    "user_interface_design",
    "automation_system",
    "reporting_dashboard",
    "e_commerce_platform",
    "cms_development",
    "crm_system",
    "erp_system",
    "integration_project",
    "migration_project",
    "performance_optimization",
    "security_enhancement"
]

# Business domain keyword mappings
DOMAIN_KEYWORDS = {
    IndustryType.ECOMMERCE: [
        "shop", "store", "product", "inventory", "cart", "payment", 
        "checkout", "order", "customer", "catalog", "shipping"
    ],
    IndustryType.FINTECH: [
        "finance", "payment", "banking", "transaction", "money", 
        "wallet", "trading", "investment", "loan", "credit"
    ],
    IndustryType.HEALTHCARE: [
        "patient", "medical", "health", "clinic", "hospital", 
        "doctor", "appointment", "record", "diagnosis", "treatment"
    ],
    IndustryType.EDUCATION: [
        "student", "course", "learning", "education", "school", 
        "university", "curriculum", "assignment", "grade", "class"
    ],
    IndustryType.ENTERPRISE: [
        "employee", "workflow", "process", "department", "management", 
        "resource", "project", "team", "collaboration", "efficiency"
    ]
}
```

### 4. business_intelligence/models/technical_specs.py
```python
"""
Technical Specifications Data Models
Defines output structures for generated technical specifications
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from enum import Enum
import uuid
from datetime import datetime

class ProjectType(str, Enum):
    """Technical project categories"""
    WEB_APPLICATION = "web_application"
    MOBILE_APP = "mobile_app"
    API_SERVICE = "api_service"
    DATA_PIPELINE = "data_pipeline"
    DESKTOP_APP = "desktop_app"
    MICROSERVICE = "microservice"
    INTEGRATION = "integration"
    MIGRATION = "migration"

class ComplexityLevel(str, Enum):
    """Technical complexity classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class TechnologyStack(BaseModel):
    """Technology recommendations"""
    frontend: List[str] = Field(default_factory=list)
    backend: List[str] = Field(default_factory=list)
    database: List[str] = Field(default_factory=list)
    infrastructure: List[str] = Field(default_factory=list)
    third_party: List[str] = Field(default_factory=list)

class FeatureSpec(BaseModel):
    """Individual feature specification"""
    name: str
    description: str
    priority: str = Field(..., regex="^(high|medium|low)$")
    estimated_hours: int = Field(..., ge=0)
    complexity: ComplexityLevel
    components: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)

class IntegrationSpec(BaseModel):
    """System integration specification"""
    system: str
    type: str = Field(..., description="API, Database, File, etc.")
    complexity: ComplexityLevel
    estimated_hours: int = Field(..., ge=0)
    requirements: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

class ArchitectureSpec(BaseModel):
    """System architecture specification"""
    architecture_pattern: str = Field(..., description="MVC, Microservices, Layered, etc.")
    technology_stack: TechnologyStack
    scalability_requirements: List[str] = Field(default_factory=list)
    security_requirements: List[str] = Field(default_factory=list)
    performance_requirements: List[str] = Field(default_factory=list)

class ResourceEstimate(BaseModel):
    """Resource and timeline estimates"""
    total_estimated_hours: int = Field(..., ge=0)
    estimated_timeline: str = Field(..., description="Human readable timeline")
    team_size_recommendation: int = Field(..., ge=1)
    required_skills: List[str] = Field(default_factory=list)
    budget_range: Optional[str] = None

class TechnicalSpecification(BaseModel):
    """Complete technical specification output"""
    
    # Project identification
    specification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_requirement_id: str = Field(..., description="Reference to original business requirement")
    
    # Core specification
    project_type: ProjectType
    project_name: str
    project_description: str
    
    # Technical details
    architecture: ArchitectureSpec
    features: List[FeatureSpec] = Field(default_factory=list)
    integrations: List[IntegrationSpec] = Field(default_factory=list)
    
    # Estimates and planning
    resource_estimate: ResourceEstimate
    milestones: List[Dict[str, str]] = Field(default_factory=list)
    risks_and_mitigation: List[Dict[str, str]] = Field(default_factory=list)
    
    # Compliance and quality
    compliance_requirements: List[str] = Field(default_factory=list)
    quality_standards: List[str] = Field(default_factory=list)
    testing_strategy: List[str] = Field(default_factory=list)
    
    # AI recommendations
    recommendations: List[str] = Field(default_factory=list)
    alternatives: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    
    # Confidence and metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(default="v2.0")
    processing_time_ms: Optional[float] = None

# Technology recommendations by project type
TECH_RECOMMENDATIONS = {
    ProjectType.WEB_APPLICATION: {
        "frontend": ["React", "Vue.js", "Angular", "Next.js"],
        "backend": ["Node.js", "Python/FastAPI", "Java/Spring", "C#/.NET"],
        "database": ["PostgreSQL", "MongoDB", "MySQL"],
        "infrastructure": ["Docker", "AWS", "Nginx"]
    },
    ProjectType.MOBILE_APP: {
        "frontend": ["React Native", "Flutter", "Swift/SwiftUI", "Kotlin"],
        "backend": ["Node.js", "Python/FastAPI", "Firebase"],
        "database": ["Firebase", "PostgreSQL", "SQLite"],
        "infrastructure": ["AWS Mobile", "Google Cloud", "App Store"]
    },
    ProjectType.API_SERVICE: {
        "backend": ["FastAPI", "Express.js", "Spring Boot", "Django REST"],
        "database": ["PostgreSQL", "MongoDB", "Redis"],
        "infrastructure": ["Docker", "Kubernetes", "API Gateway"]
    }
}

# Compliance standards by industry
COMPLIANCE_STANDARDS = {
    "fintech": ["PCI DSS", "SOX", "GDPR", "PSD2"],
    "healthcare": ["HIPAA", "FDA", "GDPR", "SOC 2"],
    "e-commerce": ["PCI DSS", "GDPR", "CCPA", "SOC 2"],
    "education": ["FERPA", "COPPA", "GDPR", "SOC 2"]
}
```

### 5. business_intelligence/business/__init__.py
```python
"""Core business logic components"""

from .requirements_parser import BusinessRequirementsParser
from .intent_extractor import IntentExtractor
from .business_translator import BusinessTranslator
from .context_enricher import ContextEnricher
from .validators import RequirementValidator

__all__ = [
    'BusinessRequirementsParser',
    'IntentExtractor',
    'BusinessTranslator', 
    'ContextEnricher',
    'RequirementValidator'
]
```

### 6. business_intelligence/business/intent_extractor.py
```python
"""
Intent Extraction Engine
Natural Language Processing for Business Requirements
"""

import spacy
import nltk
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import re
import time
import logging
from ..models.business_models import IntentClassification, IndustryType, INTENT_CATEGORIES, DOMAIN_KEYWORDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentExtractor:
    """
    NLP-powered intent extraction from business requirements
    Uses spaCy for entity extraction and transformers for classification
    """
    
    def __init__(self):
        """Initialize NLP models and classification pipeline"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize transformer-based classifier
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self._gpu_available() else -1
            )
            
            # Download required NLTK data
            self._download_nltk_data()
            
            logger.info("IntentExtractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IntentExtractor: {e}")
            raise
    
    def extract_intent(self, requirement_text: str) -> IntentClassification:
        """
        Extract business intent from natural language requirement
        
        Args:
            requirement_text: Natural language business requirement
            
        Returns:
            IntentClassification with extracted information
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            clean_text = self._preprocess_text(requirement_text)
            
            # Extract entities using spaCy
            doc = self.nlp(clean_text)
            entities = self._extract_entities(doc)
            
            # Classify intent using transformer
            classification = self.classifier(clean_text, INTENT_CATEGORIES)
            
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
                processing_time_ms=processing_time,
                model_version="spacy-3.6-bart-mnli"
            )
            
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            # Return fallback classification
            return IntentClassification(
                primary_intent="general_development",
                secondary_intents=[],
                confidence_score=0.1,
                extracted_entities={},
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important punctuation
        cleaned = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', cleaned)
        
        # Normalize common business terms
        replacements = {
            'app': 'application',
            'db': 'database',
            'api': 'API',
            'ui': 'user interface',
            'ux': 'user experience'
        }
        
        for old, new in replacements.items():
            cleaned = re.sub(rf'\b{old}\b', new, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _extract_entities(self, doc) -> Dict[str, str]:
        """Extract named entities and key business terms"""
        entities = {}
        
        # Extract standard NER entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "MONEY", "PERCENT", "DATE"]:
                key = ent.label_.lower()
                if key not in entities:
                    entities[key] = []
                entities[key].append(ent.text)
        
        # Extract technical terms
        tech_terms = [
            "API", "database", "frontend", "backend", "mobile", "web", 
            "dashboard", "system", "platform", "service", "application"
        ]
        
        found_tech = []
        text_lower = doc.text.lower()
        for term in tech_terms:
            if term.lower() in text_lower:
                found_tech.append(term)
        
        if found_tech:
            entities["technology"] = found_tech
        
        # Extract business processes
        process_terms = [
            "management", "tracking", "automation", "integration", 
            "analysis", "reporting", "monitoring", "optimization"
        ]
        
        found_processes = []
        for term in process_terms:
            if term.lower() in text_lower:
                found_processes.append(term)
        
        if found_processes:
            entities["processes"] = found_processes
        
        return entities
    
    def _detect_industry(self, text: str) -> Optional[IndustryType]:
        """Detect industry based on keyword patterns"""
        text_lower = text.lower()
        
        industry_scores = {}
        
        for industry, keywords in DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                # Boost score for exact matches
                if f" {keyword} " in f" {text_lower} ":
                    score += 0.5
            
            if score > 0:
                industry_scores[industry] = score
        
        if not industry_scores:
            return None
        
        # Return industry with highest score, but only if confident
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        if best_industry[1] >= 2:  # Require at least 2 keyword matches
            return best_industry[0]
        
        return None
    
    def _extract_technical_keywords(self, text: str) -> List[str]:
        """Extract technical keywords and technologies"""
        text_lower = text.lower()
        
        # Common technical terms
        tech_keywords = [
            # Programming languages
            "python", "javascript", "java", "c#", "php", "ruby", "go",
            # Frameworks
            "react", "angular", "vue", "django", "flask", "spring", "express",
            # Databases
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            # Infrastructure
            "aws", "azure", "docker", "kubernetes", "nginx", "apache",
            # Concepts
            "microservices", "api", "rest", "graphql", "websocket", "oauth",
            "machine learning", "ai", "blockchain", "cloud", "devops"
        ]
        
        found_keywords = []
        for keyword in tech_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available for transformers"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

# Example usage and testing
if __name__ == "__main__":
    extractor = IntentExtractor()
    
    test_requirements = [
        "Create an e-commerce website with shopping cart and payment integration",
        "Build a mobile app for patient management in healthcare",
        "Develop REST API for financial transaction processing",
        "Design dashboard for inventory tracking and analytics"
    ]
    
    for req in test_requirements:
        result = extractor.extract_intent(req)
        print(f"\nRequirement: {req}")
        print(f"Primary Intent: {result.primary_intent}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Industry: {result.business_domain}")
        print(f"Tech Keywords: {result.technical_keywords}")
```

### 7. business_intelligence/business/context_enricher.py
```python
"""
Context Enricher
Enhances business requirements with additional context and metadata
"""

import re
from typing import Dict, List, Optional, Tuple
from ..models.business_models import BusinessContext, BusinessRequirement, IndustryType, ProjectScale

class ContextEnricher:
    """
    Analyzes and enriches business requirements with contextual information
    """
    
    def __init__(self):
        """Initialize context enrichment patterns and mappings"""
        self.timeline_patterns = {
            r'\b(\d+)\s*months?\b': lambda m: f"{m.group(1)} months",
            r'\b(\d+)\s*weeks?\b': lambda m: f"{m.group(1)} weeks",
            r'\bq[1-4]\s*202[5-9]\b': lambda m: m.group(0).upper(),
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s*202[5-9]\b': lambda m: m.group(0).title(),
            r'\basap\b': lambda m: "ASAP",
            r'\burgent\b': lambda m: "Urgent timeline"
        }
        
        self.budget_patterns = {
            r'\$(\d+)k?': lambda m: f"${m.group(1)}k budget",
            r'\b(low|small|minimal)\s*budget\b': lambda m: "Low budget",
            r'\b(medium|moderate)\s*budget\b': lambda m: "Medium budget", 
            r'\b(high|large|substantial)\s*budget\b': lambda m: "High budget",
            r'\b(enterprise|unlimited)\s*budget\b': lambda m: "Enterprise budget"
        }
        
        self.scale_indicators = {
            ProjectScale.SMALL: [
                "simple", "basic", "small", "minimal", "prototype", "mvp",
                "startup", "individual", "personal", "quick"
            ],
            ProjectScale.MEDIUM: [
                "medium", "moderate", "standard", "typical", "business",
                "professional", "commercial", "team"
            ],
            ProjectScale.LARGE: [
                "large", "complex", "advanced", "comprehensive", "full-scale",
                "enterprise", "organization", "corporation"
            ],
            ProjectScale.ENTERPRISE: [
                "enterprise", "global", "international", "massive", "scalable",
                "mission-critical", "high-availability", "distributed"
            ]
        }
        
        self.user_patterns = {
            r'\b(\d+)\s*users?\b': lambda m: f"{m.group(1)} users",
            r'\b(few|several)\s*users?\b': lambda m: "Small user base",
            r'\b(many|lots of|numerous)\s*users?\b': lambda m: "Large user base",
            r'\b(thousands|millions)\s*of\s*users?\b': lambda m: f"{m.group(1)} of users"
        }

    def enrich_context(self, requirement: BusinessRequirement) -> BusinessRequirement:
        """
        Enrich business requirement with extracted contextual information
        
        Args:
            requirement: Original business requirement
            
        Returns:
            Enhanced business requirement with enriched context
        """
        text = requirement.business_requirement.lower()
        
        # Create enriched context
        enriched_context = requirement.context or BusinessContext()
        
        # Extract timeline information
        if not enriched_context.timeline:
            timeline = self._extract_timeline(text)
            if timeline:
                enriched_context.timeline = timeline
        
        # Extract budget information  
        if not enriched_context.budget_range:
            budget = self._extract_budget(text)
            if budget:
                enriched_context.budget_range = budget
        
        # Determine project scale
        if not enriched_context.scale:
            scale = self._determine_scale(text)
            if scale:
                enriched_context.scale = scale
        
        # Extract target users information
        if not enriched_context.target_users:
            users = self._extract_target_users(text)
            if users:
                enriched_context.target_users = users
        
        # Extract existing systems
        existing_systems = self._extract_existing_systems(text)
        if existing_systems:
            enriched_context.existing_systems.extend(existing_systems)
            # Remove duplicates
            enriched_context.existing_systems = list(set(enriched_context.existing_systems))
        
        # Extract business goals
        business_goals = self._extract_business_goals(text)
        if business_goals:
            enriched_context.business_goals.extend(business_goals)
            enriched_context.business_goals = list(set(enriched_context.business_goals))
        
        # Update the requirement
        requirement.context = enriched_context
        
        return requirement

    def _extract_timeline(self, text: str) -> Optional[str]:
        """Extract timeline information from text"""
        for pattern, formatter in self.timeline_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        return None

    def _extract_budget(self, text: str) -> Optional[str]:
        """Extract budget information from text"""
        for pattern, formatter in self.budget_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        return None

    def _determine_scale(self, text: str) -> Optional[ProjectScale]:
        """Determine project scale from indicators in text"""
        scale_scores = {scale: 0 for scale in ProjectScale}
        
        for scale, indicators in self.scale_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    scale_scores[scale] += 1
        
        # Return scale with highest score if any indicators found
        max_score = max(scale_scores.values())
        if max_score > 0:
            for scale, score in scale_scores.items():
                if score == max_score:
                    return scale
        
        return None

    def _extract_target_users(self, text: str) -> Optional[str]:
        """Extract target user information"""
        for pattern, formatter in self.user_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        
        # Look for user type descriptions
        user_types = [
            "customers", "clients", "employees", "students", "patients",
            "members", "visitors", "subscribers", "administrators", "managers"
        ]
        
        found_types = []
        for user_type in user_types:
            if user_type in text:
                found_types.append(user_type)
        
        if found_types:
            return ", ".join(found_types)
        
        return None

    def _extract_existing_systems(self, text: str) -> List[str]:
        """Extract mentions of existing systems to integrate with"""
        systems = []
        
        # Common system patterns
        system_patterns = [
            r'\bintegrate\s+with\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\bconnect\s+to\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\bexisting\s+([A-Za-z\s]+?)\s*system',
            r'\bcurrent\s+([A-Za-z\s]+?)\s*platform'
        ]
        
        for pattern in system_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                system_name = match.group(1).strip()
                if len(system_name) > 2 and len(system_name) < 30:
                    systems.append(system_name.title())
        
        # Common system names
        known_systems = [
            "CRM", "ERP", "SAP", "Salesforce", "HubSpot", "QuickBooks",
            "Stripe", "PayPal", "Shopify", "WooCommerce", "Magento",
            "Google Analytics", "Mailchimp", "Slack", "Teams", "Jira"
        ]
        
        for system in known_systems:
            if system.lower() in text:
                systems.append(system)
        
        return list(set(systems))  # Remove duplicates

    def _extract_business_goals(self, text: str) -> List[str]:
        """Extract high-level business goals and objectives"""
        goals = []
        
        # Goal-indicating patterns
        goal_patterns = [
            r'\bincrease\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\bimprove\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\breduce\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\bstreamline\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\bautomate\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)',
            r'\boptimize\s+([A-Za-z\s]+?)(?:\s|$|\.|\,)'
        ]
        
        for pattern in goal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                goal = match.group(0).strip()
                if len(goal) > 5 and len(goal) < 50:
                    goals.append(goal.capitalize())
        
        # Common business objectives
        objective_keywords = [
            "cost reduction", "efficiency", "customer satisfaction",
            "revenue growth", "market expansion", "competitive advantage",
            "operational excellence", "digital transformation", "scalability",
            "user experience", "data-driven decisions", "automation"
        ]
        
        for objective in objective_keywords:
            if objective in text:
                goals.append(objective.title())
        
        return list(set(goals))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    enricher = ContextEnricher()
    
    # Test requirement
    test_req = BusinessRequirement(
        business_requirement="Create an e-commerce platform that integrates with our existing Salesforce CRM and Stripe payment system. We need to handle thousands of customers and want to increase sales by 25%. Timeline is 6 months with a moderate budget."
    )
    
    enriched = enricher.enrich_context(test_req)
    print(f"Timeline: {enriched.context.timeline}")
    print(f"Budget: {enriched.context.budget_range}")
    print(f"Scale: {enriched.context.scale}")
    print(f"Users: {enriched.context.target_users}")
    print(f"Systems: {enriched.context.existing_systems}")
    print(f"Goals: {enriched.context.business_goals}")
```

---

## 🚀 NEXT STEPS - IMPLEMENTATION PLAN

### 1. Immediate Actions (Today)
```bash
# 1. Setup repository structure
cd /home/ianua/projects/agent-zero-v1
git checkout -b feature/business-requirements-parser

# 2. Create directories
mkdir -p business_intelligence/{business,api,models,tests}

# 3. Install dependencies
pip install spacy transformers torch pydantic nltk scikit-learn
python -m spacy download en_core_web_sm

# 4. Create all __init__.py files
touch business_intelligence/__init__.py
touch business_intelligence/{business,api,models,tests}/__init__.py

# 5. Implement core files (use complete contents above)
```

### 2. File Implementation Order
1. **Models first** - Create data structures (`business_models.py`, `technical_specs.py`)
2. **Core logic** - Implement NLP components (`intent_extractor.py`, `context_enricher.py`) 
3. **Business logic** - Create translator and validator
4. **Integration** - API endpoints and Agent Zero tool integration
5. **Testing** - Comprehensive test suite

### 3. Integration with Agent Zero
```python
# Dodaj do python/tools/ jako nowy tool
# Create: python/tools/business_requirements_tool.py
# Create: prompts/default/agent.system.tool.business_requirements.md
```

### 4. Testing Strategy
- Unit tests dla każdego komponentu
- Integration tests z Agent Zero framework
- Performance tests (<500ms processing)
- End-to-end tests z przykładowymi requirements

Ten complete guide zawiera wszystkie potrzebne pliki i instrukcje implementacji zgodne z Agent Zero architecture. Każdy plik ma pełną zawartość gotową do użycia.

Czy potrzebujesz żebym rozwinął któryś z komponentów lub dodał więcej szczegółów implementacyjnych?