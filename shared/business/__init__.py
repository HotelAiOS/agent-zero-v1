"""Agent Zero V1 - Business Requirements Intelligence Layer

🎯 Business Requirements Parser Module
==================================== 

V2.0 Intelligence Layer - Developer A Implementation
Week 42: Foundation Development (9-10 października 2025)

Core Components:
- Intent Extraction: Natural language business goal parsing
- Context Enrichment: Domain knowledge integration
- Constraint Analysis: Business logic validation
- Technical Translation: Business to dev spec conversion

Architecture:
    Business Input → Intent → Context → Constraints → Technical Spec
    
Story Points: 4 SP (Foundation)
Target: Multi-agent project orchestration enhancement

Author: Agent Zero V1 AI Assistant
Project: HotelAiOS/agent-zero-v1
"""

from .intent_extractor import IntentExtractor
from .context_enricher import ContextEnricher
from .requirements_parser import BusinessRequirementsParser
from .constraint_analyzer import ConstraintAnalyzer
from .business_translator import BusinessToTechnicalTranslator

__version__ = "2.0.0-alpha"
__author__ = "Agent Zero V1 - Developer A"
__date__ = "2025-10-08"

# Export main classes for easy import
__all__ = [
    "IntentExtractor",
    "ContextEnricher", 
    "BusinessRequirementsParser",
    "ConstraintAnalyzer",
    "BusinessToTechnicalTranslator"
]

# Module metadata for integration
MODULE_INFO = {
    "name": "business_requirements_parser",
    "version": __version__,
    "phase": "V2.0 Intelligence Layer",
    "story_points": 4,
    "completion": "Foundation - 25%",
    "next_milestone": "Context Enrichment Engine",
    "dependencies": ["shared.core", "shared.llm", "shared.models"],
    "integration_ready": False
}
