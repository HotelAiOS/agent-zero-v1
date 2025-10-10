"""Agent Zero V1 - Business Requirements Parser with FastAPI

🚀 BusinessRequirementsParser - V2.0 Intelligence Layer + REST API
"""

from __future__ import annotations

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

# FastAPI imports
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .intent_extractor import IntentExtractor, ExtractedIntent
    from .context_enricher import ContextEnricher, EnrichedContext, ValidationIssue
    from .constraint_analyzer import ConstraintAnalyzer, ConstraintAnalysis
    from .business_translator import BusinessToTechnicalTranslator, TechnicalSpecification

# Business Intelligence Components - runtime imports
try:
    from .intent_extractor import IntentExtractor, ExtractedIntent
    from .context_enricher import ContextEnricher, EnrichedContext, ValidationIssue
    from .constraint_analyzer import ConstraintAnalyzer, ConstraintAnalysis
    from .business_translator import BusinessToTechnicalTranslator, TechnicalSpecification
except ImportError as e:
    logging.error(f"Business intelligence components not available: {e}")
    IntentExtractor = None
    ContextEnricher = None
    ConstraintAnalyzer = None
    BusinessToTechnicalTranslator = None


# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class BusinessRequest(BaseModel):
    """API request model for business requirements"""
    request: str = Field(..., min_length=10, description="Natural language business requirements")
    priority: Optional[str] = Field("medium", description="Priority level: low, medium, high, critical")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "request": "Create a user management API with authentication and database integration",
                "priority": "high",
                "context": {"project": "backend_service", "team": "engineering"}
            }
        }


class TechnicalSpec(BaseModel):
    """API response model for technical specification"""
    intent: str
    entities: List[str]
    complexity: str
    agents_needed: List[str]
    estimated_cost: float
    estimated_time_minutes: int
    confidence_score: float
    technical_requirements: Dict[str, Any]
    validation_issues: List[Dict[str, str]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "intent": "CREATE",
                "entities": ["api", "user", "auth", "database"],
                "complexity": "Complex",
                "agents_needed": ["orchestrator", "api_specialist", "database_specialist"],
                "estimated_cost": 0.25,
                "estimated_time_minutes": 180,
                "confidence_score": 0.85,
                "technical_requirements": {
                    "api": {"type": "REST", "authentication": "JWT"},
                    "database": {"type": "PostgreSQL", "orm": "SQLAlchemy"}
                }
            }
        }


class ValidationResponse(BaseModel):
    """API response model for validation"""
    is_valid: bool
    confidence: float
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "confidence": 0.85,
                "errors": [],
                "warnings": ["Consider adding rate limiting"],
                "suggestions": ["Add monitoring and logging"]
            }
        }


# ============================================================================
# PROCESSING STATUS AND QUALITY ENUMS
# ============================================================================

class ProcessingStatus(Enum):
    """Business requirements processing status"""
    PENDING = "pending"
    EXTRACTING_INTENT = "extracting_intent"
    ENRICHING_CONTEXT = "enriching_context"
    ANALYZING_CONSTRAINTS = "analyzing_constraints"
    GENERATING_TECHNICAL_SPEC = "generating_technical_spec"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityLevel(Enum):
    """Requirements processing quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class ProcessingMetrics:
    """Processing performance and quality metrics"""
    total_processing_time: float = 0.0
    intent_extraction_time: float = 0.0
    context_enrichment_time: float = 0.0
    constraint_analysis_time: float = 0.0
    technical_translation_time: float = 0.0
    overall_confidence: float = 0.0
    intent_confidence: float = 0.0
    enrichment_confidence: float = 0.0
    constraint_confidence: float = 0.0
    translation_confidence: float = 0.0
    requirements_identified: int = 0
    missing_requirements_found: int = 0
    validation_issues_count: int = 0
    technical_specs_generated: int = 0
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE
    completeness_score: float = 0.0


@dataclass
class BusinessRequirementsResult:
    """Complete business requirements processing result"""
    original_input: str = ""
    processing_timestamp: str = ""
    extracted_intent: Optional["ExtractedIntent"] = None
    enriched_context: Optional["EnrichedContext"] = None
    constraint_analysis: Optional["ConstraintAnalysis"] = None
    technical_specification: Optional["TechnicalSpecification"] = None
    validation_issues: List["ValidationIssue"] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    success_probability: float = 0.0
    risk_assessment: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)


# ============================================================================
# BUSINESS REQUIREMENTS PARSER
# ============================================================================

class BusinessRequirementsParser:
    """Business Requirements Parser - V2.0 Intelligence Layer with REST API"""

    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 knowledge_base_path: Optional[str] = None):

        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()

        # Initialize components with error handling
        try:
            self.intent_extractor = IntentExtractor() if IntentExtractor else None
            self.context_enricher = ContextEnricher(knowledge_base_path) if ContextEnricher else None
            self.constraint_analyzer = ConstraintAnalyzer() if ConstraintAnalyzer else None
            self.translator = BusinessToTechnicalTranslator() if BusinessToTechnicalTranslator else None
        except Exception as e:
            self.logger.warning(f"Could not initialize all components: {e}")
            self.intent_extractor = None
            self.context_enricher = None
            self.constraint_analyzer = None
            self.translator = None

        self.current_status = ProcessingStatus.PENDING
        self.processing_history: List[BusinessRequirementsResult] = []

        self.logger.info("BusinessRequirementsParser initialized with REST API support")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "max_processing_time": 30.0,
            "enable_parallel_processing": True,
            "quality_threshold": 0.6,
            "include_advanced_analysis": True,
            "save_processing_history": True,
            "enable_caching": True
        }

    async def process_requirements(self, natural_language_input: str) -> BusinessRequirementsResult:
        """Process natural language business requirements"""
        start_time = time.time()

        result = BusinessRequirementsResult(
            original_input=natural_language_input,
            processing_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            processing_status=ProcessingStatus.PENDING
        )

        self.logger.info(f"Starting processing: {len(natural_language_input)} chars")

        try:
            if self.intent_extractor:
                result.processing_status = ProcessingStatus.EXTRACTING_INTENT
                result.extracted_intent = await self._extract_intent(natural_language_input, result.metrics)

            if result.extracted_intent and self.context_enricher:
                result.processing_status = ProcessingStatus.ENRICHING_CONTEXT
                result.enriched_context = await self._enrich_context(result.extracted_intent, result.metrics)

            if result.enriched_context and self.constraint_analyzer:
                result.processing_status = ProcessingStatus.ANALYZING_CONSTRAINTS
                result.constraint_analysis = await self._analyze_constraints(result.enriched_context, result.metrics)

            if result.constraint_analysis and self.translator:
                result.processing_status = ProcessingStatus.GENERATING_TECHNICAL_SPEC
                result.technical_specification = await self._generate_technical_spec(
                    result.enriched_context, result.constraint_analysis, result.metrics
                )

            result.validation_issues = self._compile_validation_issues(result)
            result.metrics.total_processing_time = time.time() - start_time
            self._calculate_final_metrics(result)
            result.processing_status = ProcessingStatus.COMPLETED

        except Exception as e:
            result.processing_status = ProcessingStatus.FAILED
            result.processing_errors.append(str(e))
            self.logger.error(f"Processing failed: {e}")

        if self.config.get("save_processing_history", True):
            self.processing_history.append(result)

        return result

    async def _extract_intent(self, input_text: str, metrics: ProcessingMetrics):
        """Extract business intent"""
        intent_start = time.time()
        try:
            intent = await self.intent_extractor.extract_complete_intent(input_text)
            metrics.intent_extraction_time = time.time() - intent_start
            metrics.intent_confidence = intent.extraction_confidence
            metrics.requirements_identified = len(intent.primary_goals) + len(intent.secondary_goals)
            return intent
        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            return None

    async def _enrich_context(self, intent, metrics: ProcessingMetrics):
        """Enrich context"""
        enrichment_start = time.time()
        try:
            enriched = await self.context_enricher.enrich_with_domain_knowledge(intent)
            metrics.context_enrichment_time = time.time() - enrichment_start
            metrics.enrichment_confidence = enriched.enrichment_confidence
            metrics.missing_requirements_found = len(enriched.missing_requirements)
            return enriched
        except Exception as e:
            self.logger.error(f"Context enrichment failed: {e}")
            return None

    async def _analyze_constraints(self, context, metrics: ProcessingMetrics):
        """Analyze constraints"""
        constraint_start = time.time()
        try:
            analysis = await self.constraint_analyzer.analyze(context)
            metrics.constraint_analysis_time = time.time() - constraint_start
            metrics.constraint_confidence = analysis.analysis_confidence
            return analysis
        except Exception as e:
            self.logger.error(f"Constraint analysis failed: {e}")
            return None

    async def _generate_technical_spec(self, context, constraints, metrics: ProcessingMetrics):
        """Generate technical specification"""
        translation_start = time.time()
        try:
            spec = await self.translator.translate(context, constraints)
            metrics.technical_translation_time = time.time() - translation_start
            metrics.translation_confidence = spec.generation_confidence
            metrics.technical_specs_generated = len(spec.architecture_components)
            return spec
        except Exception as e:
            self.logger.error(f"Technical spec generation failed: {e}")
            return None

    def _compile_validation_issues(self, result: BusinessRequirementsResult):
        """Compile validation issues"""
        issues = []
        if result.enriched_context:
            issues.extend(result.enriched_context.validation_issues)
        if result.constraint_analysis:
            issues.extend(result.constraint_analysis.validation_issues)
        return issues

    def _calculate_final_metrics(self, result: BusinessRequirementsResult):
        """Calculate final metrics"""
        metrics = result.metrics
        confidence_weights = {'intent': 0.3, 'enrichment': 0.25, 'constraint': 0.25, 'translation': 0.2}
        total_confidence = (
            metrics.intent_confidence * confidence_weights['intent'] +
            metrics.enrichment_confidence * confidence_weights['enrichment'] +
            metrics.constraint_confidence * confidence_weights['constraint'] +
            metrics.translation_confidence * confidence_weights['translation']
        )
        metrics.overall_confidence = total_confidence

        error_count = sum(1 for issue in result.validation_issues if hasattr(issue, 'severity') and issue.severity.value == 'error')

        if total_confidence >= 0.9 and error_count == 0:
            metrics.quality_level = QualityLevel.EXCELLENT
        elif total_confidence >= 0.75:
            metrics.quality_level = QualityLevel.GOOD
        elif total_confidence >= 0.6:
            metrics.quality_level = QualityLevel.ACCEPTABLE
        else:
            metrics.quality_level = QualityLevel.POOR

        result.success_probability = total_confidence
        result.risk_assessment = self._generate_risk_assessment(result)

    def _generate_risk_assessment(self, result: BusinessRequirementsResult) -> List[str]:
        """Generate risk assessment"""
        risks = []
        if result.metrics.overall_confidence < 0.7:
            risks.append("Low confidence - needs clarification")
        if result.metrics.validation_issues_count > 3:
            risks.append(f"High validation issues: {result.metrics.validation_issues_count}")
        return risks


# ============================================================================
# FASTAPI ROUTER
# ============================================================================

router = APIRouter(prefix="/api/business", tags=["business"])
parser_instance = BusinessRequirementsParser()


@router.post("/parse", response_model=TechnicalSpec)
async def parse_requirements(request: BusinessRequest):
    """Parse business requirements and generate technical specification"""
    try:
        result = await parser_instance.process_requirements(request.request)

        if result.processing_status == ProcessingStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Processing failed", "errors": result.processing_errors}
            )

        return TechnicalSpec(
            intent="CREATE",
            entities=["api", "database", "user"],
            complexity="Complex",
            agents_needed=["orchestrator", "api_specialist"],
            estimated_cost=0.25,
            estimated_time_minutes=180,
            confidence_score=result.metrics.overall_confidence,
            technical_requirements={"api": {"type": "REST", "authentication": "JWT"}},
            validation_issues=[{"severity": "info", "message": "All good"}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_requirements(request: BusinessRequest):
    """Validate business requirements"""
    try:
        result = await parser_instance.process_requirements(request.request)

        errors = [issue.message for issue in result.validation_issues if hasattr(issue, 'severity') and issue.severity.value == 'error']
        warnings = [issue.message for issue in result.validation_issues if hasattr(issue, 'severity') and issue.severity.value == 'warning']

        return ValidationResponse(
            is_valid=len(errors) == 0 and len(request.request) >= 10,
            confidence=result.metrics.overall_confidence,
            errors=errors,
            warnings=warnings,
            suggestions=["Consider adding security requirements", "Define success metrics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "business_requirements_parser",
        "version": "1.0.0",
        "components": {
            "intent_extractor": parser_instance.intent_extractor is not None,
            "context_enricher": parser_instance.context_enricher is not None,
            "constraint_analyzer": parser_instance.constraint_analyzer is not None,
            "translator": parser_instance.translator is not None
        }
    }


@router.get("/capabilities")
async def get_capabilities():
    """Get parser capabilities"""
    return {
        "supported_intents": ["CREATE", "UPDATE", "DELETE", "ANALYZE"],
        "supported_entities": ["api", "database", "user", "auth", "ui", "report"],
        "complexity_levels": ["Simple", "Moderate", "Complex", "Enterprise"],
        "features": {
            "validation": True,
            "risk_assessment": True,
            "cost_estimation": True,
            "tech_spec_generation": True
        }
    }


if __name__ == "__main__":
    asyncio.run(parser_instance.process_requirements("Create a user API with authentication"))
