"""Agent Zero V1 - Business Requirements Parser (Main Orchestrator)

🚀 BusinessRequirementsParser - V2.0 Intelligence Layer Core
========================================================

Developer A Implementation - Week 42 (9-10 października 2025)
Main orchestrator integrating all business intelligence components

Pipeline:
1. Intent Extraction (Natural Language → Structured Intent)
2. Context Enrichment (Domain Knowledge Integration)
3. Constraint Analysis (Business Logic Validation)
4. Technical Translation (Business → Technical Specifications)

Integration: Multi-agent project orchestration for Agent Zero V1
"""

from __future__ import annotations  # CRITICAL FIX: Enable forward references

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

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
    # Create mock classes for testing
    IntentExtractor = None
    ContextEnricher = None
    ConstraintAnalyzer = None
    BusinessToTechnicalTranslator = None


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
    EXCELLENT = "excellent"    # >90% confidence, complete analysis
    GOOD = "good"             # >75% confidence, most features covered
    ACCEPTABLE = "acceptable"  # >60% confidence, basic analysis
    POOR = "poor"             # <60% confidence, incomplete analysis


@dataclass
class ProcessingMetrics:
    """Processing performance and quality metrics"""

    # Performance metrics
    total_processing_time: float = 0.0
    intent_extraction_time: float = 0.0
    context_enrichment_time: float = 0.0
    constraint_analysis_time: float = 0.0
    technical_translation_time: float = 0.0

    # Quality metrics
    overall_confidence: float = 0.0
    intent_confidence: float = 0.0
    enrichment_confidence: float = 0.0
    constraint_confidence: float = 0.0
    translation_confidence: float = 0.0

    # Coverage metrics
    requirements_identified: int = 0
    missing_requirements_found: int = 0
    validation_issues_count: int = 0
    technical_specs_generated: int = 0

    # Quality assessment
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE
    completeness_score: float = 0.0


@dataclass
class BusinessRequirementsResult:
    """Complete business requirements processing result"""

    # Input
    original_input: str = ""
    processing_timestamp: str = ""

    # Processing stages - FIXED: Using string annotations (forward references)
    extracted_intent: Optional["ExtractedIntent"] = None
    enriched_context: Optional["EnrichedContext"] = None
    constraint_analysis: Optional["ConstraintAnalysis"] = None  # FIXED LINE 97
    technical_specification: Optional["TechnicalSpecification"] = None

    # Validation and issues
    validation_issues: List["ValidationIssue"] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)

    # Metadata
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)

    # Success indicators
    success_probability: float = 0.0
    risk_assessment: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)


class BusinessRequirementsParser:
    """Business Requirements Parser - V2.0 Intelligence Layer

    Main orchestrator for processing natural language business requirements
    into structured technical specifications for multi-agent systems.

    Pipeline:
    --------
    Natural Language Input
           ↓
    1. Intent Extraction (Goals, Stakeholders, Classification)
           ↓  
    2. Context Enrichment (Domain Knowledge, Best Practices)
           ↓
    3. Constraint Analysis (Validation, Risk Assessment)
           ↓
    4. Technical Translation (Architecture, Implementation Plan)
           ↓
    Technical Specification Output
    """

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

        # Processing state
        self.current_status = ProcessingStatus.PENDING
        self.processing_history: List[BusinessRequirementsResult] = []

        self.logger.info("BusinessRequirementsParser initialized for V2.0 Intelligence Layer")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the parser"""
        return {
            "max_processing_time": 30.0,  # seconds
            "enable_parallel_processing": True,
            "quality_threshold": 0.6,
            "include_advanced_analysis": True,
            "save_processing_history": True,
            "enable_caching": True
        }

    async def process_requirements(self, natural_language_input: str) -> BusinessRequirementsResult:
        """Process natural language business requirements

        Main entry point for the business requirements intelligence pipeline.

        Args:
            natural_language_input: Raw business requirements in natural language

        Returns:
            BusinessRequirementsResult with complete analysis and technical specs
        """
        start_time = time.time()

        # Initialize result container
        result = BusinessRequirementsResult(
            original_input=natural_language_input,
            processing_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            processing_status=ProcessingStatus.PENDING
        )

        self.logger.info(f"Starting business requirements processing: {len(natural_language_input)} chars")

        try:
            # Stage 1: Intent Extraction
            result.processing_status = ProcessingStatus.EXTRACTING_INTENT
            if self.intent_extractor:
                result.extracted_intent = await self._extract_intent(
                    natural_language_input, result.metrics
                )

            # Stage 2: Context Enrichment
            result.processing_status = ProcessingStatus.ENRICHING_CONTEXT
            if result.extracted_intent and self.context_enricher:
                result.enriched_context = await self._enrich_context(
                    result.extracted_intent, result.metrics
                )

            # Stage 3: Constraint Analysis
            result.processing_status = ProcessingStatus.ANALYZING_CONSTRAINTS
            if result.enriched_context and self.constraint_analyzer:
                result.constraint_analysis = await self._analyze_constraints(
                    result.enriched_context, result.metrics
                )

            # Stage 4: Technical Translation
            result.processing_status = ProcessingStatus.GENERATING_TECHNICAL_SPEC
            if result.constraint_analysis and self.translator:
                result.technical_specification = await self._generate_technical_spec(
                    result.enriched_context, result.constraint_analysis, result.metrics
                )

            # Compile validation issues
            result.validation_issues = self._compile_validation_issues(result)

            # Calculate final metrics and quality assessment
            result.metrics.total_processing_time = time.time() - start_time
            self._calculate_final_metrics(result)

            result.processing_status = ProcessingStatus.COMPLETED

            self.logger.info(
                f"Requirements processing completed in {result.metrics.total_processing_time:.2f}s "
                f"with {result.metrics.quality_level.value} quality"
            )

        except Exception as e:
            result.processing_status = ProcessingStatus.FAILED
            result.processing_errors.append(str(e))
            self.logger.error(f"Requirements processing failed: {e}")

        # Save to history if enabled
        if self.config.get("save_processing_history", True):
            self.processing_history.append(result)

        return result

    async def _extract_intent(self, input_text: str, metrics: ProcessingMetrics) -> Optional["ExtractedIntent"]:
        """Extract business intent from natural language"""
        intent_start = time.time()

        try:
            intent = await self.intent_extractor.extract_complete_intent(input_text)

            metrics.intent_extraction_time = time.time() - intent_start
            metrics.intent_confidence = intent.extraction_confidence
            metrics.requirements_identified = len(intent.primary_goals) + len(intent.secondary_goals)

            self.logger.debug(f"Intent extracted with {intent.extraction_confidence:.2f} confidence")
            return intent

        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            return None

    async def _enrich_context(self, intent: "ExtractedIntent", metrics: ProcessingMetrics) -> Optional["EnrichedContext"]:
        """Enrich business context with domain knowledge"""
        enrichment_start = time.time()

        try:
            enriched = await self.context_enricher.enrich_with_domain_knowledge(intent)

            metrics.context_enrichment_time = time.time() - enrichment_start
            metrics.enrichment_confidence = enriched.enrichment_confidence
            metrics.missing_requirements_found = len(enriched.missing_requirements)

            self.logger.debug(f"Context enriched with {enriched.enrichment_confidence:.2f} confidence")
            return enriched

        except Exception as e:
            self.logger.error(f"Context enrichment failed: {e}")
            return None

    async def _analyze_constraints(self, context: "EnrichedContext", metrics: ProcessingMetrics) -> Optional["ConstraintAnalysis"]:
        """Analyze business constraints and validate logic"""
        constraint_start = time.time()

        try:
            analysis = await self.constraint_analyzer.analyze(context)

            metrics.constraint_analysis_time = time.time() - constraint_start
            metrics.constraint_confidence = analysis.analysis_confidence

            self.logger.debug(f"Constraints analyzed with {analysis.analysis_confidence:.2f} confidence")
            return analysis

        except Exception as e:
            self.logger.error(f"Constraint analysis failed: {e}")
            return None

    async def _generate_technical_spec(self, 
                                     context: "EnrichedContext", 
                                     constraints: "ConstraintAnalysis",
                                     metrics: ProcessingMetrics) -> Optional["TechnicalSpecification"]:
        """Generate technical specification from business requirements"""
        translation_start = time.time()

        try:
            spec = await self.translator.translate(context, constraints)

            metrics.technical_translation_time = time.time() - translation_start
            metrics.translation_confidence = spec.generation_confidence
            metrics.technical_specs_generated = len(spec.architecture_components)

            self.logger.debug(f"Technical spec generated with {spec.generation_confidence:.2f} confidence")
            return spec

        except Exception as e:
            self.logger.error(f"Technical specification generation failed: {e}")
            return None

    def _compile_validation_issues(self, result: BusinessRequirementsResult) -> List["ValidationIssue"]:
        """Compile all validation issues from processing stages"""
        issues = []

        # Add issues from context enrichment
        if result.enriched_context:
            issues.extend(result.enriched_context.validation_issues)

        # Add issues from constraint analysis
        if result.constraint_analysis:
            issues.extend(result.constraint_analysis.validation_issues)

        return issues

    def _calculate_final_metrics(self, result: BusinessRequirementsResult) -> None:
        """Calculate final quality metrics and assessment"""
        metrics = result.metrics

        # Calculate overall confidence (weighted average)
        confidence_weights = {
            'intent': 0.3,
            'enrichment': 0.25,
            'constraint': 0.25,
            'translation': 0.2
        }

        total_confidence = (
            metrics.intent_confidence * confidence_weights['intent'] +
            metrics.enrichment_confidence * confidence_weights['enrichment'] +
            metrics.constraint_confidence * confidence_weights['constraint'] +
            metrics.translation_confidence * confidence_weights['translation']
        )

        metrics.overall_confidence = total_confidence

        # Count validation issues by severity
        error_count = sum(1 for issue in result.validation_issues if hasattr(issue, 'severity') and issue.severity.value == 'error')
        warning_count = sum(1 for issue in result.validation_issues if hasattr(issue, 'severity') and issue.severity.value == 'warning')
        metrics.validation_issues_count = len(result.validation_issues)

        # Calculate completeness score
        completeness_factors = []

        if result.extracted_intent:
            completeness_factors.append(0.8 if hasattr(result.extracted_intent, 'primary_goals') and result.extracted_intent.primary_goals else 0.2)

        if result.enriched_context:
            completeness_factors.append(0.8 if hasattr(result.enriched_context, 'missing_requirements') and result.enriched_context.missing_requirements else 0.6)

        if result.technical_specification:
            completeness_factors.append(0.9 if hasattr(result.technical_specification, 'architecture_components') and result.technical_specification.architecture_components else 0.3)

        metrics.completeness_score = sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0

        # Determine quality level
        if total_confidence >= 0.9 and error_count == 0:
            metrics.quality_level = QualityLevel.EXCELLENT
        elif total_confidence >= 0.75 and error_count <= 1:
            metrics.quality_level = QualityLevel.GOOD
        elif total_confidence >= 0.6 and error_count <= 2:
            metrics.quality_level = QualityLevel.ACCEPTABLE
        else:
            metrics.quality_level = QualityLevel.POOR

        # Calculate success probability
        success_factors = [
            total_confidence,
            metrics.completeness_score,
            max(0, 1.0 - (error_count * 0.2)),  # Penalize errors
            max(0, 1.0 - (warning_count * 0.1))  # Penalize warnings
        ]

        result.success_probability = sum(success_factors) / len(success_factors)

        # Generate risk assessment
        result.risk_assessment = self._generate_risk_assessment(result)

    def _generate_risk_assessment(self, result: BusinessRequirementsResult) -> List[str]:
        """Generate risk assessment based on processing results"""
        risks = []

        if result.metrics.overall_confidence < 0.7:
            risks.append("Low confidence in requirements analysis - may need clarification")

        if result.metrics.validation_issues_count > 3:
            risks.append(f"High number of validation issues ({result.metrics.validation_issues_count})")

        if result.extracted_intent and hasattr(result.extracted_intent, 'urgency') and result.extracted_intent.urgency.value == 'critical':
            if hasattr(result.extracted_intent, 'primary_goals') and len(result.extracted_intent.primary_goals) > 3:
                risks.append("Critical timeline with extensive scope - high delivery risk")

        if result.metrics.total_processing_time > self.config.get("max_processing_time", 30):
            risks.append("Processing time exceeded threshold - complexity risk")

        return risks

    async def get_processing_summary(self, result: BusinessRequirementsResult) -> Dict[str, Any]:
        """Generate human-readable processing summary"""
        return {
            "status": result.processing_status.value,
            "quality": result.metrics.quality_level.value,
            "confidence": f"{result.metrics.overall_confidence:.1%}",
            "success_probability": f"{result.success_probability:.1%}",
            "processing_time": f"{result.metrics.total_processing_time:.2f}s",
            "requirements_found": result.metrics.requirements_identified,
            "missing_requirements": result.metrics.missing_requirements_found,
            "validation_issues": result.metrics.validation_issues_count,
            "technical_components": result.metrics.technical_specs_generated,
            "main_risks": result.risk_assessment[:3],  # Top 3 risks
            "project_type": result.extracted_intent.project_type.value if result.extracted_intent and hasattr(result.extracted_intent, 'project_type') else "unknown",
            "urgency": result.extracted_intent.urgency.value if result.extracted_intent and hasattr(result.extracted_intent, 'urgency') else "unknown"
        }


# Development and testing utilities
async def test_business_requirements_parser():
    """Test the complete business requirements parser"""
    parser = BusinessRequirementsParser()

    # Test with comprehensive e-commerce requirements
    test_requirements = """
    We need to build a modern e-commerce platform for our electronics startup.

    The CEO and CTO want customers to browse our product catalog, add items to their shopping cart, 
    and complete secure payments using credit cards or PayPal. Users should be able to create accounts,
    track their orders, and leave product reviews.

    This is urgent - we need to launch before Black Friday to capture holiday sales.
    The platform should be mobile-responsive and integrate with our existing inventory system.

    Success will be measured by conversion rate, average order value, and customer retention.
    We prefer using React for frontend and Python for backend, with PostgreSQL database.
    """

    print("\n🚀 Testing Business Requirements Parser - V2.0 Intelligence Layer")
    print("=" * 70)

    # Process requirements
    result = await parser.process_requirements(test_requirements)

    # Generate summary
    summary = await parser.get_processing_summary(result)

    print("📊 Processing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n✅ Processing completed in {result.metrics.total_processing_time:.2f}s")
    print(f"Quality Level: {result.metrics.quality_level.value.upper()}")
    print(f"Success Probability: {result.success_probability:.1%}")


if __name__ == "__main__":
    asyncio.run(test_business_requirements_parser())
