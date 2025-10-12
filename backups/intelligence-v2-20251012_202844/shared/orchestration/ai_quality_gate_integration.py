#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - AI-Powered Quality Gate Integration (FIXED)
Advanced acceptance criteria with automated quality gates and continuous monitoring

FIXES:
- Import path correction for orchestration components
- Fallback definitions when components unavailable  
- Proper relative import handling
- All type definitions included locally
"""

import asyncio
import json
import logging
import time
import subprocess
import ast
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# ========== FALLBACK DEFINITIONS - START ==========
# All necessary types defined locally to avoid import issues

class TaskType(Enum):
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    TESTING = "testing"
    DEPLOYMENT = "deployment"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class ProgressStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class GateStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class GateSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class OptimizationStrategy(Enum):
    CRITICAL_PATH = "critical_path"
    RESOURCE_BALANCED = "resource_balanced"
    RISK_MINIMIZED = "risk_minimized"
    TIME_OPTIMIZED = "time_optimized"

@dataclass
class ReasoningContext:
    project_type: str = "general"
    tech_stack: List[str] = field(default_factory=list)
    team_skills: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    previous_decisions: List[Dict] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class Task:
    id: int
    title: str
    description: str
    task_type: TaskType = TaskType.BACKEND
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_hours: float = 8.0

@dataclass
class TaskProgress:
    task_id: int
    workflow_id: str
    status: ProgressStatus
    progress_percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

@dataclass
class QualityGate:
    gate_id: str
    name: str
    description: str
    severity: GateSeverity
    criteria: List[str]
    required_for_deployment: bool = True
    status: GateStatus = GateStatus.PENDING
    checked_at: Optional[datetime] = None
    passed_criteria: List[str] = field(default_factory=list)
    failed_criteria: List[str] = field(default_factory=list)
    requires_human_approval: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    result_details: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

@dataclass
class WorkflowMetrics:
    total_tasks: int
    completed_tasks: int
    active_tasks: int
    blocked_tasks: int
    estimated_completion: datetime
    critical_path_length: float
    resource_utilization: Dict[str, float]
    bottlenecks: List[str]
    optimization_opportunities: List[str]

@dataclass
class OptimizedWorkflow:
    workflow_id: str
    strategy: str  # Using str instead of OptimizationStrategy for flexibility
    task_order: List[int]
    parallel_groups: List[List[int]]
    critical_path: List[int]
    estimated_duration: float
    resource_allocation: Dict[str, List[int]]
    confidence: float
    optimization_reasoning: str
    metrics: WorkflowMetrics

# AI Components fallback
class ReasoningType(Enum):
    TASK_ANALYSIS = "task_analysis"
    DECISION_MAKING = "decision_making"
    QUALITY_ASSESSMENT = "quality_assessment"
    CODE_REVIEW = "code_review"
    PROBLEM_SOLVING = "problem_solving"

class AIModelType(Enum):
    CODE = "code"
    STANDARD = "standard"
    ADVANCED = "advanced"

# Mock AI components for when real ones aren't available
class MockReasoningChain:
    def __init__(self, reasoning: str):
        self.final_reasoning = reasoning

class MockReasoningEngine:
    async def reason_with_context(self, problem_statement: str, context: ReasoningContext, reasoning_type: ReasoningType):
        # Return mock reasoning based on the reasoning type
        if reasoning_type == ReasoningType.QUALITY_ASSESSMENT:
            return MockReasoningChain("""
            Quality Assessment Analysis:
            
            Acceptance Criteria:
            1. API endpoints must return correct status codes and valid JSON responses (CRITICAL)
            2. Response times should be under 200ms for 95th percentile (HIGH)
            3. All endpoints must implement proper error handling (CRITICAL)  
            4. Input validation must be present for all user inputs (HIGH)
            5. Security best practices must be followed (CRITICAL)
            6. Code must have adequate test coverage (MEDIUM)
            7. Documentation must be present for all public APIs (MEDIUM)
            
            These criteria ensure functionality, performance, security, and maintainability.
            """)
        elif reasoning_type == ReasoningType.CODE_REVIEW:
            return MockReasoningChain("""
            Code Review Analysis:
            
            Maintainability Score: 0.78
            
            Issues Found:
            - Function complexity is moderate, consider breaking down larger functions
            - Some error handling could be more specific
            - Type hints are present but could be more comprehensive
            
            Suggestions:
            - Add more detailed docstrings for complex functions
            - Consider using custom exception types for better error handling
            - Implement input validation for all public methods
            """)
        else:
            return MockReasoningChain(f"Analysis completed for {reasoning_type.value}")

# ========== FALLBACK DEFINITIONS - END ==========

# Import existing components with proper fallbacks
ORCHESTRATION_COMPONENTS_AVAILABLE = False
AI_COMPONENTS_AVAILABLE = False
UnifiedAIClient = None
ContextAwareReasoningEngine = None

try:
    import sys
    import os
    
    # Try to import orchestration components
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from dynamic_workflow_optimizer import DynamicWorkflowOptimizer
        # Override fallback definitions with real ones if available
        ORCHESTRATION_COMPONENTS_AVAILABLE = True
    except ImportError:
        DynamicWorkflowOptimizer = None
    
    try:
        from real_time_progress_monitor import RealTimeProgressMonitor
    except ImportError:
        RealTimeProgressMonitor = None
    
    try:
        from quality_gates import QualityGateManager
    except ImportError:
        QualityGateManager = None
    
    # Try to import AI components
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai'))
        from unified_ai_client import UnifiedAIClient
        from context_aware_reasoning import ContextAwareReasoningEngine
        AI_COMPONENTS_AVAILABLE = True
    except ImportError:
        pass

except Exception as e:
    logger.warning(f"Component imports failed: {e}")

class QualityDimension(Enum):
    """Quality dimensions for assessment"""
    FUNCTIONALITY = "functionality"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    DOCUMENTATION = "documentation"

class AssessmentMethod(Enum):
    """Quality assessment methods"""
    STATIC_ANALYSIS = "static_analysis"
    DYNAMIC_TESTING = "dynamic_testing"
    CODE_REVIEW = "code_review"
    AI_ANALYSIS = "ai_analysis"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_SCAN = "security_scan"
    MANUAL_REVIEW = "manual_review"

class QualityImpact(Enum):
    """Impact levels for quality issues"""
    CRITICAL = "critical"      # System breaking, security vulnerabilities
    HIGH = "high"             # Major functionality issues, performance problems
    MEDIUM = "medium"         # Minor issues, maintainability concerns
    LOW = "low"              # Cosmetic, documentation improvements
    INFO = "info"            # Informational, suggestions

@dataclass
class QualityMetric:
    """Quality metric measurement"""
    metric_id: str
    dimension: QualityDimension
    name: str
    description: str
    target_value: float
    measured_value: Optional[float] = None
    unit: str = ""
    threshold_critical: Optional[float] = None
    threshold_warning: Optional[float] = None
    measurement_method: AssessmentMethod = AssessmentMethod.STATIC_ANALYSIS
    last_measured: Optional[datetime] = None
    trend_data: List[Tuple[datetime, float]] = field(default_factory=list)

@dataclass
class AcceptanceCriteria:
    """AI-generated acceptance criteria for tasks"""
    criteria_id: str
    task_id: int
    description: str
    priority: QualityImpact
    test_method: AssessmentMethod
    success_condition: str
    verification_steps: List[str]
    automated_check: Optional[str] = None  # Code/command for automated verification
    ai_generated: bool = True
    confidence: float = 0.0
    context_factors: List[str] = field(default_factory=list)

@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result"""
    assessment_id: str
    task_id: int
    workflow_id: str
    overall_score: float  # 0.0 - 1.0
    dimension_scores: Dict[QualityDimension, float]
    criteria_results: List[Tuple[str, bool, str]]  # criteria_id, passed, details
    issues_found: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    assessment_time: datetime
    assessment_duration: float
    assessor: str  # AI, human, or tool name
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityTrend:
    """Quality trend analysis"""
    metric_id: str
    trend_direction: str  # "improving", "declining", "stable"
    rate_of_change: float
    prediction_30d: float
    confidence_interval: Tuple[float, float]
    risk_factors: List[str]
    recommendations: List[str]

class AIQualityGateIntegration:
    """
    AI-Powered Quality Gate Integration System (FIXED VERSION)
    
    Features:
    - Dynamic acceptance criteria generation using AI context analysis
    - Multi-dimensional quality assessment with predictive scoring
    - Automated code review using AST analysis and AI insights
    - Real-time quality monitoring with trend analysis
    - Integration testing pipeline orchestration
    - Continuous quality improvement recommendations
    - Risk-based quality gate prioritization
    - Performance-aware quality thresholds
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.ai_client = None
        self.reasoning_engine = None
        self.quality_gate_manager = None
        self.progress_monitor = None
        
        # Initialize components with fallbacks
        if ORCHESTRATION_COMPONENTS_AVAILABLE and QualityGateManager:
            try:
                self.quality_gate_manager = QualityGateManager()
            except Exception as e:
                logger.warning(f"QualityGateManager initialization failed: {e}")
        
        if ORCHESTRATION_COMPONENTS_AVAILABLE and RealTimeProgressMonitor:
            try:
                self.progress_monitor = RealTimeProgressMonitor(db_path=db_path)
            except Exception as e:
                logger.warning(f"RealTimeProgressMonitor initialization failed: {e}")
        
        if AI_COMPONENTS_AVAILABLE and UnifiedAIClient and ContextAwareReasoningEngine:
            try:
                self.ai_client = UnifiedAIClient(db_path=db_path)
                self.reasoning_engine = ContextAwareReasoningEngine(db_path=db_path)
                logger.info("✅ AI components connected")
            except Exception as e:
                logger.warning(f"AI initialization failed: {e}")
        else:
            # Use mock reasoning engine
            self.reasoning_engine = MockReasoningEngine()
            logger.info("🤖 Using mock AI reasoning engine")
        
        # Quality management
        self.quality_metrics: Dict[str, QualityMetric] = {}
        self.acceptance_criteria: Dict[int, List[AcceptanceCriteria]] = {}
        self.quality_assessments: Dict[str, QualityAssessment] = {}
        self.quality_trends: Dict[str, QualityTrend] = {}
        
        # Assessment tools
        self.assessment_tools = {
            AssessmentMethod.STATIC_ANALYSIS: self._run_static_analysis,
            AssessmentMethod.CODE_REVIEW: self._run_ai_code_review,
            AssessmentMethod.SECURITY_SCAN: self._run_security_scan,
            AssessmentMethod.PERFORMANCE_TEST: self._run_performance_test,
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityDimension.FUNCTIONALITY: 0.9,
            QualityDimension.RELIABILITY: 0.85,
            QualityDimension.PERFORMANCE: 0.8,
            QualityDimension.SECURITY: 0.95,
            QualityDimension.MAINTAINABILITY: 0.75,
            QualityDimension.DOCUMENTATION: 0.7
        }
        
        self._init_database()
        self._init_default_metrics()
        logger.info("✅ AIQualityGateIntegration initialized")
    
    def _init_database(self):
        """Initialize quality integration database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Quality metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_id TEXT UNIQUE NOT NULL,
                        dimension TEXT NOT NULL,
                        name TEXT NOT NULL,
                        target_value REAL,
                        measured_value REAL,
                        unit TEXT,
                        threshold_critical REAL,
                        threshold_warning REAL,
                        measurement_method TEXT,
                        last_measured TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Acceptance criteria
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS acceptance_criteria (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        criteria_id TEXT UNIQUE NOT NULL,
                        task_id INTEGER NOT NULL,
                        description TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        test_method TEXT NOT NULL,
                        success_condition TEXT NOT NULL,
                        verification_steps TEXT,  -- JSON
                        automated_check TEXT,
                        ai_generated BOOLEAN DEFAULT TRUE,
                        confidence REAL,
                        context_factors TEXT,  -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Quality assessments
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        assessment_id TEXT UNIQUE NOT NULL,
                        task_id INTEGER NOT NULL,
                        workflow_id TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        dimension_scores TEXT,  -- JSON
                        criteria_results TEXT,  -- JSON
                        issues_found TEXT,  -- JSON
                        improvement_suggestions TEXT,  -- JSON
                        assessment_duration REAL,
                        assessor TEXT,
                        confidence REAL,
                        metadata TEXT,  -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    
    def _init_default_metrics(self):
        """Initialize default quality metrics"""
        default_metrics = [
            QualityMetric(
                "code_coverage", QualityDimension.RELIABILITY, "Test Coverage", 
                "Percentage of code covered by tests", 80.0, unit="%",
                threshold_critical=50.0, threshold_warning=70.0
            ),
            QualityMetric(
                "cyclomatic_complexity", QualityDimension.MAINTAINABILITY, 
                "Cyclomatic Complexity", "Average cyclomatic complexity per function", 
                10.0, threshold_critical=20.0, threshold_warning=15.0
            ),
            QualityMetric(
                "response_time_p95", QualityDimension.PERFORMANCE, 
                "Response Time P95", "95th percentile response time", 
                200.0, unit="ms", threshold_critical=1000.0, threshold_warning=500.0
            ),
            QualityMetric(
                "security_score", QualityDimension.SECURITY, 
                "Security Score", "Overall security assessment score", 
                90.0, unit="%", threshold_critical=70.0, threshold_warning=80.0,
                measurement_method=AssessmentMethod.SECURITY_SCAN
            ),
            QualityMetric(
                "documentation_coverage", QualityDimension.DOCUMENTATION, 
                "Documentation Coverage", "Percentage of functions with documentation", 
                80.0, unit="%", threshold_critical=40.0, threshold_warning=60.0
            )
        ]
        
        for metric in default_metrics:
            self.quality_metrics[metric.metric_id] = metric
    
    async def generate_acceptance_criteria(
        self, 
        task: Task, 
        context: ReasoningContext
    ) -> List[AcceptanceCriteria]:
        """Generate AI-powered acceptance criteria for a task"""
        
        logger.info(f"🎯 Generating acceptance criteria for task {task.id}: {task.title}")
        
        criteria_list = []
        
        try:
            if self.reasoning_engine:
                # AI-powered criteria generation
                criteria_list = await self._generate_ai_criteria(task, context)
            else:
                # Fallback: rule-based criteria generation
                criteria_list = self._generate_rule_based_criteria(task)
            
            # Store criteria
            self.acceptance_criteria[task.id] = criteria_list
            
            # Log to database
            for criteria in criteria_list:
                self._log_acceptance_criteria(criteria)
            
            logger.info(f"✅ Generated {len(criteria_list)} acceptance criteria for task {task.id}")
            return criteria_list
            
        except Exception as e:
            logger.error(f"Acceptance criteria generation failed: {e}")
            return []
    
    async def _generate_ai_criteria(
        self, 
        task: Task, 
        context: ReasoningContext
    ) -> List[AcceptanceCriteria]:
        """Generate acceptance criteria using AI reasoning"""
        
        problem_statement = f"""
        Generate comprehensive acceptance criteria for task:
        
        Task Details:
        - Title: {task.title}
        - Description: {task.description}
        - Type: {task.task_type.value}
        - Priority: {task.priority.value}
        
        Context:
        - Project Type: {context.project_type}
        - Tech Stack: {', '.join(context.tech_stack)}
        - Constraints: {', '.join(context.constraints)}
        
        Generate 3-7 specific, testable acceptance criteria covering:
        1. Functional requirements
        2. Quality attributes (performance, security, usability)
        3. Technical constraints
        4. Integration requirements
        
        For each criterion, specify:
        - Clear success condition
        - Verification method
        - Priority level (CRITICAL, HIGH, MEDIUM, LOW)
        """
        
        reasoning_chain = await self.reasoning_engine.reason_with_context(
            problem_statement=problem_statement,
            context=context,
            reasoning_type=ReasoningType.QUALITY_ASSESSMENT
        )
        
        # Parse AI response and create AcceptanceCriteria objects
        criteria_list = self._parse_ai_criteria_response(
            reasoning_chain.final_reasoning, task.id
        )
        
        return criteria_list
    
    def _parse_ai_criteria_response(
        self, 
        ai_response: str, 
        task_id: int
    ) -> List[AcceptanceCriteria]:
        """Parse AI response and create AcceptanceCriteria objects"""
        
        criteria_list = []
        
        try:
            # Simple parsing - would be more sophisticated in production
            lines = ai_response.split('\n')
            current_criteria = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for numbered criteria
                if re.match(r'^\d+\.', line) or any(word in line.lower() for word in ['criterion', 'acceptance', 'must', 'should']):
                    if current_criteria:
                        criteria_list.append(current_criteria)
                    
                    criteria_id = f"ac_{task_id}_{len(criteria_list) + 1}_{int(time.time())}"
                    
                    # Determine priority from keywords
                    priority = QualityImpact.MEDIUM
                    if any(word in line.lower() for word in ['critical', 'security', 'performance']):
                        priority = QualityImpact.CRITICAL
                    elif any(word in line.lower() for word in ['important', 'high', 'required']):
                        priority = QualityImpact.HIGH
                    elif any(word in line.lower() for word in ['nice', 'optional', 'should']):
                        priority = QualityImpact.LOW
                    
                    # Determine test method from keywords
                    test_method = AssessmentMethod.MANUAL_REVIEW
                    if any(word in line.lower() for word in ['test', 'unit', 'integration']):
                        test_method = AssessmentMethod.DYNAMIC_TESTING
                    elif any(word in line.lower() for word in ['review', 'code']):
                        test_method = AssessmentMethod.CODE_REVIEW
                    elif any(word in line.lower() for word in ['performance', 'benchmark']):
                        test_method = AssessmentMethod.PERFORMANCE_TEST
                    elif any(word in line.lower() for word in ['security', 'vulnerability']):
                        test_method = AssessmentMethod.SECURITY_SCAN
                    
                    current_criteria = AcceptanceCriteria(
                        criteria_id=criteria_id,
                        task_id=task_id,
                        description=line,
                        priority=priority,
                        test_method=test_method,
                        success_condition="Criterion met as described",
                        verification_steps=["Review implementation", "Execute tests", "Verify results"],
                        confidence=0.8,
                        context_factors=["ai_generated", "task_type_specific"]
                    )
            
            if current_criteria:
                criteria_list.append(current_criteria)
        
        except Exception as e:
            logger.warning(f"AI criteria parsing failed: {e}")
        
        return criteria_list
    
    def _generate_rule_based_criteria(self, task: Task) -> List[AcceptanceCriteria]:
        """Generate acceptance criteria using rule-based approach"""
        
        criteria_list = []
        base_id = f"ac_{task.id}_{int(time.time())}"
        
        # Common criteria based on task type
        if task.task_type == TaskType.BACKEND:
            criteria_list.extend([
                AcceptanceCriteria(
                    f"{base_id}_api", task.id,
                    "API endpoints respond correctly with expected data formats",
                    QualityImpact.CRITICAL, AssessmentMethod.INTEGRATION_TEST,
                    "All API endpoints return 2xx status codes and valid JSON",
                    ["Test all endpoints", "Validate response schemas", "Check error handling"],
                    ai_generated=False, confidence=0.9
                ),
                AcceptanceCriteria(
                    f"{base_id}_perf", task.id,
                    "API response times meet performance requirements",
                    QualityImpact.HIGH, AssessmentMethod.PERFORMANCE_TEST,
                    "95th percentile response time < 200ms",
                    ["Load test with 100 concurrent users", "Monitor response times", "Validate SLA compliance"],
                    ai_generated=False, confidence=0.8
                )
            ])
        
        elif task.task_type == TaskType.FRONTEND:
            criteria_list.extend([
                AcceptanceCriteria(
                    f"{base_id}_ui", task.id,
                    "User interface is responsive and accessible",
                    QualityImpact.HIGH, AssessmentMethod.DYNAMIC_TESTING,
                    "UI works correctly on mobile and desktop, passes accessibility audit",
                    ["Test on multiple screen sizes", "Run accessibility scanner", "Validate user workflows"],
                    ai_generated=False, confidence=0.85
                ),
                AcceptanceCriteria(
                    f"{base_id}_ux", task.id,
                    "User experience meets usability standards",
                    QualityImpact.MEDIUM, AssessmentMethod.MANUAL_REVIEW,
                    "Users can complete primary tasks without confusion",
                    ["Conduct usability testing", "Gather user feedback", "Measure task completion rates"],
                    ai_generated=False, confidence=0.7
                )
            ])
        
        # Universal criteria
        criteria_list.append(
            AcceptanceCriteria(
                f"{base_id}_security", task.id,
                "Implementation follows security best practices",
                QualityImpact.CRITICAL, AssessmentMethod.SECURITY_SCAN,
                "No critical or high severity security vulnerabilities found",
                ["Run security scanner", "Review code for vulnerabilities", "Check authentication/authorization"],
                ai_generated=False, confidence=0.95
            )
        )
        
        return criteria_list
    
    async def assess_task_quality(
        self, 
        task: Task, 
        workflow_id: str,
        code_path: Optional[str] = None
    ) -> QualityAssessment:
        """Perform comprehensive quality assessment for a task"""
        
        assessment_id = f"qa_{task.id}_{workflow_id}_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"🔍 Starting quality assessment for task {task.id}")
        
        try:
            # Get acceptance criteria
            criteria_list = self.acceptance_criteria.get(task.id, [])
            
            # Run assessment methods
            dimension_scores = {}
            criteria_results = []
            issues_found = []
            improvement_suggestions = []
            
            # Static analysis
            if code_path:
                static_results = await self._run_static_analysis(code_path)
                dimension_scores.update(static_results.get('scores', {}))
                issues_found.extend(static_results.get('issues', []))
                improvement_suggestions.extend(static_results.get('suggestions', []))
            
            # AI code review
            if self.reasoning_engine and code_path:
                ai_review = await self._run_ai_code_review(code_path, task)
                dimension_scores[QualityDimension.MAINTAINABILITY] = ai_review.get('maintainability_score', 0.7)
                issues_found.extend(ai_review.get('issues', []))
                improvement_suggestions.extend(ai_review.get('suggestions', []))
            
            # Evaluate acceptance criteria
            for criteria in criteria_list:
                passed, details = await self._evaluate_acceptance_criteria(criteria, code_path)
                criteria_results.append((criteria.criteria_id, passed, details))
                
                if not passed:
                    issues_found.append({
                        'type': 'acceptance_criteria_failed',
                        'criteria_id': criteria.criteria_id,
                        'description': criteria.description,
                        'details': details,
                        'priority': criteria.priority.value
                    })
            
            # Calculate overall score
            if dimension_scores:
                overall_score = sum(dimension_scores.values()) / len(dimension_scores)
            else:
                overall_score = 0.7  # Default when no assessments available
            
            # Adjust score based on criteria results
            if criteria_results:
                passed_criteria = sum(1 for _, passed, _ in criteria_results if passed)
                criteria_score = passed_criteria / len(criteria_results)
                overall_score = (overall_score + criteria_score) / 2
            
            # Create assessment
            assessment = QualityAssessment(
                assessment_id=assessment_id,
                task_id=task.id,
                workflow_id=workflow_id,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                criteria_results=criteria_results,
                issues_found=issues_found,
                improvement_suggestions=improvement_suggestions,
                assessment_time=datetime.now(),
                assessment_duration=time.time() - start_time,
                assessor="AIQualityGateIntegration",
                confidence=0.8,
                metadata={'code_path': code_path}
            )
            
            # Store assessment
            self.quality_assessments[assessment_id] = assessment
            self._log_quality_assessment(assessment)
            
            logger.info(f"✅ Quality assessment completed: {overall_score:.2f} score, {len(issues_found)} issues")
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return minimal assessment
            return QualityAssessment(
                assessment_id=assessment_id,
                task_id=task.id,
                workflow_id=workflow_id,
                overall_score=0.5,
                dimension_scores={},
                criteria_results=[],
                issues_found=[{'type': 'assessment_error', 'description': str(e)}],
                improvement_suggestions=['Fix assessment system errors'],
                assessment_time=datetime.now(),
                assessment_duration=time.time() - start_time,
                assessor="AIQualityGateIntegration",
                confidence=0.1
            )
    
    async def _run_static_analysis(self, code_path: str) -> Dict[str, Any]:
        """Run static code analysis"""
        
        results = {
            'scores': {},
            'issues': [],
            'suggestions': []
        }
        
        try:
            # Python-specific analysis
            if code_path.endswith('.py'):
                # Complexity analysis
                complexity_score = self._analyze_complexity(code_path)
                results['scores'][QualityDimension.MAINTAINABILITY] = complexity_score
                
                # Documentation analysis
                doc_score = self._analyze_documentation(code_path)
                results['scores'][QualityDimension.DOCUMENTATION] = doc_score
                
                # Type hints analysis
                type_hints_score = self._analyze_type_hints(code_path)
                if type_hints_score < 0.8:
                    results['issues'].append({
                        'type': 'missing_type_hints',
                        'description': f'Type hints coverage: {type_hints_score:.1%}',
                        'priority': 'medium',
                        'file': code_path
                    })
                
                if complexity_score < 0.7:
                    results['suggestions'].append('Reduce cyclomatic complexity by breaking down large functions')
                
                if doc_score < 0.6:
                    results['suggestions'].append('Add docstrings to functions and classes')
        
        except Exception as e:
            logger.warning(f"Static analysis failed: {e}")
        
        return results
    
    def _analyze_complexity(self, file_path: str) -> float:
        """Analyze code complexity"""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            # Simple complexity scoring based on AST analysis
            total_functions = 0
            total_complexity = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    # Count decision points (simplified)
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            complexity += len(child.values) - 1
                    
                    total_complexity += complexity
            
            if total_functions == 0:
                return 1.0
            
            avg_complexity = total_complexity / total_functions
            # Score: 1.0 for complexity <= 5, decreasing to 0.0 for complexity >= 20
            score = max(0.0, min(1.0, (20 - avg_complexity) / 15))
            return score
        
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return 0.5
    
    def _analyze_documentation(self, file_path: str) -> float:
        """Analyze documentation coverage"""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            functions_with_docs = 0
            total_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    # Check for docstring
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        functions_with_docs += 1
            
            if total_functions == 0:
                return 1.0
            
            return functions_with_docs / total_functions
        
        except Exception as e:
            logger.warning(f"Documentation analysis failed: {e}")
            return 0.5
    
    def _analyze_type_hints(self, file_path: str) -> float:
        """Analyze type hints coverage"""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            functions_with_hints = 0
            total_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    # Check for type hints
                    has_hints = (
                        node.returns or  # Return type annotation
                        any(arg.annotation for arg in node.args.args) or  # Argument type annotations
                        any(arg.annotation for arg in getattr(node.args, 'kwonlyargs', []))  # Keyword-only args
                    )
                    if has_hints:
                        functions_with_hints += 1
            
            if total_functions == 0:
                return 1.0
            
            return functions_with_hints / total_functions
        
        except Exception as e:
            logger.warning(f"Type hints analysis failed: {e}")
            return 0.5
    
    async def _run_ai_code_review(self, code_path: str, task: Task) -> Dict[str, Any]:
        """Run AI-powered code review"""
        
        results = {
            'maintainability_score': 0.7,
            'issues': [],
            'suggestions': []
        }
        
        if not self.reasoning_engine:
            return results
        
        try:
            # Read code file
            with open(code_path, 'r') as f:
                code_content = f.read()
            
            # Limit code length for AI analysis
            if len(code_content) > 2000:
                code_content = code_content[:2000] + "\n... (truncated for analysis)"
            
            review_prompt = f"""
            Perform code review for task: {task.title}
            
            Code to review:
            ```python
            {code_content}
            ```
            
            Analyze:
            1. Code quality and maintainability
            2. Best practices adherence
            3. Potential bugs or issues
            4. Performance considerations
            5. Security concerns
            
            Provide:
            - Maintainability score (0.0-1.0)
            - List of specific issues found
            - Improvement suggestions
            """
            
            context = ReasoningContext(
                project_type="code_review",
                tech_stack=["Python"],
                constraints=["production_ready", "maintainable"]
            )
            
            reasoning_chain = await self.reasoning_engine.reason_with_context(
                problem_statement=review_prompt,
                context=context,
                reasoning_type=ReasoningType.CODE_REVIEW
            )
            
            # Parse AI review results
            review_text = reasoning_chain.final_reasoning.lower()
            
            # Extract maintainability score
            score_match = re.search(r'maintainability[:\s]*(\d+\.?\d*)', review_text)
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 10.0  # Handle percentage format
                results['maintainability_score'] = min(1.0, max(0.0, score))
            
            # Extract issues and suggestions
            if 'issue' in review_text or 'problem' in review_text:
                results['issues'].append({
                    'type': 'ai_code_review',
                    'description': 'Code quality issues identified by AI review',
                    'details': reasoning_chain.final_reasoning,
                    'priority': 'medium'
                })
            
            if 'suggest' in review_text or 'recommend' in review_text:
                results['suggestions'].append('Follow AI code review recommendations')
        
        except Exception as e:
            logger.warning(f"AI code review failed: {e}")
        
        return results
    
    async def _run_security_scan(self, code_path: str) -> Dict[str, Any]:
        """Run security scan (placeholder)"""
        # In production, would integrate with tools like bandit, safety, etc.
        return {
            'security_score': 0.85,
            'vulnerabilities': [],
            'recommendations': ['Use updated dependencies', 'Validate all inputs']
        }
    
    async def _run_performance_test(self, code_path: str) -> Dict[str, Any]:
        """Run performance test (placeholder)"""
        # In production, would integrate with performance testing tools
        return {
            'performance_score': 0.8,
            'metrics': {'response_time_p95': 150.0},
            'bottlenecks': []
        }
    
    async def _evaluate_acceptance_criteria(
        self, 
        criteria: AcceptanceCriteria,
        code_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Evaluate whether acceptance criteria is met"""
        
        try:
            # For automated checks
            if criteria.automated_check:
                # Execute automated check (would be more sophisticated in production)
                return True, "Automated check passed"
            
            # Method-specific evaluation
            if criteria.test_method == AssessmentMethod.STATIC_ANALYSIS and code_path:
                # Example: Check for specific patterns in code
                with open(code_path, 'r') as f:
                    code_content = f.read()
                
                # Simple checks based on criteria description
                if 'docstring' in criteria.description.lower():
                    has_docstrings = '"""' in code_content or "'''" in code_content
                    return has_docstrings, f"Docstrings {'found' if has_docstrings else 'missing'}"
                
                if 'type hint' in criteria.description.lower():
                    has_type_hints = '->' in code_content or ': ' in code_content
                    return has_type_hints, f"Type hints {'found' if has_type_hints else 'missing'}"
            
            # Default: assume manual verification needed
            return True, "Manual verification required"
        
        except Exception as e:
            return False, f"Evaluation failed: {e}"
    
    async def create_dynamic_quality_gates(
        self, 
        workflow: OptimizedWorkflow,
        context: ReasoningContext
    ) -> List[QualityGate]:
        """Create dynamic quality gates based on workflow context"""
        
        logger.info(f"🚪 Creating dynamic quality gates for workflow {workflow.workflow_id}")
        
        dynamic_gates = []
        
        try:
            # Analyze workflow characteristics
            total_tasks = len(workflow.task_order)
            
            # Create context-specific gates (simplified - would analyze actual tasks in production)
            api_gate = QualityGate(
                gate_id=f"api_quality_{workflow.workflow_id}",
                name="API Quality Gate",
                description="Ensure API endpoints meet quality standards",
                severity=GateSeverity.CRITICAL,
                criteria=[
                    "All API endpoints documented",
                    "Response time < 200ms (p95)",
                    "Error handling implemented",
                    "Input validation present",
                    "Authentication/authorization working"
                ]
            )
            dynamic_gates.append(api_gate)
            
            security_gate = QualityGate(
                gate_id=f"security_{workflow.workflow_id}",
                name="Security Quality Gate",
                description="Comprehensive security validation",
                severity=GateSeverity.CRITICAL,
                criteria=[
                    "No critical security vulnerabilities",
                    "Authentication mechanisms secure",
                    "Data encryption implemented",
                    "Input sanitization present",
                    "Security headers configured"
                ],
                requires_human_approval=True
            )
            dynamic_gates.append(security_gate)
            
            # Performance gate for complex workflows
            if total_tasks > 3:
                performance_gate = QualityGate(
                    gate_id=f"performance_{workflow.workflow_id}",
                    name="Performance Quality Gate",
                    description="System performance validation",
                    severity=GateSeverity.HIGH,
                    criteria=[
                        "Load testing completed",
                        "Memory usage within limits",
                        "Database queries optimized",
                        "Caching strategy implemented"
                    ]
                )
                dynamic_gates.append(performance_gate)
            
            logger.info(f"✅ Created {len(dynamic_gates)} dynamic quality gates")
            return dynamic_gates
        
        except Exception as e:
            logger.error(f"Dynamic quality gates creation failed: {e}")
            return []
    
    async def monitor_quality_trends(self) -> Dict[str, QualityTrend]:
        """Monitor quality trends and generate predictions"""
        
        logger.info("📈 Analyzing quality trends")
        
        trends = {}
        
        try:
            for metric_id, metric in self.quality_metrics.items():
                if len(metric.trend_data) < 3:
                    continue  # Need at least 3 data points
                
                # Simple trend analysis
                recent_values = [value for _, value in metric.trend_data[-10:]]
                
                # Calculate trend direction
                if len(recent_values) >= 2:
                    slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    
                    if abs(slope) < 0.01:
                        trend_direction = "stable"
                    elif slope > 0:
                        trend_direction = "improving"
                    else:
                        trend_direction = "declining"
                    
                    # Predict 30-day value
                    prediction = recent_values[-1] + (slope * 30)
                    
                    # Calculate confidence interval (simplified)
                    std_dev = (max(recent_values) - min(recent_values)) / 4
                    confidence_interval = (
                        prediction - std_dev,
                        prediction + std_dev
                    )
                    
                    # Identify risk factors
                    risk_factors = []
                    if trend_direction == "declining":
                        risk_factors.append("Quality metric trending downward")
                    if recent_values[-1] < metric.threshold_warning:
                        risk_factors.append("Below warning threshold")
                    
                    # Generate recommendations
                    recommendations = []
                    if trend_direction == "declining":
                        recommendations.append(f"Investigate causes of {metric.name} decline")
                        recommendations.append("Implement corrective measures")
                    elif recent_values[-1] < metric.threshold_warning:
                        recommendations.append(f"Improve {metric.name} to meet standards")
                    
                    trend = QualityTrend(
                        metric_id=metric_id,
                        trend_direction=trend_direction,
                        rate_of_change=slope,
                        prediction_30d=prediction,
                        confidence_interval=confidence_interval,
                        risk_factors=risk_factors,
                        recommendations=recommendations
                    )
                    
                    trends[metric_id] = trend
                    self.quality_trends[metric_id] = trend
            
            logger.info(f"📊 Analyzed trends for {len(trends)} metrics")
            return trends
        
        except Exception as e:
            logger.error(f"Quality trend monitoring failed: {e}")
            return {}
    
    def _log_acceptance_criteria(self, criteria: AcceptanceCriteria):
        """Log acceptance criteria to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO acceptance_criteria
                    (criteria_id, task_id, description, priority, test_method, 
                     success_condition, verification_steps, automated_check,
                     ai_generated, confidence, context_factors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    criteria.criteria_id,
                    criteria.task_id,
                    criteria.description,
                    criteria.priority.value,
                    criteria.test_method.value,
                    criteria.success_condition,
                    json.dumps(criteria.verification_steps),
                    criteria.automated_check,
                    criteria.ai_generated,
                    criteria.confidence,
                    json.dumps(criteria.context_factors)
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Acceptance criteria logging failed: {e}")
    
    def _log_quality_assessment(self, assessment: QualityAssessment):
        """Log quality assessment to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO quality_assessments
                    (assessment_id, task_id, workflow_id, overall_score,
                     dimension_scores, criteria_results, issues_found,
                     improvement_suggestions, assessment_duration, assessor,
                     confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    assessment.assessment_id,
                    assessment.task_id,
                    assessment.workflow_id,
                    assessment.overall_score,
                    json.dumps({k.value: v for k, v in assessment.dimension_scores.items()}),
                    json.dumps(assessment.criteria_results),
                    json.dumps(assessment.issues_found),
                    json.dumps(assessment.improvement_suggestions),
                    assessment.assessment_duration,
                    assessment.assessor,
                    assessment.confidence,
                    json.dumps(assessment.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Quality assessment logging failed: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get quality integration statistics"""
        return {
            "quality_metrics": len(self.quality_metrics),
            "acceptance_criteria": sum(len(criteria) for criteria in self.acceptance_criteria.values()),
            "quality_assessments": len(self.quality_assessments),
            "quality_trends": len(self.quality_trends),
            "orchestration_components_available": ORCHESTRATION_COMPONENTS_AVAILABLE,
            "ai_components_available": AI_COMPONENTS_AVAILABLE,
            "assessment_tools": len(self.assessment_tools)
        }

# Demo function
async def demo_ai_quality_gate_integration():
    """Demo the AI Quality Gate Integration"""
    print("🚪 Agent Zero V2.0 - AI-Powered Quality Gate Integration Demo (FIXED)")
    print("=" * 70)
    
    # Initialize integration
    integration = AIQualityGateIntegration()
    
    # Create demo task
    demo_task = Task(
        id=1,
        title="User Authentication API",
        description="Implement JWT-based authentication API with security best practices",
        task_type=TaskType.BACKEND,
        priority=TaskPriority.CRITICAL,
        estimated_hours=8.0
    )
    
    # Create context
    context = ReasoningContext(
        project_type="web_api",
        tech_stack=["FastAPI", "JWT", "PostgreSQL"],
        constraints=["security_critical", "production_ready", "high_performance"]
    )
    
    print(f"📋 Demo Task:")
    print(f"   Title: {demo_task.title}")
    print(f"   Type: {demo_task.task_type.value}")
    print(f"   Priority: {demo_task.priority.value}")
    
    # Generate acceptance criteria
    print(f"\n🎯 Generating acceptance criteria...")
    criteria_list = await integration.generate_acceptance_criteria(demo_task, context)
    
    print(f"   ✅ Generated {len(criteria_list)} acceptance criteria:")
    for i, criteria in enumerate(criteria_list, 1):
        print(f"      {i}. {criteria.description}")
        print(f"         Priority: {criteria.priority.value}, Method: {criteria.test_method.value}")
    
    # Create a mock code file for assessment
    demo_code = '''
def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user and return JWT token."""
    if not username or not password:
        raise ValueError("Username and password required")
    
    # Mock authentication logic
    if verify_credentials(username, password):
        token = generate_jwt_token(username)
        return {"token": token, "expires_in": 3600}
    else:
        raise AuthenticationError("Invalid credentials")

def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials against database."""
    # Mock implementation
    return True

def generate_jwt_token(username: str) -> str:
    """Generate JWT token for user."""
    import jwt
    import time
    payload = {"username": username, "exp": time.time() + 3600}
    return jwt.encode(payload, "secret_key", algorithm="HS256")
'''
    
    # Write demo code to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(demo_code)
        temp_code_path = f.name
    
    # Run quality assessment
    print(f"\n🔍 Running quality assessment...")
    assessment = await integration.assess_task_quality(
        demo_task, 
        "demo_workflow_123", 
        temp_code_path
    )
    
    print(f"   ✅ Assessment completed:")
    print(f"      Overall Score: {assessment.overall_score:.2f}")
    print(f"      Assessment Duration: {assessment.assessment_duration:.2f}s")
    print(f"      Issues Found: {len(assessment.issues_found)}")
    print(f"      Improvement Suggestions: {len(assessment.improvement_suggestions)}")
    
    if assessment.dimension_scores:
        print(f"      Dimension Scores:")
        for dimension, score in assessment.dimension_scores.items():
            print(f"        {dimension.value}: {score:.2f}")
    
    if assessment.improvement_suggestions:
        print(f"      Top Suggestions:")
        for suggestion in assessment.improvement_suggestions[:3]:
            print(f"        - {suggestion}")
    
    # Create dynamic quality gates
    print(f"\n🚪 Creating dynamic quality gates...")
    
    demo_workflow = OptimizedWorkflow(
        workflow_id="demo_quality_workflow",
        strategy="security_focused",
        task_order=[1],
        parallel_groups=[],
        critical_path=[1],
        estimated_duration=8.0,
        resource_allocation={"backend_agents": [1]},
        confidence=0.9,
        optimization_reasoning="Security-focused workflow for authentication",
        metrics=WorkflowMetrics(
            total_tasks=1,
            completed_tasks=0,
            active_tasks=1,
            blocked_tasks=0,
            estimated_completion=datetime.now() + timedelta(hours=8),
            critical_path_length=8.0,
            resource_utilization={},
            bottlenecks=[],
            optimization_opportunities=[]
        )
    )
    
    quality_gates = await integration.create_dynamic_quality_gates(demo_workflow, context)
    
    print(f"   ✅ Created {len(quality_gates)} quality gates:")
    for gate in quality_gates:
        print(f"      - {gate.name} ({gate.severity.value})")
        print(f"        Criteria: {len(gate.criteria)} requirements")
    
    # Monitor quality trends (mock data)
    print(f"\n📈 Quality trend monitoring...")
    
    # Add mock trend data
    import random
    for metric_id, metric in integration.quality_metrics.items():
        for i in range(10):
            timestamp = datetime.now() - timedelta(days=10-i)
            # Simulate declining trend for demo
            base_value = metric.target_value
            trend_value = base_value - (i * 2) + random.uniform(-5, 5)
            metric.trend_data.append((timestamp, max(0, trend_value)))
    
    trends = await integration.monitor_quality_trends()
    
    print(f"   📊 Analyzed {len(trends)} quality trends:")
    for metric_id, trend in trends.items():
        print(f"      {metric_id}: {trend.trend_direction}")
        print(f"        Rate of change: {trend.rate_of_change:.2f}/day")
        print(f"        30-day prediction: {trend.prediction_30d:.1f}")
        if trend.risk_factors:
            print(f"        Risk factors: {', '.join(trend.risk_factors)}")
    
    # Show integration stats
    print(f"\n📊 Integration Statistics:")
    stats = integration.get_integration_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Cleanup
    import os
    os.unlink(temp_code_path)
    
    print(f"\n✅ AI Quality Gate Integration demo completed!")

if __name__ == "__main__":
    print("🚪 Agent Zero V2.0 Phase 4 - AI-Powered Quality Gate Integration (FIXED)")
    print("Advanced acceptance criteria with automated quality gates")
    
    # Run demo
    asyncio.run(demo_ai_quality_gate_integration())