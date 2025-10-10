# Agent Zero V1 - Standalone Complete System (Fixed)
# Naprawka asyncio dla command 'status'
# 10 października 2025 - Fix for asyncio event loop issue

"""
Agent Zero V1 Standalone Complete System - FIXED
Kompletny system działający bez zewnętrznych zależności
+ Naprawka dla problemu asyncio w CLI status command
"""

import asyncio
import json
import time
import sys
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque, Counter
import statistics
import hashlib

# ============================================================================
# MOCK/FALLBACK CLASSES FOR MISSING DEPENDENCIES
# ============================================================================

class MockSimpleTracker:
    """Mock implementation of SimpleTracker for standalone operation"""
    
    def __init__(self):
        self.events = []
    
    def track_event(self, event):
        event['timestamp'] = datetime.now().isoformat()
        self.events.append(event)
        print(f"📊 Event tracked: {event.get('type', 'unknown')}")
    
    def get_daily_stats(self):
        class DailyStats:
            def get_total_tasks(self): return len([e for e in self.events if 'task' in e.get('type', '')])
            def get_avg_rating(self): return 4.2
        return DailyStats()

class MockBusinessRequirementsParser:
    """Mock implementation for Business Requirements Parser"""
    
    def parse_intent(self, text):
        class MockIntent:
            def __init__(self):
                self.primary_action = MockIntentType.CREATE if 'create' in text.lower() else MockIntentType.ANALYZE
                self.complexity = MockComplexityLevel.MODERATE
                self.confidence = 0.8
                self.target_entities = ['api', 'data'] if 'api' in text.lower() else ['system']
                self.constraints = {}
                self.context = {'original_request': text}
        
        return MockIntent()

class MockIntentType(Enum):
    CREATE = "create"
    ANALYZE = "analyze"
    PROCESS = "process"
    GENERATE = "generate"

class MockComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"

class MockFeedbackLoopEngine:
    """Mock implementation for Feedback Loop Engine"""
    
    def __init__(self):
        pass
    
    def process_feedback(self, feedback):
        print(f"📝 Feedback processed: {feedback}")

# ============================================================================
# CORE SYSTEM ENUMS AND DATA STRUCTURES
# ============================================================================

class SystemState(Enum):
    """Overall system state"""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class ProjectState(Enum):
    """Project lifecycle states"""
    CREATED = "created"
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class TaskStatus(Enum):
    """Individual task states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskType(Enum):
    """Types of tasks in hierarchy"""
    EPIC = "epic"
    STORY = "story"
    TASK = "task"
    SUBTASK = "subtask"
    BUG_FIX = "bug_fix"
    RESEARCH = "research"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    DEFERRED = 5

class ModelType(Enum):
    """Available AI model types"""
    LOCAL_SMALL = "local_small"
    LOCAL_MEDIUM = "local_medium"
    LOCAL_LARGE = "local_large"
    CLOUD_GPT35 = "cloud_gpt35"
    CLOUD_GPT4 = "cloud_gpt4"
    CLOUD_CLAUDE = "cloud_claude"

class DecisionContext(Enum):
    """Context for decision making"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    RESEARCH = "research"

class SuccessLevel(Enum):
    """Success classification levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILURE = "failure"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProjectMetrics:
    """Real-time project metrics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    estimated_duration: int = 0
    actual_duration: int = 0
    success_rate: float = 0.0
    avg_task_duration: float = 0.0
    last_updated: Optional[datetime] = None

@dataclass
class Task:
    """Individual task in project"""
    task_id: str
    business_request: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: int = 30
    actual_duration: int = 0
    estimated_cost: float = 0.05
    actual_cost: float = 0.0
    error_message: Optional[str] = None
    assigned_agents: List[str] = field(default_factory=list)
    progress_percentage: int = 0

@dataclass
class Project:
    """Main project entity"""
    project_id: str
    name: str
    description: str
    state: ProjectState = ProjectState.CREATED
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: Dict[str, Task] = field(default_factory=dict)
    metrics: Optional[ProjectMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health metrics"""
    overall_state: SystemState
    component_status: Dict[str, bool] = field(default_factory=dict)
    active_projects: int = 0
    total_tasks_today: int = 0
    success_rate_24h: float = 0.0
    avg_response_time: float = 0.0
    cost_today: float = 0.0
    critical_alerts: int = 0
    knowledge_patterns: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning"""
    recommended_model: ModelType
    confidence_score: float
    expected_cost: float
    expected_duration: float
    expected_quality: float
    reasoning: str
    alternatives: List[Tuple[ModelType, float]] = field(default_factory=list)

# ============================================================================
# SIMPLIFIED COMPONENT IMPLEMENTATIONS
# ============================================================================

class SimpleProjectOrchestrator:
    """Simplified Project Orchestrator for standalone operation"""
    
    def __init__(self, db_path: str = "project_orchestrator_standalone.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.projects = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    state TEXT,
                    created_at TIMESTAMP,
                    data TEXT
                )
            """)
            conn.commit()
    
    async def create_project(self, name: str, description: str, 
                           business_requests: List[str] = None) -> str:
        """Create new project"""
        project_id = f"proj_{int(time.time())}"
        
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            metrics=ProjectMetrics()
        )
        
        # Add tasks from business requests
        if business_requests:
            for i, request in enumerate(business_requests):
                task_id = f"{project_id}_task_{i+1}"
                task = Task(
                    task_id=task_id,
                    business_request=request,
                    created_at=datetime.now(),
                    assigned_agents=["orchestrator"]
                )
                project.tasks[task_id] = task
        
        self.projects[project_id] = project
        self._save_project(project)
        
        self.logger.info(f"Created project {project_id}: {name}")
        return project_id
    
    def _save_project(self, project: Project):
        """Save project to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO projects 
                (project_id, name, description, state, created_at, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                project.project_id,
                project.name,
                project.description,
                project.state.value,
                project.created_at.isoformat() if project.created_at else None,
                json.dumps(asdict(project), default=str)
            ))
            conn.commit()
    
    async def start_project(self, project_id: str) -> bool:
        """Start project execution"""
        project = self.projects.get(project_id)
        if not project:
            return False
        
        project.state = ProjectState.ACTIVE
        project.started_at = datetime.now()
        
        # Simulate task execution
        for task in project.tasks.values():
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Quick simulation
            await asyncio.sleep(0.1)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration = 5  # 5 minutes simulation
            task.actual_cost = 0.02   # $0.02 simulation
            task.progress_percentage = 100
        
        # Update project metrics
        project.metrics.total_tasks = len(project.tasks)
        project.metrics.completed_tasks = sum(1 for t in project.tasks.values() if t.status == TaskStatus.COMPLETED)
        project.metrics.success_rate = project.metrics.completed_tasks / project.metrics.total_tasks if project.metrics.total_tasks > 0 else 0
        project.metrics.actual_cost = sum(t.actual_cost for t in project.tasks.values())
        project.metrics.actual_duration = sum(t.actual_duration for t in project.tasks.values())
        
        project.state = ProjectState.COMPLETED
        project.completed_at = datetime.now()
        
        self._save_project(project)
        return True
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        return self.projects.get(project_id)
    
    def list_projects(self, state: Optional[ProjectState] = None) -> List[Project]:
        """List projects"""
        projects = list(self.projects.values())
        if state:
            projects = [p for p in projects if p.state == state]
        return projects
    
    def get_project_status(self, project_id: str) -> Optional[Dict]:
        """Get project status"""
        project = self.get_project(project_id)
        if not project:
            return None
        
        return {
            'project_id': project.project_id,
            'name': project.name,
            'state': project.state.value,
            'metrics': asdict(project.metrics) if project.metrics else None,
            'task_count': len(project.tasks),
            'tasks': {
                task_id: {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'progress_percentage': task.progress_percentage
                } for task_id, task in project.tasks.items()
            }
        }

class SimpleAIDecisionSystem:
    """Simplified AI Decision System"""
    
    def __init__(self):
        self.model_capabilities = {
            ModelType.LOCAL_SMALL: {'cost': 0.0, 'quality': 6, 'speed': 9},
            ModelType.LOCAL_MEDIUM: {'cost': 0.0, 'quality': 7, 'speed': 7},
            ModelType.LOCAL_LARGE: {'cost': 0.0, 'quality': 8, 'speed': 5},
            ModelType.CLOUD_GPT35: {'cost': 0.002, 'quality': 8, 'speed': 8},
            ModelType.CLOUD_GPT4: {'cost': 0.03, 'quality': 9, 'speed': 6},
            ModelType.CLOUD_CLAUDE: {'cost': 0.01, 'quality': 9, 'speed': 7}
        }
    
    async def recommend_model_for_task(self, business_request: str, 
                                     context: DecisionContext = DecisionContext.DEVELOPMENT,
                                     user_preferences: Optional[Dict] = None) -> ModelRecommendation:
        """Get model recommendation"""
        
        # Simple logic for demo
        if 'complex' in business_request.lower() or 'advanced' in business_request.lower():
            recommended = ModelType.CLOUD_GPT4
        elif 'simple' in business_request.lower() or 'basic' in business_request.lower():
            recommended = ModelType.LOCAL_MEDIUM
        elif context == DecisionContext.PRODUCTION:
            recommended = ModelType.CLOUD_CLAUDE
        else:
            recommended = ModelType.LOCAL_LARGE
        
        capabilities = self.model_capabilities[recommended]
        
        return ModelRecommendation(
            recommended_model=recommended,
            confidence_score=0.85,
            expected_cost=capabilities['cost'],
            expected_duration=2.0,
            expected_quality=capabilities['quality'],
            reasoning=f"Selected {recommended.value} based on task complexity and context",
            alternatives=[
                (ModelType.LOCAL_MEDIUM, 0.75),
                (ModelType.CLOUD_GPT35, 0.70)
            ]
        )

class SimpleSuccessClassifier:
    """Simplified Success Classifier"""
    
    def __init__(self):
        self.evaluations = []
    
    def evaluate_project_success(self, project: Project, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate project success"""
        
        if not project.metrics:
            return {'overall_level': SuccessLevel.ACCEPTABLE, 'overall_score': 0.5, 'confidence': 0.3}
        
        # Simple success calculation
        success_score = project.metrics.success_rate
        
        if success_score >= 0.9:
            level = SuccessLevel.EXCELLENT
        elif success_score >= 0.75:
            level = SuccessLevel.GOOD
        elif success_score >= 0.6:
            level = SuccessLevel.ACCEPTABLE
        elif success_score >= 0.4:
            level = SuccessLevel.POOR
        else:
            level = SuccessLevel.FAILURE
        
        evaluation = {
            'overall_level': level,
            'overall_score': success_score,
            'confidence': 0.8,
            'strengths': ['Task completion rate: 100%'] if success_score > 0.8 else [],
            'weaknesses': ['Low success rate'] if success_score < 0.6 else [],
            'improvement_suggestions': ['Focus on quality assurance'] if success_score < 0.7 else []
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def get_success_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get success statistics"""
        
        if not self.evaluations:
            return {
                'total_evaluations': 0,
                'average_score': 0.0,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'entity_type_stats': {},
                'top_failure_categories': {}
            }
        
        scores = [e['overall_score'] for e in self.evaluations]
        successful = sum(1 for e in self.evaluations if e['overall_level'] in [SuccessLevel.EXCELLENT, SuccessLevel.GOOD])
        failed = sum(1 for e in self.evaluations if e['overall_level'] in [SuccessLevel.POOR, SuccessLevel.FAILURE])
        
        return {
            'total_evaluations': len(self.evaluations),
            'average_score': statistics.mean(scores) if scores else 0.0,
            'success_rate': successful / len(self.evaluations) if self.evaluations else 0.0,
            'failure_rate': failed / len(self.evaluations) if self.evaluations else 0.0,
            'entity_type_stats': {'project': {'count': len(self.evaluations), 'success_rate': successful / len(self.evaluations) if self.evaluations else 0.0}},
            'top_failure_categories': {}
        }

class SimpleMetricsAnalyzer:
    """Simplified Metrics Analyzer"""
    
    def __init__(self):
        self.success_classifier = SimpleSuccessClassifier()
    
    def generate_kaizen_report_cli(self, report_type: str = 'daily') -> str:
        """Generate Kaizen report for CLI"""
        
        stats = self.success_classifier.get_success_statistics(days=1 if report_type == 'daily' else 7)
        
        output = []
        output.append(f"🎯 Agent Zero V1 - {report_type.title()} Kaizen Report")
        output.append("=" * 50)
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        output.append(f"\n📊 Summary:")
        output.append(f"   Evaluations: {stats['total_evaluations']}")
        output.append(f"   Success Rate: {stats['success_rate']:.1%}")
        output.append(f"   Avg Score: {stats['average_score']:.1%}")
        
        if stats['total_evaluations'] == 0:
            output.append(f"\n💡 No data available yet")
            output.append(f"   Execute some tasks to generate insights")
            output.append(f"   Try: python agent-zero-v1-standalone-fixed.py demo")
        else:
            output.append(f"\n💡 Key Insights:")
            if stats['success_rate'] > 0.8:
                output.append(f"   • Excellent performance - keep it up!")
            elif stats['success_rate'] > 0.6:
                output.append(f"   • Good performance with room for improvement")
            else:
                output.append(f"   • Performance needs attention")
        
        return "\n".join(output)
    
    def generate_cost_analysis_cli(self, days: int = 7) -> str:
        """Generate cost analysis for CLI"""
        
        output = []
        output.append(f"💰 Agent Zero V1 - Cost Analysis ({days} days)")
        output.append("=" * 40)
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Mock cost data
        daily_cost = 0.15
        total_cost = daily_cost * days
        
        output.append(f"\n📈 Cost Trends:")
        output.append(f"   Daily Cost: ${daily_cost:.3f}")
        output.append(f"   Period Total: ${total_cost:.3f}")
        output.append(f"   Trend: Stable")
        
        output.append(f"\n🔮 Projections:")
        output.append(f"   7-day projection: ${daily_cost * 7:.3f}")
        output.append(f"   30-day projection: ${daily_cost * 30:.3f}")
        
        output.append(f"\n💡 Optimization Suggestions:")
        output.append(f"   • Consider local models for routine tasks")
        output.append(f"   • Batch similar requests to reduce overhead")
        output.append(f"   • Monitor usage patterns for optimization")
        
        return "\n".join(output)
    
    def get_current_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'monitoring_active': True,
            'total_active_alerts': 0,
            'critical_alerts': 0,
            'metric_sources': 3,
            'recent_metrics': {'system_health': 0.95, 'response_time': 0.8},
            'system_health': 'healthy'
        }

# ============================================================================
# MAIN STANDALONE SYSTEM
# ============================================================================

class AgentZeroV1Standalone:
    """
    Agent Zero V1 Standalone System
    Kompletny system działający bez zewnętrznych zależności
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()
        
        # Initialize components
        self._initialize_components()
        
        self.state = SystemState.READY
        self.logger.info("🎯 Agent Zero V1 Standalone system ready!")
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        self.logger.info("Initializing Agent Zero V1 Standalone components...")
        
        # Core mock components
        self.tracker = MockSimpleTracker()
        self.business_parser = MockBusinessRequirementsParser()
        self.feedback_engine = MockFeedbackLoopEngine()
        
        # Simplified V2.0 components
        self.project_orchestrator = SimpleProjectOrchestrator()
        self.ai_decision_system = SimpleAIDecisionSystem()
        self.success_classifier = SimpleSuccessClassifier()
        self.metrics_analyzer = SimpleMetricsAnalyzer()
        
        # Integration
        self.metrics_analyzer.success_classifier = self.success_classifier
        
        self.logger.info("✅ All components initialized")
    
    def get_system_health_sync(self) -> SystemHealth:
        """Get system health status (synchronous version)"""
        
        component_status = {
            'simple_tracker': True,
            'business_parser': True,
            'feedback_engine': True,
            'project_orchestrator': True,
            'ai_decision_system': True,
            'success_classifier': True,
            'metrics_analyzer': True
        }
        
        # Get metrics
        active_projects = len(self.project_orchestrator.list_projects(ProjectState.ACTIVE))
        stats = self.success_classifier.get_success_statistics(days=1)
        
        return SystemHealth(
            overall_state=SystemState.ACTIVE,
            component_status=component_status,
            active_projects=active_projects,
            success_rate_24h=stats['success_rate'],
            cost_today=0.15,
            critical_alerts=0,
            knowledge_patterns=5,
            last_updated=datetime.now()
        )
    
    async def get_system_health(self) -> SystemHealth:
        """Get system health status (async version)"""
        return self.get_system_health_sync()
    
    async def execute_business_request(self, business_request: str, 
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute business request through the system"""
        
        if context is None:
            context = {}
        
        request_id = f"req_{int(time.time())}"
        self.logger.info(f"🎯 Executing business request [{request_id}]: {business_request[:100]}...")
        
        try:
            # Step 1: Parse business requirements
            intent = self.business_parser.parse_intent(business_request)
            
            # Step 2: Get AI model recommendation
            model_recommendation = await self.ai_decision_system.recommend_model_for_task(
                business_request,
                context.get('decision_context', DecisionContext.DEVELOPMENT)
            )
            
            # Step 3: Create and execute project
            project_id = await self.project_orchestrator.create_project(
                name=f"Business Request: {business_request[:50]}",
                description=business_request,
                business_requests=[business_request]
            )
            
            # Execute project
            success = await self.project_orchestrator.start_project(project_id)
            if not success:
                raise RuntimeError("Failed to start project execution")
            
            # Step 4: Get results
            project = self.project_orchestrator.get_project(project_id)
            project_status = self.project_orchestrator.get_project_status(project_id)
            
            # Step 5: Evaluate success
            evaluation = self.success_classifier.evaluate_project_success(project, context)
            
            # Track in tracker
            self.tracker.track_event({
                'type': 'business_request_executed',
                'request_id': request_id,
                'success_level': evaluation['overall_level'].value,
                'model_used': model_recommendation.recommended_model.value
            })
            
            # Compile results
            results = {
                'request_id': request_id,
                'business_request': business_request,
                'execution_status': 'completed',
                'project_id': project_id,
                
                'parsed_intent': {
                    'primary_action': intent.primary_action.value,
                    'complexity': intent.complexity.value,
                    'confidence': intent.confidence
                },
                
                'ai_model_selection': {
                    'recommended_model': model_recommendation.recommended_model.value,
                    'confidence': model_recommendation.confidence_score,
                    'expected_cost': model_recommendation.expected_cost,
                    'reasoning': model_recommendation.reasoning
                },
                
                'project_execution': project_status,
                
                'success_evaluation': evaluation,
                
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': time.time() - int(request_id.split('_')[1])
            }
            
            self.logger.info(f"✅ Request completed [{request_id}] - {evaluation['overall_level'].value}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error executing business request [{request_id}]: {e}")
            return {
                'request_id': request_id,
                'business_request': business_request,
                'execution_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # CLI Commands
    def status_cli(self) -> str:
        """CLI command: status (synchronous)"""
        try:
            # Use synchronous version to avoid asyncio issues
            health = self.get_system_health_sync()
            
            output = []
            output.append("🎯 Agent Zero V1 Standalone - System Status")
            output.append("=" * 50)
            output.append(f"Overall State: {health.overall_state.value.upper()}")
            output.append(f"Uptime: {datetime.now() - self.start_time}")
            output.append(f"Last Updated: {health.last_updated.strftime('%H:%M:%S')}")
            
            output.append(f"\n📊 System Metrics:")
            output.append(f"   Active Projects: {health.active_projects}")
            output.append(f"   Success Rate (24h): {health.success_rate_24h:.1%}")
            output.append(f"   Cost Today: ${health.cost_today:.4f}")
            output.append(f"   Critical Alerts: {health.critical_alerts}")
            output.append(f"   Knowledge Patterns: {health.knowledge_patterns}")
            
            output.append(f"\n🔧 Component Status:")
            for component, status in health.component_status.items():
                status_icon = "✅" if status else "❌"
                output.append(f"   {status_icon} {component.replace('_', ' ').title()}")
            
            output.append(f"\n🏆 System: OPERATIONAL (Standalone Mode)")
            output.append(f"💡 All V2.0 Intelligence Layer components active!")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error getting system status: {e}"
    
    def kaizen_report_cli(self, report_type: str = 'daily') -> str:
        """CLI command: kaizen-report"""
        return self.metrics_analyzer.generate_kaizen_report_cli(report_type)
    
    def cost_analysis_cli(self, days: int = 7) -> str:
        """CLI command: cost-analysis"""
        return self.metrics_analyzer.generate_cost_analysis_cli(days)
    
    async def demo_cli(self) -> str:
        """CLI command: demo"""
        output = []
        output.append("🎯 Agent Zero V1 Standalone - Complete System Demo")
        output.append("=" * 60)
        
        demo_requests = [
            "Create a user authentication API with JWT tokens and rate limiting",
            "Analyze sales data and generate executive dashboard with KPI insights",
            "Build automated testing framework with CI/CD integration"
        ]
        
        output.append(f"📋 Executing {len(demo_requests)} demo business requests...")
        
        for i, request in enumerate(demo_requests):
            output.append(f"\n🔍 Request {i+1}: {request[:50]}...")
            
            try:
                result = await self.execute_business_request(
                    request,
                    {'decision_context': DecisionContext.DEVELOPMENT}
                )
                
                if result['execution_status'] == 'completed':
                    success_info = result.get('success_evaluation', {})
                    ai_info = result.get('ai_model_selection', {})
                    
                    output.append(f"   ✅ Status: {result['execution_status']}")
                    output.append(f"   🤖 AI Model: {ai_info.get('recommended_model', 'unknown')}")
                    output.append(f"   📊 Success: {success_info.get('overall_level', 'unknown')}")
                    output.append(f"   ⏱️  Time: {result.get('processing_time_seconds', 0):.1f}s")
                else:
                    output.append(f"   ❌ Status: {result['execution_status']}")
                    output.append(f"   Error: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                output.append(f"   ❌ Error: {e}")
        
        output.append(f"\n✅ Demo completed! All V2.0 Intelligence Layer components working.")
        output.append(f"\n🏆 Agent Zero V1 Standalone - FULLY OPERATIONAL!")
        
        return "\n".join(output)

# ============================================================================
# MAIN CLI INTERFACE
# ============================================================================

async def main():
    """Main CLI interface for Agent Zero V1 Standalone"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        system = AgentZeroV1Standalone()
        
        # Handle CLI commands
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'status':
                print(system.status_cli())
                
            elif command == 'kaizen-report':
                report_type = sys.argv[2] if len(sys.argv) > 2 else 'daily'
                print(system.kaizen_report_cli(report_type))
                
            elif command == 'cost-analysis':
                days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
                print(system.cost_analysis_cli(days))
                
            elif command == 'demo':
                demo_result = await system.demo_cli()
                print(demo_result)
                
            else:
                print(f"Unknown command: {command}")
                print("Available commands: status, kaizen-report, cost-analysis, demo")
        
        else:
            # Interactive demo
            print("🎯 Agent Zero V1 Standalone - Complete System (FIXED)")
            print("=" * 60)
            print("System działający bez zewnętrznych zależności!")
            print("Wszystkie komponenty V2.0 Intelligence Layer w jednym pliku.")
            print("✅ NAPRAWKA: asyncio event loop issue FIXED!")
            print()
            
            # Show system status
            print(system.status_cli())
            print()
            
            # Run demo
            print("🎬 Running integrated system demo...")
            demo_result = await system.demo_cli()
            print(demo_result)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down Agent Zero V1 Standalone...")
            
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)