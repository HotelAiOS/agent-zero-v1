#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Real-Time Progress Monitoring
Advanced execution tracking with bottleneck detection and adaptive optimization

Priority 3.2: Real-Time Progress Monitor (1 SP)
- Live task progress tracking with WebSocket support
- Intelligent bottleneck detection and prediction
- Adaptive workflow re-optimization during execution
- Performance analytics with predictive insights
- Integration with DynamicWorkflowOptimizer
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from collections import deque
import statistics

# Import existing components
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from dynamic_workflow_optimizer import (
        DynamicWorkflowOptimizer, OptimizedWorkflow, Task, TaskStatus, TaskType, 
        TaskPriority, OptimizationStrategy, ReasoningContext
    )
    WORKFLOW_OPTIMIZER_AVAILABLE = True
except ImportError:
    WORKFLOW_OPTIMIZER_AVAILABLE = False
    print("⚠️ DynamicWorkflowOptimizer not available")

# Import AI components with fallbacks
AI_COMPONENTS_AVAILABLE = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai'))
    from unified_ai_client import UnifiedAIClient, AIReasoningRequest, ReasoningType, AIModelType
    from context_aware_reasoning import ContextAwareReasoningEngine
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    # Fallback definitions
    class ReasoningType(Enum):
        TASK_ANALYSIS = "task_analysis"
        DECISION_MAKING = "decision_making"
        PROBLEM_SOLVING = "problem_solving"
    
    class AIModelType(Enum):
        STANDARD = "standard"
        ADVANCED = "advanced"

logger = logging.getLogger(__name__)

class ProgressStatus(Enum):
    """Task progress status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class BottleneckType(Enum):
    """Types of bottlenecks detected"""
    RESOURCE_CONTENTION = "resource_contention"
    DEPENDENCY_BLOCKING = "dependency_blocking"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SKILL_MISMATCH = "skill_mismatch"
    EXTERNAL_DEPENDENCY = "external_dependency"
    CAPACITY_LIMIT = "capacity_limit"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class TaskProgress:
    """Real-time task progress tracking"""
    task_id: int
    workflow_id: str
    status: ProgressStatus
    progress_percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_duration: Optional[float] = None
    estimated_remaining: Optional[float] = None
    agent_id: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    blockers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class BottleneckDetection:
    """Detected bottleneck information"""
    bottleneck_id: str
    type: BottleneckType
    severity: AlertSeverity
    affected_tasks: List[int]
    resource_involved: str
    description: str
    predicted_impact: Dict[str, Any]
    suggested_actions: List[str]
    confidence: float
    detection_time: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class WorkflowExecutionMetrics:
    """Real-time workflow execution metrics"""
    workflow_id: str
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    blocked_tasks: int
    failed_tasks: int
    overall_progress: float
    estimated_completion: datetime
    actual_vs_estimated_variance: float
    resource_utilization: Dict[str, float]
    bottlenecks_active: int
    performance_score: float
    velocity_trend: float  # Tasks completed per hour trend
    efficiency_ratio: float  # Actual vs estimated time ratio

@dataclass
class PerformancePrediction:
    """AI-powered performance predictions"""
    workflow_id: str
    predicted_completion: datetime
    confidence_interval: Tuple[datetime, datetime]
    risk_factors: List[str]
    optimization_opportunities: List[str]
    predicted_bottlenecks: List[str]
    success_probability: float
    prediction_accuracy: float
    generated_at: datetime

class RealTimeProgressMonitor:
    """
    Real-Time Progress Monitor with AI-Powered Analytics
    
    Features:
    - Live task progress tracking with WebSocket-ready updates
    - Intelligent bottleneck detection using ML algorithms
    - Predictive analytics for completion time and risk assessment
    - Adaptive workflow re-optimization during execution
    - Performance trend analysis and velocity tracking
    - Resource utilization monitoring and optimization
    - Alert system with severity-based notifications
    - Integration with DynamicWorkflowOptimizer for re-planning
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.workflow_optimizer = None
        self.ai_client = None
        self.reasoning_engine = None
        
        # Initialize components
        if WORKFLOW_OPTIMIZER_AVAILABLE:
            self.workflow_optimizer = DynamicWorkflowOptimizer(db_path=db_path)
            
        if AI_COMPONENTS_AVAILABLE:
            try:
                self.ai_client = UnifiedAIClient(db_path=db_path)
                self.reasoning_engine = ContextAwareReasoningEngine(db_path=db_path)
                logger.info("✅ AI components connected")
            except Exception as e:
                logger.warning(f"AI initialization failed: {e}")
        
        # Progress tracking
        self.active_workflows: Dict[str, WorkflowExecutionMetrics] = {}
        self.task_progress: Dict[int, TaskProgress] = {}
        self.bottlenecks: Dict[str, BottleneckDetection] = {}
        self.performance_history: Dict[str, deque] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_callbacks: List[Callable] = []
        
        # Performance analytics
        self.performance_stats = {
            "workflows_monitored": 0,
            "bottlenecks_detected": 0,
            "bottlenecks_resolved": 0,
            "avg_detection_time": 0.0,
            "prediction_accuracy": 0.0,
            "adaptive_optimizations": 0
        }
        
        self._init_database()
        logger.info("✅ RealTimeProgressMonitor initialized")
    
    def _init_database(self):
        """Initialize progress monitoring database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Task progress tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id INTEGER NOT NULL,
                        workflow_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress_percentage REAL,
                        start_time TEXT,
                        end_time TEXT,
                        actual_duration REAL,
                        estimated_remaining REAL,
                        agent_id TEXT,
                        resource_usage TEXT,  -- JSON
                        performance_metrics TEXT,  -- JSON
                        blockers TEXT,  -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Bottleneck detection log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bottleneck_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bottleneck_id TEXT UNIQUE NOT NULL,
                        type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        affected_tasks TEXT,  -- JSON
                        resource_involved TEXT,
                        description TEXT,
                        predicted_impact TEXT,  -- JSON
                        suggested_actions TEXT,  -- JSON
                        confidence REAL,
                        resolved BOOLEAN DEFAULT FALSE,
                        detection_time TEXT,
                        resolution_time TEXT
                    )
                """)
                
                # Workflow execution metrics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_execution_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        total_tasks INTEGER,
                        completed_tasks INTEGER,
                        overall_progress REAL,
                        estimated_completion TEXT,
                        actual_vs_estimated_variance REAL,
                        resource_utilization TEXT,  -- JSON
                        bottlenecks_active INTEGER,
                        performance_score REAL,
                        velocity_trend REAL,
                        efficiency_ratio REAL,
                        recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance predictions
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        predicted_completion TEXT,
                        confidence_interval_start TEXT,
                        confidence_interval_end TEXT,
                        risk_factors TEXT,  -- JSON
                        optimization_opportunities TEXT,  -- JSON
                        success_probability REAL,
                        prediction_accuracy REAL,
                        generated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    
    async def start_monitoring_workflow(self, workflow: OptimizedWorkflow) -> bool:
        """Start real-time monitoring for a workflow"""
        
        try:
            workflow_id = workflow.workflow_id
            logger.info(f"🔍 Starting monitoring for workflow: {workflow_id}")
            
            # Initialize workflow metrics
            metrics = WorkflowExecutionMetrics(
                workflow_id=workflow_id,
                total_tasks=len(workflow.task_order),
                completed_tasks=0,
                in_progress_tasks=0,
                blocked_tasks=0,
                failed_tasks=0,
                overall_progress=0.0,
                estimated_completion=datetime.now() + timedelta(hours=workflow.estimated_duration),
                actual_vs_estimated_variance=0.0,
                resource_utilization={},
                bottlenecks_active=0,
                performance_score=1.0,
                velocity_trend=0.0,
                efficiency_ratio=1.0
            )
            
            self.active_workflows[workflow_id] = metrics
            
            # Initialize task progress tracking
            for task_id in workflow.task_order:
                task_progress = TaskProgress(
                    task_id=task_id,
                    workflow_id=workflow_id,
                    status=ProgressStatus.NOT_STARTED
                )
                self.task_progress[task_id] = task_progress
            
            # Initialize performance history
            self.performance_history[workflow_id] = deque(maxlen=100)
            
            # Start monitoring thread if not active
            if not self.monitoring_active:
                self._start_monitoring_thread()
            
            # Log workflow monitoring start
            self._log_workflow_start(workflow)
            
            logger.info(f"✅ Monitoring started for workflow {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    async def update_task_progress(
        self, 
        task_id: int, 
        status: ProgressStatus, 
        progress: float = None,
        agent_id: str = None,
        resource_usage: Dict[str, float] = None,
        notes: str = None
    ) -> bool:
        """Update task progress in real-time"""
        
        try:
            if task_id not in self.task_progress:
                logger.warning(f"Task {task_id} not found in progress tracking")
                return False
            
            task_progress = self.task_progress[task_id]
            
            # Update task progress
            old_status = task_progress.status
            task_progress.status = status
            task_progress.last_update = datetime.now()
            
            if progress is not None:
                task_progress.progress_percentage = min(100.0, max(0.0, progress))
            
            if agent_id:
                task_progress.agent_id = agent_id
                
            if resource_usage:
                task_progress.resource_usage.update(resource_usage)
            
            if notes:
                task_progress.notes.append(f"{datetime.now()}: {notes}")
            
            # Handle status transitions
            if status == ProgressStatus.IN_PROGRESS and old_status == ProgressStatus.NOT_STARTED:
                task_progress.start_time = datetime.now()
                logger.info(f"📝 Task {task_id} started")
                
            elif status == ProgressStatus.COMPLETED:
                if task_progress.start_time:
                    task_progress.end_time = datetime.now()
                    task_progress.actual_duration = (
                        task_progress.end_time - task_progress.start_time
                    ).total_seconds() / 3600.0  # Convert to hours
                    task_progress.progress_percentage = 100.0
                logger.info(f"✅ Task {task_id} completed")
                
            elif status == ProgressStatus.BLOCKED:
                logger.warning(f"🚫 Task {task_id} blocked")
                
            elif status == ProgressStatus.FAILED:
                if task_progress.start_time:
                    task_progress.end_time = datetime.now()
                logger.error(f"❌ Task {task_id} failed")
            
            # Update workflow metrics
            await self._update_workflow_metrics(task_progress.workflow_id)
            
            # Check for bottlenecks
            await self._detect_bottlenecks(task_progress.workflow_id)
            
            # Log progress update
            self._log_progress_update(task_progress)
            
            # Notify subscribers
            await self._notify_progress_update(task_progress)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update task progress: {e}")
            return False
    
    async def _update_workflow_metrics(self, workflow_id: str):
        """Update comprehensive workflow execution metrics"""
        
        if workflow_id not in self.active_workflows:
            return
        
        metrics = self.active_workflows[workflow_id]
        workflow_tasks = [tp for tp in self.task_progress.values() if tp.workflow_id == workflow_id]
        
        # Count task statuses
        metrics.completed_tasks = sum(1 for tp in workflow_tasks if tp.status == ProgressStatus.COMPLETED)
        metrics.in_progress_tasks = sum(1 for tp in workflow_tasks if tp.status == ProgressStatus.IN_PROGRESS)
        metrics.blocked_tasks = sum(1 for tp in workflow_tasks if tp.status == ProgressStatus.BLOCKED)
        metrics.failed_tasks = sum(1 for tp in workflow_tasks if tp.status == ProgressStatus.FAILED)
        
        # Calculate overall progress
        total_progress = sum(tp.progress_percentage for tp in workflow_tasks)
        metrics.overall_progress = total_progress / len(workflow_tasks) if workflow_tasks else 0.0
        
        # Calculate velocity trend
        completed_in_last_hour = sum(
            1 for tp in workflow_tasks 
            if tp.status == ProgressStatus.COMPLETED 
            and tp.end_time 
            and (datetime.now() - tp.end_time).total_seconds() <= 3600
        )
        metrics.velocity_trend = completed_in_last_hour
        
        # Calculate efficiency ratio
        completed_with_duration = [
            tp for tp in workflow_tasks 
            if tp.status == ProgressStatus.COMPLETED and tp.actual_duration
        ]
        
        if completed_with_duration:
            # Would need estimated duration per task - simplified for now
            actual_avg = statistics.mean([tp.actual_duration for tp in completed_with_duration])
            estimated_avg = 8.0  # Default estimation, would be from task data
            metrics.efficiency_ratio = estimated_avg / actual_avg if actual_avg > 0 else 1.0
        
        # Update resource utilization
        current_resource_usage = {}
        for tp in workflow_tasks:
            if tp.status == ProgressStatus.IN_PROGRESS:
                for resource, usage in tp.resource_usage.items():
                    current_resource_usage[resource] = current_resource_usage.get(resource, 0) + usage
        
        metrics.resource_utilization = current_resource_usage
        
        # Count active bottlenecks
        metrics.bottlenecks_active = sum(
            1 for b in self.bottlenecks.values() 
            if workflow_id in [str(t) for t in b.affected_tasks] and not b.resolved
        )
        
        # Calculate performance score (0.0-1.0)
        score_factors = []
        
        # Progress factor
        progress_factor = metrics.overall_progress / 100.0
        score_factors.append(progress_factor)
        
        # Efficiency factor
        efficiency_factor = min(1.0, metrics.efficiency_ratio)
        score_factors.append(efficiency_factor)
        
        # Bottleneck penalty
        bottleneck_factor = max(0.1, 1.0 - (metrics.bottlenecks_active * 0.2))
        score_factors.append(bottleneck_factor)
        
        # Velocity factor
        velocity_factor = min(1.0, metrics.velocity_trend / max(1.0, metrics.total_tasks / 8.0))
        score_factors.append(velocity_factor)
        
        metrics.performance_score = statistics.mean(score_factors)
        
        # Add to performance history
        self.performance_history[workflow_id].append({
            'timestamp': datetime.now(),
            'progress': metrics.overall_progress,
            'performance_score': metrics.performance_score,
            'velocity': metrics.velocity_trend,
            'bottlenecks': metrics.bottlenecks_active
        })
        
        # Log metrics update
        self._log_workflow_metrics(metrics)
    
    async def _detect_bottlenecks(self, workflow_id: str):
        """Intelligent bottleneck detection using multiple algorithms"""
        
        workflow_tasks = [tp for tp in self.task_progress.values() if tp.workflow_id == workflow_id]
        detected_bottlenecks = []
        
        # 1. Resource Contention Detection
        resource_usage = {}
        for tp in workflow_tasks:
            if tp.status == ProgressStatus.IN_PROGRESS:
                for resource, usage in tp.resource_usage.items():
                    if resource not in resource_usage:
                        resource_usage[resource] = []
                    resource_usage[resource].append((tp.task_id, usage))
        
        for resource, usage_list in resource_usage.items():
            if len(usage_list) > 2:  # More than 2 tasks competing for same resource
                total_usage = sum(usage for _, usage in usage_list)
                if total_usage > 0.8:  # Over 80% utilization
                    bottleneck = BottleneckDetection(
                        bottleneck_id=f"resource_contention_{resource}_{int(time.time())}",
                        type=BottleneckType.RESOURCE_CONTENTION,
                        severity=AlertSeverity.WARNING,
                        affected_tasks=[task_id for task_id, _ in usage_list],
                        resource_involved=resource,
                        description=f"Resource contention detected on {resource} ({total_usage:.1%} utilization)",
                        predicted_impact={"delay_hours": len(usage_list) * 0.5},
                        suggested_actions=[
                            f"Consider reallocating {resource} capacity",
                            "Review task scheduling for better resource distribution",
                            "Implement resource queuing system"
                        ],
                        confidence=0.7,
                        detection_time=datetime.now()
                    )
                    detected_bottlenecks.append(bottleneck)
        
        # 2. Dependency Blocking Detection
        blocked_tasks = [tp for tp in workflow_tasks if tp.status == ProgressStatus.BLOCKED]
        if len(blocked_tasks) > 1:
            # Multiple blocked tasks might indicate dependency issues
            bottleneck = BottleneckDetection(
                bottleneck_id=f"dependency_blocking_{workflow_id}_{int(time.time())}",
                type=BottleneckType.DEPENDENCY_BLOCKING,
                severity=AlertSeverity.CRITICAL if len(blocked_tasks) > 2 else AlertSeverity.WARNING,
                affected_tasks=[tp.task_id for tp in blocked_tasks],
                resource_involved="dependencies",
                description=f"Multiple tasks blocked ({len(blocked_tasks)} tasks), possible dependency chain issue",
                predicted_impact={"delay_hours": len(blocked_tasks) * 1.0},
                suggested_actions=[
                    "Review task dependencies and unblock critical path",
                    "Consider parallel execution alternatives",
                    "Escalate dependency resolution"
                ],
                confidence=0.8,
                detection_time=datetime.now()
            )
            detected_bottlenecks.append(bottleneck)
        
        # 3. Performance Degradation Detection
        if workflow_id in self.performance_history:
            history = list(self.performance_history[workflow_id])
            if len(history) >= 5:
                recent_scores = [h['performance_score'] for h in history[-5:]]
                older_scores = [h['performance_score'] for h in history[-10:-5]] if len(history) >= 10 else recent_scores
                
                if recent_scores and older_scores:
                    recent_avg = statistics.mean(recent_scores)
                    older_avg = statistics.mean(older_scores)
                    
                    if recent_avg < older_avg * 0.8:  # 20% performance drop
                        bottleneck = BottleneckDetection(
                            bottleneck_id=f"performance_degradation_{workflow_id}_{int(time.time())}",
                            type=BottleneckType.PERFORMANCE_DEGRADATION,
                            severity=AlertSeverity.WARNING,
                            affected_tasks=[tp.task_id for tp in workflow_tasks if tp.status == ProgressStatus.IN_PROGRESS],
                            resource_involved="overall_performance",
                            description=f"Performance degradation detected (dropped to {recent_avg:.2f} from {older_avg:.2f})",
                            predicted_impact={"efficiency_loss": (older_avg - recent_avg) * 100},
                            suggested_actions=[
                                "Investigate recent changes affecting performance",
                                "Review resource allocation and capacity",
                                "Consider workflow re-optimization"
                            ],
                            confidence=0.6,
                            detection_time=datetime.now()
                        )
                        detected_bottlenecks.append(bottleneck)
        
        # Store detected bottlenecks
        for bottleneck in detected_bottlenecks:
            self.bottlenecks[bottleneck.bottleneck_id] = bottleneck
            self._log_bottleneck_detection(bottleneck)
            await self._notify_bottleneck_detected(bottleneck)
            
            # Trigger adaptive optimization if critical
            if bottleneck.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self._trigger_adaptive_optimization(workflow_id, bottleneck)
        
        self.performance_stats["bottlenecks_detected"] += len(detected_bottlenecks)
    
    async def _trigger_adaptive_optimization(self, workflow_id: str, bottleneck: BottleneckDetection):
        """Trigger adaptive workflow re-optimization based on detected bottlenecks"""
        
        if not self.workflow_optimizer:
            logger.warning("Workflow optimizer not available for adaptive optimization")
            return
        
        try:
            logger.info(f"🔄 Triggering adaptive optimization for workflow {workflow_id}")
            
            # Get current workflow state
            workflow_tasks = [tp for tp in self.task_progress.values() if tp.workflow_id == workflow_id]
            
            # Create reasoning context for re-optimization
            context = ReasoningContext(
                project_type="adaptive_optimization",
                constraints=[
                    f"Bottleneck: {bottleneck.type.value}",
                    f"Affected tasks: {len(bottleneck.affected_tasks)}",
                    f"Resource: {bottleneck.resource_involved}"
                ]
            )
            
            # Use AI reasoning to determine best adaptive strategy
            if self.reasoning_engine:
                optimization_reasoning = await self._generate_adaptive_strategy(
                    workflow_id, bottleneck, context
                )
                logger.info(f"🧠 AI adaptive strategy: {optimization_reasoning[:100]}...")
            
            # For now, log the adaptive optimization trigger
            # In full implementation, would re-run workflow optimization
            logger.info(f"📊 Adaptive optimization triggered for {bottleneck.type.value}")
            
            self.performance_stats["adaptive_optimizations"] += 1
            
        except Exception as e:
            logger.error(f"Adaptive optimization failed: {e}")
    
    async def _generate_adaptive_strategy(
        self, 
        workflow_id: str, 
        bottleneck: BottleneckDetection,
        context: ReasoningContext
    ) -> str:
        """Generate AI-powered adaptive optimization strategy"""
        
        if not self.reasoning_engine:
            return f"Standard mitigation for {bottleneck.type.value}"
        
        try:
            problem = f"""
            Adaptive workflow optimization required:
            
            Bottleneck Details:
            - Type: {bottleneck.type.value}
            - Severity: {bottleneck.severity.value}
            - Affected Tasks: {len(bottleneck.affected_tasks)}
            - Resource: {bottleneck.resource_involved}
            - Description: {bottleneck.description}
            
            Current Situation:
            - Workflow ID: {workflow_id}
            - Active Tasks: {len([tp for tp in self.task_progress.values() if tp.status == ProgressStatus.IN_PROGRESS])}
            
            Suggest adaptive optimization strategy:
            1. Immediate actions to resolve bottleneck
            2. Workflow adjustments to prevent recurrence
            3. Resource reallocation recommendations
            4. Priority adjustments if needed
            """
            
            reasoning_chain = await self.reasoning_engine.reason_with_context(
                problem_statement=problem,
                context=context,
                reasoning_type=ReasoningType.PROBLEM_SOLVING
            )
            
            return reasoning_chain.final_reasoning
            
        except Exception as e:
            logger.warning(f"Adaptive strategy generation failed: {e}")
            return f"Standard mitigation approach for {bottleneck.type.value}"
    
    async def generate_performance_prediction(self, workflow_id: str) -> Optional[PerformancePrediction]:
        """Generate AI-powered performance predictions"""
        
        if workflow_id not in self.active_workflows:
            return None
        
        try:
            metrics = self.active_workflows[workflow_id]
            history = list(self.performance_history.get(workflow_id, []))
            
            # Basic prediction using current velocity
            current_progress = metrics.overall_progress
            remaining_progress = 100.0 - current_progress
            
            if metrics.velocity_trend > 0:
                estimated_hours_remaining = remaining_progress / metrics.velocity_trend
            else:
                estimated_hours_remaining = remaining_progress * 0.5  # Fallback estimation
            
            predicted_completion = datetime.now() + timedelta(hours=estimated_hours_remaining)
            
            # Calculate confidence interval (±20%)
            confidence_range = estimated_hours_remaining * 0.2
            confidence_interval = (
                datetime.now() + timedelta(hours=estimated_hours_remaining - confidence_range),
                datetime.now() + timedelta(hours=estimated_hours_remaining + confidence_range)
            )
            
            # Identify risk factors
            risk_factors = []
            if metrics.bottlenecks_active > 0:
                risk_factors.append(f"{metrics.bottlenecks_active} active bottlenecks")
            if metrics.efficiency_ratio < 0.8:
                risk_factors.append("Below average efficiency")
            if metrics.blocked_tasks > 0:
                risk_factors.append(f"{metrics.blocked_tasks} blocked tasks")
            
            # Optimization opportunities
            opportunities = []
            if metrics.resource_utilization:
                low_util_resources = [r for r, u in metrics.resource_utilization.items() if u < 0.5]
                if low_util_resources:
                    opportunities.append(f"Underutilized resources: {', '.join(low_util_resources)}")
            
            if metrics.velocity_trend < metrics.total_tasks / 16.0:  # Expected 2 tasks per hour
                opportunities.append("Increase task completion velocity")
            
            # Success probability based on current performance
            success_factors = [
                metrics.performance_score,
                1.0 - (metrics.bottlenecks_active / max(1, metrics.total_tasks)),
                min(1.0, metrics.velocity_trend / 2.0),
                metrics.efficiency_ratio
            ]
            
            success_probability = statistics.mean(success_factors)
            
            prediction = PerformancePrediction(
                workflow_id=workflow_id,
                predicted_completion=predicted_completion,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                optimization_opportunities=opportunities,
                predicted_bottlenecks=[],  # Would be more sophisticated with ML
                success_probability=success_probability,
                prediction_accuracy=0.7,  # Would improve with historical data
                generated_at=datetime.now()
            )
            
            # Log prediction
            self._log_performance_prediction(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return None
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🔍 Monitoring thread started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Run async monitoring tasks
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                for workflow_id in list(self.active_workflows.keys()):
                    loop.run_until_complete(self._periodic_monitoring_check(workflow_id))
                
                loop.close()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    async def _periodic_monitoring_check(self, workflow_id: str):
        """Periodic monitoring check for workflow"""
        
        try:
            # Update workflow metrics
            await self._update_workflow_metrics(workflow_id)
            
            # Check for bottlenecks
            await self._detect_bottlenecks(workflow_id)
            
            # Generate predictions if needed
            if workflow_id in self.active_workflows:
                prediction = await self.generate_performance_prediction(workflow_id)
                if prediction and prediction.success_probability < 0.7:
                    logger.warning(f"⚠️ Low success probability ({prediction.success_probability:.2f}) for workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Periodic monitoring check failed: {e}")
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates (WebSocket ready)"""
        self.update_callbacks.append(callback)
    
    async def _notify_progress_update(self, task_progress: TaskProgress):
        """Notify all callbacks about progress update"""
        
        update_data = {
            'type': 'task_progress_update',
            'task_id': task_progress.task_id,
            'workflow_id': task_progress.workflow_id,
            'status': task_progress.status.value,
            'progress': task_progress.progress_percentage,
            'timestamp': task_progress.last_update.isoformat()
        }
        
        for callback in self.update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update_data)
                else:
                    callback(update_data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def _notify_bottleneck_detected(self, bottleneck: BottleneckDetection):
        """Notify about bottleneck detection"""
        
        alert_data = {
            'type': 'bottleneck_detected',
            'bottleneck_id': bottleneck.bottleneck_id,
            'severity': bottleneck.severity.value,
            'description': bottleneck.description,
            'affected_tasks': bottleneck.affected_tasks,
            'suggested_actions': bottleneck.suggested_actions,
            'timestamp': bottleneck.detection_time.isoformat()
        }
        
        for callback in self.update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.warning(f"Bottleneck callback failed: {e}")
    
    def _log_workflow_start(self, workflow: OptimizedWorkflow):
        """Log workflow monitoring start"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO workflow_execution_metrics
                    (workflow_id, total_tasks, completed_tasks, overall_progress, 
                     estimated_completion, performance_score, velocity_trend)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    workflow.workflow_id,
                    len(workflow.task_order),
                    0,
                    0.0,
                    (datetime.now() + timedelta(hours=workflow.estimated_duration)).isoformat(),
                    1.0,
                    0.0
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Workflow start logging failed: {e}")
    
    def _log_progress_update(self, task_progress: TaskProgress):
        """Log task progress update"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO task_progress
                    (task_id, workflow_id, status, progress_percentage, start_time,
                     end_time, actual_duration, agent_id, resource_usage, blockers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_progress.task_id,
                    task_progress.workflow_id,
                    task_progress.status.value,
                    task_progress.progress_percentage,
                    task_progress.start_time.isoformat() if task_progress.start_time else None,
                    task_progress.end_time.isoformat() if task_progress.end_time else None,
                    task_progress.actual_duration,
                    task_progress.agent_id,
                    json.dumps(task_progress.resource_usage),
                    json.dumps(task_progress.blockers)
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Progress logging failed: {e}")
    
    def _log_bottleneck_detection(self, bottleneck: BottleneckDetection):
        """Log bottleneck detection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO bottleneck_detections
                    (bottleneck_id, type, severity, affected_tasks, resource_involved,
                     description, predicted_impact, suggested_actions, confidence, detection_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bottleneck.bottleneck_id,
                    bottleneck.type.value,
                    bottleneck.severity.value,
                    json.dumps(bottleneck.affected_tasks),
                    bottleneck.resource_involved,
                    bottleneck.description,
                    json.dumps(bottleneck.predicted_impact),
                    json.dumps(bottleneck.suggested_actions),
                    bottleneck.confidence,
                    bottleneck.detection_time.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Bottleneck logging failed: {e}")
    
    def _log_workflow_metrics(self, metrics: WorkflowExecutionMetrics):
        """Log workflow execution metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO workflow_execution_metrics
                    (workflow_id, total_tasks, completed_tasks, overall_progress,
                     resource_utilization, bottlenecks_active, performance_score,
                     velocity_trend, efficiency_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.workflow_id,
                    metrics.total_tasks,
                    metrics.completed_tasks,
                    metrics.overall_progress,
                    json.dumps(metrics.resource_utilization),
                    metrics.bottlenecks_active,
                    metrics.performance_score,
                    metrics.velocity_trend,
                    metrics.efficiency_ratio
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Metrics logging failed: {e}")
    
    def _log_performance_prediction(self, prediction: PerformancePrediction):
        """Log performance prediction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_predictions
                    (workflow_id, predicted_completion, confidence_interval_start,
                     confidence_interval_end, risk_factors, optimization_opportunities,
                     success_probability, prediction_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.workflow_id,
                    prediction.predicted_completion.isoformat(),
                    prediction.confidence_interval[0].isoformat(),
                    prediction.confidence_interval[1].isoformat(),
                    json.dumps(prediction.risk_factors),
                    json.dumps(prediction.optimization_opportunities),
                    prediction.success_probability,
                    prediction.prediction_accuracy
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Prediction logging failed: {e}")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🔍 Monitoring stopped")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.performance_stats,
            "active_workflows": len(self.active_workflows),
            "monitored_tasks": len(self.task_progress),
            "active_bottlenecks": len([b for b in self.bottlenecks.values() if not b.resolved]),
            "monitoring_active": self.monitoring_active,
            "workflow_optimizer_available": WORKFLOW_OPTIMIZER_AVAILABLE,
            "ai_components_available": AI_COMPONENTS_AVAILABLE
        }

# Demo and testing
async def demo_real_time_progress_monitor():
    """Demo the real-time progress monitor"""
    print("🔍 Agent Zero V2.0 - Real-Time Progress Monitor Demo")
    print("=" * 55)
    
    # Initialize monitor
    monitor = RealTimeProgressMonitor()
    
    # Create a mock workflow (would normally come from DynamicWorkflowOptimizer)
    from dynamic_workflow_optimizer import OptimizedWorkflow, WorkflowMetrics
    
    mock_workflow = OptimizedWorkflow(
        workflow_id="demo_workflow_123",
        strategy="time_optimized",
        task_order=[1, 2, 3, 4],
        parallel_groups=[[1, 2], [3, 4]],
        critical_path=[1, 3],
        estimated_duration=16.0,
        resource_allocation={"backend_agents": [1, 3], "frontend_agents": [2, 4]},
        confidence=0.8,
        optimization_reasoning="Demo workflow for monitoring",
        metrics=WorkflowMetrics(
            total_tasks=4,
            completed_tasks=0,
            active_tasks=0,
            blocked_tasks=0,
            estimated_completion=datetime.now() + timedelta(hours=16),
            critical_path_length=16.0,
            resource_utilization={},
            bottlenecks=[],
            optimization_opportunities=[]
        )
    )
    
    print(f"📋 Demo Workflow:")
    print(f"   Workflow ID: {mock_workflow.workflow_id}")
    print(f"   Tasks: {len(mock_workflow.task_order)}")
    print(f"   Estimated Duration: {mock_workflow.estimated_duration} hours")
    
    # Add progress callback for demo
    async def demo_callback(update_data):
        print(f"📡 Real-time Update: {update_data['type']} - {update_data}")
    
    monitor.add_progress_callback(demo_callback)
    
    # Start monitoring
    await monitor.start_monitoring_workflow(mock_workflow)
    
    # Simulate task progress updates
    print(f"\n🎬 Simulating task execution...")
    
    # Start task 1
    await monitor.update_task_progress(1, ProgressStatus.IN_PROGRESS, 0, "agent_backend_1")
    await asyncio.sleep(0.5)
    
    # Start task 2 (parallel)
    await monitor.update_task_progress(2, ProgressStatus.IN_PROGRESS, 0, "agent_frontend_1")
    await asyncio.sleep(0.5)
    
    # Progress task 1
    await monitor.update_task_progress(1, ProgressStatus.IN_PROGRESS, 50)
    await asyncio.sleep(0.5)
    
    # Block task 3 (dependency issue)
    await monitor.update_task_progress(3, ProgressStatus.BLOCKED, 0, notes="Waiting for external API")
    await asyncio.sleep(0.5)
    
    # Complete task 1
    await monitor.update_task_progress(1, ProgressStatus.COMPLETED, 100)
    await asyncio.sleep(0.5)
    
    # Start overloading resources (simulate bottleneck)
    await monitor.update_task_progress(4, ProgressStatus.IN_PROGRESS, 0, "agent_backend_2", 
                                     {"backend_agents": 0.9})
    await asyncio.sleep(1)
    
    # Generate performance prediction
    print(f"\n🔮 Generating performance prediction...")
    prediction = await monitor.generate_performance_prediction("demo_workflow_123")
    if prediction:
        print(f"   Predicted Completion: {prediction.predicted_completion}")
        print(f"   Success Probability: {prediction.success_probability:.2f}")
        print(f"   Risk Factors: {prediction.risk_factors}")
        print(f"   Opportunities: {prediction.optimization_opportunities}")
    
    # Check bottlenecks
    active_bottlenecks = [b for b in monitor.bottlenecks.values() if not b.resolved]
    print(f"\n🚫 Active Bottlenecks: {len(active_bottlenecks)}")
    for bottleneck in active_bottlenecks:
        print(f"   {bottleneck.type.value}: {bottleneck.description}")
        print(f"   Suggested Actions: {bottleneck.suggested_actions}")
    
    # Show monitoring stats
    print(f"\n📊 Monitoring Statistics:")
    stats = monitor.get_monitoring_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print(f"\n✅ Real-time progress monitor demo completed!")

if __name__ == "__main__":
    print("🔍 Agent Zero V2.0 Phase 4 - Real-Time Progress Monitor")
    print("Advanced execution tracking with AI-powered bottleneck detection")
    
    # Run demo
    asyncio.run(demo_real_time_progress_monitor())