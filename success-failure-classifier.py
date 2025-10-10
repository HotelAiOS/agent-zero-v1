# Success/Failure Classification System - Analytics Module for Agent Zero V1
# Task: A0-25 Success/Failure Classification System (Week 44-45)
# Focus: Multi-dimensional success criteria (correctness, efficiency, cost, latency)
# Impact: Foundation dla całego Kaizen learning

"""
Success/Failure Classification System for Agent Zero V1
Multi-dimensional success criteria evaluation and learning

This system provides:
- Multi-dimensional success/failure classification
- Adaptive threshold learning based on context
- Predictive success probability estimation
- Detailed failure analysis and categorization
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3
import statistics
import numpy as np
from collections import defaultdict, Counter
from contextlib import contextmanager

# Import existing components
try:
    import sys
    sys.path.insert(0, '.')
    from simple_tracker import SimpleTracker
    from project_orchestrator import Project, Task, ProjectState, TaskStatus, ProjectMetrics
    from hierarchical_task_planner import HierarchicalTask, TaskType, TaskPriority
    from feedback_loop_engine import FeedbackLoopEngine
except ImportError as e:
    logging.warning(f"Could not import existing components: {e}")
    # Fallback classes
    class SimpleTracker:
        def track_event(self, event): pass
    class ProjectState:
        COMPLETED = "completed"
        FAILED = "failed"
    class TaskStatus:
        COMPLETED = "completed"
        FAILED = "failed"

class SuccessMetric(Enum):
    """Success evaluation metrics"""
    CORRECTNESS = "correctness"           # Did it solve the problem correctly?
    EFFICIENCY = "efficiency"             # Resource usage optimization
    COST = "cost"                        # Financial cost optimization
    LATENCY = "latency"                  # Time to completion
    QUALITY = "quality"                  # Output quality assessment
    RELIABILITY = "reliability"          # Consistency and stability
    USABILITY = "usability"             # User satisfaction and experience
    MAINTAINABILITY = "maintainability" # Long-term sustainability
    SCALABILITY = "scalability"         # Performance under load
    SECURITY = "security"               # Security and safety measures

class SuccessLevel(Enum):
    """Success classification levels"""
    EXCELLENT = "excellent"    # > 90% success across all metrics
    GOOD = "good"             # > 75% success across all metrics
    ACCEPTABLE = "acceptable"  # > 60% success across all metrics
    POOR = "poor"             # > 40% success across all metrics
    FAILURE = "failure"       # < 40% success across all metrics

class FailureCategory(Enum):
    """Failure categorization"""
    TECHNICAL = "technical"              # Code/system failures
    RESOURCE = "resource"                # Insufficient resources
    DESIGN = "design"                    # Poor design decisions
    REQUIREMENTS = "requirements"        # Misunderstood requirements
    DEPENDENCY = "dependency"            # External dependency failures
    TIMEOUT = "timeout"                  # Time limit exceeded
    BUDGET = "budget"                    # Budget constraints
    QUALITY = "quality"                  # Quality standards not met
    HUMAN_ERROR = "human_error"          # Human mistakes
    EXTERNAL = "external"                # External factors

@dataclass
class SuccessThresholds:
    """Dynamic thresholds for success classification"""
    metric: SuccessMetric
    context: str  # e.g., "development", "production", "simple_task"
    
    # Threshold values (0-1 scale)
    excellent_threshold: float = 0.9
    good_threshold: float = 0.75
    acceptable_threshold: float = 0.6
    poor_threshold: float = 0.4
    
    # Confidence metrics
    sample_size: int = 0
    confidence: float = 0.5
    last_updated: Optional[datetime] = None

@dataclass
class SuccessEvaluation:
    """Comprehensive success evaluation result"""
    evaluation_id: str
    entity_id: str  # Project or Task ID
    entity_type: str  # "project" or "task"
    timestamp: datetime
    
    # Metric scores (0-1 scale)
    metric_scores: Dict[SuccessMetric, float] = field(default_factory=dict)
    
    # Overall classification
    overall_success_level: SuccessLevel = SuccessLevel.ACCEPTABLE
    overall_score: float = 0.0
    
    # Detailed analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Failure analysis (if applicable)
    failure_categories: List[FailureCategory] = field(default_factory=list)
    failure_root_causes: List[str] = field(default_factory=list)
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Prediction confidence
    confidence: float = 0.0

@dataclass
class PredictionResult:
    """Success probability prediction"""
    entity_id: str
    predicted_success_probability: float
    predicted_success_level: SuccessLevel
    confidence: float
    risk_factors: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class MetricCalculator:
    """Calculates success metrics for projects and tasks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_project_metrics(self, project) -> Dict[SuccessMetric, float]:
        """Calculate success metrics for a project"""
        
        metrics = {}
        
        # Correctness - based on completion rate and task success
        if hasattr(project, 'metrics') and project.metrics:
            completion_rate = project.metrics.completed_tasks / max(project.metrics.total_tasks, 1)
            failure_rate = project.metrics.failed_tasks / max(project.metrics.total_tasks, 1)
            correctness = max(0, completion_rate - failure_rate * 0.5)
            metrics[SuccessMetric.CORRECTNESS] = min(1.0, correctness)
        else:
            metrics[SuccessMetric.CORRECTNESS] = 0.5  # Default
        
        # Cost efficiency
        if hasattr(project, 'metrics') and project.metrics:
            estimated_cost = project.metrics.estimated_cost
            actual_cost = project.metrics.actual_cost
            
            if estimated_cost > 0 and actual_cost > 0:
                if actual_cost <= estimated_cost:
                    # Under or on budget - excellent
                    cost_efficiency = 1.0
                elif actual_cost <= estimated_cost * 1.2:
                    # Up to 20% over budget - good
                    cost_efficiency = 0.8
                elif actual_cost <= estimated_cost * 1.5:
                    # Up to 50% over budget - acceptable
                    cost_efficiency = 0.6
                elif actual_cost <= estimated_cost * 2.0:
                    # Up to 100% over budget - poor
                    cost_efficiency = 0.4
                else:
                    # More than 100% over budget - failure
                    cost_efficiency = 0.2
                
                metrics[SuccessMetric.COST] = cost_efficiency
            else:
                metrics[SuccessMetric.COST] = 0.5
        else:
            metrics[SuccessMetric.COST] = 0.5
        
        # Time efficiency (latency)
        if hasattr(project, 'metrics') and project.metrics:
            estimated_duration = project.metrics.estimated_duration
            actual_duration = project.metrics.actual_duration
            
            if estimated_duration > 0 and actual_duration > 0:
                if actual_duration <= estimated_duration:
                    time_efficiency = 1.0
                elif actual_duration <= estimated_duration * 1.2:
                    time_efficiency = 0.8
                elif actual_duration <= estimated_duration * 1.5:
                    time_efficiency = 0.6
                elif actual_duration <= estimated_duration * 2.0:
                    time_efficiency = 0.4
                else:
                    time_efficiency = 0.2
                
                metrics[SuccessMetric.LATENCY] = time_efficiency
            else:
                metrics[SuccessMetric.LATENCY] = 0.5
        else:
            metrics[SuccessMetric.LATENCY] = 0.5
        
        # Quality - based on success rate and completion quality
        if hasattr(project, 'metrics') and project.metrics:
            success_rate = project.metrics.success_rate
            metrics[SuccessMetric.QUALITY] = success_rate
        else:
            metrics[SuccessMetric.QUALITY] = 0.5
        
        # Efficiency - combination of cost and time efficiency
        efficiency = (metrics[SuccessMetric.COST] + metrics[SuccessMetric.LATENCY]) / 2
        metrics[SuccessMetric.EFFICIENCY] = efficiency
        
        # Reliability - based on failure patterns and consistency
        if hasattr(project, 'metrics') and project.metrics:
            failure_rate = project.metrics.failed_tasks / max(project.metrics.total_tasks, 1)
            reliability = max(0, 1.0 - failure_rate * 2)  # Penalize failures heavily
            metrics[SuccessMetric.RELIABILITY] = reliability
        else:
            metrics[SuccessMetric.RELIABILITY] = 0.5
        
        return metrics
    
    def calculate_task_metrics(self, task) -> Dict[SuccessMetric, float]:
        """Calculate success metrics for a task"""
        
        metrics = {}
        
        # Correctness - based on task status
        if hasattr(task, 'status'):
            if task.status == TaskStatus.COMPLETED:
                metrics[SuccessMetric.CORRECTNESS] = 1.0
            elif task.status == TaskStatus.FAILED:
                metrics[SuccessMetric.CORRECTNESS] = 0.0
            else:
                # In progress or pending
                progress = getattr(task, 'progress_percentage', 0) / 100.0
                metrics[SuccessMetric.CORRECTNESS] = progress * 0.5  # Partial credit
        else:
            metrics[SuccessMetric.CORRECTNESS] = 0.5
        
        # Cost efficiency
        estimated_cost = getattr(task, 'estimated_cost', 0)
        actual_cost = getattr(task, 'actual_cost', 0)
        
        if estimated_cost > 0 and actual_cost > 0:
            cost_ratio = actual_cost / estimated_cost
            if cost_ratio <= 1.0:
                cost_efficiency = 1.0
            elif cost_ratio <= 1.2:
                cost_efficiency = 0.8
            elif cost_ratio <= 1.5:
                cost_efficiency = 0.6
            else:
                cost_efficiency = max(0.2, 1.0 / cost_ratio)
            
            metrics[SuccessMetric.COST] = cost_efficiency
        else:
            metrics[SuccessMetric.COST] = 0.5
        
        # Time efficiency
        estimated_duration = getattr(task, 'estimated_duration', 0)
        actual_duration = getattr(task, 'actual_duration', 0)
        
        if estimated_duration > 0 and actual_duration > 0:
            time_ratio = actual_duration / estimated_duration
            if time_ratio <= 1.0:
                time_efficiency = 1.0
            elif time_ratio <= 1.2:
                time_efficiency = 0.8
            elif time_ratio <= 1.5:
                time_efficiency = 0.6
            else:
                time_efficiency = max(0.2, 1.0 / time_ratio)
            
            metrics[SuccessMetric.LATENCY] = time_efficiency
        else:
            metrics[SuccessMetric.LATENCY] = 0.5
        
        # Quality - based on completion and progress
        progress = getattr(task, 'progress_percentage', 0) / 100.0
        if hasattr(task, 'status') and task.status == TaskStatus.COMPLETED:
            quality = 1.0
        elif hasattr(task, 'status') and task.status == TaskStatus.FAILED:
            quality = 0.0
        else:
            quality = progress
        
        metrics[SuccessMetric.QUALITY] = quality
        
        # Overall efficiency
        efficiency = (metrics[SuccessMetric.COST] + metrics[SuccessMetric.LATENCY]) / 2
        metrics[SuccessMetric.EFFICIENCY] = efficiency
        
        # Reliability - assume high for completed tasks, low for failed
        if hasattr(task, 'status'):
            if task.status == TaskStatus.COMPLETED:
                metrics[SuccessMetric.RELIABILITY] = 1.0
            elif task.status == TaskStatus.FAILED:
                metrics[SuccessMetric.RELIABILITY] = 0.0
            else:
                metrics[SuccessMetric.RELIABILITY] = 0.5
        else:
            metrics[SuccessMetric.RELIABILITY] = 0.5
        
        return metrics

class ThresholdManager:
    """Manages adaptive thresholds for success classification"""
    
    def __init__(self, db_path: str = "success_thresholds.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        
        # Cache for thresholds
        self._threshold_cache = {}
        self._load_thresholds()
    
    def _initialize_database(self):
        """Initialize SQLite database for thresholds"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS success_thresholds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    context TEXT NOT NULL,
                    excellent_threshold REAL NOT NULL,
                    good_threshold REAL NOT NULL,
                    acceptable_threshold REAL NOT NULL,
                    poor_threshold REAL NOT NULL,
                    sample_size INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    last_updated TIMESTAMP,
                    UNIQUE(metric, context)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    context TEXT NOT NULL,
                    threshold_type TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
            # Insert default thresholds if none exist
            self._insert_default_thresholds(cursor)
            conn.commit()
    
    def _insert_default_thresholds(self, cursor):
        """Insert default threshold values"""
        
        default_thresholds = [
            # General context
            ('correctness', 'general', 0.9, 0.75, 0.6, 0.4),
            ('efficiency', 'general', 0.85, 0.7, 0.55, 0.35),
            ('cost', 'general', 0.9, 0.8, 0.6, 0.4),
            ('latency', 'general', 0.9, 0.8, 0.6, 0.4),
            ('quality', 'general', 0.9, 0.75, 0.6, 0.4),
            ('reliability', 'general', 0.95, 0.85, 0.7, 0.5),
            
            # Development context (more lenient)
            ('correctness', 'development', 0.8, 0.65, 0.5, 0.3),
            ('cost', 'development', 0.7, 0.6, 0.4, 0.2),
            ('latency', 'development', 0.7, 0.6, 0.4, 0.2),
            
            # Production context (more strict)
            ('correctness', 'production', 0.95, 0.85, 0.7, 0.5),
            ('reliability', 'production', 0.98, 0.9, 0.8, 0.6),
            ('quality', 'production', 0.95, 0.85, 0.7, 0.5),
        ]
        
        for metric, context, excellent, good, acceptable, poor in default_thresholds:
            cursor.execute("""
                INSERT OR IGNORE INTO success_thresholds 
                (metric, context, excellent_threshold, good_threshold, acceptable_threshold, poor_threshold)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric, context, excellent, good, acceptable, poor))
    
    def _load_thresholds(self):
        """Load thresholds from database into cache"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT metric, context, excellent_threshold, good_threshold, 
                       acceptable_threshold, poor_threshold, sample_size, confidence, last_updated
                FROM success_thresholds
            """)
            
            for row in cursor.fetchall():
                key = (row[0], row[1])  # (metric, context)
                threshold = SuccessThresholds(
                    metric=SuccessMetric(row[0]),
                    context=row[1],
                    excellent_threshold=row[2],
                    good_threshold=row[3],
                    acceptable_threshold=row[4],
                    poor_threshold=row[5],
                    sample_size=row[6],
                    confidence=row[7],
                    last_updated=datetime.fromisoformat(row[8]) if row[8] else None
                )
                self._threshold_cache[key] = threshold
    
    def get_thresholds(self, metric: SuccessMetric, context: str = "general") -> SuccessThresholds:
        """Get thresholds for specific metric and context"""
        
        key = (metric.value, context)
        
        # Check cache first
        if key in self._threshold_cache:
            return self._threshold_cache[key]
        
        # Fallback to general context
        general_key = (metric.value, "general")
        if general_key in self._threshold_cache:
            return self._threshold_cache[general_key]
        
        # Create default thresholds
        default_threshold = SuccessThresholds(
            metric=metric,
            context=context,
            excellent_threshold=0.9,
            good_threshold=0.75,
            acceptable_threshold=0.6,
            poor_threshold=0.4
        )
        
        # Save to cache and database
        self._threshold_cache[key] = default_threshold
        self._save_threshold(default_threshold)
        
        return default_threshold
    
    def _save_threshold(self, threshold: SuccessThresholds):
        """Save threshold to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO success_thresholds
                (metric, context, excellent_threshold, good_threshold, acceptable_threshold, 
                 poor_threshold, sample_size, confidence, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threshold.metric.value,
                threshold.context,
                threshold.excellent_threshold,
                threshold.good_threshold,
                threshold.acceptable_threshold,
                threshold.poor_threshold,
                threshold.sample_size,
                threshold.confidence,
                threshold.last_updated.isoformat() if threshold.last_updated else None
            ))
            conn.commit()
    
    def update_thresholds_from_data(self, evaluations: List[SuccessEvaluation]):
        """Update thresholds based on actual performance data"""
        
        # Group evaluations by metric and context
        metric_data = defaultdict(lambda: defaultdict(list))
        
        for evaluation in evaluations:
            context = evaluation.context.get('environment', 'general')
            
            for metric, score in evaluation.metric_scores.items():
                metric_data[metric.value][context].append(score)
        
        # Update thresholds based on data distribution
        updates = []
        
        for metric_name, context_data in metric_data.items():
            for context, scores in context_data.items():
                if len(scores) < 10:  # Need minimum sample size
                    continue
                
                try:
                    metric = SuccessMetric(metric_name)
                    current_threshold = self.get_thresholds(metric, context)
                    
                    # Calculate new thresholds based on percentiles
                    scores.sort()
                    
                    # Use percentiles to set thresholds
                    excellent_new = np.percentile(scores, 90)  # Top 10%
                    good_new = np.percentile(scores, 75)       # Top 25%
                    acceptable_new = np.percentile(scores, 50) # Median
                    poor_new = np.percentile(scores, 25)       # Bottom 25%
                    
                    # Apply smoothing to avoid dramatic changes
                    smoothing_factor = 0.2  # 20% new, 80% old
                    
                    new_threshold = SuccessThresholds(
                        metric=metric,
                        context=context,
                        excellent_threshold=min(0.95, 
                            current_threshold.excellent_threshold * (1 - smoothing_factor) +
                            excellent_new * smoothing_factor),
                        good_threshold=min(0.9,
                            current_threshold.good_threshold * (1 - smoothing_factor) +
                            good_new * smoothing_factor),
                        acceptable_threshold=min(0.8,
                            current_threshold.acceptable_threshold * (1 - smoothing_factor) +
                            acceptable_new * smoothing_factor),
                        poor_threshold=min(0.6,
                            current_threshold.poor_threshold * (1 - smoothing_factor) +
                            poor_new * smoothing_factor),
                        sample_size=len(scores),
                        confidence=min(1.0, len(scores) / 50),  # Confidence based on sample size
                        last_updated=datetime.now()
                    )
                    
                    # Update cache and save
                    key = (metric.value, context)
                    self._threshold_cache[key] = new_threshold
                    self._save_threshold(new_threshold)
                    
                    # Record threshold change
                    self._record_threshold_change(
                        metric, context, "excellent", 
                        current_threshold.excellent_threshold,
                        new_threshold.excellent_threshold,
                        f"Updated from {len(scores)} samples"
                    )
                    
                    updates.append(f"Updated {metric.value} thresholds for {context}")
                
                except Exception as e:
                    self.logger.error(f"Error updating threshold for {metric_name}/{context}: {e}")
        
        return updates
    
    def _record_threshold_change(self, metric: SuccessMetric, context: str,
                               threshold_type: str, old_value: float, 
                               new_value: float, reason: str):
        """Record threshold change in history"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO threshold_history
                (metric, context, threshold_type, old_value, new_value, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric.value, context, threshold_type, old_value, new_value, reason))
            conn.commit()

class FailureAnalyzer:
    """Analyzes failures to identify patterns and root causes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Failure pattern indicators
        self.failure_indicators = {
            FailureCategory.TECHNICAL: [
                'error', 'exception', 'crash', 'bug', 'syntax', 'runtime',
                'segmentation fault', 'null pointer', 'stack overflow'
            ],
            FailureCategory.RESOURCE: [
                'memory', 'cpu', 'disk', 'network', 'quota', 'limit',
                'insufficient', 'exhausted', 'overload'
            ],
            FailureCategory.TIMEOUT: [
                'timeout', 'deadline', 'expired', 'slow', 'hanging',
                'unresponsive', 'blocked'
            ],
            FailureCategory.DEPENDENCY: [
                'dependency', 'service unavailable', 'connection refused',
                'external', 'third party', 'api down'
            ],
            FailureCategory.BUDGET: [
                'budget', 'cost', 'expensive', 'over limit', 'quota exceeded'
            ],
            FailureCategory.QUALITY: [
                'quality', 'standard', 'test failed', 'validation',
                'incorrect', 'invalid', 'wrong'
            ]
        }
    
    def analyze_failure(self, entity, context: Dict[str, Any] = None) -> Tuple[List[FailureCategory], List[str]]:
        """Analyze failure to identify categories and root causes"""
        
        categories = []
        root_causes = []
        
        # Analyze error messages if available
        error_messages = []
        
        if hasattr(entity, 'error_message') and entity.error_message:
            error_messages.append(entity.error_message)
        
        if hasattr(entity, 'tasks'):
            for task in entity.tasks.values():
                if hasattr(task, 'error_message') and task.error_message:
                    error_messages.append(task.error_message)
        
        # Categorize based on error messages
        for error_msg in error_messages:
            error_lower = error_msg.lower()
            
            for category, indicators in self.failure_indicators.items():
                if any(indicator in error_lower for indicator in indicators):
                    if category not in categories:
                        categories.append(category)
                    
                    # Extract specific root cause
                    for indicator in indicators:
                        if indicator in error_lower:
                            root_causes.append(f"{category.value}: {indicator}")
                            break
        
        # Analyze performance metrics for additional categorization
        if hasattr(entity, 'metrics') and entity.metrics:
            metrics = entity.metrics
            
            # Cost overrun
            if (metrics.estimated_cost > 0 and 
                metrics.actual_cost > metrics.estimated_cost * 2):
                categories.append(FailureCategory.BUDGET)
                root_causes.append(f"Cost overrun: {metrics.actual_cost:.3f} vs {metrics.estimated_cost:.3f}")
            
            # Time overrun
            if (metrics.estimated_duration > 0 and 
                metrics.actual_duration > metrics.estimated_duration * 2):
                categories.append(FailureCategory.TIMEOUT)
                root_causes.append(f"Time overrun: {metrics.actual_duration} vs {metrics.estimated_duration} minutes")
            
            # High failure rate
            if (metrics.total_tasks > 0 and 
                metrics.failed_tasks / metrics.total_tasks > 0.5):
                categories.append(FailureCategory.TECHNICAL)
                root_causes.append(f"High task failure rate: {metrics.failed_tasks}/{metrics.total_tasks}")
        
        # Default categorization if none found
        if not categories:
            categories.append(FailureCategory.TECHNICAL)
            root_causes.append("General failure - insufficient diagnostic information")
        
        return categories, root_causes

class SuccessClassifier:
    """Main success/failure classification system"""
    
    def __init__(self, threshold_db_path: str = "success_thresholds.db",
                 evaluation_db_path: str = "success_evaluations.db"):
        
        self.metric_calculator = MetricCalculator()
        self.threshold_manager = ThresholdManager(threshold_db_path)
        self.failure_analyzer = FailureAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        self.evaluation_db_path = evaluation_db_path
        self._initialize_evaluation_database()
        
        # Integration components
        try:
            self.tracker = SimpleTracker()
        except:
            self.tracker = None
            self.logger.warning("Could not initialize SimpleTracker")
    
    def _initialize_evaluation_database(self):
        """Initialize database for storing evaluations"""
        
        with sqlite3.connect(self.evaluation_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS success_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    overall_success_level TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    metric_scores TEXT NOT NULL,
                    strengths TEXT,
                    weaknesses TEXT,
                    improvement_suggestions TEXT,
                    failure_categories TEXT,
                    failure_root_causes TEXT,
                    context TEXT,
                    confidence REAL NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_id ON success_evaluations(entity_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON success_evaluations(timestamp)
            """)
            
            conn.commit()
    
    def evaluate_project_success(self, project, context: Dict[str, Any] = None) -> SuccessEvaluation:
        """Evaluate project success across multiple dimensions"""
        
        if context is None:
            context = {}
        
        # Calculate metrics
        metric_scores = self.metric_calculator.calculate_project_metrics(project)
        
        # Determine overall success level
        overall_score = sum(metric_scores.values()) / len(metric_scores)
        success_level = self._classify_success_level(overall_score, context.get('environment', 'general'))
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(metric_scores, context.get('environment', 'general'))
        
        # Generate improvement suggestions
        improvements = self._generate_improvement_suggestions(metric_scores, weaknesses)
        
        # Analyze failures if applicable
        failure_categories = []
        failure_causes = []
        
        if success_level in [SuccessLevel.POOR, SuccessLevel.FAILURE]:
            failure_categories, failure_causes = self.failure_analyzer.analyze_failure(project, context)
        
        # Create evaluation
        evaluation = SuccessEvaluation(
            evaluation_id=f"eval_proj_{project.project_id}_{int(time.time())}",
            entity_id=project.project_id,
            entity_type="project",
            timestamp=datetime.now(),
            metric_scores=metric_scores,
            overall_success_level=success_level,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements,
            failure_categories=failure_categories,
            failure_root_causes=failure_causes,
            context=context,
            confidence=self._calculate_evaluation_confidence(metric_scores, context)
        )
        
        # Store evaluation
        self._store_evaluation(evaluation)
        
        # Track evaluation
        if self.tracker:
            self.tracker.track_event({
                'type': 'success_evaluation_completed',
                'entity_type': 'project',
                'entity_id': project.project_id,
                'success_level': success_level.value,
                'overall_score': overall_score,
                'confidence': evaluation.confidence
            })
        
        return evaluation
    
    def evaluate_task_success(self, task, context: Dict[str, Any] = None) -> SuccessEvaluation:
        """Evaluate task success across multiple dimensions"""
        
        if context is None:
            context = {}
        
        # Calculate metrics
        metric_scores = self.metric_calculator.calculate_task_metrics(task)
        
        # Determine overall success level
        overall_score = sum(metric_scores.values()) / len(metric_scores)
        success_level = self._classify_success_level(overall_score, context.get('environment', 'general'))
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(metric_scores, context.get('environment', 'general'))
        
        # Generate improvement suggestions
        improvements = self._generate_improvement_suggestions(metric_scores, weaknesses)
        
        # Analyze failures if applicable
        failure_categories = []
        failure_causes = []
        
        if success_level in [SuccessLevel.POOR, SuccessLevel.FAILURE]:
            failure_categories, failure_causes = self.failure_analyzer.analyze_failure(task, context)
        
        # Create evaluation
        evaluation = SuccessEvaluation(
            evaluation_id=f"eval_task_{task.task_id}_{int(time.time())}",
            entity_id=task.task_id,
            entity_type="task",
            timestamp=datetime.now(),
            metric_scores=metric_scores,
            overall_success_level=success_level,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements,
            failure_categories=failure_categories,
            failure_root_causes=failure_causes,
            context=context,
            confidence=self._calculate_evaluation_confidence(metric_scores, context)
        )
        
        # Store evaluation
        self._store_evaluation(evaluation)
        
        return evaluation
    
    def _classify_success_level(self, overall_score: float, context: str) -> SuccessLevel:
        """Classify success level based on score and context"""
        
        # Use correctness thresholds as general success indicators
        thresholds = self.threshold_manager.get_thresholds(SuccessMetric.CORRECTNESS, context)
        
        if overall_score >= thresholds.excellent_threshold:
            return SuccessLevel.EXCELLENT
        elif overall_score >= thresholds.good_threshold:
            return SuccessLevel.GOOD
        elif overall_score >= thresholds.acceptable_threshold:
            return SuccessLevel.ACCEPTABLE
        elif overall_score >= thresholds.poor_threshold:
            return SuccessLevel.POOR
        else:
            return SuccessLevel.FAILURE
    
    def _analyze_strengths_weaknesses(self, metric_scores: Dict[SuccessMetric, float], 
                                    context: str) -> Tuple[List[str], List[str]]:
        """Analyze strengths and weaknesses based on metric scores"""
        
        strengths = []
        weaknesses = []
        
        for metric, score in metric_scores.items():
            thresholds = self.threshold_manager.get_thresholds(metric, context)
            
            if score >= thresholds.excellent_threshold:
                strengths.append(f"Excellent {metric.value}: {score:.1%}")
            elif score >= thresholds.good_threshold:
                strengths.append(f"Good {metric.value}: {score:.1%}")
            elif score < thresholds.acceptable_threshold:
                weaknesses.append(f"Poor {metric.value}: {score:.1%}")
        
        return strengths, weaknesses
    
    def _generate_improvement_suggestions(self, metric_scores: Dict[SuccessMetric, float],
                                        weaknesses: List[str]) -> List[str]:
        """Generate specific improvement suggestions"""
        
        suggestions = []
        
        for metric, score in metric_scores.items():
            if score < 0.6:  # Below acceptable threshold
                if metric == SuccessMetric.COST:
                    suggestions.append("Implement better cost estimation and monitoring")
                    suggestions.append("Consider more cost-effective resource allocation")
                elif metric == SuccessMetric.LATENCY:
                    suggestions.append("Optimize task sequencing and parallel execution")
                    suggestions.append("Review time estimates and add buffer time")
                elif metric == SuccessMetric.CORRECTNESS:
                    suggestions.append("Improve testing and validation processes")
                    suggestions.append("Enhance requirement analysis and clarity")
                elif metric == SuccessMetric.QUALITY:
                    suggestions.append("Implement more rigorous quality assurance")
                    suggestions.append("Add automated quality checks")
                elif metric == SuccessMetric.RELIABILITY:
                    suggestions.append("Add error handling and retry mechanisms")
                    suggestions.append("Implement better monitoring and alerting")
        
        # Generic suggestions if many weaknesses
        if len(weaknesses) > 3:
            suggestions.append("Consider breaking down complex tasks into smaller components")
            suggestions.append("Implement more frequent progress reviews and checkpoints")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _calculate_evaluation_confidence(self, metric_scores: Dict[SuccessMetric, float],
                                       context: Dict[str, Any]) -> float:
        """Calculate confidence in the evaluation"""
        
        confidence = 0.8  # Base confidence
        
        # Adjust based on available data
        if len(metric_scores) >= 5:
            confidence += 0.1  # More metrics = higher confidence
        
        # Adjust based on metric score variance
        if metric_scores:
            variance = statistics.variance(metric_scores.values())
            if variance < 0.1:  # Low variance = more consistent = higher confidence
                confidence += 0.1
            elif variance > 0.3:  # High variance = less consistent = lower confidence
                confidence -= 0.1
        
        # Adjust based on context completeness
        if context and len(context) > 2:
            confidence += 0.05
        
        return max(0.5, min(1.0, confidence))
    
    def _store_evaluation(self, evaluation: SuccessEvaluation):
        """Store evaluation in database"""
        
        with sqlite3.connect(self.evaluation_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO success_evaluations
                (evaluation_id, entity_id, entity_type, timestamp, overall_success_level,
                 overall_score, metric_scores, strengths, weaknesses, improvement_suggestions,
                 failure_categories, failure_root_causes, context, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation.evaluation_id,
                evaluation.entity_id,
                evaluation.entity_type,
                evaluation.timestamp.isoformat(),
                evaluation.overall_success_level.value,
                evaluation.overall_score,
                json.dumps({k.value: v for k, v in evaluation.metric_scores.items()}),
                json.dumps(evaluation.strengths),
                json.dumps(evaluation.weaknesses),
                json.dumps(evaluation.improvement_suggestions),
                json.dumps([c.value for c in evaluation.failure_categories]),
                json.dumps(evaluation.failure_root_causes),
                json.dumps(evaluation.context),
                evaluation.confidence
            ))
            
            conn.commit()
    
    def predict_success_probability(self, entity, context: Dict[str, Any] = None) -> PredictionResult:
        """Predict probability of success for entity before execution"""
        
        if context is None:
            context = {}
        
        # Get historical similar evaluations
        similar_evaluations = self._get_similar_evaluations(entity, context)
        
        if not similar_evaluations:
            # No historical data - use default prediction
            return PredictionResult(
                entity_id=getattr(entity, 'project_id', getattr(entity, 'task_id', 'unknown')),
                predicted_success_probability=0.7,  # Optimistic default
                predicted_success_level=SuccessLevel.GOOD,
                confidence=0.3,  # Low confidence without data
                recommendations=["No historical data available - proceed with caution"]
            )
        
        # Calculate prediction based on historical data
        success_scores = []
        failure_patterns = Counter()
        success_patterns = Counter()
        
        for eval_data in similar_evaluations:
            overall_score = eval_data['overall_score']
            
            success_scores.append(overall_score)
            
            if eval_data['overall_success_level'] in ['poor', 'failure']:
                failure_categories = json.loads(eval_data['failure_categories'] or '[]')
                for category in failure_categories:
                    failure_patterns[category] += 1
            else:
                strengths = json.loads(eval_data['strengths'] or '[]')
                for strength in strengths:
                    success_patterns[strength] += 1
        
        # Calculate predicted probability
        avg_success_score = statistics.mean(success_scores)
        success_probability = avg_success_score
        
        # Predict success level
        predicted_level = self._classify_success_level(success_probability, context.get('environment', 'general'))
        
        # Calculate confidence based on sample size and variance
        confidence = min(1.0, len(similar_evaluations) / 20)  # Max confidence with 20+ samples
        if len(success_scores) > 1:
            score_variance = statistics.variance(success_scores)
            confidence *= max(0.5, 1.0 - score_variance)  # Lower confidence for high variance
        
        # Generate risk and success factors
        risk_factors = [f"{category}: {count} occurrences" for category, count in failure_patterns.most_common(3)]
        success_factors = [f"{factor}" for factor, count in success_patterns.most_common(3)]
        
        # Generate recommendations
        recommendations = []
        if avg_success_score < 0.7:
            recommendations.append("Consider additional planning and risk mitigation")
        if failure_patterns:
            top_risk = failure_patterns.most_common(1)[0][0]
            recommendations.append(f"Pay special attention to {top_risk} risks")
        if success_factors:
            recommendations.append(f"Leverage successful patterns: {success_factors[0] if success_factors else 'N/A'}")
        
        return PredictionResult(
            entity_id=getattr(entity, 'project_id', getattr(entity, 'task_id', 'unknown')),
            predicted_success_probability=success_probability,
            predicted_success_level=predicted_level,
            confidence=confidence,
            risk_factors=risk_factors,
            success_factors=success_factors,
            recommendations=recommendations
        )
    
    def _get_similar_evaluations(self, entity, context: Dict[str, Any], limit: int = 50) -> List[Dict]:
        """Get similar historical evaluations for prediction"""
        
        with sqlite3.connect(self.evaluation_db_path) as conn:
            cursor = conn.cursor()
            
            entity_type = "project" if hasattr(entity, 'project_id') else "task"
            
            # Simple similarity based on entity type and recent evaluations
            cursor.execute("""
                SELECT overall_score, overall_success_level, failure_categories, strengths, context
                FROM success_evaluations
                WHERE entity_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (entity_type, limit))
            
            columns = ['overall_score', 'overall_success_level', 'failure_categories', 'strengths', 'context']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_success_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get success statistics for specified period"""
        
        with sqlite3.connect(self.evaluation_db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(overall_score) as avg_score,
                    COUNT(CASE WHEN overall_success_level IN ('excellent', 'good') THEN 1 END) as successful,
                    COUNT(CASE WHEN overall_success_level IN ('poor', 'failure') THEN 1 END) as failed
                FROM success_evaluations
                WHERE timestamp > ?
            """, (cutoff_date,))
            
            overall_stats = cursor.fetchone()
            
            # Success by entity type
            cursor.execute("""
                SELECT 
                    entity_type,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score,
                    COUNT(CASE WHEN overall_success_level IN ('excellent', 'good') THEN 1 END) as successful
                FROM success_evaluations
                WHERE timestamp > ?
                GROUP BY entity_type
            """, (cutoff_date,))
            
            entity_stats = cursor.fetchall()
            
            # Most common failure categories
            cursor.execute("""
                SELECT failure_categories
                FROM success_evaluations
                WHERE timestamp > ? AND overall_success_level IN ('poor', 'failure')
                AND failure_categories IS NOT NULL AND failure_categories != '[]'
            """, (cutoff_date,))
            
            failure_data = cursor.fetchall()
            
            # Count failure categories
            failure_counter = Counter()
            for row in failure_data:
                categories = json.loads(row[0])
                for category in categories:
                    failure_counter[category] += 1
        
        return {
            'period_days': days,
            'total_evaluations': overall_stats[0] if overall_stats else 0,
            'average_score': round(overall_stats[1], 3) if overall_stats and overall_stats[1] else 0,
            'success_rate': (overall_stats[2] / max(overall_stats[0], 1)) if overall_stats else 0,
            'failure_rate': (overall_stats[3] / max(overall_stats[0], 1)) if overall_stats else 0,
            'entity_type_stats': {
                row[0]: {
                    'count': row[1],
                    'avg_score': round(row[2], 3),
                    'success_rate': row[3] / max(row[1], 1)
                } for row in entity_stats
            },
            'top_failure_categories': dict(failure_counter.most_common(5))
        }
    
    def update_thresholds(self, days: int = 30) -> List[str]:
        """Update success thresholds based on recent evaluations"""
        
        with sqlite3.connect(self.evaluation_db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT evaluation_id, entity_id, entity_type, overall_success_level,
                       overall_score, metric_scores, context
                FROM success_evaluations
                WHERE timestamp > ?
            """, (cutoff_date,))
            
            evaluations = []
            for row in cursor.fetchall():
                try:
                    metric_scores_dict = json.loads(row[5])
                    context_dict = json.loads(row[6]) if row[6] else {}
                    
                    evaluation = SuccessEvaluation(
                        evaluation_id=row[0],
                        entity_id=row[1],
                        entity_type=row[2],
                        timestamp=datetime.now(),
                        overall_success_level=SuccessLevel(row[3]),
                        overall_score=row[4],
                        metric_scores={SuccessMetric(k): v for k, v in metric_scores_dict.items()},
                        context=context_dict
                    )
                    evaluations.append(evaluation)
                except Exception as e:
                    self.logger.warning(f"Error parsing evaluation {row[0]}: {e}")
        
        return self.threshold_manager.update_thresholds_from_data(evaluations)

# CLI interface for testing
async def main():
    """CLI interface for testing Success/Failure Classification System"""
    
    classifier = SuccessClassifier()
    
    print("📊 Agent Zero V1 - Success/Failure Classification System")
    print("=" * 70)
    
    # Create sample project for testing
    from datetime import datetime
    
    # Mock project class for testing
    class MockProject:
        def __init__(self, project_id, name, state, metrics):
            self.project_id = project_id
            self.name = name
            self.state = state
            self.metrics = metrics
    
    class MockMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockTask:
        def __init__(self, task_id, business_request, status, **kwargs):
            self.task_id = task_id
            self.business_request = business_request
            self.status = status
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # Test projects
    successful_project = MockProject(
        "proj_success_001",
        "Successful API Development",
        "completed",
        MockMetrics(
            total_tasks=5, completed_tasks=5, failed_tasks=0, success_rate=1.0,
            estimated_cost=0.15, actual_cost=0.12,
            estimated_duration=180, actual_duration=165
        )
    )
    
    failed_project = MockProject(
        "proj_fail_001", 
        "Failed Complex Project",
        "failed",
        MockMetrics(
            total_tasks=8, completed_tasks=3, failed_tasks=5, success_rate=0.375,
            estimated_cost=0.50, actual_cost=0.85,
            estimated_duration=300, actual_duration=420
        )
    )
    
    test_projects = [successful_project, failed_project]
    
    print(f"\n📋 Evaluating {len(test_projects)} test projects...")
    
    evaluations = []
    
    for project in test_projects:
        context = {
            'environment': 'development',
            'complexity': 'moderate',
            'team_size': 3
        }
        
        evaluation = classifier.evaluate_project_success(project, context)
        evaluations.append(evaluation)
        
        print(f"\n🔍 Project: {project.name}")
        print(f"   📊 Overall Score: {evaluation.overall_score:.1%}")
        print(f"   🏆 Success Level: {evaluation.overall_success_level.value}")
        print(f"   🎯 Confidence: {evaluation.confidence:.1%}")
        
        if evaluation.strengths:
            print(f"   ✅ Strengths: {', '.join(evaluation.strengths[:2])}")
        
        if evaluation.weaknesses:
            print(f"   ⚠️  Weaknesses: {', '.join(evaluation.weaknesses[:2])}")
        
        if evaluation.improvement_suggestions:
            print(f"   💡 Suggestions: {evaluation.improvement_suggestions[0]}")
        
        if evaluation.failure_categories:
            print(f"   🚨 Failure Categories: {', '.join([c.value for c in evaluation.failure_categories])}")
    
    # Test prediction
    print(f"\n🔮 Testing success prediction...")
    
    # Mock future project
    future_project = MockProject(
        "proj_future_001",
        "Future Project Prediction",
        "planning",
        None
    )
    
    prediction = classifier.predict_success_probability(future_project, {'environment': 'development'})
    
    print(f"   📈 Predicted Success: {prediction.predicted_success_probability:.1%}")
    print(f"   🎯 Predicted Level: {prediction.predicted_success_level.value}")
    print(f"   🔍 Confidence: {prediction.confidence:.1%}")
    
    if prediction.risk_factors:
        print(f"   ⚠️  Risk Factors: {prediction.risk_factors[0] if prediction.risk_factors else 'None'}")
    
    if prediction.recommendations:
        print(f"   💡 Recommendations: {prediction.recommendations[0]}")
    
    # System statistics
    print(f"\n📊 System Statistics (Last 30 days):")
    stats = classifier.get_success_statistics(30)
    
    print(f"   Total Evaluations: {stats['total_evaluations']}")
    print(f"   Average Score: {stats['average_score']:.1%}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    print(f"   Failure Rate: {stats['failure_rate']:.1%}")
    
    if stats['entity_type_stats']:
        print(f"   📈 By Entity Type:")
        for entity_type, type_stats in stats['entity_type_stats'].items():
            print(f"      {entity_type}: {type_stats['success_rate']:.1%} success ({type_stats['count']} evaluations)")
    
    if stats['top_failure_categories']:
        print(f"   🚨 Top Failure Categories:")
        for category, count in list(stats['top_failure_categories'].items())[:3]:
            print(f"      {category}: {count} occurrences")
    
    # Test threshold updates
    print(f"\n🎛️  Testing threshold updates...")
    threshold_updates = classifier.update_thresholds(30)
    
    if threshold_updates:
        print(f"   ✅ Threshold Updates:")
        for update in threshold_updates[:3]:
            print(f"      • {update}")
    else:
        print(f"   ℹ️  No threshold updates needed (insufficient data)")
    
    print(f"\n✅ Success/Failure Classification System test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())