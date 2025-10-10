# Active Metrics Analyzer - Analytics Module for Agent Zero V1
# Task: A0-26 Active Metrics Analyzer (Week 44-45)
# Focus: Real-time Kaizen z alertami i optimization suggestions
# Zakres: KaizenMetricsAnalyzer, cost optimization engine, daily reports
# CLI: a0 kaizen-report, a0 cost-analysis

"""
Active Metrics Analyzer for Agent Zero V1
Real-time Kaizen analytics with alerts and optimization

This system provides:
- Real-time metrics monitoring and analysis
- Automated alert generation for anomalies
- Cost optimization recommendations
- Performance trend analysis
- Daily/weekly Kaizen reports
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3
import statistics
import numpy as np
from collections import defaultdict, deque
import schedule
import threading

# Import existing components
try:
    import sys
    sys.path.insert(0, '.')
    from simple_tracker import SimpleTracker
    from success_failure_classifier import SuccessClassifier, SuccessEvaluation, SuccessLevel
    from project_orchestrator import ProjectOrchestrator, Project, ProjectState, ProjectMetrics
    from feedback_loop_engine import FeedbackLoopEngine
except ImportError as e:
    logging.warning(f"Could not import existing components: {e}")
    # Fallback classes for testing
    class SimpleTracker:
        def track_event(self, event): pass
        def get_daily_stats(self): 
            return type('', (), {'get_total_tasks': lambda: 0, 'get_avg_rating': lambda: 0})()

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types of alerts"""
    COST_OVERRUN = "cost_overrun"
    TIME_OVERRUN = "time_overrun" 
    QUALITY_DEGRADATION = "quality_degradation"
    FAILURE_SPIKE = "failure_spike"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    RESOURCE_SHORTAGE = "resource_shortage"
    SLA_VIOLATION = "sla_violation"
    TREND_ANOMALY = "trend_anomaly"

class MetricTrend(Enum):
    """Metric trend directions"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"

@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    entity_id: str  # Project/Task/System ID
    entity_type: str
    metric_name: str
    current_value: float
    threshold_value: float
    trend: Optional[MetricTrend] = None
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class MetricSnapshot:
    """Point-in-time metric reading"""
    metric_name: str
    entity_id: str
    entity_type: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    entity_id: str
    trend_direction: MetricTrend
    trend_strength: float  # 0-1, how strong the trend is
    prediction_7d: Optional[float] = None
    prediction_30d: Optional[float] = None
    confidence: float = 0.0
    data_points: int = 0
    analysis_period_days: int = 7

@dataclass
class OptimizationSuggestion:
    """Optimization recommendation"""
    suggestion_id: str
    category: str
    title: str
    description: str
    expected_impact: str
    confidence: float
    implementation_effort: str  # "low", "medium", "high"
    priority: int  # 1-10
    related_metrics: List[str] = field(default_factory=list)
    success_probability: float = 0.0

class MetricsCollector:
    """Collects metrics from various system components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.collection_interval = 60  # seconds
        self._collectors = {}
        self._running = False
    
    def register_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register a metric collector function"""
        self._collectors[name] = collector_func
        self.logger.info(f"Registered collector: {name}")
    
    async def start_collection(self, callback: Optional[Callable] = None):
        """Start continuous metric collection"""
        
        self._running = True
        self.logger.info("Started metrics collection")
        
        while self._running:
            try:
                timestamp = datetime.now()
                
                for collector_name, collector_func in self._collectors.items():
                    try:
                        metrics = collector_func()
                        
                        for metric_name, value in metrics.items():
                            snapshot = MetricSnapshot(
                                metric_name=metric_name,
                                entity_id=collector_name,
                                entity_type="system_component",
                                value=value,
                                timestamp=timestamp
                            )
                            
                            if callback:
                                await callback(snapshot)
                    
                    except Exception as e:
                        self.logger.error(f"Error in collector {collector_name}: {e}")
                
                await asyncio.sleep(self.collection_interval)
            
            except Exception as e:
                self.logger.error(f"Error in metric collection loop: {e}")
                await asyncio.sleep(5)
    
    def stop_collection(self):
        """Stop metric collection"""
        self._running = False
        self.logger.info("Stopped metrics collection")

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        
        # Alert rules and thresholds
        self.alert_rules = {
            AlertType.COST_OVERRUN: {
                'threshold_multiplier': 1.2,  # 20% over estimate
                'severity': AlertSeverity.WARNING
            },
            AlertType.TIME_OVERRUN: {
                'threshold_multiplier': 1.3,  # 30% over estimate
                'severity': AlertSeverity.WARNING
            },
            AlertType.QUALITY_DEGRADATION: {
                'threshold_value': 0.7,  # Below 70%
                'severity': AlertSeverity.CRITICAL
            },
            AlertType.FAILURE_SPIKE: {
                'threshold_rate': 0.3,  # 30% failure rate
                'severity': AlertSeverity.CRITICAL
            }
        }
        
        # Active alerts cache
        self._active_alerts = {}
        self._load_active_alerts()
    
    def _initialize_database(self):
        """Initialize alert database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    trend TEXT,
                    recommendations TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_entity ON alerts(entity_id, entity_type)
            """)
            
            conn.commit()
    
    def _load_active_alerts(self):
        """Load active alerts from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT alert_id, alert_type, severity, title, description, entity_id,
                       entity_type, metric_name, current_value, threshold_value, trend,
                       recommendations, timestamp, acknowledged, resolved
                FROM alerts
                WHERE resolved = FALSE
            """)
            
            for row in cursor.fetchall():
                try:
                    alert = Alert(
                        alert_id=row[0],
                        alert_type=AlertType(row[1]),
                        severity=AlertSeverity(row[2]),
                        title=row[3],
                        description=row[4],
                        entity_id=row[5],
                        entity_type=row[6],
                        metric_name=row[7],
                        current_value=row[8],
                        threshold_value=row[9],
                        trend=MetricTrend(row[10]) if row[10] else None,
                        recommendations=json.loads(row[11]) if row[11] else [],
                        timestamp=datetime.fromisoformat(row[12]),
                        acknowledged=bool(row[13]),
                        resolved=bool(row[14])
                    )
                    self._active_alerts[alert.alert_id] = alert
                
                except Exception as e:
                    self.logger.error(f"Error loading alert {row[0]}: {e}")
    
    def check_metric_for_alerts(self, snapshot: MetricSnapshot, 
                              historical_data: List[MetricSnapshot]) -> List[Alert]:
        """Check metric snapshot for alert conditions"""
        
        alerts = []
        
        # Cost overrun check
        if 'cost' in snapshot.metric_name.lower():
            estimated = snapshot.context.get('estimated_value')
            if estimated and snapshot.value > estimated * 1.2:
                alert = self._create_alert(
                    AlertType.COST_OVERRUN,
                    f"Cost Overrun - {snapshot.entity_id}",
                    f"Current cost {snapshot.value:.3f} exceeds estimate {estimated:.3f} by {((snapshot.value/estimated)-1)*100:.1f}%",
                    snapshot,
                    estimated * 1.2,
                    ["Review budget allocation", "Optimize resource usage", "Consider alternative approaches"]
                )
                alerts.append(alert)
        
        # Time overrun check
        if 'duration' in snapshot.metric_name.lower() or 'time' in snapshot.metric_name.lower():
            estimated = snapshot.context.get('estimated_value')
            if estimated and snapshot.value > estimated * 1.3:
                alert = self._create_alert(
                    AlertType.TIME_OVERRUN,
                    f"Time Overrun - {snapshot.entity_id}",
                    f"Current duration {snapshot.value:.1f} exceeds estimate {estimated:.1f} by {((snapshot.value/estimated)-1)*100:.1f}%",
                    snapshot,
                    estimated * 1.3,
                    ["Optimize task scheduling", "Identify bottlenecks", "Consider parallel execution"]
                )
                alerts.append(alert)
        
        # Quality degradation check
        if 'quality' in snapshot.metric_name.lower() or 'success' in snapshot.metric_name.lower():
            if snapshot.value < 0.7:
                alert = self._create_alert(
                    AlertType.QUALITY_DEGRADATION,
                    f"Quality Degradation - {snapshot.entity_id}",
                    f"Quality score {snapshot.value:.1%} below acceptable threshold",
                    snapshot,
                    0.7,
                    ["Review quality assurance processes", "Increase testing coverage", "Analyze failure patterns"]
                )
                alerts.append(alert)
        
        # Trend-based anomaly detection
        if len(historical_data) >= 5:
            trend_alert = self._check_trend_anomaly(snapshot, historical_data)
            if trend_alert:
                alerts.append(trend_alert)
        
        return alerts
    
    def _create_alert(self, alert_type: AlertType, title: str, description: str,
                     snapshot: MetricSnapshot, threshold: float,
                     recommendations: List[str]) -> Alert:
        """Create alert from metric snapshot"""
        
        severity = self.alert_rules.get(alert_type, {}).get('severity', AlertSeverity.WARNING)
        
        return Alert(
            alert_id=f"alert_{alert_type.value}_{snapshot.entity_id}_{int(time.time())}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            entity_id=snapshot.entity_id,
            entity_type=snapshot.entity_type,
            metric_name=snapshot.metric_name,
            current_value=snapshot.value,
            threshold_value=threshold,
            recommendations=recommendations,
            timestamp=snapshot.timestamp
        )
    
    def _check_trend_anomaly(self, snapshot: MetricSnapshot, 
                           historical_data: List[MetricSnapshot]) -> Optional[Alert]:
        """Check for trend anomalies in metric data"""
        
        # Extract recent values
        recent_values = [s.value for s in historical_data[-10:]]  # Last 10 readings
        
        if len(recent_values) < 5:
            return None
        
        # Calculate trend
        avg_recent = statistics.mean(recent_values[-3:])  # Last 3 readings
        avg_historical = statistics.mean(recent_values[:-3])  # Earlier readings
        
        # Detect significant changes
        if avg_historical > 0:
            change_ratio = (avg_recent - avg_historical) / avg_historical
            
            # Significant degradation
            if change_ratio < -0.2:  # 20% degradation
                return self._create_alert(
                    AlertType.TREND_ANOMALY,
                    f"Negative Trend Detected - {snapshot.entity_id}",
                    f"Metric {snapshot.metric_name} showing {change_ratio:.1%} degradation over recent readings",
                    snapshot,
                    avg_historical,
                    [
                        "Investigate root cause of performance degradation",
                        "Review recent changes that may have impacted performance",
                        "Consider rollback if degradation continues"
                    ]
                )
        
        return None
    
    def store_alert(self, alert: Alert):
        """Store alert in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts
                (alert_id, alert_type, severity, title, description, entity_id, entity_type,
                 metric_name, current_value, threshold_value, trend, recommendations, timestamp,
                 acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.alert_type.value,
                alert.severity.value,
                alert.title,
                alert.description,
                alert.entity_id,
                alert.entity_type,
                alert.metric_name,
                alert.current_value,
                alert.threshold_value,
                alert.trend.value if alert.trend else None,
                json.dumps(alert.recommendations),
                alert.timestamp.isoformat(),
                alert.acknowledged,
                alert.resolved
            ))
            conn.commit()
        
        self._active_alerts[alert.alert_id] = alert
        
        self.logger.info(f"Generated {alert.severity.value} alert: {alert.title}")

class CostOptimizationEngine:
    """Analyzes costs and provides optimization recommendations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cost optimization strategies
        self.optimization_strategies = {
            'model_selection': {
                'impact': 'high',
                'effort': 'low',
                'description': 'Switch to more cost-effective models for similar quality'
            },
            'batch_processing': {
                'impact': 'medium',
                'effort': 'medium', 
                'description': 'Group similar tasks to reduce overhead costs'
            },
            'caching': {
                'impact': 'high',
                'effort': 'medium',
                'description': 'Cache frequent computations to avoid recomputation'
            },
            'resource_right_sizing': {
                'impact': 'medium',
                'effort': 'low',
                'description': 'Adjust resource allocation based on actual usage patterns'
            }
        }
    
    def analyze_cost_trends(self, cost_history: List[MetricSnapshot],
                          period_days: int = 30) -> Dict[str, Any]:
        """Analyze cost trends and identify optimization opportunities"""
        
        if len(cost_history) < 7:  # Need minimum data
            return {'status': 'insufficient_data', 'message': 'Need at least 7 days of cost data'}
        
        # Calculate trend
        values = [s.value for s in cost_history[-period_days:]]
        timestamps = [s.timestamp for s in cost_history[-period_days:]]
        
        # Simple linear trend calculation
        if len(values) >= 2:
            # Calculate daily cost change
            days = [(t - timestamps[0]).days for t in timestamps]
            if days[-1] > 0:
                slope = (values[-1] - values[0]) / days[-1]  # Cost change per day
                
                trend_direction = MetricTrend.IMPROVING if slope < -0.01 else \
                                MetricTrend.DEGRADING if slope > 0.01 else \
                                MetricTrend.STABLE
                
                # Project future costs
                projected_7d = values[-1] + (slope * 7)
                projected_30d = values[-1] + (slope * 30)
                
                analysis = {
                    'current_daily_cost': values[-1] if values else 0,
                    'trend_direction': trend_direction.value,
                    'daily_change': slope,
                    'projected_cost_7d': max(0, projected_7d),
                    'projected_cost_30d': max(0, projected_30d),
                    'total_period_cost': sum(values),
                    'avg_daily_cost': statistics.mean(values),
                    'cost_volatility': statistics.stdev(values) if len(values) > 1 else 0
                }
                
                # Generate optimization suggestions
                suggestions = self._generate_cost_optimizations(analysis, cost_history)
                analysis['optimization_suggestions'] = suggestions
                
                return analysis
        
        return {'status': 'calculation_error', 'message': 'Could not calculate cost trends'}
    
    def _generate_cost_optimizations(self, cost_analysis: Dict[str, Any],
                                   cost_history: List[MetricSnapshot]) -> List[OptimizationSuggestion]:
        """Generate specific cost optimization suggestions"""
        
        suggestions = []
        
        # High cost suggestion
        if cost_analysis['avg_daily_cost'] > 0.50:  # Above $0.50/day
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"cost_opt_{int(time.time())}_1",
                category="model_selection",
                title="Consider More Cost-Effective Models",
                description="Daily costs are high. Evaluate switching to local models or cheaper cloud options for routine tasks.",
                expected_impact="20-40% cost reduction",
                confidence=0.8,
                implementation_effort="low",
                priority=8,
                related_metrics=["cost", "efficiency"],
                success_probability=0.85
            ))
        
        # Volatile cost suggestion
        if cost_analysis['cost_volatility'] > 0.1:  # High volatility
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"cost_opt_{int(time.time())}_2",
                category="resource_planning",
                title="Stabilize Cost Patterns",
                description="Cost patterns are volatile. Implement better resource planning and budget controls.",
                expected_impact="Reduce cost variance by 50%",
                confidence=0.7,
                implementation_effort="medium",
                priority=6,
                related_metrics=["cost", "reliability"],
                success_probability=0.70
            ))
        
        # Increasing trend suggestion
        if cost_analysis.get('daily_change', 0) > 0.05:  # Increasing costs
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"cost_opt_{int(time.time())}_3",
                category="cost_control",
                title="Implement Cost Controls",
                description="Costs are trending upward. Set up automated cost limits and optimization alerts.",
                expected_impact="Prevent further cost increases",
                confidence=0.9,
                implementation_effort="low",
                priority=9,
                related_metrics=["cost"],
                success_probability=0.95
            ))
        
        # Sort by priority
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        
        return suggestions

class KaizenReporter:
    """Generates Kaizen reports and insights"""
    
    def __init__(self, success_classifier: SuccessClassifier):
        self.success_classifier = success_classifier
        self.logger = logging.getLogger(__name__)
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate daily Kaizen report"""
        
        if date is None:
            date = datetime.now()
        
        # Get statistics for the day
        stats = self.success_classifier.get_success_statistics(days=1)
        
        # Calculate key metrics
        report = {
            'report_date': date.strftime('%Y-%m-%d'),
            'report_type': 'daily_kaizen',
            'generated_at': datetime.now().isoformat(),
            
            # Daily summary
            'daily_summary': {
                'total_evaluations': stats['total_evaluations'],
                'average_success_score': stats['average_score'],
                'success_rate': stats['success_rate'],
                'failure_rate': stats['failure_rate']
            },
            
            # Performance insights
            'performance_insights': self._generate_performance_insights(stats),
            
            # Learning indicators
            'learning_indicators': {
                'evaluation_count': stats['total_evaluations'],
                'data_quality': 'good' if stats['total_evaluations'] > 5 else 'limited',
                'confidence_level': min(1.0, stats['total_evaluations'] / 20)
            },
            
            # Improvement areas
            'improvement_areas': self._identify_improvement_areas(stats),
            
            # Tomorrow's focus
            'tomorrow_focus': self._suggest_tomorrow_focus(stats)
        }
        
        return report
    
    def generate_weekly_report(self, week_end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate weekly Kaizen report"""
        
        if week_end_date is None:
            week_end_date = datetime.now()
        
        # Get weekly statistics
        stats = self.success_classifier.get_success_statistics(days=7)
        
        # Get previous week for comparison
        prev_week_stats = self.success_classifier.get_success_statistics(days=14)  # Last 14 days
        
        report = {
            'report_date': week_end_date.strftime('%Y-%m-%d'),
            'report_type': 'weekly_kaizen',
            'generated_at': datetime.now().isoformat(),
            
            # Weekly summary
            'weekly_summary': {
                'total_evaluations': stats['total_evaluations'],
                'average_success_score': stats['average_score'],
                'success_rate': stats['success_rate'],
                'improvement_vs_previous': self._calculate_improvement(stats, prev_week_stats)
            },
            
            # Trend analysis
            'trends': self._analyze_weekly_trends(stats, prev_week_stats),
            
            # Success patterns
            'success_patterns': self._identify_success_patterns(stats),
            
            # Failure analysis
            'failure_analysis': {
                'top_failure_categories': stats.get('top_failure_categories', {}),
                'failure_trends': 'decreasing' if stats['failure_rate'] < prev_week_stats.get('failure_rate', 1) else 'stable'
            },
            
            # Strategic recommendations
            'strategic_recommendations': self._generate_strategic_recommendations(stats),
            
            # Next week priorities
            'next_week_priorities': self._suggest_next_week_priorities(stats)
        }
        
        return report
    
    def _generate_performance_insights(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance insights from statistics"""
        
        insights = []
        
        if stats['success_rate'] > 0.8:
            insights.append(f"Excellent performance with {stats['success_rate']:.1%} success rate")
        elif stats['success_rate'] > 0.6:
            insights.append(f"Good performance with room for improvement ({stats['success_rate']:.1%} success)")
        else:
            insights.append(f"Performance needs attention - only {stats['success_rate']:.1%} success rate")
        
        if stats['average_score'] > 0.8:
            insights.append("High quality execution across all metrics")
        elif stats['average_score'] < 0.6:
            insights.append("Quality metrics below target - review processes")
        
        if stats['total_evaluations'] < 5:
            insights.append("Limited data available - need more task completions for better insights")
        
        return insights
    
    def _identify_improvement_areas(self, stats: Dict[str, Any]) -> List[str]:
        """Identify key improvement areas"""
        
        areas = []
        
        if stats['failure_rate'] > 0.2:
            areas.append("Reduce task failure rate through better planning")
        
        if stats.get('top_failure_categories'):
            top_failure = list(stats['top_failure_categories'].keys())[0]
            areas.append(f"Address {top_failure} failures - primary failure mode")
        
        if stats['average_score'] < 0.7:
            areas.append("Improve overall execution quality")
        
        if not areas:
            areas.append("Maintain current performance levels")
        
        return areas
    
    def _suggest_tomorrow_focus(self, stats: Dict[str, Any]) -> List[str]:
        """Suggest focus areas for tomorrow"""
        
        focus_areas = []
        
        if stats['total_evaluations'] < 3:
            focus_areas.append("Complete more tasks to gather performance data")
        
        if stats.get('top_failure_categories'):
            top_category = list(stats['top_failure_categories'].keys())[0]
            focus_areas.append(f"Prevent {top_category} failures in new tasks")
        
        if stats['success_rate'] > 0.8:
            focus_areas.append("Maintain high performance while taking on more challenging tasks")
        else:
            focus_areas.append("Focus on quality over quantity until success rate improves")
        
        return focus_areas
    
    def _calculate_improvement(self, current: Dict[str, Any], 
                             previous: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement vs previous period"""
        
        if not previous or previous.get('total_evaluations', 0) == 0:
            return {'status': 'no_comparison_data'}
        
        current_rate = current.get('success_rate', 0)
        prev_rate = previous.get('success_rate', 0)
        
        improvement = {
            'success_rate_change': current_rate - prev_rate,
            'score_change': current.get('average_score', 0) - previous.get('average_score', 0),
            'evaluation_count_change': current.get('total_evaluations', 0) - previous.get('total_evaluations', 0)
        }
        
        # Overall assessment
        if improvement['success_rate_change'] > 0.05:
            improvement['overall'] = 'significant_improvement'
        elif improvement['success_rate_change'] > 0.01:
            improvement['overall'] = 'slight_improvement'
        elif improvement['success_rate_change'] > -0.01:
            improvement['overall'] = 'stable'
        else:
            improvement['overall'] = 'degradation'
        
        return improvement

class ActiveMetricsAnalyzer:
    """Main Active Metrics Analyzer system"""
    
    def __init__(self, 
                 success_classifier_db: str = "success_classifier.db",
                 metrics_db: str = "active_metrics.db"):
        
        self.success_classifier = SuccessClassifier(evaluation_db_path=success_classifier_db)
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.cost_optimizer = CostOptimizationEngine()
        self.kaizen_reporter = KaizenReporter(self.success_classifier)
        
        self.metrics_db_path = metrics_db
        self.logger = logging.getLogger(__name__)
        self._initialize_metrics_database()
        
        # Real-time monitoring state
        self._monitoring_active = False
        self._metric_history = defaultdict(deque)
        self._max_history_size = 1000
        
        # Integration components
        try:
            self.tracker = SimpleTracker()
        except:
            self.tracker = None
            self.logger.warning("Could not initialize SimpleTracker")
    
    def _initialize_metrics_database(self):
        """Initialize metrics storage database"""
        
        with sqlite3.connect(self.metrics_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metric_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    context TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_timestamp ON metric_snapshots(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_entity ON metric_snapshots(entity_id, metric_name)
            """)
            
            conn.commit()
    
    async def start_monitoring(self):
        """Start real-time metrics monitoring"""
        
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Register default collectors
        self._register_default_collectors()
        
        # Start metric collection
        collection_task = asyncio.create_task(
            self.metrics_collector.start_collection(self._process_metric_snapshot)
        )
        
        # Start alert checking
        alert_task = asyncio.create_task(self._run_alert_checks())
        
        # Start scheduled reports
        report_task = asyncio.create_task(self._run_scheduled_reports())
        
        self.logger.info("Started Active Metrics Analyzer monitoring")
        
        try:
            await asyncio.gather(collection_task, alert_task, report_task)
        except asyncio.CancelledError:
            self.logger.info("Monitoring cancelled")
        finally:
            self._monitoring_active = False
    
    def stop_monitoring(self):
        """Stop metrics monitoring"""
        self._monitoring_active = False
        self.metrics_collector.stop_collection()
    
    def _register_default_collectors(self):
        """Register default metric collectors"""
        
        # SimpleTracker metrics
        if self.tracker:
            def collect_tracker_metrics():
                try:
                    daily_stats = self.tracker.get_daily_stats()
                    return {
                        'total_tasks': daily_stats.get_total_tasks(),
                        'avg_rating': daily_stats.get_avg_rating(),
                        'completion_rate': 0.85  # Mock for now
                    }
                except:
                    return {}
            
            self.metrics_collector.register_collector('simple_tracker', collect_tracker_metrics)
        
        # System metrics
        def collect_system_metrics():
            return {
                'memory_usage': 0.6,  # Mock - would get real system metrics
                'cpu_usage': 0.4,
                'active_connections': 5,
                'response_time': 0.8
            }
        
        self.metrics_collector.register_collector('system', collect_system_metrics)
    
    async def _process_metric_snapshot(self, snapshot: MetricSnapshot):
        """Process incoming metric snapshot"""
        
        # Store in database
        self._store_metric_snapshot(snapshot)
        
        # Update in-memory history
        history_key = f"{snapshot.entity_id}_{snapshot.metric_name}"
        history = self._metric_history[history_key]
        
        history.append(snapshot)
        if len(history) > self._max_history_size:
            history.popleft()
        
        # Check for alerts
        historical_data = list(history)
        alerts = self.alert_manager.check_metric_for_alerts(snapshot, historical_data)
        
        # Process alerts
        for alert in alerts:
            self.alert_manager.store_alert(alert)
            
            # Log critical alerts immediately
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                self.logger.critical(f"ALERT: {alert.title} - {alert.description}")
    
    def _store_metric_snapshot(self, snapshot: MetricSnapshot):
        """Store metric snapshot in database"""
        
        with sqlite3.connect(self.metrics_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metric_snapshots
                (metric_name, entity_id, entity_type, value, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                snapshot.metric_name,
                snapshot.entity_id, 
                snapshot.entity_type,
                snapshot.value,
                snapshot.timestamp.isoformat(),
                json.dumps(snapshot.context)
            ))
            conn.commit()
    
    async def _run_alert_checks(self):
        """Run periodic alert checks"""
        
        while self._monitoring_active:
            try:
                # Check for system-wide anomalies
                await self._check_system_alerts()
                
                # Clean up resolved alerts
                self._cleanup_old_alerts()
                
                await asyncio.sleep(300)  # Check every 5 minutes
            
            except Exception as e:
                self.logger.error(f"Error in alert checking: {e}")
                await asyncio.sleep(60)
    
    async def _check_system_alerts(self):
        """Check for system-wide alerts"""
        
        # Check for too many active alerts
        active_count = len([a for a in self._active_alerts.values() if not a.resolved])
        
        if active_count > 10:
            system_alert = Alert(
                alert_id=f"system_alert_{int(time.time())}",
                alert_type=AlertType.PERFORMANCE_ANOMALY,
                severity=AlertSeverity.WARNING,
                title="High Alert Volume",
                description=f"System has {active_count} active alerts - may indicate systemic issues",
                entity_id="system",
                entity_type="system",
                metric_name="alert_count",
                current_value=active_count,
                threshold_value=10,
                recommendations=[
                    "Review alert rules for false positives",
                    "Investigate common root causes",
                    "Consider system-wide improvements"
                ]
            )
            self.alert_manager.store_alert(system_alert)
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        
        cutoff_date = datetime.now() - timedelta(days=7)
        
        with sqlite3.connect(self.alert_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM alerts 
                WHERE resolved = TRUE AND resolved_at < ?
            """, (cutoff_date.isoformat(),))
            conn.commit()
    
    async def _run_scheduled_reports(self):
        """Run scheduled report generation"""
        
        # Daily report at 23:00
        schedule.every().day.at("23:00").do(self._generate_daily_report_job)
        
        # Weekly report on Sunday at 23:30  
        schedule.every().sunday.at("23:30").do(self._generate_weekly_report_job)
        
        while self._monitoring_active:
            try:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in scheduled reports: {e}")
                await asyncio.sleep(300)
    
    def _generate_daily_report_job(self):
        """Job wrapper for daily report generation"""
        try:
            report = self.kaizen_reporter.generate_daily_report()
            self._save_report(report)
            
            if self.tracker:
                self.tracker.track_event({
                    'type': 'daily_report_generated',
                    'evaluations_count': report['daily_summary']['total_evaluations'],
                    'success_rate': report['daily_summary']['success_rate']
                })
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    def _generate_weekly_report_job(self):
        """Job wrapper for weekly report generation"""
        try:
            report = self.kaizen_reporter.generate_weekly_report()
            self._save_report(report)
            
            if self.tracker:
                self.tracker.track_event({
                    'type': 'weekly_report_generated',
                    'evaluations_count': report['weekly_summary']['total_evaluations'],
                    'success_rate': report['weekly_summary']['success_rate']
                })
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
    
    def _save_report(self, report: Dict[str, Any]):
        """Save report to file system"""
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_type = report['report_type']
        report_date = report['report_date']
        filename = f"{report_type}_{report_date}.json"
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Saved {report_type} report to {filepath}")
    
    def get_current_system_status(self) -> Dict[str, Any]:
        """Get current comprehensive system status"""
        
        # Active alerts
        active_alerts = [a for a in self.alert_manager._active_alerts.values() if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        # Recent metrics
        recent_metrics = {}
        for key, history in self._metric_history.items():
            if history:
                recent_metrics[key] = history[-1].value
        
        return {
            'monitoring_active': self._monitoring_active,
            'total_active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'metric_sources': len(self.metrics_collector._collectors),
            'recent_metrics': recent_metrics,
            'system_health': 'healthy' if len(critical_alerts) == 0 else 'degraded'
        }
    
    # CLI Command Methods
    def generate_kaizen_report_cli(self, report_type: str = 'daily') -> str:
        """CLI command: a0 kaizen-report"""
        
        if report_type == 'daily':
            report = self.kaizen_reporter.generate_daily_report()
        elif report_type == 'weekly':
            report = self.kaizen_reporter.generate_weekly_report()
        else:
            return f"Error: Unknown report type '{report_type}'. Use 'daily' or 'weekly'"
        
        # Format report for CLI display
        output = []
        output.append(f"🎯 Agent Zero V1 - {report_type.title()} Kaizen Report")
        output.append("=" * 60)
        output.append(f"Generated: {report['generated_at']}")
        
        if report_type == 'daily':
            summary = report['daily_summary']
            output.append(f"\n📊 Daily Summary ({report['report_date']}):")
            output.append(f"   Evaluations: {summary['total_evaluations']}")
            output.append(f"   Success Rate: {summary['success_rate']:.1%}")
            output.append(f"   Avg Score: {summary['average_success_score']:.1%}")
            
            if report.get('performance_insights'):
                output.append(f"\n💡 Key Insights:")
                for insight in report['performance_insights'][:3]:
                    output.append(f"   • {insight}")
            
            if report.get('improvement_areas'):
                output.append(f"\n🎯 Improvement Areas:")
                for area in report['improvement_areas'][:3]:
                    output.append(f"   • {area}")
        
        return "\n".join(output)
    
    def generate_cost_analysis_cli(self, days: int = 7) -> str:
        """CLI command: a0 cost-analysis"""
        
        # Get cost history
        with sqlite3.connect(self.metrics_db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT metric_name, entity_id, value, timestamp
                FROM metric_snapshots
                WHERE metric_name LIKE '%cost%' AND timestamp > ?
                ORDER BY timestamp
            """, (cutoff_date,))
            
            cost_snapshots = []
            for row in cursor.fetchall():
                snapshot = MetricSnapshot(
                    metric_name=row[0],
                    entity_id=row[1],
                    entity_type="unknown",  # Not stored in this simple version
                    value=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                )
                cost_snapshots.append(snapshot)
        
        if not cost_snapshots:
            return "📊 No cost data available for analysis"
        
        # Analyze costs
        cost_analysis = self.cost_optimizer.analyze_cost_trends(cost_snapshots, days)
        
        if cost_analysis.get('status') == 'insufficient_data':
            return f"📊 Insufficient cost data (need at least 7 days)"
        
        # Format output
        output = []
        output.append(f"💰 Agent Zero V1 - Cost Analysis ({days} days)")
        output.append("=" * 50)
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        output.append(f"\n📈 Cost Trends:")
        output.append(f"   Current Daily Cost: ${cost_analysis['current_daily_cost']:.4f}")
        output.append(f"   Trend: {cost_analysis['trend_direction']}")
        output.append(f"   Daily Change: ${cost_analysis['daily_change']:.4f}")
        output.append(f"   Period Total: ${cost_analysis['total_period_cost']:.4f}")
        
        output.append(f"\n🔮 Projections:")
        output.append(f"   7-day projection: ${cost_analysis['projected_cost_7d']:.4f}")
        output.append(f"   30-day projection: ${cost_analysis['projected_cost_30d']:.4f}")
        
        if cost_analysis.get('optimization_suggestions'):
            output.append(f"\n💡 Optimization Suggestions:")
            for suggestion in cost_analysis['optimization_suggestions'][:3]:
                output.append(f"   • {suggestion.title}")
                output.append(f"     Impact: {suggestion.expected_impact}")
                output.append(f"     Priority: {suggestion.priority}/10")
        
        return "\n".join(output)

# CLI interface for testing
async def main():
    """CLI interface for testing Active Metrics Analyzer"""
    
    analyzer = ActiveMetricsAnalyzer()
    
    print("📊 Agent Zero V1 - Active Metrics Analyzer")
    print("=" * 60)
    
    # Test CLI commands
    print(f"\n🎯 Testing CLI Commands:")
    
    # Test kaizen report
    print(f"\n1. Kaizen Report (Daily):")
    daily_report = analyzer.generate_kaizen_report_cli('daily')
    print(daily_report)
    
    # Test cost analysis
    print(f"\n2. Cost Analysis:")
    cost_analysis = analyzer.generate_cost_analysis_cli(7)
    print(cost_analysis)
    
    # Test system status
    print(f"\n📊 System Status:")
    status = analyzer.get_current_system_status()
    print(f"   Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
    print(f"   Active Alerts: {status['total_active_alerts']}")
    print(f"   Critical Alerts: {status['critical_alerts']}")
    print(f"   Metric Sources: {status['metric_sources']}")
    print(f"   System Health: {status['system_health']}")
    
    # Show recent metrics if available
    if status['recent_metrics']:
        print(f"\n📈 Recent Metrics:")
        for key, value in list(status['recent_metrics'].items())[:5]:
            print(f"   {key}: {value:.3f}")
    
    # Test monitoring (brief demo)
    print(f"\n🔄 Starting brief monitoring demo...")
    
    # Start monitoring for 10 seconds
    monitoring_task = asyncio.create_task(analyzer.start_monitoring())
    
    # Let it run for a short time
    await asyncio.sleep(5)
    
    # Stop monitoring
    analyzer.stop_monitoring()
    
    # Cancel monitoring task
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    print(f"✅ Monitoring demo completed")
    
    print(f"\n✅ Active Metrics Analyzer test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())