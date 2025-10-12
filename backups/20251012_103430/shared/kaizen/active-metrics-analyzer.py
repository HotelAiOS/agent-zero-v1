#!/usr/bin/env python3
"""
Agent Zero V1 - Active Metrics Analyzer
V2.0 Intelligence Layer - Real-time Performance Monitoring

This module provides real-time metrics monitoring, threshold alerting,
and automated Kaizen reporting for Agent Zero V1.

Author: Developer A (Backend Architect)
Date: 10 października 2025  
Linear Issue: A0-28
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from threading import Thread, Event
import logging
from enum import Enum
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    """Types of metrics to monitor"""
    COST_PER_TASK = "cost_per_task"
    SUCCESS_RATE = "success_rate"
    LATENCY_MS = "latency_ms"
    OVERRIDE_RATE = "override_rate"
    THROUGHPUT = "throughput"

@dataclass
class MetricThreshold:
    """Threshold configuration for metrics"""
    metric_type: MetricType
    warning_value: float
    critical_value: float
    operator: str  # ">", "<", ">=", "<="
    window_minutes: int = 15  # Time window for calculation

@dataclass
class Alert:
    """Alert generated when threshold exceeded"""
    id: str
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str]
    context: Dict[str, Any]

@dataclass
class KaizenReport:
    """Daily Kaizen improvement report"""
    date: datetime
    summary: Dict[str, Any]
    top_performers: List[Dict[str, Any]]
    improvement_opportunities: List[Dict[str, Any]]
    cost_analysis: Dict[str, Any]
    success_trends: Dict[str, Any]
    recommendations: List[str]
    action_items: List[str]

class ActiveMetricsAnalyzer:
    """
    Real-time metrics analyzer for Agent Zero V1 V2.0 Intelligence Layer
    
    Monitors key performance indicators:
    - Cost per task (threshold: $0.02)
    - Success rate (minimum: 85%)
    - Latency (maximum: 5 seconds)
    - Human override rate (maximum: 20%)
    
    Provides automated alerting and daily Kaizen reports.
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = Event()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Default thresholds based on Agent Zero V1 requirements
        self.thresholds = {
            MetricType.COST_PER_TASK: MetricThreshold(
                metric_type=MetricType.COST_PER_TASK,
                warning_value=0.02,
                critical_value=0.05,
                operator=">=",
                window_minutes=15
            ),
            MetricType.SUCCESS_RATE: MetricThreshold(
                metric_type=MetricType.SUCCESS_RATE,
                warning_value=0.85,
                critical_value=0.70,
                operator="<=",
                window_minutes=30
            ),
            MetricType.LATENCY_MS: MetricThreshold(
                metric_type=MetricType.LATENCY_MS,
                warning_value=5000,
                critical_value=10000,
                operator=">=",
                window_minutes=15
            ),
            MetricType.OVERRIDE_RATE: MetricThreshold(
                metric_type=MetricType.OVERRIDE_RATE,
                warning_value=0.20,
                critical_value=0.35,
                operator=">=",
                window_minutes=60
            )
        }
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database tables for metrics tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_active_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    context TEXT,
                    window_start DATETIME,
                    window_end DATETIME
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    message TEXT NOT NULL,
                    recommendations TEXT,
                    context TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME DEFAULT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_kaizen_reports (
                    date DATE PRIMARY KEY,
                    report_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    def start_monitoring(self, interval_seconds: int = 60):
        """Start real-time monitoring in background thread"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitoring_thread = Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"✅ Started active monitoring (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Stopped active monitoring")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)

    def analyze_task_completion(self, task_result: Dict[str, Any]):
        """Analyze completed task and update metrics"""
        timestamp = datetime.now()
        
        # Extract metrics from task result
        cost = task_result.get('cost_usd', 0.0)
        latency = task_result.get('execution_time_ms', 0)
        success = task_result.get('success', True)
        human_override = task_result.get('human_override', False)
        
        # Store individual metrics
        self._store_metric(MetricType.COST_PER_TASK, cost, timestamp, task_result)
        self._store_metric(MetricType.LATENCY_MS, latency, timestamp, task_result)
        
        # Calculate and store aggregate metrics
        self._update_success_rate(success, timestamp)
        self._update_override_rate(human_override, timestamp)
        
        # Check thresholds (immediate)
        self._check_immediate_thresholds(task_result, timestamp)

    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop running in background"""
        while not self.stop_event.wait(interval_seconds):
            try:
                logger.debug("🔄 Running monitoring cycle")
                
                # Calculate windowed metrics
                current_time = datetime.now()
                
                for metric_type, threshold in self.thresholds.items():
                    window_start = current_time - timedelta(minutes=threshold.window_minutes)
                    
                    metric_value = self._calculate_windowed_metric(
                        metric_type, window_start, current_time
                    )
                    
                    if metric_value is not None:
                        # Store calculated metric
                        context = {
                            'window_start': window_start.isoformat(),
                            'window_end': current_time.isoformat(),
                            'window_minutes': threshold.window_minutes
                        }
                        
                        self._store_metric(metric_type, metric_value, current_time, context)
                        
                        # Check threshold
                        alert = self._check_threshold(metric_type, metric_value, threshold, current_time)
                        if alert:
                            self._handle_alert(alert)
                
                # Generate daily Kaizen report if needed
                self._check_daily_kaizen_report()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _store_metric(self, metric_type: MetricType, value: float, timestamp: datetime, context: Any = None):
        """Store metric value in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO v2_active_metrics (timestamp, metric_type, value, context)
                    VALUES (?, ?, ?, ?)
                ''', (
                    timestamp,
                    metric_type.value,
                    value,
                    json.dumps(context) if context else None
                ))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store metric {metric_type.value}: {e}")

    def _calculate_windowed_metric(self, metric_type: MetricType, window_start: datetime, window_end: datetime) -> Optional[float]:
        """Calculate metric value over time window"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                
                if metric_type == MetricType.COST_PER_TASK:
                    cursor = conn.execute('''
                        SELECT AVG(cost_usd) FROM v2_success_evaluations
                        WHERE timestamp BETWEEN ? AND ?
                    ''', (window_start, window_end))
                    
                elif metric_type == MetricType.SUCCESS_RATE:
                    cursor = conn.execute('''
                        SELECT AVG(
                            CASE WHEN success_level IN ('SUCCESS', 'PARTIAL') 
                            THEN 1.0 ELSE 0.0 END
                        ) FROM v2_success_evaluations
                        WHERE timestamp BETWEEN ? AND ?
                    ''', (window_start, window_end))
                    
                elif metric_type == MetricType.LATENCY_MS:
                    cursor = conn.execute('''
                        SELECT AVG(execution_time_ms) FROM v2_success_evaluations
                        WHERE timestamp BETWEEN ? AND ?
                    ''', (window_start, window_end))
                    
                elif metric_type == MetricType.OVERRIDE_RATE:
                    # Calculate from feedback data
                    cursor = conn.execute('''
                        SELECT AVG(
                            CASE WHEN human_feedback IS NOT NULL AND human_feedback <= 2
                            THEN 1.0 ELSE 0.0 END
                        ) FROM v2_model_decisions
                        WHERE timestamp BETWEEN ? AND ?
                    ''', (window_start, window_end))
                
                else:
                    return None
                
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else None
                
        except sqlite3.Error as e:
            logger.error(f"Failed to calculate windowed metric {metric_type.value}: {e}")
            return None

    def _check_threshold(self, metric_type: MetricType, value: float, threshold: MetricThreshold, timestamp: datetime) -> Optional[Alert]:
        """Check if metric value exceeds threshold"""
        
        def check_operator(val: float, threshold_val: float, op: str) -> bool:
            if op == ">=":
                return val >= threshold_val
            elif op == "<=":
                return val <= threshold_val
            elif op == ">":
                return val > threshold_val
            elif op == "<":
                return val < threshold_val
            return False
        
        # Determine alert level
        alert_level = None
        threshold_value = None
        
        if check_operator(value, threshold.critical_value, threshold.operator):
            alert_level = AlertLevel.CRITICAL
            threshold_value = threshold.critical_value
        elif check_operator(value, threshold.warning_value, threshold.operator):
            alert_level = AlertLevel.WARNING
            threshold_value = threshold.warning_value
        
        if alert_level:
            alert_id = f"{metric_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            return Alert(
                id=alert_id,
                timestamp=timestamp,
                level=alert_level,
                metric_type=metric_type,
                current_value=value,
                threshold_value=threshold_value,
                message=self._generate_alert_message(metric_type, value, threshold_value, alert_level),
                recommendations=self._generate_alert_recommendations(metric_type, alert_level),
                context={'window_minutes': threshold.window_minutes}
            )
        
        return None

    def _generate_alert_message(self, metric_type: MetricType, current_value: float, threshold_value: float, level: AlertLevel) -> str:
        """Generate human-readable alert message"""
        
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨"
        }
        
        metric_names = {
            MetricType.COST_PER_TASK: "Cost per task",
            MetricType.SUCCESS_RATE: "Success rate",
            MetricType.LATENCY_MS: "Response latency",
            MetricType.OVERRIDE_RATE: "Human override rate"
        }
        
        emoji = level_emoji.get(level, "")
        metric_name = metric_names.get(metric_type, metric_type.value)
        
        if metric_type == MetricType.COST_PER_TASK:
            return f"{emoji} {metric_name}: ${current_value:.4f} exceeds threshold ${threshold_value:.4f}"
        elif metric_type == MetricType.SUCCESS_RATE:
            return f"{emoji} {metric_name}: {current_value:.1%} below threshold {threshold_value:.1%}"
        elif metric_type == MetricType.LATENCY_MS:
            return f"{emoji} {metric_name}: {current_value:.0f}ms exceeds threshold {threshold_value:.0f}ms"
        elif metric_type == MetricType.OVERRIDE_RATE:
            return f"{emoji} {metric_name}: {current_value:.1%} exceeds threshold {threshold_value:.1%}"
        else:
            return f"{emoji} {metric_name}: {current_value} vs threshold {threshold_value}"

    def _generate_alert_recommendations(self, metric_type: MetricType, level: AlertLevel) -> List[str]:
        """Generate actionable recommendations for alerts"""
        
        recommendations = {
            MetricType.COST_PER_TASK: [
                "💡 Consider switching to local Ollama models for routine tasks",
                "🔄 Batch similar requests to reduce overhead",
                "⚡ Use lighter models (e.g., gpt-4o-mini instead of gpt-4o)",
                "📊 Review task complexity - break down large tasks"
            ],
            MetricType.SUCCESS_RATE: [
                "🎯 Review failed tasks to identify patterns",
                "🔧 Improve prompt engineering and context",
                "🤖 Consider switching to more capable models",
                "📋 Add validation steps and error handling"
            ],
            MetricType.LATENCY_MS: [
                "⚡ Switch to faster models (e.g., llama3.2:3b)",
                "🔀 Implement asynchronous processing",
                "🗂️ Add caching for repeated operations",
                "✂️ Break large tasks into smaller chunks"
            ],
            MetricType.OVERRIDE_RATE: [
                "🎯 Analyze override patterns to improve AI decisions",
                "📚 Enhance training data and feedback loops",
                "🤝 Provide better AI explanation and transparency",
                "⚙️ Adjust decision criteria weights"
            ]
        }
        
        base_recs = recommendations.get(metric_type, [])
        
        if level == AlertLevel.CRITICAL:
            base_recs.insert(0, "🚨 CRITICAL: Immediate intervention required")
        
        return base_recs

    def _handle_alert(self, alert: Alert):
        """Handle generated alert"""
        # Store alert in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO v2_alerts (
                        id, timestamp, level, metric_type, current_value, 
                        threshold_value, message, recommendations, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.timestamp,
                    alert.level.value,
                    alert.metric_type.value,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    json.dumps(alert.recommendations),
                    json.dumps(alert.context)
                ))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store alert: {e}")
        
        # Log alert
        log_level = logging.WARNING if alert.level == AlertLevel.WARNING else logging.CRITICAL
        logger.log(log_level, alert.message)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _check_immediate_thresholds(self, task_result: Dict[str, Any], timestamp: datetime):
        """Check for immediate threshold violations"""
        
        # High individual task cost
        cost = task_result.get('cost_usd', 0.0)
        if cost > 0.10:  # $0.10 per individual task
            alert = Alert(
                id=f"high_cost_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                level=AlertLevel.WARNING,
                metric_type=MetricType.COST_PER_TASK,
                current_value=cost,
                threshold_value=0.10,
                message=f"⚠️ High individual task cost: ${cost:.4f}",
                recommendations=[
                    "💡 Consider using a more cost-effective model",
                    "📏 Review task complexity and size"
                ],
                context=task_result
            )
            self._handle_alert(alert)
        
        # Extremely long latency
        latency = task_result.get('execution_time_ms', 0)
        if latency > 30000:  # 30 seconds
            alert = Alert(
                id=f"high_latency_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                level=AlertLevel.WARNING,
                metric_type=MetricType.LATENCY_MS,
                current_value=latency,
                threshold_value=30000,
                message=f"⚠️ Very high task latency: {latency:.0f}ms",
                recommendations=[
                    "⚡ Switch to faster model",
                    "✂️ Break task into smaller parts"
                ],
                context=task_result
            )
            self._handle_alert(alert)

    def _check_daily_kaizen_report(self):
        """Check if daily Kaizen report needs to be generated"""
        today = datetime.now().date()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM v2_kaizen_reports WHERE date = ?',
                    (today,)
                )
                
                if cursor.fetchone()[0] == 0:
                    # Generate today's report
                    logger.info("📊 Generating daily Kaizen report")
                    report = self.generate_kaizen_report(today)
                    self._store_kaizen_report(report)
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to check Kaizen report: {e}")

    def generate_kaizen_report(self, date: datetime.date = None) -> KaizenReport:
        """Generate comprehensive Kaizen improvement report"""
        if date is None:
            date = datetime.now().date()
        
        # Calculate metrics for the day
        day_start = datetime.combine(date, datetime.min.time())
        day_end = day_start + timedelta(days=1)
        
        report_data = self._gather_kaizen_data(day_start, day_end)
        
        return KaizenReport(
            date=datetime.combine(date, datetime.min.time()),
            summary=report_data['summary'],
            top_performers=report_data['top_performers'],
            improvement_opportunities=report_data['improvements'],
            cost_analysis=report_data['cost_analysis'],
            success_trends=report_data['trends'],
            recommendations=report_data['recommendations'],
            action_items=report_data['action_items']
        )

    def _gather_kaizen_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Gather data for Kaizen report"""
        data = {
            'summary': {},
            'top_performers': [],
            'improvements': [],
            'cost_analysis': {},
            'trends': {},
            'recommendations': [],
            'action_items': []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Daily summary
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_tasks,
                        AVG(overall_score) as avg_score,
                        SUM(cost_usd) as total_cost,
                        AVG(execution_time_ms) as avg_latency,
                        COUNT(CASE WHEN success_level = 'SUCCESS' THEN 1 END) as successes
                    FROM v2_success_evaluations
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_time, end_time))
                
                row = cursor.fetchone()
                if row and row[0] > 0:
                    total_tasks, avg_score, total_cost, avg_latency, successes = row
                    success_rate = successes / total_tasks if total_tasks > 0 else 0
                    
                    data['summary'] = {
                        'total_tasks': total_tasks,
                        'success_rate': round(success_rate, 3),
                        'avg_score': round(avg_score or 0, 3),
                        'total_cost': round(total_cost or 0, 4),
                        'avg_latency_ms': round(avg_latency or 0, 0)
                    }
                
                # Top performers
                cursor = conn.execute('''
                    SELECT 
                        model_used,
                        COUNT(*) as task_count,
                        AVG(overall_score) as avg_score,
                        AVG(cost_usd) as avg_cost
                    FROM v2_success_evaluations
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY model_used
                    HAVING COUNT(*) >= 2
                    ORDER BY AVG(overall_score) DESC
                    LIMIT 5
                ''', (start_time, end_time))
                
                data['top_performers'] = [
                    {
                        'model': row[0],
                        'task_count': row[1],
                        'avg_score': round(row[2], 3),
                        'avg_cost': round(row[3], 4)
                    }
                    for row in cursor.fetchall()
                ]
                
                # Generate recommendations
                data['recommendations'] = self._generate_kaizen_recommendations(data['summary'])
                data['action_items'] = self._generate_action_items(data['summary'])
                
        except sqlite3.Error as e:
            logger.error(f"Failed to gather Kaizen data: {e}")
        
        return data

    def _generate_kaizen_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate Kaizen improvement recommendations"""
        recommendations = []
        
        if summary.get('success_rate', 0) < 0.85:
            recommendations.append("🎯 Focus on improving success rate - review failed tasks")
        
        if summary.get('total_cost', 0) > 1.0:  # $1 per day threshold
            recommendations.append("💰 Optimize costs - consider more efficient models")
        
        if summary.get('avg_latency_ms', 0) > 8000:  # 8 second average
            recommendations.append("⚡ Reduce latency - switch to faster models for routine tasks")
        
        if not recommendations:
            recommendations.append("✅ System performing well - maintain current patterns")
        
        return recommendations

    def _generate_action_items(self, summary: Dict[str, Any]) -> List[str]:
        """Generate specific action items"""
        actions = []
        
        if summary.get('total_tasks', 0) < 10:
            actions.append("📈 Increase system usage to gather more performance data")
        
        actions.append("📊 Review daily metrics and adjust thresholds if needed")
        actions.append("🔍 Analyze pattern discovery for optimization opportunities")
        
        return actions

    def _store_kaizen_report(self, report: KaizenReport):
        """Store Kaizen report in database"""
        try:
            report_data = {
                'date': report.date.isoformat(),
                'summary': report.summary,
                'top_performers': report.top_performers,
                'improvement_opportunities': report.improvement_opportunities,
                'cost_analysis': report.cost_analysis,
                'success_trends': report.success_trends,
                'recommendations': report.recommendations,
                'action_items': report.action_items
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO v2_kaizen_reports (date, report_data)
                    VALUES (?, ?)
                ''', (report.date.date(), json.dumps(report_data)))
                conn.commit()
                
            logger.info(f"✅ Stored Kaizen report for {report.date.date()}")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store Kaizen report: {e}")

    def _update_success_rate(self, success: bool, timestamp: datetime):
        """Update rolling success rate metric"""
        success_value = 1.0 if success else 0.0
        self._store_metric(MetricType.SUCCESS_RATE, success_value, timestamp, {'individual_task': True})

    def _update_override_rate(self, override: bool, timestamp: datetime):
        """Update human override rate metric"""
        override_value = 1.0 if override else 0.0
        self._store_metric(MetricType.OVERRIDE_RATE, override_value, timestamp, {'individual_task': True})

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        metrics = {}
        current_time = datetime.now()
        
        for metric_type, threshold in self.thresholds.items():
            window_start = current_time - timedelta(minutes=threshold.window_minutes)
            value = self._calculate_windowed_metric(metric_type, window_start, current_time)
            
            if value is not None:
                metrics[metric_type.value] = value
        
        return metrics

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        alerts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, timestamp, level, metric_type, current_value,
                           threshold_value, message, recommendations
                    FROM v2_alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                for row in cursor.fetchall():
                    alerts.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'level': row[2],
                        'metric_type': row[3],
                        'current_value': row[4],
                        'threshold_value': row[5],
                        'message': row[6],
                        'recommendations': json.loads(row[7]) if row[7] else []
                    })
        
        except sqlite3.Error as e:
            logger.error(f"Failed to get recent alerts: {e}")
        
        return alerts

def console_alert_handler(alert: Alert):
    """Example alert handler that prints to console"""
    level_colors = {
        AlertLevel.INFO: "cyan",
        AlertLevel.WARNING: "yellow", 
        AlertLevel.CRITICAL: "red"
    }
    
    print(f"[{alert.level.value}] {alert.message}")
    for rec in alert.recommendations:
        print(f"  → {rec}")

# Example usage and testing
def main():
    """Test the ActiveMetricsAnalyzer"""
    print("🚀 Testing Agent Zero V1 - ActiveMetricsAnalyzer")
    
    analyzer = ActiveMetricsAnalyzer()
    
    # Add console alert handler
    analyzer.add_alert_callback(console_alert_handler)
    
    # Simulate some task completions
    print("\n📊 Simulating task completions...")
    
    task_results = [
        {'cost_usd': 0.015, 'execution_time_ms': 3500, 'success': True, 'human_override': False},
        {'cost_usd': 0.055, 'execution_time_ms': 2000, 'success': True, 'human_override': False},  # High cost
        {'cost_usd': 0.008, 'execution_time_ms': 12000, 'success': False, 'human_override': True},  # High latency, failure
        {'cost_usd': 0.002, 'execution_time_ms': 1500, 'success': True, 'human_override': False},
    ]
    
    for i, task in enumerate(task_results):
        print(f"Processing task {i+1}...")
        analyzer.analyze_task_completion(task)
        time.sleep(0.5)  # Small delay
    
    # Get current metrics
    print("\n📈 Current Metrics:")
    current_metrics = analyzer.get_current_metrics()
    for metric, value in current_metrics.items():
        print(f"  {metric}: {value}")
    
    # Get recent alerts
    print("\n🚨 Recent Alerts:")
    recent_alerts = analyzer.get_recent_alerts(5)
    for alert in recent_alerts:
        print(f"  [{alert['level']}] {alert['message']}")
    
    # Generate Kaizen report
    print("\n📊 Generating Kaizen Report...")
    report = analyzer.generate_kaizen_report()
    print(f"Report for {report.date.date()}:")
    print(f"  Summary: {report.summary}")
    print(f"  Top Performers: {len(report.top_performers)}")
    print(f"  Recommendations: {len(report.recommendations)}")

if __name__ == "__main__":
    main()