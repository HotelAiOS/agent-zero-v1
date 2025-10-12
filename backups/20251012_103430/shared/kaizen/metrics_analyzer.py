#!/usr/bin/env python3
"""
Agent Zero V1 - Active Metrics Analyzer
V2.0 Intelligence Layer - Real-time Performance Monitoring

Author: Developer A (Backend Architect)
Date: 10 października 2025  
Linear Issue: A0-28
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from threading import Thread, Event
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    COST_PER_TASK = "cost_per_task"
    SUCCESS_RATE = "success_rate"
    LATENCY_MS = "latency_ms"
    OVERRIDE_RATE = "override_rate"

@dataclass
class Alert:
    id: str
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str]

class ActiveMetricsAnalyzer:
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = Event()
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        self.thresholds = {
            MetricType.COST_PER_TASK: 0.02,
            MetricType.SUCCESS_RATE: 0.85,
            MetricType.LATENCY_MS: 5000,
            MetricType.OVERRIDE_RATE: 0.20
        }
        
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_active_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    context TEXT
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
                    message TEXT NOT NULL
                )
            ''')
            conn.commit()

    def start_monitoring(self, interval_seconds: int = 60):
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
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Stopped active monitoring")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        self.alert_callbacks.append(callback)

    def analyze_task_completion(self, task_result: Dict[str, Any]):
        timestamp = datetime.now()
        
        cost = task_result.get('cost_usd', 0.0)
        latency = task_result.get('execution_time_ms', 0)
        success = task_result.get('success', True)
        
        self._store_metric(MetricType.COST_PER_TASK, cost, timestamp)
        self._store_metric(MetricType.LATENCY_MS, latency, timestamp)
        
        # Check immediate thresholds
        if cost > 0.10:  # High individual cost
            self._generate_alert(MetricType.COST_PER_TASK, cost, 0.10, timestamp)
        
        if latency > 30000:  # Very high latency
            self._generate_alert(MetricType.LATENCY_MS, latency, 30000, timestamp)

    def _monitoring_loop(self, interval_seconds: int):
        while not self.stop_event.wait(interval_seconds):
            try:
                logger.debug("🔄 Running monitoring cycle")
                
                current_time = datetime.now()
                window_start = current_time - timedelta(minutes=15)
                
                # Calculate windowed metrics
                avg_cost = self._calculate_average_metric(MetricType.COST_PER_TASK, window_start, current_time)
                avg_latency = self._calculate_average_metric(MetricType.LATENCY_MS, window_start, current_time)
                success_rate = self._calculate_success_rate(window_start, current_time)
                
                # Check thresholds
                if avg_cost and avg_cost > self.thresholds[MetricType.COST_PER_TASK]:
                    self._generate_alert(MetricType.COST_PER_TASK, avg_cost, self.thresholds[MetricType.COST_PER_TASK], current_time)
                
                if avg_latency and avg_latency > self.thresholds[MetricType.LATENCY_MS]:
                    self._generate_alert(MetricType.LATENCY_MS, avg_latency, self.thresholds[MetricType.LATENCY_MS], current_time)
                
                if success_rate is not None and success_rate < self.thresholds[MetricType.SUCCESS_RATE]:
                    self._generate_alert(MetricType.SUCCESS_RATE, success_rate, self.thresholds[MetricType.SUCCESS_RATE], current_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _store_metric(self, metric_type: MetricType, value: float, timestamp: datetime):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO v2_active_metrics (timestamp, metric_type, value)
                    VALUES (?, ?, ?)
                ''', (timestamp, metric_type.value, value))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to store metric: {e}")

    def _calculate_average_metric(self, metric_type: MetricType, start_time: datetime, end_time: datetime) -> Optional[float]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT AVG(value) FROM v2_active_metrics
                    WHERE metric_type = ? AND timestamp BETWEEN ? AND ?
                ''', (metric_type.value, start_time, end_time))
                
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else None
        except sqlite3.Error:
            return None

    def _calculate_success_rate(self, start_time: datetime, end_time: datetime) -> Optional[float]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT AVG(
                        CASE WHEN success_level IN ('SUCCESS', 'PARTIAL') 
                        THEN 1.0 ELSE 0.0 END
                    ) FROM v2_success_evaluations
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_time, end_time))
                
                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else None
        except sqlite3.Error:
            return None

    def _generate_alert(self, metric_type: MetricType, current_value: float, threshold_value: float, timestamp: datetime):
        alert_id = f"{metric_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        level = AlertLevel.WARNING
        if current_value > threshold_value * 2:
            level = AlertLevel.CRITICAL
        
        message = self._generate_alert_message(metric_type, current_value, threshold_value, level)
        recommendations = self._generate_recommendations(metric_type)
        
        alert = Alert(
            id=alert_id,
            timestamp=timestamp,
            level=level,
            metric_type=metric_type,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            recommendations=recommendations
        )
        
        self._handle_alert(alert)

    def _generate_alert_message(self, metric_type: MetricType, current_value: float, threshold_value: float, level: AlertLevel) -> str:
        level_emoji = {"WARNING": "⚠️", "CRITICAL": "🚨"}
        emoji = level_emoji.get(level.value, "")
        
        if metric_type == MetricType.COST_PER_TASK:
            return f"{emoji} Cost per task: ${current_value:.4f} exceeds threshold ${threshold_value:.4f}"
        elif metric_type == MetricType.SUCCESS_RATE:
            return f"{emoji} Success rate: {current_value:.1%} below threshold {threshold_value:.1%}"
        elif metric_type == MetricType.LATENCY_MS:
            return f"{emoji} Response latency: {current_value:.0f}ms exceeds threshold {threshold_value:.0f}ms"
        else:
            return f"{emoji} {metric_type.value}: {current_value} vs threshold {threshold_value}"

    def _generate_recommendations(self, metric_type: MetricType) -> List[str]:
        recommendations = {
            MetricType.COST_PER_TASK: [
                "💡 Consider switching to local Ollama models",
                "🔄 Batch similar requests to reduce overhead"
            ],
            MetricType.SUCCESS_RATE: [
                "🎯 Review failed tasks to identify patterns",
                "🔧 Improve prompt engineering and context"
            ],
            MetricType.LATENCY_MS: [
                "⚡ Switch to faster models (e.g., llama3.2:3b)",
                "🔀 Implement asynchronous processing"
            ]
        }
        
        return recommendations.get(metric_type, [])

    def _handle_alert(self, alert: Alert):
        # Store alert
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO v2_alerts (
                        id, timestamp, level, metric_type, current_value, 
                        threshold_value, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.timestamp, alert.level.value,
                    alert.metric_type.value, alert.current_value,
                    alert.threshold_value, alert.message
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to store alert: {e}")
        
        # Log alert
        logger.warning(alert.message)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_current_metrics(self) -> Dict[str, float]:
        metrics = {}
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=15)
        
        for metric_type in [MetricType.COST_PER_TASK, MetricType.LATENCY_MS]:
            value = self._calculate_average_metric(metric_type, window_start, current_time)
            if value is not None:
                metrics[metric_type.value] = value
        
        success_rate = self._calculate_success_rate(window_start, current_time)
        if success_rate is not None:
            metrics[MetricType.SUCCESS_RATE.value] = success_rate
        
        return metrics

def console_alert_handler(alert: Alert):
    print(f"[{alert.level.value}] {alert.message}")
    for rec in alert.recommendations:
        print(f"  → {rec}")

if __name__ == "__main__":
    analyzer = ActiveMetricsAnalyzer()
    analyzer.add_alert_callback(console_alert_handler)
    
    # Test
    task_result = {'cost_usd': 0.055, 'execution_time_ms': 12000, 'success': False}
    analyzer.analyze_task_completion(task_result)
    
    print("\nCurrent Metrics:")
    metrics = analyzer.get_current_metrics()
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
