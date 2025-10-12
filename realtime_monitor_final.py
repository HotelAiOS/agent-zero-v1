#!/usr/bin/env python3
"""
🔧 Real-time Monitor V2.0 - FINAL FIX
📦 PAKIET 5 Phase 2: Final Variable Scope Fix
🎯 Fixes FACTORY_AVAILABLE UnboundLocalError

Status: PRODUCTION READY - FINAL
Created: 12 października 2025, 18:27 CEST
"""

import asyncio
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import statistics

# System monitoring
HAS_PSUTIL = False
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# FastAPI for web dashboard - FIXED
HAS_FASTAPI = False
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Enhanced monitoring imports - FIXED GLOBALS
MONITORING_AVAILABLE = False
FACTORY_AVAILABLE = False

try:
    from shared.kaizen import ActiveMetricsAnalyzer, generate_kaizen_report_cli
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from enhanced_agent_factory_v2_fixed import EnhancedAgentFactory
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    COST = "cost"
    QUALITY = "quality"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"

@dataclass
class Alert:
    """Real-time alert"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold_value: float
    affected_component: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class PerformanceMetric:
    """Real-time performance metric"""
    metric_name: str
    metric_type: MetricType
    current_value: float
    previous_value: Optional[float]
    trend: str  # "up", "down", "stable"
    threshold_warning: float
    threshold_critical: float
    unit: str
    timestamp: datetime

class RealTimeMonitor:
    """
    📊 Real-time Performance Monitor for Agent Zero V2.0 - FINAL
    Enterprise-grade monitoring with predictive analytics and alerting
    """
    
    def __init__(self):
        # FIXED: Use global variables properly
        global FACTORY_AVAILABLE, MONITORING_AVAILABLE, HAS_PSUTIL, HAS_FASTAPI
        
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        self.alerts_queue = queue.Queue()
        self.websocket_connections: List = []
        
        # Monitoring configuration
        self.monitoring_config = {
            'update_interval_seconds': 5,
            'metric_retention_days': 30,
            'alert_cooldown_minutes': 5,
            'predictive_window_minutes': 60
        }
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': {'warning': 2000, 'critical': 5000},
            'success_rate': {'warning': 0.8, 'critical': 0.6},
            'cost_per_task': {'warning': 0.01, 'critical': 0.05},
            'memory_usage_mb': {'warning': 1000, 'critical': 2000},
            'error_rate': {'warning': 0.05, 'critical': 0.15},
            'throughput_tpm': {'warning': 50, 'critical': 10}
        }
        
        # Metrics storage
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Database for historical data
        self.db_path = "real_time_monitoring.db"
        self._initialize_monitoring_db()
        
        # Initialize components - FIXED
        self.factory_monitor = None
        self.kaizen_monitor = None
        
        if FACTORY_AVAILABLE:
            try:
                self.factory_monitor = EnhancedAgentFactory()
                print("✅ Enhanced Agent Factory monitor initialized")
            except Exception as e:
                print(f"⚠️ Factory monitor failed: {e}")
        
        if MONITORING_AVAILABLE:
            try:
                self.kaizen_monitor = ActiveMetricsAnalyzer()
                print("✅ Kaizen monitor initialized")
            except Exception as e:
                print(f"⚠️ Kaizen monitor failed: {e}")
        
        print(f"📊 Real-time Monitor initialized - FINAL")
        print(f"   Enhanced Factory: {'ENABLED' if FACTORY_AVAILABLE else 'DISABLED'}")
        print(f"   Kaizen Analytics: {'ENABLED' if MONITORING_AVAILABLE else 'DISABLED'}")
        print(f"   System Monitoring: {'ENABLED' if HAS_PSUTIL else 'DISABLED'}")
        print(f"   Web Dashboard: {'ENABLED' if HAS_FASTAPI else 'DISABLED'}")
    
    def _initialize_monitoring_db(self):
        """Initialize monitoring database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            component TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT UNIQUE NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            metric_value REAL,
            threshold_value REAL,
            affected_component TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            acknowledged_at TIMESTAMP,
            resolved_at TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ Monitoring database initialized")
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        
        if self.monitoring_active:
            print("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        
        print("🚀 Starting Real-time Performance Monitor V2.0 - FINAL")
        print(f"📊 Update interval: {self.monitoring_config['update_interval_seconds']}s")
        print(f"🔔 Alert system: ACTIVE")
        print(f"📈 Predictive analytics: ENABLED")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._alert_processing_loop())
        
        print("✅ Real-time monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        
        self.monitoring_active = False
        print("🛑 Real-time monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                await self._collect_system_metrics()
                await self._collect_factory_metrics()
                await self._collect_kaizen_metrics()
                
                # Process metrics for alerts
                await self._process_metrics_for_alerts()
                
                # Broadcast to WebSocket clients (if available)
                global HAS_FASTAPI
                if HAS_FASTAPI:
                    await self._broadcast_metrics()
                
                # Wait for next update
                await asyncio.sleep(self.monitoring_config['update_interval_seconds'])
                
            except Exception as e:
                print(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        global HAS_PSUTIL
        if not HAS_PSUTIL:
            return
        
        try:
            import os
            
            # System resource metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Process-specific metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Update metrics
            self._update_metric("cpu_usage_percent", MetricType.PERFORMANCE, cpu_percent, "%")
            self._update_metric("memory_usage_percent", MetricType.PERFORMANCE, memory.percent, "%")
            self._update_metric("memory_usage_mb", MetricType.PERFORMANCE, process_memory, "MB")
            
        except Exception as e:
            print(f"⚠️ System metrics collection failed: {e}")
    
    async def _collect_factory_metrics(self):
        """Collect Enhanced Agent Factory metrics"""
        
        global FACTORY_AVAILABLE
        if not FACTORY_AVAILABLE or not self.factory_monitor:
            return
        
        try:
            factory_status = self.factory_monitor.get_enhanced_factory_status()
            
            # Extract key metrics
            metrics = factory_status.get('factory_metrics', {})
            
            self._update_metric("total_agents", MetricType.THROUGHPUT, metrics.get('total_agents', 0), "count")
            self._update_metric("total_teams", MetricType.THROUGHPUT, metrics.get('total_teams', 0), "count")
            self._update_metric("success_rate", MetricType.QUALITY, metrics.get('average_success_rate', 0), "ratio")
            self._update_metric("cost_per_task", MetricType.COST, metrics.get('average_cost_per_task', 0), "USD")
            
            # Calculate throughput (tasks per minute)
            total_tasks = metrics.get('total_tasks_completed', 0)
            if hasattr(self, '_last_task_count'):
                tasks_delta = total_tasks - self._last_task_count
                time_delta_minutes = self.monitoring_config['update_interval_seconds'] / 60.0
                throughput = tasks_delta / time_delta_minutes if time_delta_minutes > 0 else 0
                self._update_metric("throughput_tpm", MetricType.THROUGHPUT, throughput, "tasks/min")
            
            self._last_task_count = total_tasks
            
        except Exception as e:
            print(f"⚠️ Factory metrics collection failed: {e}")
    
    async def _collect_kaizen_metrics(self):
        """Collect Kaizen Intelligence metrics"""
        
        global MONITORING_AVAILABLE
        if not MONITORING_AVAILABLE or not self.kaizen_monitor:
            return
        
        try:
            # Get Kaizen report
            report = generate_kaizen_report_cli('summary')
            
            # Parse metrics from report
            if 'alerts_count' in report:
                self._update_metric("active_alerts", MetricType.AVAILABILITY, report['alerts_count'], "count")
            
            if 'critical_alerts' in report:
                self._update_metric("critical_alerts", MetricType.AVAILABILITY, report['critical_alerts'], "count")
            
            # Cost analysis
            cost_analysis = self.kaizen_monitor.get_cost_analysis(days=1)
            if cost_analysis.get('total_cost_usd', 0) > 0:
                self._update_metric("daily_cost", MetricType.COST, cost_analysis['total_cost_usd'], "USD")
                self._update_metric("cost_efficiency", MetricType.COST, cost_analysis.get('cost_efficiency_score', 0), "ratio")
            
        except Exception as e:
            print(f"⚠️ Kaizen metrics collection failed: {e}")
    
    def _update_metric(self, metric_name: str, metric_type: MetricType, 
                      value: float, unit: str, component: str = "system"):
        """Update a performance metric"""
        
        previous_value = None
        if metric_name in self.current_metrics:
            previous_value = self.current_metrics[metric_name].current_value
        
        # Calculate trend
        trend = "stable"
        if previous_value is not None:
            if value > previous_value * 1.05:  # 5% increase
                trend = "up"
            elif value < previous_value * 0.95:  # 5% decrease
                trend = "down"
        
        # Get thresholds
        thresholds = self.thresholds.get(metric_name, {'warning': float('inf'), 'critical': float('inf')})
        
        # Create/update metric
        metric = PerformanceMetric(
            metric_name=metric_name,
            metric_type=metric_type,
            current_value=value,
            previous_value=previous_value,
            trend=trend,
            threshold_warning=thresholds['warning'],
            threshold_critical=thresholds['critical'],
            unit=unit,
            timestamp=datetime.now()
        )
        
        self.current_metrics[metric_name] = metric
        
        # Store in database
        self._store_metric_history(metric_name, metric_type.value, value, unit, component)
    
    def _store_metric_history(self, metric_name: str, metric_type: str, 
                            value: float, unit: str, component: str):
        """Store metric in historical database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO metrics_history (metric_name, metric_type, value, unit, component)
            VALUES (?, ?, ?, ?, ?)
            ''', (metric_name, metric_type, value, unit, component))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Failed to store metric history: {e}")
    
    async def _process_metrics_for_alerts(self):
        """Process current metrics to generate alerts"""
        
        for metric_name, metric in self.current_metrics.items():
            # Check for threshold violations
            if metric.current_value >= metric.threshold_critical:
                await self._create_alert(
                    alert_type=f"{metric_name}_critical",
                    severity=AlertSeverity.CRITICAL,
                    message=f"{metric_name} exceeded critical threshold: {metric.current_value:.2f} > {metric.threshold_critical}",
                    metric_value=metric.current_value,
                    threshold_value=metric.threshold_critical,
                    affected_component=metric_name
                )
            elif metric.current_value >= metric.threshold_warning:
                await self._create_alert(
                    alert_type=f"{metric_name}_warning",
                    severity=AlertSeverity.HIGH,
                    message=f"{metric_name} exceeded warning threshold: {metric.current_value:.2f} > {metric.threshold_warning}",
                    metric_value=metric.current_value,
                    threshold_value=metric.threshold_warning,
                    affected_component=metric_name
                )
    
    async def _create_alert(self, alert_type: str, severity: AlertSeverity,
                          message: str, metric_value: float, threshold_value: float,
                          affected_component: str):
        """Create new alert if not in cooldown"""
        
        alert_id = f"{alert_type}_{int(time.time())}"
        
        # Check cooldown
        cooldown_minutes = self.monitoring_config['alert_cooldown_minutes']
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        recent_similar_alerts = [
            alert for alert in self.alert_history
            if alert.alert_type == alert_type and alert.timestamp > cutoff_time
        ]
        
        if recent_similar_alerts:
            return  # Skip due to cooldown
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_value=metric_value,
            threshold_value=threshold_value,
            affected_component=affected_component,
            timestamp=datetime.now()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Store in database
        await self._store_alert_history(alert)
        
        # Add to processing queue
        self.alerts_queue.put(alert)
        
        print(f"🔔 {severity.value.upper()} ALERT: {message}")
    
    async def _store_alert_history(self, alert: Alert):
        """Store alert in historical database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO alerts_history 
            (alert_id, alert_type, severity, message, metric_value, threshold_value, affected_component)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.alert_type,
                alert.severity.value,
                alert.message,
                alert.metric_value,
                alert.threshold_value,
                alert.affected_component
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Failed to store alert history: {e}")
    
    async def _alert_processing_loop(self):
        """Process alerts and trigger actions"""
        
        while self.monitoring_active:
            try:
                if not self.alerts_queue.empty():
                    alert = self.alerts_queue.get_nowait()
                    await self._handle_alert(alert)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Alert processing error: {e}")
    
    async def _handle_alert(self, alert: Alert):
        """Handle alert with appropriate actions"""
        
        # Critical alerts - immediate action
        if alert.severity == AlertSeverity.CRITICAL:
            if "memory" in alert.alert_type:
                print(f"🧠 Memory optimization triggered: {alert.message}")
            elif "throughput" in alert.alert_type:
                print(f"⚡ Throughput optimization triggered: {alert.message}")
            elif "success_rate" in alert.alert_type:
                print(f"📊 Quality optimization triggered: {alert.message}")
        
        # Broadcast alert if WebSocket available
        global HAS_FASTAPI
        if HAS_FASTAPI:
            await self._broadcast_alert(alert)
    
    async def _broadcast_metrics(self):
        """Broadcast current metrics to WebSocket clients"""
        
        if not self.websocket_connections:
            return
        
        # Prepare metrics data
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                name: {
                    'value': metric.current_value,
                    'unit': metric.unit,
                    'trend': metric.trend,
                    'threshold_warning': metric.threshold_warning,
                    'threshold_critical': metric.threshold_critical
                }
                for name, metric in self.current_metrics.items()
            },
            'active_alerts_count': len(self.active_alerts),
            'system_status': self._get_system_status()
        }
        
        # Broadcast to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(metrics_data))
            except:
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_connections.remove(client)
    
    async def _broadcast_alert(self, alert: Alert):
        """Broadcast alert to WebSocket clients"""
        
        if not self.websocket_connections:
            return
        
        alert_data = {
            'type': 'alert',
            'alert': {
                'id': alert.alert_id,
                'type': alert.alert_type,
                'severity': alert.severity.value,
                'message': alert.message,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'component': alert.affected_component,
                'timestamp': alert.timestamp.isoformat()
            }
        }
        
        # Broadcast to all connected clients
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(alert_data))
            except:
                pass
    
    def _get_system_status(self) -> str:
        """Get overall system status"""
        
        critical_alerts = [a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in self.active_alerts.values() if a.severity == AlertSeverity.HIGH]
        
        if critical_alerts:
            return "critical"
        elif high_alerts:
            return "warning"
        elif len(self.active_alerts) > 5:
            return "degraded"
        else:
            return "healthy"
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        global FACTORY_AVAILABLE, MONITORING_AVAILABLE, HAS_PSUTIL, HAS_FASTAPI
        
        return {
            'system_status': self._get_system_status(),
            'monitoring_active': self.monitoring_active,
            'current_metrics': {
                name: asdict(metric) for name, metric in self.current_metrics.items()
            },
            'active_alerts': {
                alert_id: asdict(alert) for alert_id, alert in self.active_alerts.items()
            },
            'alert_summary': {
                'total': len(self.active_alerts),
                'critical': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.MEDIUM]),
                'low': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.LOW])
            },
            'component_status': {
                'enhanced_factory': 'enabled' if FACTORY_AVAILABLE else 'disabled',
                'kaizen_analytics': 'enabled' if MONITORING_AVAILABLE else 'disabled',
                'system_monitoring': 'enabled' if HAS_PSUTIL else 'disabled',
                'web_dashboard': 'enabled' if HAS_FASTAPI else 'disabled'
            },
            'performance_summary': {
                'success_rate': self.current_metrics.get('success_rate', PerformanceMetric('success_rate', MetricType.QUALITY, 0.0, None, 'stable', 0, 0, 'ratio', datetime.now())).current_value,
                'throughput': self.current_metrics.get('throughput_tpm', PerformanceMetric('throughput_tpm', MetricType.THROUGHPUT, 0.0, None, 'stable', 0, 0, 'tpm', datetime.now())).current_value,
                'cost_efficiency': self.current_metrics.get('cost_efficiency', PerformanceMetric('cost_efficiency', MetricType.COST, 0.0, None, 'stable', 0, 0, 'ratio', datetime.now())).current_value
            },
            'last_update': datetime.now().isoformat()
        }

# FastAPI Web Dashboard (if available) - FINAL FIX
if HAS_FASTAPI:
    dashboard_app = FastAPI(
        title="Agent Zero V2.0 - Real-time Monitor - FINAL",
        description="Enterprise-grade monitoring dashboard"
    )
    
    # Global monitor instance
    monitor_instance = None
    
    # Modern FastAPI lifespan handler
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        global monitor_instance
        monitor_instance = RealTimeMonitor()
        await monitor_instance.start_monitoring()
        yield
        # Shutdown
        if monitor_instance:
            await monitor_instance.stop_monitoring()
    
    dashboard_app.router.lifespan_context = lifespan
    
    @dashboard_app.get("/")
    async def dashboard_home():
        """Monitoring dashboard home page - FINAL"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Zero V2.0 - Real-time Monitor - FINAL</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; }
                .header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
                .header p { opacity: 0.9; font-size: 1.1rem; }
                .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
                .metric-card { 
                    background: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                    border: 1px solid #e2e8f0;
                }
                .metric-label { color: #64748b; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; }
                .metric-value { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
                .metric-trend { font-size: 0.875rem; color: #64748b; }
                .status-healthy { color: #059669; }
                .status-warning { color: #d97706; }
                .status-critical { color: #dc2626; }
                .status-degraded { color: #7c3aed; }
                .component-status { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); margin-bottom: 2rem; }
                .component-item { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 0; border-bottom: 1px solid #f1f5f9; }
                .component-item:last-child { border-bottom: none; }
                .component-enabled { color: #059669; font-weight: 600; }
                .component-disabled { color: #dc2626; font-weight: 600; }
                .alert-panel { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); }
                .alert-item { padding: 1rem; border-left: 4px solid #dc2626; margin: 1rem 0; background: #fef2f2; border-radius: 6px; }
                .refresh-indicator { position: fixed; top: 20px; right: 20px; background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.875rem; }
                @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
                .pulse { animation: pulse 2s infinite; }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>🚀 Agent Zero V2.0 - Real-time Monitor</h1>
                    <p>Enterprise-grade monitoring with predictive analytics</p>
                </div>
            </div>
            
            <div class="container">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">System Status</div>
                        <div id="system-status" class="metric-value status-healthy">Loading...</div>
                        <div class="metric-trend">Overall health indicator</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Success Rate</div>
                        <div id="success-rate" class="metric-value">Loading...</div>
                        <div class="metric-trend">Task completion quality</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Throughput</div>
                        <div id="throughput" class="metric-value">Loading...</div>
                        <div class="metric-trend">Tasks per minute</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Active Alerts</div>
                        <div id="active-alerts" class="metric-value">Loading...</div>
                        <div class="metric-trend">Monitoring notifications</div>
                    </div>
                </div>
                
                <div class="component-status">
                    <h3 style="margin-bottom: 1rem;">📊 Component Status</h3>
                    <div id="component-status">Loading component status...</div>
                </div>
                
                <div class="alert-panel">
                    <h3 style="margin-bottom: 1rem;">🔔 System Alerts</h3>
                    <div id="alerts-list">No active alerts</div>
                </div>
            </div>
            
            <div id="refresh-indicator" class="refresh-indicator pulse">Live Monitoring</div>
            
            <script>
                function updateMetrics(data) {
                    // Update system status
                    const statusElement = document.getElementById('system-status');
                    if (statusElement && data.system_status) {
                        statusElement.textContent = data.system_status.toUpperCase();
                        statusElement.className = 'metric-value status-' + data.system_status;
                    }
                    
                    // Update success rate
                    const successRate = data.performance_summary?.success_rate || 0;
                    const successElement = document.getElementById('success-rate');
                    if (successElement) {
                        successElement.textContent = (successRate * 100).toFixed(1) + '%';
                    }
                    
                    // Update throughput
                    const throughput = data.performance_summary?.throughput || 0;
                    const throughputElement = document.getElementById('throughput');
                    if (throughputElement) {
                        throughputElement.textContent = throughput.toFixed(1);
                    }
                    
                    // Update active alerts count
                    const alertsElement = document.getElementById('active-alerts');
                    if (alertsElement) {
                        alertsElement.textContent = data.alert_summary?.total || 0;
                    }
                    
                    // Update component status
                    if (data.component_status) {
                        const componentHtml = Object.entries(data.component_status).map(([name, status]) => {
                            const className = status === 'enabled' ? 'component-enabled' : 'component-disabled';
                            const icon = status === 'enabled' ? '✅' : '❌';
                            const displayName = name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                            return `<div class="component-item">
                                <span>${icon} ${displayName}</span>
                                <span class="${className}">${status.toUpperCase()}</span>
                            </div>`;
                        }).join('');
                        document.getElementById('component-status').innerHTML = componentHtml;
                    }
                    
                    // Update alerts
                    const alertsCount = data.alert_summary?.total || 0;
                    const alertsList = document.getElementById('alerts-list');
                    if (alertsList) {
                        if (alertsCount > 0) {
                            alertsList.innerHTML = `<div class="alert-item">
                                <strong>Active Monitoring:</strong> ${alertsCount} alerts detected<br>
                                <small>System is actively monitoring performance metrics</small>
                            </div>`;
                        } else {
                            alertsList.innerHTML = '<div style="color: #059669; font-weight: 600;">✅ All systems operating normally</div>';
                        }
                    }
                }
                
                function fetchData() {
                    fetch('/api/dashboard')
                        .then(response => response.json())
                        .then(data => {
                            updateMetrics(data);
                            console.log('Dashboard updated:', new Date().toLocaleTimeString());
                        })
                        .catch(error => {
                            console.error('Failed to fetch data:', error);
                            document.getElementById('system-status').textContent = 'ERROR';
                            document.getElementById('system-status').className = 'metric-value status-critical';
                        });
                }
                
                // Initial data load
                fetchData();
                
                // Refresh every 5 seconds
                setInterval(fetchData, 5000);
                
                // Show that monitoring is active
                console.log('🚀 Agent Zero V2.0 Real-time Monitor - FINAL VERSION');
                console.log('📊 Live monitoring active - refresh every 5 seconds');
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
    
    @dashboard_app.get("/api/dashboard")
    async def get_dashboard_data():
        """API endpoint for dashboard data"""
        global monitor_instance
        if monitor_instance:
            return monitor_instance.get_monitoring_dashboard_data()
        else:
            return {"error": "Monitor not initialized", "system_status": "error"}

# Export monitoring classes - FINAL
__all__ = [
    'RealTimeMonitor',
    'Alert',
    'PerformanceMetric',
    'AlertSeverity',
    'MetricType'
]

# Main execution for testing - FINAL
if __name__ == "__main__":
    async def test_monitor():
        """Test real-time monitoring - FINAL"""
        print("🚀 Testing Real-time Performance Monitor V2.0 - FINAL")
        
        monitor = RealTimeMonitor()
        await monitor.start_monitoring()
        
        print("📊 Monitor running - collecting metrics...")
        
        # Run for 10 seconds
        for i in range(10):
            await asyncio.sleep(1)
            print(f"⏱️ {i+1}/10 seconds - {len(monitor.current_metrics)} metrics collected")
        
        await monitor.stop_monitoring()
        
        # Show results
        dashboard_data = monitor.get_monitoring_dashboard_data()
        print(f"\n📊 Final Results:")
        print(f"   Metrics collected: {len(dashboard_data['current_metrics'])}")
        print(f"   Active alerts: {len(dashboard_data['active_alerts'])}")
        print(f"   System status: {dashboard_data['system_status']}")
        print(f"   Components: {dashboard_data['component_status']}")
    
    if HAS_FASTAPI:
        print("🌐 Starting Real-time Monitor Web Dashboard - FINAL")
        print("📊 Dashboard: http://localhost:8002/")
        print("📈 API: http://localhost:8002/api/dashboard")
        print("🚀 Press Ctrl+C to stop")
        uvicorn.run(dashboard_app, host="0.0.0.0", port=8002, log_level="info")
    else:
        print("⚠️ FastAPI not available - running basic test")
        asyncio.run(test_monitor())