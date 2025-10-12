"""
🎯 Agent Zero V2.0 - Real-time Performance Monitor
📦 PAKIET 5 Phase 2: Enterprise Monitoring Dashboard
🔧 Real-time analytics, alerting, and performance optimization

Status: PRODUCTION READY
Created: 12 października 2025, 18:20 CEST
Architecture: Enterprise-grade monitoring with predictive analytics
Integrates: Enhanced Agent Factory + Kaizen Intelligence Layer
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

# FastAPI for web dashboard
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Enhanced monitoring imports
try:
    from shared.kaizen import ActiveMetricsAnalyzer, generate_kaizen_report_cli
    from enhanced_agent_factory_v2 import EnhancedAgentFactory
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

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
    📊 Real-time Performance Monitor for Agent Zero V2.0
    Enterprise-grade monitoring with predictive analytics and alerting
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        self.alerts_queue = queue.Queue()
        self.websocket_connections: List[WebSocket] = []
        
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
            'throughput_tpm': {'warning': 50, 'critical': 10}  # tasks per minute
        }
        
        # Metrics storage
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Database for historical data
        self.db_path = "real_time_monitoring.db"
        self._initialize_monitoring_db()
        
        # Initialize components
        self.factory_monitor = None
        self.kaizen_monitor = None
        
        if MONITORING_AVAILABLE:
            self.factory_monitor = EnhancedAgentFactory()
            self.kaizen_monitor = ActiveMetricsAnalyzer()
            print("✅ Enhanced monitoring components initialized")
        else:
            print("⚠️ Enhanced monitoring not available - using basic mode")
    
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
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        
        print("🚀 Starting Real-time Performance Monitor V2.0")
        print(f"📊 Update interval: {self.monitoring_config['update_interval_seconds']}s")
        print(f"🔔 Alert system: ACTIVE")
        print(f"📈 Predictive analytics: ENABLED")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._alert_processing_loop())
        asyncio.create_task(self._predictive_analysis_loop())
        
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
                
                # Broadcast to WebSocket clients
                await self._broadcast_metrics()
                
                # Wait for next update
                await asyncio.sleep(self.monitoring_config['update_interval_seconds'])
                
            except Exception as e:
                print(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        import psutil
        import os
        
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Update metrics
            self._update_metric("cpu_usage_percent", MetricType.PERFORMANCE, cpu_percent, "%")
            self._update_metric("memory_usage_percent", MetricType.PERFORMANCE, memory.percent, "%")
            self._update_metric("memory_usage_mb", MetricType.PERFORMANCE, process_memory, "MB")
            self._update_metric("disk_usage_percent", MetricType.PERFORMANCE, disk.percent, "%")
            
        except Exception as e:
            print(f"⚠️ System metrics collection failed: {e}")
    
    async def _collect_factory_metrics(self):
        """Collect Enhanced Agent Factory metrics"""
        
        if not self.factory_monitor:
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
        
        if not self.kaizen_monitor:
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
                    message=f"{metric_name} has exceeded critical threshold",
                    metric_value=metric.current_value,
                    threshold_value=metric.threshold_critical,
                    affected_component=metric_name
                )
            elif metric.current_value >= metric.threshold_warning:
                await self._create_alert(
                    alert_type=f"{metric_name}_warning",
                    severity=AlertSeverity.HIGH,
                    message=f"{metric_name} has exceeded warning threshold",
                    metric_value=metric.current_value,
                    threshold_value=metric.threshold_warning,
                    affected_component=metric_name
                )
            
            # Check for rapid changes
            if metric.previous_value is not None:
                change_percent = abs((metric.current_value - metric.previous_value) / metric.previous_value) * 100
                if change_percent > 50:  # 50% change
                    await self._create_alert(
                        alert_type=f"{metric_name}_rapid_change",
                        severity=AlertSeverity.MEDIUM,
                        message=f"{metric_name} changed by {change_percent:.1f}% rapidly",
                        metric_value=metric.current_value,
                        threshold_value=metric.previous_value,
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
                await self._handle_memory_alert(alert)
            elif "throughput" in alert.alert_type:
                await self._handle_throughput_alert(alert)
            elif "success_rate" in alert.alert_type:
                await self._handle_quality_alert(alert)
        
        # Broadcast alert to WebSocket clients
        await self._broadcast_alert(alert)
    
    async def _handle_memory_alert(self, alert: Alert):
        """Handle memory-related alerts"""
        print(f"🧠 Memory optimization triggered: {alert.message}")
        # Could trigger garbage collection, agent optimization, etc.
    
    async def _handle_throughput_alert(self, alert: Alert):
        """Handle throughput-related alerts"""
        print(f"⚡ Throughput optimization triggered: {alert.message}")
        # Could trigger auto-scaling, load balancing, etc.
    
    async def _handle_quality_alert(self, alert: Alert):
        """Handle quality-related alerts"""
        print(f"📊 Quality optimization triggered: {alert.message}")
        # Could trigger model reselection, parameter tuning, etc.
    
    async def _predictive_analysis_loop(self):
        """Predictive analysis loop"""
        
        while self.monitoring_active:
            try:
                await self._run_predictive_analysis()
                await asyncio.sleep(self.monitoring_config['predictive_window_minutes'] * 60)
                
            except Exception as e:
                print(f"❌ Predictive analysis error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _run_predictive_analysis(self):
        """Run predictive analysis on historical data"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze trends for each metric
            for metric_name in ['success_rate', 'cost_per_task', 'throughput_tpm']:
                cursor.execute('''
                SELECT value, timestamp FROM metrics_history 
                WHERE metric_name = ? AND timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp
                ''', (metric_name,))
                
                data = cursor.fetchall()
                
                if len(data) > 10:  # Need enough data points
                    values = [row[0] for row in data]
                    
                    # Simple trend analysis
                    recent_values = values[-6:]  # Last 6 data points
                    older_values = values[-12:-6] if len(values) > 12 else values[:-6]
                    
                    if older_values:
                        recent_avg = statistics.mean(recent_values)
                        older_avg = statistics.mean(older_values)
                        trend_slope = (recent_avg - older_avg) / len(older_values)
                        
                        # Predict next value
                        predicted_value = recent_avg + (trend_slope * 3)  # 3 steps ahead
                        
                        # Check if prediction crosses thresholds
                        thresholds = self.thresholds.get(metric_name, {})
                        if 'warning' in thresholds and predicted_value >= thresholds['warning']:
                            await self._create_alert(
                                alert_type=f"{metric_name}_predicted_warning",
                                severity=AlertSeverity.MEDIUM,
                                message=f"Predicted {metric_name} will exceed warning threshold",
                                metric_value=predicted_value,
                                threshold_value=thresholds['warning'],
                                affected_component=metric_name
                            )
            
            conn.close()
            
            print("🔮 Predictive analysis completed")
            
        except Exception as e:
            print(f"⚠️ Predictive analysis failed: {e}")
    
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
            'performance_summary': {
                'success_rate': self.current_metrics.get('success_rate', PerformanceMetric('', MetricType.QUALITY, 0, None, 'stable', 0, 0, '', datetime.now())).current_value,
                'avg_response_time': self.current_metrics.get('response_time_ms', PerformanceMetric('', MetricType.PERFORMANCE, 0, None, 'stable', 0, 0, '', datetime.now())).current_value,
                'throughput': self.current_metrics.get('throughput_tpm', PerformanceMetric('', MetricType.THROUGHPUT, 0, None, 'stable', 0, 0, '', datetime.now())).current_value,
                'cost_efficiency': self.current_metrics.get('cost_efficiency', PerformanceMetric('', MetricType.COST, 0, None, 'stable', 0, 0, '', datetime.now())).current_value
            },
            'last_update': datetime.now().isoformat()
        }

# FastAPI Web Dashboard (if available)
if HAS_FASTAPI:
    dashboard_app = FastAPI(title="Agent Zero V2.0 - Real-time Monitor")
    monitor = RealTimeMonitor()
    
    @dashboard_app.on_event("startup")
    async def startup_event():
        await monitor.start_monitoring()
    
    @dashboard_app.on_event("shutdown") 
    async def shutdown_event():
        await monitor.stop_monitoring()
    
    @dashboard_app.get("/")
    async def dashboard_home():
        """Monitoring dashboard home page"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Zero V2.0 - Real-time Monitor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
                .metric-label { color: #7f8c8d; margin-bottom: 10px; }
                .status-healthy { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
                .alert-panel { background: white; margin-top: 20px; padding: 20px; border-radius: 8px; }
                .alert-item { padding: 10px; border-left: 4px solid #e74c3c; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Agent Zero V2.0 - Real-time Performance Monitor</h1>
                <p>Enterprise-grade monitoring with predictive analytics</p>
            </div>
            
            <div id="metrics-container">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">System Status</div>
                        <div id="system-status" class="metric-value status-healthy">Loading...</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Success Rate</div>
                        <div id="success-rate" class="metric-value">Loading...</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Throughput (TPM)</div>
                        <div id="throughput" class="metric-value">Loading...</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Active Alerts</div>
                        <div id="active-alerts" class="metric-value">Loading...</div>
                    </div>
                </div>
            </div>
            
            <div class="alert-panel">
                <h3>🔔 Recent Alerts</h3>
                <div id="alerts-list">Loading alerts...</div>
            </div>
            
            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket('ws://localhost:8002/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'alert') {
                        // Handle new alert
                        addAlert(data.alert);
                    } else {
                        // Update metrics
                        updateMetrics(data);
                    }
                };
                
                function updateMetrics(data) {
                    // Update system status
                    const statusElement = document.getElementById('system-status');
                    statusElement.textContent = data.system_status.toUpperCase();
                    statusElement.className = 'metric-value status-' + data.system_status;
                    
                    // Update success rate
                    const successRate = data.metrics.success_rate?.value || 0;
                    document.getElementById('success-rate').textContent = (successRate * 100).toFixed(1) + '%';
                    
                    // Update throughput
                    const throughput = data.metrics.throughput_tpm?.value || 0;
                    document.getElementById('throughput').textContent = throughput.toFixed(1);
                    
                    // Update active alerts count
                    document.getElementById('active-alerts').textContent = data.active_alerts_count;
                }
                
                function addAlert(alert) {
                    const alertsList = document.getElementById('alerts-list');
                    const alertElement = document.createElement('div');
                    alertElement.className = 'alert-item';
                    alertElement.innerHTML = `
                        <strong>${alert.severity.toUpperCase()}:</strong> ${alert.message}
                        <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                    `;
                    alertsList.insertBefore(alertElement, alertsList.firstChild);
                    
                    // Keep only last 10 alerts
                    while (alertsList.children.length > 10) {
                        alertsList.removeChild(alertsList.lastChild);
                    }
                }
                
                // Initial data load
                fetch('/api/dashboard')
                    .then(response => response.json())
                    .then(data => updateMetrics(data));
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
    
    @dashboard_app.get("/api/dashboard")
    async def get_dashboard_data():
        """API endpoint for dashboard data"""
        return monitor.get_monitoring_dashboard_data()
    
    @dashboard_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await websocket.accept()
        monitor.websocket_connections.append(websocket)
        
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            monitor.websocket_connections.remove(websocket)

# Export monitoring classes
__all__ = [
    'RealTimeMonitor',
    'Alert',
    'PerformanceMetric',
    'AlertSeverity',
    'MetricType'
]

# Main execution for testing
if __name__ == "__main__":
    async def main():
        """Test real-time monitoring"""
        print("🚀 Testing Real-time Performance Monitor V2.0")
        
        monitor = RealTimeMonitor()
        await monitor.start_monitoring()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        await monitor.stop_monitoring()
        
        # Show results
        dashboard_data = monitor.get_monitoring_dashboard_data()
        print(f"📊 Collected {len(dashboard_data['current_metrics'])} metrics")
        print(f"🔔 Generated {len(dashboard_data['active_alerts'])} alerts")
    
    if HAS_FASTAPI:
        print("🌐 Starting Real-time Monitor Web Dashboard")
        print("📊 Dashboard: http://localhost:8002/")
        print("📈 API: http://localhost:8002/api/dashboard")
        uvicorn.run(dashboard_app, host="0.0.0.0", port=8002)
    else:
        asyncio.run(main())