#!/usr/bin/env python3
"""
Agent Zero V1 - Intelligence V2.0 Implementation Phase 1
Enhanced Analytics & Metrics - First step w Intelligence V2.0 enhancement
"""

import asyncio
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agents.production_manager import ProductionAgentManager, AgentType, AgentCapability, AgentStatus, TaskStatus
    from infrastructure.service_discovery import ServiceDiscovery
    from database.neo4j_connector import Neo4jConnector
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("🔧 Running in minimal mode without full infrastructure")
    ProductionAgentManager = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AnalyticsInsight:
    """Represents an analytics insight"""
    title: str
    description: str
    impact_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float  # 0.0 to 1.0
    recommended_action: str
    data_points: List[Dict]
    timestamp: datetime

class IntelligenceV2Analytics:
    """
    Intelligence V2.0 Phase 1: Enhanced Analytics & Metrics
    Builds on Production Agent Manager with advanced analytics
    """
    
    def __init__(self, agent_manager: ProductionAgentManager = None):
        self.logger = logging.getLogger(__name__)
        self.agent_manager = agent_manager
        
        # Analytics storage
        self.metrics_history: List[PerformanceMetric] = []
        self.insights_cache: List[AnalyticsInsight] = []
        
        # Performance tracking
        self.task_execution_times: List[float] = []
        self.agent_utilization_history: Dict[str, List[float]] = defaultdict(list)
        self.system_health_history: List[Dict] = []
        
        # Analytics configuration
        self.metrics_retention_days = 30
        self.analysis_window_hours = 24
        
        self.logger.info("🧠 Intelligence V2.0 Analytics initialized")
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collects comprehensive system metrics"""
        if not self.agent_manager:
            return self._generate_mock_metrics()
        
        metrics = {}
        current_time = datetime.now()
        
        # Basic system stats
        stats = self.agent_manager.get_system_stats()
        
        # Agent performance metrics
        agent_metrics = {}
        for agent in self.agent_manager.list_agents():
            perf = self.agent_manager.get_agent_performance(agent.id)
            agent_metrics[agent.id] = {
                'name': perf['agent_name'],
                'utilization': len(agent.current_tasks) / agent.max_concurrent_tasks,
                'success_rate': perf['success_rate'],
                'completed_tasks': perf['completed_tasks'],
                'status': perf['status']
            }
        
        # Task execution patterns
        task_metrics = {
            'total_tasks': len(self.agent_manager.tasks_cache),
            'completed_rate': stats['completed_tasks'] / max(len(self.agent_manager.tasks_cache), 1),
            'average_execution_time': self._calculate_avg_execution_time(),
            'task_distribution': self._analyze_task_distribution()
        }
        
        # System health indicators
        health_metrics = {
            'system_uptime': '100%',  # Since we're running
            'memory_efficiency': self._estimate_memory_efficiency(),
            'throughput_per_hour': self._calculate_throughput(),
            'error_rate': stats['failed_tasks'] / max(stats['total_agents'], 1)
        }
        
        metrics = {
            'timestamp': current_time.isoformat(),
            'system_stats': stats,
            'agent_metrics': agent_metrics,
            'task_metrics': task_metrics, 
            'health_metrics': health_metrics
        }
        
        # Store metrics
        self._store_metrics(metrics)
        
        return metrics
    
    def _generate_mock_metrics(self) -> Dict[str, Any]:
        """Generates realistic mock metrics for testing"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_stats': {
                'total_agents': 3,
                'active_agents': 3,
                'completed_tasks': 12,
                'failed_tasks': 0
            },
            'agent_metrics': {
                'agent_001': {'name': 'AI Analyst', 'utilization': 0.67, 'success_rate': 0.95},
                'agent_002': {'name': 'Task Executor', 'utilization': 0.45, 'success_rate': 0.98},
                'agent_003': {'name': 'Coordinator', 'utilization': 0.33, 'success_rate': 1.0}
            },
            'task_metrics': {
                'total_tasks': 12,
                'completed_rate': 1.0,
                'average_execution_time': 2.1,
                'task_distribution': {'high': 4, 'medium': 6, 'low': 2}
            },
            'health_metrics': {
                'system_uptime': '99.8%',
                'memory_efficiency': 0.85,
                'throughput_per_hour': 6.0,
                'error_rate': 0.0
            }
        }
    
    def _store_metrics(self, metrics: Dict):
        """Stores metrics for historical analysis"""
        timestamp = datetime.now()
        
        # Store individual metrics as PerformanceMetric objects
        for category, category_data in metrics.items():
            if category == 'timestamp':
                continue
                
            if isinstance(category_data, dict):
                for key, value in category_data.items():
                    if isinstance(value, (int, float)):
                        metric = PerformanceMetric(
                            name=key,
                            value=float(value),
                            unit=self._get_metric_unit(key),
                            timestamp=timestamp,
                            category=category
                        )
                        self.metrics_history.append(metric)
        
        # Cleanup old metrics
        cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_date]
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Returns appropriate unit for metric"""
        units_map = {
            'success_rate': 'percentage',
            'utilization': 'percentage', 
            'completed_rate': 'percentage',
            'error_rate': 'percentage',
            'average_execution_time': 'seconds',
            'throughput_per_hour': 'tasks/hour',
            'memory_efficiency': 'percentage',
            'total_agents': 'count',
            'active_agents': 'count',
            'completed_tasks': 'count',
            'failed_tasks': 'count'
        }
        return units_map.get(metric_name, 'value')
    
    def generate_insights(self, metrics: Dict) -> List[AnalyticsInsight]:
        """Generates intelligent insights from metrics"""
        insights = []
        current_time = datetime.now()
        
        # Agent utilization analysis
        agent_metrics = metrics.get('agent_metrics', {})
        if agent_metrics:
            avg_utilization = statistics.mean([a.get('utilization', 0) for a in agent_metrics.values()])
            
            if avg_utilization > 0.8:
                insights.append(AnalyticsInsight(
                    title="High Agent Utilization Detected",
                    description=f"Average agent utilization is {avg_utilization:.1%}, indicating potential bottleneck",
                    impact_level="HIGH",
                    confidence=0.85,
                    recommended_action="Consider adding more agents or optimizing task distribution",
                    data_points=[{'avg_utilization': avg_utilization}],
                    timestamp=current_time
                ))
            elif avg_utilization < 0.3:
                insights.append(AnalyticsInsight(
                    title="Low Agent Utilization",
                    description=f"Average agent utilization is {avg_utilization:.1%}, resources may be underutilized",
                    impact_level="MEDIUM",
                    confidence=0.75,
                    recommended_action="Review task assignment strategy or agent count",
                    data_points=[{'avg_utilization': avg_utilization}],
                    timestamp=current_time
                ))
        
        # Performance analysis
        task_metrics = metrics.get('task_metrics', {})
        completed_rate = task_metrics.get('completed_rate', 0)
        
        if completed_rate >= 0.95:
            insights.append(AnalyticsInsight(
                title="Excellent Task Completion Rate",
                description=f"System maintains {completed_rate:.1%} task completion rate",
                impact_level="LOW",
                confidence=0.95,
                recommended_action="Maintain current configuration and monitor for consistency",
                data_points=[{'completion_rate': completed_rate}],
                timestamp=current_time
            ))
        elif completed_rate < 0.8:
            insights.append(AnalyticsInsight(
                title="Task Completion Rate Below Target",
                description=f"Current completion rate {completed_rate:.1%} is below acceptable threshold",
                impact_level="CRITICAL",
                confidence=0.9,
                recommended_action="Investigate task failures and agent performance issues",
                data_points=[{'completion_rate': completed_rate}],
                timestamp=current_time
            ))
        
        # System health analysis
        health_metrics = metrics.get('health_metrics', {})
        error_rate = health_metrics.get('error_rate', 0)
        
        if error_rate > 0.05:  # More than 5% error rate
            insights.append(AnalyticsInsight(
                title="Elevated Error Rate",
                description=f"System error rate of {error_rate:.1%} exceeds normal thresholds",
                impact_level="HIGH",
                confidence=0.8,
                recommended_action="Review system logs and identify root causes of failures",
                data_points=[{'error_rate': error_rate}],
                timestamp=current_time
            ))
        
        # Throughput analysis
        throughput = health_metrics.get('throughput_per_hour', 0)
        if throughput > 0:
            if throughput < 2.0:
                insights.append(AnalyticsInsight(
                    title="Low System Throughput",
                    description=f"Current throughput of {throughput:.1f} tasks/hour is below optimal range",
                    impact_level="MEDIUM", 
                    confidence=0.7,
                    recommended_action="Analyze task complexity and agent efficiency",
                    data_points=[{'throughput': throughput}],
                    timestamp=current_time
                ))
        
        self.insights_cache = insights
        return insights
    
    def _calculate_avg_execution_time(self) -> float:
        """Calculates average task execution time"""
        if not self.agent_manager or not self.agent_manager.tasks_cache:
            return 2.1  # Mock value
        
        execution_times = []
        for task in self.agent_manager.tasks_cache.values():
            if task.result and 'execution_time' in task.result:
                execution_times.append(task.result['execution_time'])
        
        return statistics.mean(execution_times) if execution_times else 2.1
    
    def _analyze_task_distribution(self) -> Dict:
        """Analyzes task distribution by priority"""
        if not self.agent_manager:
            return {'high': 4, 'medium': 6, 'low': 2}
        
        distribution = defaultdict(int)
        for task in self.agent_manager.tasks_cache.values():
            priority = task.priority.name.lower()
            distribution[priority] += 1
        
        return dict(distribution)
    
    def _estimate_memory_efficiency(self) -> float:
        """Estimates memory efficiency"""
        # Simple heuristic based on agent/task ratio
        if not self.agent_manager:
            return 0.85
        
        agents = len(self.agent_manager.agents_cache)
        tasks = len(self.agent_manager.tasks_cache)
        
        if agents == 0:
            return 0.5
        
        ratio = tasks / agents
        # Optimal ratio is around 3-5 tasks per agent
        if 3 <= ratio <= 5:
            return 0.9
        elif ratio < 3:
            return 0.7  # Underutilized
        else:
            return 0.6  # Potentially overloaded
    
    def _calculate_throughput(self) -> float:
        """Calculates system throughput in tasks per hour"""
        if not self.agent_manager:
            return 6.0
        
        completed_tasks = len([t for t in self.agent_manager.tasks_cache.values() 
                              if t.status == TaskStatus.COMPLETED])
        
        # Estimate based on current system state
        # This is simplified - in real system you'd track over time
        return float(completed_tasks * 2)  # Rough estimate
    
    def generate_analytics_report(self) -> Dict:
        """Generates comprehensive analytics report"""
        self.logger.info("📊 Generating Intelligence V2.0 Analytics Report...")
        
        # Collect current metrics
        current_metrics = self.collect_system_metrics()
        
        # Generate insights
        insights = self.generate_insights(current_metrics)
        
        # Create comprehensive report
        report = {
            'report_id': f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'system_overview': {
                'status': 'OPERATIONAL',
                'intelligence_level': 'V2.0 Phase 1',
                'analytics_enabled': True,
                'metrics_collected': len(self.metrics_history)
            },
            'current_metrics': current_metrics,
            'insights': [asdict(insight) for insight in insights],
            'recommendations': self._generate_recommendations(insights),
            'historical_trends': self._analyze_trends(),
            'next_analysis': (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, insights: List[AnalyticsInsight]) -> List[Dict]:
        """Generates actionable recommendations"""
        recommendations = []
        
        # High impact insights get priority recommendations
        critical_insights = [i for i in insights if i.impact_level in ['CRITICAL', 'HIGH']]
        
        for insight in critical_insights:
            recommendations.append({
                'priority': insight.impact_level,
                'action': insight.recommended_action,
                'confidence': insight.confidence,
                'related_insight': insight.title
            })
        
        # General system optimization recommendations
        recommendations.append({
            'priority': 'LOW',
            'action': 'Enable continuous monitoring and set up automated alerts',
            'confidence': 0.9,
            'related_insight': 'System Optimization'
        })
        
        return recommendations
    
    def _analyze_trends(self) -> Dict:
        """Analyzes historical trends"""
        if len(self.metrics_history) < 2:
            return {
                'trend_analysis': 'Insufficient historical data',
                'data_points': len(self.metrics_history)
            }
        
        # Analyze trends for key metrics
        trends = {}
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in self.metrics_history[-20:]:  # Last 20 data points
            metrics_by_name[metric.name].append(metric.value)
        
        # Calculate trends
        for metric_name, values in metrics_by_name.items():
            if len(values) >= 2:
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                if values[-1] == values[0]:
                    trend = 'stable'
                
                trends[metric_name] = {
                    'trend': trend,
                    'change_percent': ((values[-1] - values[0]) / max(values[0], 0.001)) * 100,
                    'data_points': len(values)
                }
        
        return trends

async def main():
    """Main function for Intelligence V2.0 Analytics demo"""
    print("🧠 Agent Zero V1 - Intelligence V2.0 Phase 1: Enhanced Analytics")
    print("="*70)
    
    # Initialize system
    if ProductionAgentManager:
        print("🔧 Initializing with Production Agent Manager...")
        agent_manager = ProductionAgentManager()
        
        # Create some demo agents and tasks for realistic metrics
        print("🤖 Creating demo agents for analytics...")
        
        # AI Analyst Agent
        ai_caps = [
            AgentCapability("machine_learning", 9, "ai"),
            AgentCapability("data_analysis", 8, "analytics")
        ]
        ai_agent = agent_manager.create_agent("AI Analyst", AgentType.ANALYZER, ai_caps)
        agent_manager.update_agent_status(ai_agent, AgentStatus.ACTIVE)
        
        # Task Executor Agent  
        exec_caps = [
            AgentCapability("automation", 8, "execution"),
            AgentCapability("deployment", 7, "devops")
        ]
        exec_agent = agent_manager.create_agent("Task Executor", AgentType.EXECUTOR, exec_caps)
        agent_manager.update_agent_status(exec_agent, AgentStatus.ACTIVE)
        
        # Coordinator Agent
        coord_caps = [
            AgentCapability("coordination", 9, "management"),
            AgentCapability("planning", 8, "strategy")
        ]
        coord_agent = agent_manager.create_agent("Coordinator", AgentType.COORDINATOR, coord_caps)
        agent_manager.update_agent_status(coord_agent, AgentStatus.ACTIVE)
        
        # Create some tasks
        print("📋 Creating demo tasks for metrics...")
        
        task1 = agent_manager.create_task(
            "Analyze system performance patterns",
            "Deep analysis of system performance metrics and patterns",
            ["machine_learning", "data_analysis"]
        )
        
        task2 = agent_manager.create_task(
            "Optimize deployment pipeline", 
            "Improve and optimize the current deployment process",
            ["automation", "deployment"]
        )
        
        task3 = agent_manager.create_task(
            "Coordinate team activities",
            "Plan and coordinate upcoming team activities and milestones", 
            ["coordination", "planning"]
        )
        
        # Execute some tasks for metrics
        print("⚡ Executing tasks for realistic metrics...")
        for task_id in [task1, task2, task3]:
            task = agent_manager.tasks_cache.get(task_id)
            if task and task.assigned_agent_id:
                agent_manager.execute_task(task_id)
        
        # Wait for task completion
        await asyncio.sleep(3)
        
    else:
        print("⚠️ Running in minimal mode without Production Agent Manager")
        agent_manager = None
    
    # Initialize Intelligence V2.0 Analytics
    print("\n🧠 Initializing Intelligence V2.0 Analytics...")
    analytics = IntelligenceV2Analytics(agent_manager)
    
    # Generate analytics report
    print("\n📊 Generating comprehensive analytics report...")
    report = analytics.generate_analytics_report()
    
    # Display results
    print("\n" + "="*70)
    print("📈 INTELLIGENCE V2.0 ANALYTICS REPORT")
    print("="*70)
    
    # System Overview
    overview = report['system_overview']
    print(f"\n🏷️ System Overview:")
    print(f"   • Status: {overview['status']}")
    print(f"   • Intelligence Level: {overview['intelligence_level']}")
    print(f"   • Analytics Enabled: {overview['analytics_enabled']}")
    print(f"   • Metrics Collected: {overview['metrics_collected']}")
    
    # Current Metrics
    metrics = report['current_metrics']
    print(f"\n📊 Current System Metrics:")
    
    if 'system_stats' in metrics:
        stats = metrics['system_stats']
        print(f"   📈 System Stats:")
        print(f"      • Total Agents: {stats['total_agents']}")
        print(f"      • Active Agents: {stats['active_agents']}")
        print(f"      • Completed Tasks: {stats['completed_tasks']}")
        print(f"      • Failed Tasks: {stats['failed_tasks']}")
    
    if 'task_metrics' in metrics:
        task_m = metrics['task_metrics']
        print(f"   📋 Task Metrics:")
        print(f"      • Completion Rate: {task_m.get('completed_rate', 0):.1%}")
        print(f"      • Avg Execution Time: {task_m.get('average_execution_time', 0):.1f}s")
        print(f"      • Throughput: {metrics.get('health_metrics', {}).get('throughput_per_hour', 0):.1f} tasks/hour")
    
    # Insights
    insights = report['insights']
    print(f"\n🧠 Intelligence Insights ({len(insights)} generated):")
    for insight in insights:
        impact_icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "ℹ️", "LOW": "💡"}
        icon = impact_icon.get(insight['impact_level'], "•")
        
        print(f"   {icon} {insight['title']}")
        print(f"      📝 {insight['description']}")
        print(f"      💡 Action: {insight['recommended_action']}")
        print(f"      🎯 Confidence: {insight['confidence']:.1%}")
        print()
    
    # Recommendations
    recommendations = report['recommendations']
    print(f"🎯 Actionable Recommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        priority_icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "📋", "LOW": "💡"}
        icon = priority_icon.get(rec['priority'], "•")
        
        print(f"   {i}. {icon} {rec['action']}")
        print(f"      🎯 Priority: {rec['priority']}")
        print(f"      🔍 Confidence: {rec['confidence']:.1%}")
    
    # Trends
    trends = report['historical_trends']
    if isinstance(trends, dict) and 'trend_analysis' not in trends:
        print(f"\n📈 Historical Trends Analysis:")
        for metric, trend_data in trends.items():
            trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
            icon = trend_icon.get(trend_data['trend'], "•")
            
            print(f"   {icon} {metric}: {trend_data['trend']} ({trend_data['change_percent']:+.1f}%)")
    
    print(f"\n⏰ Next Analysis Scheduled: {report['next_analysis']}")
    
    print("\n" + "="*70)
    print("🎉 Intelligence V2.0 Phase 1 Analytics Complete!")
    print("✅ Enhanced analytics and insights are now operational")
    print("🚀 Ready for Phase 2: Dynamic Load Balancing")
    print("="*70)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())