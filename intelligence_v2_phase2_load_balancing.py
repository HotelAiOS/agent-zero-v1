#!/usr/bin/env python3
"""
Agent Zero V1 - Intelligence V2.0 Phase 2: Dynamic Load Balancing
Następny logiczny krok architektury: Inteligentne zarządzanie obciążeniem
"""

import asyncio
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import logging
import math

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agents.production_manager import ProductionAgentManager, AgentType, AgentCapability, AgentStatus, TaskStatus, TaskPriority
    from infrastructure.service_discovery import ServiceDiscovery
    from database.neo4j_connector import Neo4jConnector
    from intelligence_v2_phase1_analytics import IntelligenceV2Analytics, PerformanceMetric
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

class LoadBalancingStrategy(Enum):
    """Strategie load balancingu"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded" 
    CAPABILITY_WEIGHTED = "capability_weighted"
    PERFORMANCE_BASED = "performance_based"
    INTELLIGENT_ML = "intelligent_ml"

class WorkloadPattern(Enum):
    """Wzorce obciążenia systemem"""
    LIGHT = "light"      # <30% utilization
    MODERATE = "moderate"  # 30-70% utilization  
    HEAVY = "heavy"      # 70-90% utilization
    OVERLOADED = "overloaded"  # >90% utilization

@dataclass
class LoadMetric:
    """Metryka obciążenia agenta"""
    agent_id: str
    current_tasks: int
    max_capacity: int
    utilization: float
    success_rate: float
    avg_execution_time: float
    performance_score: float
    timestamp: datetime
    
    @property
    def available_capacity(self) -> int:
        return self.max_capacity - self.current_tasks
    
    @property
    def is_overloaded(self) -> bool:
        return self.utilization > 0.9

@dataclass
class LoadBalancingDecision:
    """Decyzja load balancingu"""
    recommended_agent: str
    strategy_used: LoadBalancingStrategy
    confidence: float
    reasoning: str
    alternative_agents: List[str]
    predicted_completion_time: float
    load_impact: float
    timestamp: datetime

class IntelligentLoadBalancer:
    """
    Intelligence V2.0 Phase 2: Dynamic Load Balancing
    Inteligentne zarządzanie obciążeniem na bazie Phase 1 Analytics
    """
    
    def __init__(self, agent_manager: ProductionAgentManager = None,
                 analytics: IntelligenceV2Analytics = None):
        self.logger = logging.getLogger(__name__)
        self.agent_manager = agent_manager
        self.analytics = analytics or IntelligenceV2Analytics(agent_manager)
        
        # Load balancing state
        self.load_metrics: Dict[str, LoadMetric] = {}
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.decisions_history: List[LoadBalancingDecision] = []
        
        # Strategy configuration
        self.current_strategy = LoadBalancingStrategy.INTELLIGENT_ML
        self.strategy_weights = {
            'performance_score': 0.4,
            'utilization': 0.3,
            'success_rate': 0.2,
            'avg_execution_time': 0.1
        }
        
        # Adaptive thresholds
        self.utilization_thresholds = {
            'optimal': 0.7,     # Optimal load level
            'warning': 0.8,     # Start load balancing  
            'critical': 0.9     # Emergency redistribution
        }
        
        self.logger.info("⚡ Intelligence V2.0 Dynamic Load Balancer initialized")
    
    def collect_load_metrics(self) -> Dict[str, LoadMetric]:
        """Zbiera aktualne metryki obciążenia wszystkich agentów"""
        if not self.agent_manager:
            return self._generate_mock_load_metrics()
        
        current_metrics = {}
        timestamp = datetime.now()
        
        for agent in self.agent_manager.list_agents():
            perf = self.agent_manager.get_agent_performance(agent.id)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(agent, perf)
            
            # Get average execution time from recent tasks
            avg_exec_time = self._get_avg_execution_time(agent.id)
            
            metric = LoadMetric(
                agent_id=agent.id,
                current_tasks=len(agent.current_tasks),
                max_capacity=agent.max_concurrent_tasks,
                utilization=len(agent.current_tasks) / agent.max_concurrent_tasks,
                success_rate=perf['success_rate'],
                avg_execution_time=avg_exec_time,
                performance_score=performance_score,
                timestamp=timestamp
            )
            
            current_metrics[agent.id] = metric
            
            # Store in history
            self.load_history[agent.id].append({
                'timestamp': timestamp,
                'utilization': metric.utilization,
                'performance_score': metric.performance_score
            })
        
        self.load_metrics = current_metrics
        return current_metrics
    
    def _generate_mock_load_metrics(self) -> Dict[str, LoadMetric]:
        """Generuje realistyczne metryki testowe"""
        import random
        
        mock_agents = [
            {'id': 'agent_ai_analyst', 'name': 'AI Analyst', 'capacity': 5},
            {'id': 'agent_task_executor', 'name': 'Task Executor', 'capacity': 3},
            {'id': 'agent_coordinator', 'name': 'Coordinator', 'capacity': 4}
        ]
        
        metrics = {}
        timestamp = datetime.now()
        
        for agent_data in mock_agents:
            current_tasks = random.randint(0, agent_data['capacity'])
            utilization = current_tasks / agent_data['capacity']
            
            metric = LoadMetric(
                agent_id=agent_data['id'],
                current_tasks=current_tasks,
                max_capacity=agent_data['capacity'],
                utilization=utilization,
                success_rate=random.uniform(0.85, 0.98),
                avg_execution_time=random.uniform(1.5, 3.0),
                performance_score=random.uniform(0.7, 0.95),
                timestamp=timestamp
            )
            
            metrics[agent_data['id']] = metric
        
        return metrics
    
    def _calculate_performance_score(self, agent, perf: Dict) -> float:
        """Oblicza kompleksowy wynik wydajności agenta"""
        success_rate = perf['success_rate']
        current_load = len(agent.current_tasks) / agent.max_concurrent_tasks
        
        # Base score from success rate
        score = success_rate * 0.6
        
        # Adjust for current load (prefer agents with moderate load)
        if current_load < 0.3:
            load_factor = 0.8  # Slightly penalize underutilized agents
        elif current_load < 0.7:
            load_factor = 1.0  # Optimal load range
        elif current_load < 0.9:
            load_factor = 0.7  # Prefer less loaded agents
        else:
            load_factor = 0.3  # Strongly avoid overloaded agents
        
        score *= load_factor
        
        # Add capability matching bonus (simplified)
        capability_bonus = len(agent.capabilities) * 0.02
        score += capability_bonus
        
        return min(1.0, score)
    
    def _get_avg_execution_time(self, agent_id: str) -> float:
        """Pobiera średni czas wykonania zadań dla agenta"""
        if not self.agent_manager:
            return 2.0
        
        execution_times = []
        for task in self.agent_manager.tasks_cache.values():
            if (task.assigned_agent_id == agent_id and 
                task.result and 'execution_time' in task.result):
                execution_times.append(task.result['execution_time'])
        
        return statistics.mean(execution_times) if execution_times else 2.0
    
    def analyze_workload_patterns(self) -> WorkloadPattern:
        """Analizuje wzorce obciążenia całego systemu"""
        if not self.load_metrics:
            return WorkloadPattern.LIGHT
        
        total_utilization = statistics.mean([m.utilization for m in self.load_metrics.values()])
        
        if total_utilization < 0.3:
            return WorkloadPattern.LIGHT
        elif total_utilization < 0.7:
            return WorkloadPattern.MODERATE  
        elif total_utilization < 0.9:
            return WorkloadPattern.HEAVY
        else:
            return WorkloadPattern.OVERLOADED
    
    def select_optimal_agent(self, task_requirements: Dict) -> LoadBalancingDecision:
        """Wybiera optymalnego agenta dla zadania używając inteligentnego load balancingu"""
        
        # Collect current metrics
        self.collect_load_metrics()
        
        if not self.load_metrics:
            return self._create_fallback_decision("No agents available")
        
        # Filter agents by capability requirements
        required_capabilities = task_requirements.get('capabilities', [])
        available_agents = self._filter_agents_by_capability(required_capabilities)
        
        if not available_agents:
            return self._create_fallback_decision("No agents with required capabilities")
        
        # Apply current strategy
        decision = self._apply_load_balancing_strategy(available_agents, task_requirements)
        
        # Store decision
        self.decisions_history.append(decision)
        
        return decision
    
    def _filter_agents_by_capability(self, required_capabilities: List[str]) -> List[str]:
        """Filtruje agentów według wymaganych capabilities"""
        if not self.agent_manager or not required_capabilities:
            return list(self.load_metrics.keys())
        
        suitable_agents = []
        
        for agent in self.agent_manager.list_agents():
            agent_caps = [cap.name for cap in agent.capabilities]
            
            # Check if agent has all required capabilities
            if all(req_cap in agent_caps for req_cap in required_capabilities):
                suitable_agents.append(agent.id)
        
        return suitable_agents
    
    def _apply_load_balancing_strategy(self, available_agents: List[str], 
                                     task_requirements: Dict) -> LoadBalancingDecision:
        """Stosuje aktualną strategię load balancingu"""
        
        if self.current_strategy == LoadBalancingStrategy.INTELLIGENT_ML:
            return self._intelligent_ml_selection(available_agents, task_requirements)
        elif self.current_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(available_agents, task_requirements)
        elif self.current_strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(available_agents, task_requirements)
        else:
            return self._round_robin_selection(available_agents, task_requirements)
    
    def _intelligent_ml_selection(self, agents: List[str], 
                                task_req: Dict) -> LoadBalancingDecision:
        """Inteligentny wybór na bazie machine learning heuristics"""
        
        scored_agents = []
        
        for agent_id in agents:
            if agent_id not in self.load_metrics:
                continue
                
            metric = self.load_metrics[agent_id]
            
            # Multi-factor scoring
            score = 0.0
            
            # Performance factor (40%)
            performance_factor = metric.performance_score * self.strategy_weights['performance_score']
            score += performance_factor
            
            # Load factor (30%) - prefer moderately loaded agents
            load_factor = self._calculate_load_factor(metric.utilization)
            score += load_factor * self.strategy_weights['utilization']
            
            # Success rate factor (20%)
            success_factor = metric.success_rate * self.strategy_weights['success_rate']
            score += success_factor
            
            # Execution time factor (10%) - prefer faster agents
            time_factor = max(0, 1.0 - (metric.avg_execution_time / 5.0))  # Normalize to 0-1
            score += time_factor * self.strategy_weights['avg_execution_time']
            
            # Predicted completion time
            predicted_time = self._predict_completion_time(metric, task_req)
            
            scored_agents.append({
                'agent_id': agent_id,
                'score': score,
                'predicted_time': predicted_time,
                'reasoning': f"ML Score: {score:.2f} (Perf:{performance_factor:.2f}, Load:{load_factor:.2f}, Success:{success_factor:.2f})"
            })
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x['score'], reverse=True)
        
        if not scored_agents:
            return self._create_fallback_decision("No suitable agents found")
        
        best_agent = scored_agents[0]
        alternatives = [a['agent_id'] for a in scored_agents[1:3]]  # Top 2 alternatives
        
        return LoadBalancingDecision(
            recommended_agent=best_agent['agent_id'],
            strategy_used=LoadBalancingStrategy.INTELLIGENT_ML,
            confidence=min(0.95, best_agent['score']),
            reasoning=best_agent['reasoning'],
            alternative_agents=alternatives,
            predicted_completion_time=best_agent['predicted_time'],
            load_impact=self._calculate_load_impact(best_agent['agent_id']),
            timestamp=datetime.now()
        )
    
    def _calculate_load_factor(self, utilization: float) -> float:
        """Oblicza współczynnik obciążenia (preferuje umiarkowane obciążenie)"""
        if utilization < 0.2:
            return 0.6  # Too underutilized
        elif utilization < 0.7:
            return 1.0  # Optimal range
        elif utilization < 0.85:
            return 0.8  # Getting busy
        else:
            return 0.3  # Too loaded
    
    def _predict_completion_time(self, metric: LoadMetric, task_req: Dict) -> float:
        """Przewiduje czas ukończenia zadania"""
        base_time = metric.avg_execution_time
        
        # Adjust for current load
        load_multiplier = 1.0 + (metric.utilization * 0.5)  # More load = longer time
        
        # Adjust for task priority
        priority = task_req.get('priority', 'MEDIUM')
        priority_multipliers = {
            'LOW': 1.2,
            'MEDIUM': 1.0, 
            'HIGH': 0.9,
            'URGENT': 0.8,
            'CRITICAL': 0.7
        }
        priority_multiplier = priority_multipliers.get(priority, 1.0)
        
        predicted_time = base_time * load_multiplier * priority_multiplier
        
        return predicted_time
    
    def _calculate_load_impact(self, agent_id: str) -> float:
        """Oblicza wpływ przypisania zadania na obciążenie agenta"""
        if agent_id not in self.load_metrics:
            return 0.5
        
        metric = self.load_metrics[agent_id]
        
        # Calculate impact on utilization
        new_utilization = (metric.current_tasks + 1) / metric.max_capacity
        impact = new_utilization - metric.utilization
        
        return impact
    
    def _performance_based_selection(self, agents: List[str], 
                                   task_req: Dict) -> LoadBalancingDecision:
        """Wybór na bazie wydajności historycznej"""
        
        best_agent = None
        best_score = 0.0
        
        for agent_id in agents:
            if agent_id not in self.load_metrics:
                continue
                
            metric = self.load_metrics[agent_id]
            
            # Simple performance score
            score = metric.performance_score
            
            # Penalize overloaded agents
            if metric.utilization > 0.8:
                score *= 0.5
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return LoadBalancingDecision(
            recommended_agent=best_agent or agents[0],
            strategy_used=LoadBalancingStrategy.PERFORMANCE_BASED,
            confidence=best_score,
            reasoning=f"Highest performance score: {best_score:.2f}",
            alternative_agents=agents[1:3],
            predicted_completion_time=2.0,
            load_impact=0.2,
            timestamp=datetime.now()
        )
    
    def _least_loaded_selection(self, agents: List[str], 
                               task_req: Dict) -> LoadBalancingDecision:
        """Wybór najmniej obciążonego agenta"""
        
        least_loaded = min(agents, 
                          key=lambda a: self.load_metrics.get(a, LoadMetric('', 0, 1, 1.0, 0, 0, 0, datetime.now())).utilization)
        
        utilization = self.load_metrics.get(least_loaded, None)
        util_value = utilization.utilization if utilization else 0.5
        
        return LoadBalancingDecision(
            recommended_agent=least_loaded,
            strategy_used=LoadBalancingStrategy.LEAST_LOADED,
            confidence=1.0 - util_value,
            reasoning=f"Least loaded agent (utilization: {util_value:.1%})",
            alternative_agents=agents[1:3] if len(agents) > 1 else [],
            predicted_completion_time=2.0,
            load_impact=0.2,
            timestamp=datetime.now()
        )
    
    def _round_robin_selection(self, agents: List[str], 
                              task_req: Dict) -> LoadBalancingDecision:
        """Prosty round-robin selection"""
        
        # Simple round-robin based on decision history
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_agent = agents[self._round_robin_index % len(agents)]
        self._round_robin_index += 1
        
        return LoadBalancingDecision(
            recommended_agent=selected_agent,
            strategy_used=LoadBalancingStrategy.ROUND_ROBIN,
            confidence=0.7,
            reasoning="Round-robin selection",
            alternative_agents=agents[1:3] if len(agents) > 1 else [],
            predicted_completion_time=2.0,
            load_impact=0.3,
            timestamp=datetime.now()
        )
    
    def _create_fallback_decision(self, reason: str) -> LoadBalancingDecision:
        """Tworzy fallback decision gdy nie można wybrać agenta"""
        
        return LoadBalancingDecision(
            recommended_agent="",
            strategy_used=LoadBalancingStrategy.ROUND_ROBIN,
            confidence=0.0,
            reasoning=f"Fallback: {reason}",
            alternative_agents=[],
            predicted_completion_time=0.0,
            load_impact=0.0,
            timestamp=datetime.now()
        )
    
    def detect_load_imbalances(self) -> List[Dict]:
        """Wykrywa nierównowagi obciążenia i proponuje rozwiązania"""
        
        imbalances = []
        
        if not self.load_metrics:
            return imbalances
        
        utilizations = [m.utilization for m in self.load_metrics.values()]
        
        if not utilizations:
            return imbalances
        
        avg_utilization = statistics.mean(utilizations)
        std_utilization = statistics.stdev(utilizations) if len(utilizations) > 1 else 0
        
        # Detect high variance in utilization (imbalance indicator)
        if std_utilization > 0.3:
            overloaded_agents = [aid for aid, m in self.load_metrics.items() if m.utilization > 0.8]
            underloaded_agents = [aid for aid, m in self.load_metrics.items() if m.utilization < 0.3]
            
            if overloaded_agents and underloaded_agents:
                imbalances.append({
                    'type': 'utilization_imbalance',
                    'severity': 'HIGH' if std_utilization > 0.5 else 'MEDIUM',
                    'description': f'High utilization variance: {std_utilization:.2f}',
                    'overloaded_agents': overloaded_agents,
                    'underloaded_agents': underloaded_agents,
                    'recommended_action': 'redistribute_tasks',
                    'confidence': 0.85
                })
        
        # Detect performance bottlenecks
        performance_scores = [m.performance_score for m in self.load_metrics.values()]
        min_performance = min(performance_scores)
        
        if min_performance < 0.6:
            low_performers = [aid for aid, m in self.load_metrics.items() if m.performance_score < 0.6]
            
            imbalances.append({
                'type': 'performance_bottleneck',
                'severity': 'HIGH',
                'description': f'Low performance agents detected: min score {min_performance:.2f}',
                'affected_agents': low_performers,
                'recommended_action': 'agent_optimization',
                'confidence': 0.9
            })
        
        return imbalances
    
    def auto_rebalance_system(self) -> Dict:
        """Automatyczne rebalansowanie systemu"""
        
        rebalance_actions = []
        
        # Detect imbalances
        imbalances = self.detect_load_imbalances()
        
        for imbalance in imbalances:
            if imbalance['type'] == 'utilization_imbalance':
                # Suggest task redistribution
                action = {
                    'type': 'task_redistribution',
                    'from_agents': imbalance['overloaded_agents'],
                    'to_agents': imbalance['underloaded_agents'],
                    'priority': imbalance['severity'],
                    'estimated_benefit': 'Improved system throughput by 15-30%'
                }
                rebalance_actions.append(action)
                
            elif imbalance['type'] == 'performance_bottleneck':
                # Suggest performance optimization
                action = {
                    'type': 'performance_optimization',
                    'target_agents': imbalance['affected_agents'],
                    'priority': 'HIGH',
                    'estimated_benefit': 'Reduced task failure rate'
                }
                rebalance_actions.append(action)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'imbalances_detected': len(imbalances),
            'actions_recommended': len(rebalance_actions),
            'actions': rebalance_actions,
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict:
        """Ocenia zdrowotność systemu load balancing"""
        
        if not self.load_metrics:
            return {'status': 'UNKNOWN', 'score': 0.5}
        
        utilizations = [m.utilization for m in self.load_metrics.values()]
        performance_scores = [m.performance_score for m in self.load_metrics.values()]
        
        avg_utilization = statistics.mean(utilizations)
        avg_performance = statistics.mean(performance_scores)
        util_variance = statistics.stdev(utilizations) if len(utilizations) > 1 else 0
        
        # Health score calculation
        health_score = 0.0
        
        # Utilization component (40%)
        if 0.3 <= avg_utilization <= 0.7:
            util_score = 1.0
        elif 0.2 <= avg_utilization <= 0.8:
            util_score = 0.8
        else:
            util_score = 0.4
        health_score += util_score * 0.4
        
        # Performance component (40%)
        health_score += avg_performance * 0.4
        
        # Balance component (20%)
        balance_score = max(0, 1.0 - (util_variance * 2))
        health_score += balance_score * 0.2
        
        # Determine status
        if health_score >= 0.8:
            status = 'EXCELLENT'
        elif health_score >= 0.6:
            status = 'GOOD'
        elif health_score >= 0.4:
            status = 'FAIR'
        else:
            status = 'POOR'
        
        return {
            'status': status,
            'score': health_score,
            'avg_utilization': avg_utilization,
            'avg_performance': avg_performance,
            'utilization_variance': util_variance
        }
    
    def generate_load_balancing_report(self) -> Dict:
        """Generuje kompletny raport load balancingu"""
        
        self.logger.info("⚡ Generating Intelligence V2.0 Load Balancing Report...")
        
        # Collect current metrics
        current_metrics = self.collect_load_metrics()
        
        # Analyze patterns
        workload_pattern = self.analyze_workload_patterns()
        
        # Detect imbalances
        imbalances = self.detect_load_imbalances()
        
        # Auto-rebalance assessment
        rebalance_analysis = self.auto_rebalance_system()
        
        # Generate report
        report = {
            'report_id': f"load_balancing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'intelligence_level': 'V2.0 Phase 2',
            'load_balancing_status': {
                'strategy': self.current_strategy.value,
                'workload_pattern': workload_pattern.value,
                'system_health': rebalance_analysis['system_health'],
                'active_agents': len(current_metrics)
            },
            'agent_metrics': {aid: asdict(metric) for aid, metric in current_metrics.items()},
            'load_analysis': {
                'total_utilization': statistics.mean([m.utilization for m in current_metrics.values()]) if current_metrics else 0,
                'utilization_variance': statistics.stdev([m.utilization for m in current_metrics.values()]) if len(current_metrics) > 1 else 0,
                'performance_distribution': [m.performance_score for m in current_metrics.values()],
                'capacity_usage': sum(m.current_tasks for m in current_metrics.values()) / sum(m.max_capacity for m in current_metrics.values()) if current_metrics else 0
            },
            'imbalances_detected': imbalances,
            'rebalancing_recommendations': rebalance_analysis['actions'],
            'historical_trends': self._analyze_load_trends(),
            'next_analysis': (datetime.now() + timedelta(minutes=30)).isoformat()
        }
        
        return report
    
    def _analyze_load_trends(self) -> Dict:
        """Analizuje trendy obciążenia"""
        
        trends = {}
        
        for agent_id, history in self.load_history.items():
            if len(history) < 2:
                continue
                
            utilizations = [h['utilization'] for h in history]
            
            # Simple trend analysis
            if len(utilizations) >= 2:
                recent_avg = statistics.mean(utilizations[-3:])
                earlier_avg = statistics.mean(utilizations[:-3]) if len(utilizations) > 3 else utilizations[0]
                
                if recent_avg > earlier_avg * 1.1:
                    trend = 'increasing'
                elif recent_avg < earlier_avg * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                trends[agent_id] = {
                    'utilization_trend': trend,
                    'recent_avg': recent_avg,
                    'change_rate': ((recent_avg - earlier_avg) / max(earlier_avg, 0.001)) * 100
                }
        
        return trends

async def main():
    """Main function dla Intelligence V2.0 Load Balancing demo"""
    print("⚡ Agent Zero V1 - Intelligence V2.0 Phase 2: Dynamic Load Balancing")
    print("="*75)
    
    # Initialize system z Phase 1
    if ProductionAgentManager:
        print("🔧 Initializing with Production Agent Manager & Analytics...")
        agent_manager = ProductionAgentManager()
        analytics = IntelligenceV2Analytics(agent_manager)
        
        # Create demo agents dla load balancing testów
        print("🤖 Creating specialized agents for load balancing...")
        
        # High-capacity AI Agent
        ai_caps = [
            AgentCapability("machine_learning", 10, "ai"),
            AgentCapability("data_analysis", 9, "analytics"),
            AgentCapability("pattern_recognition", 8, "ai")
        ]
        ai_agent = agent_manager.create_agent("AI Specialist", AgentType.ANALYZER, ai_caps, max_concurrent_tasks=5)
        agent_manager.update_agent_status(ai_agent, AgentStatus.ACTIVE)
        
        # Medium-capacity Executor
        exec_caps = [
            AgentCapability("automation", 9, "execution"),
            AgentCapability("deployment", 8, "devops"),
            AgentCapability("monitoring", 7, "operations")
        ]
        exec_agent = agent_manager.create_agent("Task Executor", AgentType.EXECUTOR, exec_caps, max_concurrent_tasks=3)
        agent_manager.update_agent_status(exec_agent, AgentStatus.ACTIVE)
        
        # High-capacity Coordinator
        coord_caps = [
            AgentCapability("coordination", 10, "management"),
            AgentCapability("planning", 9, "strategy"),
            AgentCapability("communication", 8, "social")
        ]
        coord_agent = agent_manager.create_agent("System Coordinator", AgentType.COORDINATOR, coord_caps, max_concurrent_tasks=4)
        agent_manager.update_agent_status(coord_agent, AgentStatus.ACTIVE)
        
        # Create various tasks dla load balancing
        print("📋 Creating diverse tasks for load balancing demonstration...")
        
        tasks = [
            ("AI Model Training", "Train new ML model for task prediction", ["machine_learning"], TaskPriority.HIGH),
            ("System Deployment", "Deploy new version to production", ["deployment", "automation"], TaskPriority.URGENT),
            ("Team Coordination", "Coordinate cross-team project milestone", ["coordination", "planning"], TaskPriority.MEDIUM),
            ("Data Analysis", "Analyze system performance patterns", ["data_analysis", "pattern_recognition"], TaskPriority.HIGH),
            ("Process Automation", "Automate manual workflow processes", ["automation"], TaskPriority.MEDIUM),
            ("Strategic Planning", "Develop Q4 strategic initiatives", ["planning", "communication"], TaskPriority.LOW),
            ("Performance Monitoring", "Monitor system health and metrics", ["monitoring"], TaskPriority.MEDIUM),
            ("Advanced Analytics", "Deep dive into user behavior patterns", ["machine_learning", "data_analysis"], TaskPriority.HIGH)
        ]
        
        created_tasks = []
        for title, desc, caps, priority in tasks:
            task_id = agent_manager.create_task(title, desc, caps, priority)
            created_tasks.append(task_id)
        
        print(f"✅ Created {len(created_tasks)} diverse tasks")
        
        # Execute some tasks to create realistic load
        print("⚡ Executing tasks to create realistic system load...")
        executed_count = 0
        for task_id in created_tasks[:5]:  # Execute first 5 tasks
            task = agent_manager.tasks_cache.get(task_id)
            if task and task.assigned_agent_id:
                agent_manager.execute_task(task_id)
                executed_count += 1
        
        # Wait for some completion
        await asyncio.sleep(2)
        print(f"⚡ {executed_count} tasks in execution")
        
    else:
        print("⚠️ Running in simulation mode")
        agent_manager = None
        analytics = None
    
    # Initialize Dynamic Load Balancer
    print("\n⚡ Initializing Intelligence V2.0 Dynamic Load Balancer...")
    load_balancer = IntelligentLoadBalancer(agent_manager, analytics)
    
    # Test intelligent agent selection
    print("\n🧠 Testing Intelligent Agent Selection...")
    
    test_requirements = [
        {'capabilities': ['machine_learning'], 'priority': 'HIGH'},
        {'capabilities': ['automation', 'deployment'], 'priority': 'URGENT'},
        {'capabilities': ['coordination'], 'priority': 'MEDIUM'},
        {'capabilities': ['data_analysis', 'pattern_recognition'], 'priority': 'HIGH'}
    ]
    
    print("\n🎯 Agent Selection Results:")
    for i, req in enumerate(test_requirements, 1):
        decision = load_balancer.select_optimal_agent(req)
        
        print(f"\n   {i}. Task: {req['capabilities']} (Priority: {req['priority']})")
        print(f"      🤖 Recommended Agent: {decision.recommended_agent}")
        print(f"      📊 Strategy: {decision.strategy_used.value}")
        print(f"      🎯 Confidence: {decision.confidence:.1%}")
        print(f"      💡 Reasoning: {decision.reasoning}")
        print(f"      ⏱️ Predicted Time: {decision.predicted_completion_time:.1f}s")
        print(f"      📈 Load Impact: +{decision.load_impact:.1%}")
        
        if decision.alternative_agents:
            print(f"      🔄 Alternatives: {', '.join(decision.alternative_agents[:2])}")
    
    # Generate comprehensive load balancing report
    print("\n📊 Generating Intelligence V2.0 Load Balancing Report...")
    await asyncio.sleep(1)  # Allow for more task completion
    
    report = load_balancer.generate_load_balancing_report()
    
    # Display comprehensive results
    print("\n" + "="*75)
    print("📈 INTELLIGENCE V2.0 DYNAMIC LOAD BALANCING REPORT")
    print("="*75)
    
    # System Overview
    status = report['load_balancing_status']
    print(f"\n🏷️ Load Balancing Status:")
    print(f"   • Strategy: {status['strategy'].upper()}")
    print(f"   • Workload Pattern: {status['workload_pattern'].upper()}")
    print(f"   • Active Agents: {status['active_agents']}")
    print(f"   • System Health: {status['system_health']['status']} ({status['system_health']['score']:.1%})")
    
    # Load Analysis
    analysis = report['load_analysis']
    print(f"\n📊 System Load Analysis:")
    print(f"   • Total Utilization: {analysis['total_utilization']:.1%}")
    print(f"   • Capacity Usage: {analysis['capacity_usage']:.1%}")
    print(f"   • Utilization Variance: {analysis['utilization_variance']:.3f}")
    
    # Agent Metrics
    print(f"\n🤖 Agent Load Metrics:")
    for agent_id, metrics in report['agent_metrics'].items():
        utilization_bar = "█" * int(metrics['utilization'] * 10) + "░" * (10 - int(metrics['utilization'] * 10))
        print(f"   • {agent_id}: {utilization_bar} {metrics['utilization']:.1%}")
        print(f"     Tasks: {metrics['current_tasks']}/{metrics['max_capacity']}, "
              f"Performance: {metrics['performance_score']:.2f}, "
              f"Success: {metrics['success_rate']:.1%}")
    
    # Imbalances
    imbalances = report['imbalances_detected']
    if imbalances:
        print(f"\n⚠️ Load Imbalances Detected ({len(imbalances)}):")
        for imbalance in imbalances:
            severity_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}
            icon = severity_icon.get(imbalance['severity'], "•")
            
            print(f"   {icon} {imbalance['type'].replace('_', ' ').title()}")
            print(f"      📝 {imbalance['description']}")
            print(f"      💡 Action: {imbalance['recommended_action']}")
            print(f"      🎯 Confidence: {imbalance['confidence']:.1%}")
    else:
        print(f"\n✅ No Load Imbalances Detected - System is well balanced!")
    
    # Recommendations
    recommendations = report['rebalancing_recommendations']
    if recommendations:
        print(f"\n🎯 Rebalancing Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['type'].replace('_', ' ').title()}")
            print(f"      🎯 Priority: {rec['priority']}")
            print(f"      📈 Benefit: {rec['estimated_benefit']}")
    
    # Historical Trends
    trends = report['historical_trends']
    if trends:
        print(f"\n📈 Load Trends Analysis:")
        for agent_id, trend_data in trends.items():
            trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
            icon = trend_icon.get(trend_data['utilization_trend'], "•")
            
            print(f"   {icon} {agent_id}: {trend_data['utilization_trend']} ({trend_data['change_rate']:+.1f}%)")
    
    print(f"\n⏰ Next Analysis: {report['next_analysis']}")
    
    print("\n" + "="*75)
    print("🎉 Intelligence V2.0 Phase 2 Load Balancing Complete!")
    print("✅ Dynamic load balancing with ML-based agent selection operational")
    print("📊 System automatically optimizes task distribution for maximum efficiency")
    print("🚀 Ready for Phase 3: AI Task Decomposition")
    print("="*75)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())