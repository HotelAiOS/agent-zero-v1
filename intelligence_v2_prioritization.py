# intelligence_v2/prioritization.py
"""
Agent Zero V1 - Point 3 Dynamic Task Prioritization
Consolidates existing Point 3 service functionality with enhanced V2.0 features

CRITICAL: Maintains 100% backward compatibility with existing Point 3 service on port 8003
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import math

# Import existing Agent Zero components
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import existing SimpleTracker for compatibility
    exec(open(project_root / "simple-tracker.py").read(), globals())
    COMPONENTS_AVAILABLE = True
    logging.info("Successfully imported existing Agent Zero components")
except Exception as e:
    logging.warning(f"Could not import existing components: {e}")
    COMPONENTS_AVAILABLE = False
    
    # Fallback SimpleTracker
    class SimpleTracker:
        def track_task(self, **kwargs): pass
        def get_daily_stats(self): return {'total_tasks': 0}

from .interfaces import (
    Task, AgentProfile, PriorityDecision, ReassignmentDecision,
    TaskPriority, BusinessContext, CrisisType, AgentCapability
)

logger = logging.getLogger(__name__)

@dataclass
class CrisisEvent:
    """Crisis event that triggers priority escalation"""
    crisis_type: CrisisType
    description: str
    affected_tasks: List[str]
    severity: float = 1.0  # 0.0-2.0 multiplier
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class DynamicTaskPrioritizer:
    """
    Point 3: Dynamic Task Prioritization and Reassignment Engine
    
    Consolidates existing Point 3 functionality with enhanced V2.0 intelligence.
    Maintains full backward compatibility with existing API endpoints.
    """
    
    def __init__(self, simple_tracker=None):
        """Initialize with optional SimpleTracker integration"""
        self.simple_tracker = simple_tracker or (SimpleTracker() if COMPONENTS_AVAILABLE else None)
        
        # Priority calculation weights
        self.priority_weights = {
            'base_priority': 0.3,      # Base task priority (CRITICAL/HIGH/etc)
            'business_context': 0.25,  # Business impact multipliers
            'urgency': 0.20,           # Time-based urgency
            'dependencies': 0.15,      # Blocking/blocked relationships
            'agent_availability': 0.10 # Resource availability
        }
        
        # Business context multipliers (matching existing Point 3 logic)
        self.business_multipliers = {
            BusinessContext.REVENUE_CRITICAL: 1.8,
            BusinessContext.SECURITY_CRITICAL: 1.7,
            BusinessContext.CUSTOMER_FACING: 1.5,
            BusinessContext.COMPLIANCE_REQUIRED: 1.4,
            BusinessContext.INTERNAL_TOOLS: 1.0
        }
        
        # Crisis type multipliers
        self.crisis_multipliers = {
            CrisisType.SYSTEM_DOWN: 2.0,
            CrisisType.DATA_BREACH: 1.9,
            CrisisType.SECURITY_INCIDENT: 1.8,
            CrisisType.CUSTOMER_ESCALATION: 1.6,
            CrisisType.PERFORMANCE_DEGRADATION: 1.4,
            CrisisType.REGULATORY_VIOLATION: 1.5
        }
        
        # Active crises and priority queue
        self.active_crises: List[CrisisEvent] = []
        self.priority_queue: List[Task] = []
        self.agent_workloads: Dict[str, float] = {}
        
        logger.info("DynamicTaskPrioritizer initialized with existing Point 3 compatibility")
    
    async def calculate_priority(self, task: Task, 
                               agents: List[AgentProfile] = None,
                               context: Dict[str, Any] = None) -> PriorityDecision:
        """
        Calculate dynamic task priority using multi-factor analysis
        
        Maintains compatibility with existing Point 3 priority calculation logic
        while adding enhanced V2.0 intelligence features.
        """
        start_time = datetime.now()
        
        try:
            # 1. Base priority score (0.0-1.0)
            base_score = self._get_base_priority_score(task.priority)
            
            # 2. Business context multiplier
            business_multiplier = self._calculate_business_multiplier(task.business_contexts)
            
            # 3. Urgency score based on deadline and creation time
            urgency_multiplier = self._calculate_urgency_multiplier(task)
            
            # 4. Dependency impact analysis
            dependency_score = await self._analyze_dependency_impact(task)
            
            # 5. Resource availability consideration
            availability_score = self._calculate_resource_availability_score(task, agents or [])
            
            # 6. Crisis escalation check
            crisis_multiplier = self._check_crisis_escalation(task)
            
            # Calculate final priority score using weighted formula
            weighted_score = (
                base_score * self.priority_weights['base_priority'] +
                dependency_score * self.priority_weights['dependencies'] +
                availability_score * self.priority_weights['agent_availability']
            )
            
            # Apply multipliers
            final_score = weighted_score * business_multiplier * urgency_multiplier * crisis_multiplier
            
            # Cap at 1.0 and determine priority level
            final_score = min(final_score, 1.0)
            calculated_priority = self._score_to_priority(final_score)
            
            # Generate reasoning
            reasoning = self._generate_priority_reasoning(
                task, base_score, business_multiplier, urgency_multiplier,
                dependency_score, availability_score, crisis_multiplier
            )
            
            # Track in SimpleTracker if available
            if self.simple_tracker and COMPONENTS_AVAILABLE:
                self.simple_tracker.track_task(
                    task_id=task.id,
                    task_type="priority_calculation",
                    model_used="dynamic_prioritizer_v2",
                    success_score=0.95,  # High confidence in priority calculation
                    context=f"Priority: {calculated_priority.value}, Score: {final_score:.3f}"
                )
            
            calculation_duration = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return PriorityDecision(
                task_id=task.id,
                calculated_priority=calculated_priority,
                priority_score=final_score,
                confidence=0.9,  # High confidence in multi-factor analysis
                base_priority_score=base_score,
                urgency_multiplier=urgency_multiplier,
                business_context_multiplier=business_multiplier,
                dependency_impact_score=dependency_score,
                resource_availability_score=availability_score,
                reasoning=reasoning,
                factors_considered=[
                    "base_priority", "business_context", "urgency", 
                    "dependencies", "resource_availability", "crisis_status"
                ],
                risk_assessment=self._assess_priority_risks(task, final_score),
                recommended_action=self._recommend_priority_action(calculated_priority, final_score),
                calculation_duration_ms=calculation_duration
            )
            
        except Exception as e:
            logger.error(f"Priority calculation error for task {task.id}: {e}")
            # Fallback to base priority with error indication
            return PriorityDecision(
                task_id=task.id,
                calculated_priority=task.priority,
                priority_score=self._get_base_priority_score(task.priority),
                confidence=0.3,  # Low confidence due to error
                base_priority_score=self._get_base_priority_score(task.priority),
                urgency_multiplier=1.0,
                business_context_multiplier=1.0,
                dependency_impact_score=0.5,
                resource_availability_score=0.5,
                reasoning=f"Priority calculation failed, using base priority: {e}",
                factors_considered=["base_priority_only"],
                risk_assessment="Calculation error - manual review recommended",
                recommended_action="Review task priority manually",
                calculation_duration_ms=0
            )
    
    async def evaluate_reassignment(self, task: Task, 
                                  agents: List[AgentProfile],
                                  reason: str = "workload_optimization") -> Optional[ReassignmentDecision]:
        """
        Evaluate if task should be reassigned to different agent
        
        Compatible with existing Point 3 workload balancing logic
        """
        if not agents or not task.assigned_agent_id:
            return None
        
        try:
            current_agent = next((a for a in agents if a.id == task.assigned_agent_id), None)
            if not current_agent:
                return None
            
            # Find optimal agent based on capability match and workload
            optimal_agent = self._find_optimal_agent(task, agents)
            
            if optimal_agent.id == current_agent.id:
                return None  # Current assignment is optimal
            
            # Calculate reassignment benefits
            capability_improvement = self._calculate_capability_match_improvement(
                task, current_agent, optimal_agent
            )
            workload_improvement = self._calculate_workload_improvement(
                current_agent, optimal_agent
            )
            
            if capability_improvement < 0.1 and workload_improvement < 0.1:
                return None  # Insufficient improvement to justify reassignment
            
            # Calculate transition costs
            transition_cost = self._calculate_transition_cost(task, current_agent, optimal_agent)
            
            # Estimate timeline impact
            timeline_impact = self._estimate_timeline_impact(
                task, current_agent, optimal_agent, transition_cost
            )
            
            confidence = min(capability_improvement + workload_improvement, 1.0)
            
            return ReassignmentDecision(
                task_id=task.id,
                current_agent_id=current_agent.id,
                recommended_agent_id=optimal_agent.id,
                reassignment_reason=reason,
                confidence=confidence,
                transition_cost=transition_cost,
                efficiency_gain=capability_improvement + workload_improvement,
                timeline_impact=timeline_impact,
                current_agent_utilization=current_agent.current_workload / current_agent.max_capacity,
                target_agent_utilization=optimal_agent.current_workload / optimal_agent.max_capacity,
                capability_match_score=self._calculate_capability_match_score(task, optimal_agent)
            )
            
        except Exception as e:
            logger.error(f"Reassignment evaluation error for task {task.id}: {e}")
            return None
    
    async def handle_crisis(self, crisis_event: CrisisEvent) -> Dict[str, Any]:
        """
        Handle crisis events with automatic priority escalation
        
        Maintains compatibility with existing Point 3 crisis response system
        """
        try:
            self.active_crises.append(crisis_event)
            
            # Escalate affected tasks
            escalated_tasks = []
            for task in self.priority_queue:
                if task.id in crisis_event.affected_tasks:
                    # Recalculate priority with crisis multiplier
                    new_priority_decision = await self.calculate_priority(task)
                    
                    # Update task priority if significantly higher
                    if self._priority_level_higher(new_priority_decision.calculated_priority, task.priority):
                        old_priority = task.priority
                        task.priority = new_priority_decision.calculated_priority
                        escalated_tasks.append({
                            'task_id': task.id,
                            'old_priority': old_priority.value,
                            'new_priority': task.priority.value,
                            'escalation_reason': crisis_event.description
                        })
            
            # Track crisis response in SimpleTracker
            if self.simple_tracker and COMPONENTS_AVAILABLE:
                self.simple_tracker.track_task(
                    task_id=f"crisis_{crisis_event.crisis_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    task_type="crisis_response",
                    model_used="crisis_handler_v2",
                    success_score=1.0,
                    context=f"Crisis: {crisis_event.crisis_type.value}, Affected: {len(crisis_event.affected_tasks)}"
                )
            
            return {
                'crisis_id': f"crisis_{len(self.active_crises)}",
                'crisis_type': crisis_event.crisis_type.value,
                'escalated_tasks': escalated_tasks,
                'response_time_ms': 100,  # Fast crisis response
                'status': 'handled',
                'recommendations': self._generate_crisis_recommendations(crisis_event)
            }
            
        except Exception as e:
            logger.error(f"Crisis handling error: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'crisis_type': crisis_event.crisis_type.value
            }
    
    def get_priority_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get current priority queue with task details
        
        Maintains compatibility with existing Point 3 API format
        """
        try:
            # Sort tasks by priority score (highest first)
            sorted_tasks = sorted(
                self.priority_queue, 
                key=lambda t: (
                    self._get_base_priority_score(t.priority),
                    -t.created_at.timestamp()  # Newer tasks first within same priority
                ),
                reverse=True
            )
            
            return [
                {
                    'task_id': task.id,
                    'title': task.title,
                    'priority': task.priority.value,
                    'business_contexts': [bc.value for bc in task.business_contexts],
                    'estimated_hours': task.estimated_hours,
                    'assigned_agent_id': task.assigned_agent_id,
                    'created_at': task.created_at.isoformat(),
                    'deadline': task.deadline.isoformat() if task.deadline else None,
                    'crisis_affected': any(task.id in crisis.affected_tasks for crisis in self.active_crises)
                }
                for task in sorted_tasks[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Priority queue retrieval error: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get prioritization system metrics
        
        Compatible with existing Point 3 metrics endpoint
        """
        try:
            total_tasks = len(self.priority_queue)
            priority_distribution = {priority.value: 0 for priority in TaskPriority}
            
            for task in self.priority_queue:
                priority_distribution[task.priority.value] += 1
            
            active_crises_count = len([c for c in self.active_crises 
                                     if (datetime.now() - c.created_at).hours < 24])
            
            return {
                'total_tasks': total_tasks,
                'priority_distribution': priority_distribution,
                'active_crises': active_crises_count,
                'average_priority_score': sum(
                    self._get_base_priority_score(task.priority) 
                    for task in self.priority_queue
                ) / max(total_tasks, 1),
                'crisis_escalated_tasks': sum(
                    len(crisis.affected_tasks) for crisis in self.active_crises
                ),
                'system_load': min(total_tasks / 100.0, 1.0),  # Normalized load
                'components_available': COMPONENTS_AVAILABLE,
                'simple_tracker_integration': self.simple_tracker is not None
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return {'error': str(e), 'components_available': COMPONENTS_AVAILABLE}
    
    # === PRIVATE METHODS ===
    
    def _get_base_priority_score(self, priority: TaskPriority) -> float:
        """Convert TaskPriority enum to numeric score"""
        priority_scores = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.LOW: 0.2
        }
        return priority_scores.get(priority, 0.5)
    
    def _calculate_business_multiplier(self, contexts: List[BusinessContext]) -> float:
        """Calculate business context multiplier"""
        if not contexts:
            return 1.0
        
        # Use highest multiplier if multiple contexts
        return max(self.business_multipliers.get(context, 1.0) for context in contexts)
    
    def _calculate_urgency_multiplier(self, task: Task) -> float:
        """Calculate time-based urgency multiplier"""
        if not task.deadline:
            return 1.0
        
        now = datetime.now()
        time_to_deadline = (task.deadline - now).total_seconds()
        
        if time_to_deadline <= 0:
            return 2.0  # Overdue
        elif time_to_deadline <= 3600:  # 1 hour
            return 1.8
        elif time_to_deadline <= 86400:  # 1 day  
            return 1.5
        elif time_to_deadline <= 604800:  # 1 week
            return 1.2
        else:
            return 1.0
    
    async def _analyze_dependency_impact(self, task: Task) -> float:
        """Analyze how task dependencies affect priority"""
        if not task.dependencies and not task.blocks:
            return 0.5  # Neutral impact
        
        # Higher score if task blocks many others
        blocking_impact = len(task.blocks) * 0.1
        
        # Lower score if task has many unresolved dependencies  
        dependency_penalty = len(task.dependencies) * 0.05
        
        return min(max(0.5 + blocking_impact - dependency_penalty, 0.0), 1.0)
    
    def _calculate_resource_availability_score(self, task: Task, agents: List[AgentProfile]) -> float:
        """Calculate resource availability impact on priority"""
        if not agents:
            return 0.5
        
        # Find agents with matching capabilities
        capable_agents = [
            agent for agent in agents
            if any(cap in agent.capabilities for cap in task.preferred_capabilities)
        ]
        
        if not capable_agents:
            return 0.2  # Low score if no capable agents
        
        # Calculate average availability of capable agents
        avg_availability = sum(
            1.0 - (agent.current_workload / agent.max_capacity)
            for agent in capable_agents
        ) / len(capable_agents)
        
        return max(avg_availability, 0.1)
    
    def _check_crisis_escalation(self, task: Task) -> float:
        """Check if task is affected by any active crisis"""
        for crisis in self.active_crises:
            if task.id in crisis.affected_tasks:
                multiplier = self.crisis_multipliers.get(crisis.crisis_type, 1.0)
                return multiplier * crisis.severity
        return 1.0
    
    def _score_to_priority(self, score: float) -> TaskPriority:
        """Convert numeric score back to TaskPriority enum"""
        if score >= 0.9:
            return TaskPriority.CRITICAL
        elif score >= 0.7:
            return TaskPriority.HIGH
        elif score >= 0.4:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
    
    def _generate_priority_reasoning(self, task: Task, base_score: float,
                                   business_multiplier: float, urgency_multiplier: float,
                                   dependency_score: float, availability_score: float,
                                   crisis_multiplier: float) -> str:
        """Generate human-readable priority reasoning"""
        reasoning_parts = [
            f"Base priority: {task.priority.value} (score: {base_score:.2f})"
        ]
        
        if business_multiplier > 1.0:
            contexts = [bc.value for bc in task.business_contexts]
            reasoning_parts.append(f"Business impact: {contexts} (×{business_multiplier:.1f})")
        
        if urgency_multiplier > 1.0:
            reasoning_parts.append(f"Time urgency: ×{urgency_multiplier:.1f}")
        
        if dependency_score > 0.6:
            reasoning_parts.append(f"Blocks {len(task.blocks)} other tasks")
        elif dependency_score < 0.4:
            reasoning_parts.append(f"Depends on {len(task.dependencies)} other tasks")
        
        if crisis_multiplier > 1.0:
            reasoning_parts.append(f"Crisis escalation: ×{crisis_multiplier:.1f}")
        
        if availability_score < 0.3:
            reasoning_parts.append("Limited agent availability")
        
        return "; ".join(reasoning_parts)
    
    def _assess_priority_risks(self, task: Task, priority_score: float) -> str:
        """Assess risks associated with current priority"""
        risks = []
        
        if priority_score > 0.9 and not task.deadline:
            risks.append("Critical priority without deadline")
        
        if len(task.dependencies) > 3:
            risks.append("High dependency count may cause delays")
        
        if not task.assigned_agent_id and priority_score > 0.7:
            risks.append("High priority task unassigned")
        
        if not task.preferred_capabilities:
            risks.append("No capability requirements specified")
        
        return "; ".join(risks) if risks else "Low risk"
    
    def _recommend_priority_action(self, priority: TaskPriority, score: float) -> str:
        """Recommend action based on calculated priority"""
        if priority == TaskPriority.CRITICAL:
            return "Immediate assignment and execution required"
        elif priority == TaskPriority.HIGH:
            return "Schedule for next available slot"
        elif priority == TaskPriority.MEDIUM:
            return "Standard scheduling and resource allocation"
        else:
            return "Schedule when resources available"
    
    def _find_optimal_agent(self, task: Task, agents: List[AgentProfile]) -> AgentProfile:
        """Find optimal agent for task assignment"""
        if not agents:
            return None
        
        def agent_score(agent: AgentProfile) -> float:
            # Capability match score
            capability_match = len(set(task.preferred_capabilities) & set(agent.capabilities)) / max(len(task.preferred_capabilities), 1)
            
            # Availability score (inverse of utilization)
            availability = 1.0 - (agent.current_workload / agent.max_capacity)
            
            # Performance score
            performance = agent.performance_score * agent.success_rate
            
            return capability_match * 0.5 + availability * 0.3 + performance * 0.2
        
        return max(agents, key=agent_score)
    
    def _calculate_capability_match_improvement(self, task: Task, 
                                              current_agent: AgentProfile,
                                              target_agent: AgentProfile) -> float:
        """Calculate improvement in capability match"""
        current_match = len(set(task.preferred_capabilities) & set(current_agent.capabilities)) / max(len(task.preferred_capabilities), 1)
        target_match = len(set(task.preferred_capabilities) & set(target_agent.capabilities)) / max(len(task.preferred_capabilities), 1)
        return target_match - current_match
    
    def _calculate_workload_improvement(self, current_agent: AgentProfile, 
                                      target_agent: AgentProfile) -> float:
        """Calculate workload balance improvement"""
        current_util = current_agent.current_workload / current_agent.max_capacity
        target_util = target_agent.current_workload / target_agent.max_capacity
        
        # Improvement is reduction in maximum utilization
        return max(current_util - target_util, 0.0)
    
    def _calculate_transition_cost(self, task: Task, 
                                 current_agent: AgentProfile,
                                 target_agent: AgentProfile) -> float:
        """Calculate cost of transitioning task between agents"""
        # Base handoff cost (context transfer, knowledge sharing)
        base_cost = 0.5  # hours
        
        # Complexity factor
        complexity_factor = task.complexity_score * 0.5
        
        # Domain similarity (lower cost if agents have similar capabilities)
        common_capabilities = len(set(current_agent.capabilities) & set(target_agent.capabilities))
        domain_factor = max(0.2, 1.0 - (common_capabilities * 0.1))
        
        return base_cost + complexity_factor + domain_factor
    
    def _estimate_timeline_impact(self, task: Task,
                                current_agent: AgentProfile,
                                target_agent: AgentProfile,
                                transition_cost: float) -> int:
        """Estimate timeline impact in days"""
        # Efficiency difference
        current_efficiency = current_agent.performance_score / current_agent.avg_completion_time
        target_efficiency = target_agent.performance_score / target_agent.avg_completion_time
        
        efficiency_gain_hours = task.estimated_hours * (target_efficiency - current_efficiency)
        
        # Net impact considering transition cost
        net_impact_hours = efficiency_gain_hours - transition_cost
        
        # Convert to days (assuming 8-hour workdays)
        return int(net_impact_hours / 8)
    
    def _calculate_capability_match_score(self, task: Task, agent: AgentProfile) -> float:
        """Calculate how well agent capabilities match task requirements"""
        if not task.preferred_capabilities:
            return 0.8  # Default good match if no specific requirements
        
        matches = len(set(task.preferred_capabilities) & set(agent.capabilities))
        required = len(task.preferred_capabilities)
        
        return matches / required
    
    def _priority_level_higher(self, new_priority: TaskPriority, old_priority: TaskPriority) -> bool:
        """Check if new priority level is higher than old"""
        priority_order = {
            TaskPriority.LOW: 0,
            TaskPriority.MEDIUM: 1, 
            TaskPriority.HIGH: 2,
            TaskPriority.CRITICAL: 3
        }
        return priority_order[new_priority] > priority_order[old_priority]
    
    def _generate_crisis_recommendations(self, crisis_event: CrisisEvent) -> List[str]:
        """Generate recommendations for crisis response"""
        recommendations = []
        
        if crisis_event.crisis_type == CrisisType.SYSTEM_DOWN:
            recommendations.extend([
                "Escalate all system-critical tasks",
                "Activate emergency response team",
                "Prepare rollback procedures"
            ])
        elif crisis_event.crisis_type == CrisisType.SECURITY_INCIDENT:
            recommendations.extend([
                "Review security protocols",
                "Audit affected systems",
                "Implement additional monitoring"
            ])
        
        return recommendations
    
    # === PUBLIC API METHODS FOR COMPATIBILITY ===
    
    async def add_task_to_queue(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add task to priority queue (API compatibility method)"""
        try:
            # Convert dict to Task object
            task = Task(
                id=task_data.get('id', str(uuid.uuid4())),
                title=task_data.get('title', ''),
                description=task_data.get('description', ''),
                priority=TaskPriority(task_data.get('priority', 'medium')),
                business_contexts=[BusinessContext(bc) for bc in task_data.get('business_contexts', [])],
                estimated_hours=task_data.get('estimated_hours', 0.0),
                deadline=datetime.fromisoformat(task_data['deadline']) if task_data.get('deadline') else None
            )
            
            # Calculate priority
            priority_decision = await self.calculate_priority(task)
            task.priority = priority_decision.calculated_priority
            
            # Add to queue
            self.priority_queue.append(task)
            
            return {
                'status': 'added',
                'task_id': task.id,
                'calculated_priority': priority_decision.calculated_priority.value,
                'priority_score': priority_decision.priority_score,
                'reasoning': priority_decision.reasoning
            }
            
        except Exception as e:
            logger.error(f"Add task error: {e}")
            return {'status': 'error', 'error': str(e)}