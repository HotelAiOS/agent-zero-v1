"""
Agent Zero V1 - Context-Aware Agent Selection & Task Assignment
Point 2 of 6 Critical AI Features - Week 43 Implementation

Inteligentny system wyboru agentów na podstawie:
- Task context i requirements
- Technology stack expertise  
- Agent availability i current workload
- Historical performance
- Project priorities
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math

# Import our existing components
try:
    from nlp_enhanced_task_decomposer import (
        NLUTaskDecomposer, DomainContext, TaskBreakdown, 
        Task, TaskType, TaskPriority
    )
    from team_formation import TeamFormationEngine, Team, TeamMember
    from task_decomposer import TaskDependency
except ImportError:
    # Fallback imports for standalone testing
    from dataclasses import dataclass
    from enum import Enum
    
    class TaskType(Enum):
        FRONTEND = "frontend"
        BACKEND = "backend"
        DATABASE = "database"
        DEVOPS = "devops"
        TESTING = "testing"
        ARCHITECTURE = "architecture"
    
    class TaskPriority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    proficiency_level: float  # 0.0 - 1.0
    years_experience: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 1.0  # Historical success rate

@dataclass
class AgentProfile:
    """Complete agent profile for selection"""
    agent_id: str
    agent_type: str
    primary_expertise: List[str]
    capabilities: List[AgentCapability]
    current_workload: float = 0.0
    max_workload: float = 40.0
    availability_score: float = 1.0  # 0.0 - 1.0
    performance_history: Dict[str, float] = field(default_factory=dict)
    technology_expertise: Dict[str, float] = field(default_factory=dict)  # tech -> score
    preferred_task_types: List[TaskType] = field(default_factory=list)
    collaboration_score: float = 0.8  # How well works with others
    location: str = "remote"
    time_zone: str = "UTC"
    
    def get_capability_score(self, capability_name: str) -> float:
        """Get proficiency score for a capability"""
        for cap in self.capabilities:
            if cap.name.lower() == capability_name.lower():
                return cap.proficiency_level
        return 0.0
    
    def get_technology_score(self, technology: str) -> float:
        """Get expertise score for a technology"""
        return self.technology_expertise.get(technology, 0.0)
    
    def get_availability(self) -> float:
        """Calculate current availability (0.0 - 1.0)"""
        utilization = self.current_workload / self.max_workload
        return max(0.0, 1.0 - utilization)
    
    def get_task_type_affinity(self, task_type: TaskType) -> float:
        """Get affinity score for a task type"""
        if task_type in self.preferred_task_types:
            return 1.0
        elif task_type.value == self.agent_type:
            return 0.9
        else:
            return 0.5

@dataclass
class TaskAssignment:
    """Task assignment with scoring"""
    task: Task
    assigned_agent: AgentProfile
    assignment_score: float
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    estimated_completion: Optional[datetime] = None
    dependencies_met: bool = True

@dataclass
class TeamComposition:
    """Intelligent team composition result"""
    team_id: str
    project_id: str
    selected_agents: List[AgentProfile]
    task_assignments: List[TaskAssignment]
    team_score: float
    coverage_analysis: Dict[str, float]
    potential_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_timeline: Dict[str, Any] = field(default_factory=dict)

class SelectionStrategy(Enum):
    """Agent selection strategies"""
    BALANCED = "balanced"  # Balance between expertise and availability
    EXPERTISE_FIRST = "expertise_first"  # Prioritize expertise over availability
    AVAILABILITY_FIRST = "availability_first"  # Prioritize availability
    PERFORMANCE_BASED = "performance_based"  # Based on historical performance
    COST_OPTIMIZED = "cost_optimized"  # Optimize for cost
    COLLABORATIVE = "collaborative"  # Optimize for team collaboration

@dataclass
class SelectionContext:
    """Context for agent selection"""
    project_priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[datetime] = None
    budget_constraints: Optional[float] = None
    required_skills: List[str] = field(default_factory=list)
    preferred_location: Optional[str] = None
    strategy: SelectionStrategy = SelectionStrategy.BALANCED
    allow_overallocation: bool = False
    max_team_size: int = 10

class ContextAwareAgentSelector:
    """
    Context-Aware Agent Selection Engine
    Inteligentnie wybiera agentów na podstawie kontekstu zadań i projektów
    """
    
    def __init__(self, nlu_decomposer: Optional[NLUTaskDecomposer] = None):
        self.nlu_decomposer = nlu_decomposer or NLUTaskDecomposer()
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.selection_history: List[Dict[str, Any]] = []
        self.performance_tracker = PerformanceTracker()
        logger.info("🧠 Context-Aware Agent Selector initialized")
    
    def register_agent(self, profile: AgentProfile):
        """Register an agent profile"""
        self.agent_profiles[profile.agent_id] = profile
        logger.info(f"📝 Registered agent {profile.agent_id} ({profile.agent_type})")
    
    def populate_demo_agents(self):
        """Populate with demo agent profiles for testing"""
        demo_agents = [
            # Backend Specialists
            AgentProfile(
                agent_id="agent_backend_001",
                agent_type="backend",
                primary_expertise=["Python", "FastAPI", "PostgreSQL"],
                capabilities=[
                    AgentCapability("Python", 0.95, 5, datetime.now(), 0.92),
                    AgentCapability("FastAPI", 0.90, 3, datetime.now(), 0.95),
                    AgentCapability("PostgreSQL", 0.85, 4, datetime.now(), 0.88),
                    AgentCapability("Docker", 0.80, 2, datetime.now(), 0.90),
                ],
                technology_expertise={"FastAPI": 0.95, "Python": 0.95, "PostgreSQL": 0.85, "Docker": 0.80},
                preferred_task_types=[TaskType.BACKEND, TaskType.ARCHITECTURE],
                performance_history={"last_month": 0.92, "last_quarter": 0.90},
                current_workload=20.0,
                collaboration_score=0.85
            ),
            AgentProfile(
                agent_id="agent_backend_002",
                agent_type="backend",
                primary_expertise=["Node.js", "Express", "MongoDB"],
                capabilities=[
                    AgentCapability("Node.js", 0.88, 4, datetime.now(), 0.89),
                    AgentCapability("Express", 0.85, 3, datetime.now(), 0.91),
                    AgentCapability("MongoDB", 0.82, 3, datetime.now(), 0.87),
                ],
                technology_expertise={"Node.js": 0.88, "Express": 0.85, "MongoDB": 0.82},
                preferred_task_types=[TaskType.BACKEND],
                performance_history={"last_month": 0.88, "last_quarter": 0.89},
                current_workload=35.0,
                collaboration_score=0.90
            ),
            
            # Frontend Specialists
            AgentProfile(
                agent_id="agent_frontend_001",
                agent_type="frontend",
                primary_expertise=["React", "TypeScript", "Material-UI"],
                capabilities=[
                    AgentCapability("React", 0.92, 4, datetime.now(), 0.94),
                    AgentCapability("TypeScript", 0.88, 3, datetime.now(), 0.91),
                    AgentCapability("CSS", 0.85, 5, datetime.now(), 0.88),
                    AgentCapability("JavaScript", 0.90, 5, datetime.now(), 0.92),
                ],
                technology_expertise={"React": 0.92, "TypeScript": 0.88, "JavaScript": 0.90},
                preferred_task_types=[TaskType.FRONTEND],
                performance_history={"last_month": 0.91, "last_quarter": 0.93},
                current_workload=15.0,
                collaboration_score=0.88
            ),
            
            # DevOps Specialist
            AgentProfile(
                agent_id="agent_devops_001",
                agent_type="devops",
                primary_expertise=["Docker", "Kubernetes", "AWS"],
                capabilities=[
                    AgentCapability("Docker", 0.95, 4, datetime.now(), 0.96),
                    AgentCapability("Kubernetes", 0.88, 3, datetime.now(), 0.89),
                    AgentCapability("AWS", 0.85, 4, datetime.now(), 0.91),
                    AgentCapability("Terraform", 0.80, 2, datetime.now(), 0.87),
                ],
                technology_expertise={"Docker": 0.95, "Kubernetes": 0.88, "AWS": 0.85},
                preferred_task_types=[TaskType.DEVOPS, TaskType.ARCHITECTURE],
                performance_history={"last_month": 0.94, "last_quarter": 0.91},
                current_workload=25.0,
                collaboration_score=0.82
            ),
            
            # Database Expert
            AgentProfile(
                agent_id="agent_database_001",
                agent_type="database",
                primary_expertise=["PostgreSQL", "Neo4j", "Redis"],
                capabilities=[
                    AgentCapability("PostgreSQL", 0.93, 6, datetime.now(), 0.95),
                    AgentCapability("Neo4j", 0.87, 2, datetime.now(), 0.88),
                    AgentCapability("Redis", 0.85, 3, datetime.now(), 0.90),
                    AgentCapability("SQL Optimization", 0.91, 5, datetime.now(), 0.93),
                ],
                technology_expertise={"PostgreSQL": 0.93, "Neo4j": 0.87, "Redis": 0.85},
                preferred_task_types=[TaskType.DATABASE, TaskType.ARCHITECTURE],
                performance_history={"last_month": 0.92, "last_quarter": 0.94},
                current_workload=18.0,
                collaboration_score=0.75
            ),
            
            # Testing Specialist
            AgentProfile(
                agent_id="agent_tester_001",
                agent_type="testing",
                primary_expertise=["Pytest", "Selenium", "Jest"],
                capabilities=[
                    AgentCapability("Pytest", 0.90, 3, datetime.now(), 0.92),
                    AgentCapability("Selenium", 0.85, 4, datetime.now(), 0.89),
                    AgentCapability("Jest", 0.82, 2, datetime.now(), 0.87),
                    AgentCapability("Test Automation", 0.88, 3, datetime.now(), 0.90),
                ],
                technology_expertise={"Pytest": 0.90, "Selenium": 0.85, "Jest": 0.82},
                preferred_task_types=[TaskType.TESTING],
                performance_history={"last_month": 0.89, "last_quarter": 0.88},
                current_workload=12.0,
                collaboration_score=0.95
            ),
        ]
        
        for agent in demo_agents:
            self.register_agent(agent)
        
        logger.info(f"✅ Populated {len(demo_agents)} demo agents")
    
    async def intelligent_team_selection(
        self, 
        project_description: str,
        domain_context: DomainContext,
        selection_context: SelectionContext
    ) -> TeamComposition:
        """
        Main intelligent team selection method
        """
        logger.info(f"🎯 Starting intelligent team selection for: {project_description[:50]}...")
        
        # Step 1: Analyze project with NLU
        task_breakdown = await self.nlu_decomposer.enhanced_decompose(
            project_description, domain_context
        )
        
        # Step 2: Analyze requirements from tasks
        requirements = self._analyze_task_requirements(task_breakdown.subtasks)
        logger.info(f"📊 Task requirements: {len(requirements)} agent types needed")
        
        # Step 3: Select optimal agents
        selected_agents = await self._select_optimal_agents(
            requirements, selection_context
        )
        
        # Step 4: Assign tasks to selected agents
        task_assignments = self._assign_tasks_intelligently(
            task_breakdown.subtasks, selected_agents, selection_context
        )
        
        # Step 5: Analyze team composition
        team_score, coverage_analysis, risks, recommendations = self._analyze_team_composition(
            selected_agents, task_assignments, requirements
        )
        
        # Step 6: Calculate timeline
        timeline = self._calculate_project_timeline(task_assignments, task_breakdown.dependencies_graph)
        
        team_composition = TeamComposition(
            team_id=f"team_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            project_id=project_description[:20].replace(" ", "_"),
            selected_agents=selected_agents,
            task_assignments=task_assignments,
            team_score=team_score,
            coverage_analysis=coverage_analysis,
            potential_risks=risks,
            recommendations=recommendations,
            estimated_timeline=timeline
        )
        
        # Store selection for learning
        self._record_selection(team_composition, task_breakdown, domain_context, selection_context)
        
        logger.info(f"✅ Team selection completed: {len(selected_agents)} agents, score: {team_score:.2f}")
        return team_composition
    
    def _analyze_task_requirements(self, tasks: List[Task]) -> Dict[str, Dict[str, Any]]:
        """Analyze requirements from task list"""
        requirements = {}
        
        for task in tasks:
            agent_type = task.required_agent_type
            
            if agent_type not in requirements:
                requirements[agent_type] = {
                    'total_hours': 0.0,
                    'task_count': 0,
                    'priority_score': 0.0,
                    'required_skills': set(),
                    'tasks': []
                }
            
            req = requirements[agent_type]
            req['total_hours'] += task.estimated_hours
            req['task_count'] += 1
            req['tasks'].append(task)
            
            # Calculate weighted priority score
            priority_weights = {
                TaskPriority.LOW: 1.0,
                TaskPriority.MEDIUM: 2.0,
                TaskPriority.HIGH: 3.0,
                TaskPriority.CRITICAL: 4.0
            }
            req['priority_score'] += priority_weights.get(task.priority, 2.0)
            
            # Extract skills from task description (simple keyword matching)
            task_text = f"{task.title} {task.description}".lower()
            for tech in ["fastapi", "react", "postgresql", "docker", "neo4j", "redis"]:
                if tech in task_text:
                    req['required_skills'].add(tech)
        
        # Convert sets to lists for JSON serialization
        for req in requirements.values():
            req['required_skills'] = list(req['required_skills'])
            req['priority_score'] /= req['task_count']  # Average priority
        
        return requirements
    
    async def _select_optimal_agents(
        self, 
        requirements: Dict[str, Dict[str, Any]], 
        context: SelectionContext
    ) -> List[AgentProfile]:
        """Select optimal agents based on requirements and context"""
        
        selected_agents = []
        
        for agent_type, req_data in requirements.items():
            logger.info(f"🔍 Selecting {agent_type} agent...")
            
            # Get candidates of this type
            candidates = [
                agent for agent in self.agent_profiles.values()
                if agent.agent_type == agent_type
            ]
            
            if not candidates:
                logger.warning(f"⚠️ No candidates found for {agent_type}")
                continue
            
            # Score each candidate
            scored_candidates = []
            for candidate in candidates:
                score = self._calculate_agent_score(
                    candidate, req_data, context
                )
                scored_candidates.append((candidate, score))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select top candidate(s)
            selected_count = min(
                1 + (req_data['total_hours'] // 40),  # 1 agent per 40 hours
                len(scored_candidates),
                context.max_team_size - len(selected_agents)
            )
            
            for i in range(selected_count):
                if scored_candidates[i][1] > 0.3:  # Minimum acceptable score
                    selected_agents.append(scored_candidates[i][0])
                    logger.info(
                        f"✅ Selected {scored_candidates[i][0].agent_id} "
                        f"(score: {scored_candidates[i][1]:.2f})"
                    )
        
        return selected_agents
    
    def _calculate_agent_score(
        self, 
        agent: AgentProfile, 
        requirements: Dict[str, Any], 
        context: SelectionContext
    ) -> float:
        """Calculate agent fitness score for requirements"""
        
        scores = {
            'expertise': 0.0,
            'availability': 0.0,
            'performance': 0.0,
            'technology_match': 0.0,
            'collaboration': 0.0
        }
        
        # Expertise score
        task_type_score = agent.get_task_type_affinity(TaskType(agent.agent_type))
        avg_capability = sum(cap.proficiency_level for cap in agent.capabilities) / max(len(agent.capabilities), 1)
        scores['expertise'] = (task_type_score + avg_capability) / 2
        
        # Availability score
        scores['availability'] = agent.get_availability()
        if not context.allow_overallocation and agent.get_availability() < 0.2:
            scores['availability'] = 0.1  # Heavily penalize overloaded agents
        
        # Performance score
        recent_performance = agent.performance_history.get('last_month', 0.8)
        scores['performance'] = recent_performance
        
        # Technology match score
        required_skills = requirements.get('required_skills', [])
        if required_skills:
            tech_scores = [agent.get_technology_score(skill) for skill in required_skills]
            scores['technology_match'] = sum(tech_scores) / len(tech_scores)
        else:
            scores['technology_match'] = 0.5
        
        # Collaboration score
        scores['collaboration'] = agent.collaboration_score
        
        # Apply strategy-based weights
        weights = self._get_strategy_weights(context.strategy)
        
        total_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        logger.debug(
            f"Agent {agent.agent_id} scores: {scores} -> total: {total_score:.2f}"
        )
        
        return total_score
    
    def _get_strategy_weights(self, strategy: SelectionStrategy) -> Dict[str, float]:
        """Get scoring weights based on selection strategy"""
        weight_sets = {
            SelectionStrategy.BALANCED: {
                'expertise': 0.25, 'availability': 0.25, 'performance': 0.2,
                'technology_match': 0.2, 'collaboration': 0.1
            },
            SelectionStrategy.EXPERTISE_FIRST: {
                'expertise': 0.4, 'availability': 0.1, 'performance': 0.2,
                'technology_match': 0.25, 'collaboration': 0.05
            },
            SelectionStrategy.AVAILABILITY_FIRST: {
                'expertise': 0.15, 'availability': 0.45, 'performance': 0.15,
                'technology_match': 0.15, 'collaboration': 0.1
            },
            SelectionStrategy.PERFORMANCE_BASED: {
                'expertise': 0.2, 'availability': 0.2, 'performance': 0.4,
                'technology_match': 0.15, 'collaboration': 0.05
            },
            SelectionStrategy.COLLABORATIVE: {
                'expertise': 0.2, 'availability': 0.2, 'performance': 0.15,
                'technology_match': 0.15, 'collaboration': 0.3
            }
        }
        return weight_sets.get(strategy, weight_sets[SelectionStrategy.BALANCED])
    
    def _assign_tasks_intelligently(
        self, 
        tasks: List[Task], 
        agents: List[AgentProfile], 
        context: SelectionContext
    ) -> List[TaskAssignment]:
        """Assign tasks to agents intelligently"""
        
        assignments = []
        
        # Create agent availability map
        agent_workloads = {agent.agent_id: agent.current_workload for agent in agents}
        
        # Sort tasks by priority and dependencies
        sorted_tasks = sorted(tasks, key=lambda t: (
            -t.priority.value.count('high'),  # High priority first
            t.id  # Then by task ID for stable sorting
        ))
        
        for task in sorted_tasks:
            # Find suitable agents for this task type
            suitable_agents = [
                agent for agent in agents
                if agent.agent_type == task.required_agent_type
            ]
            
            if not suitable_agents:
                logger.warning(f"⚠️ No suitable agent for task {task.id} ({task.task_type})")
                continue
            
            # Score each suitable agent for this specific task
            agent_scores = []
            for agent in suitable_agents:
                score = self._calculate_task_assignment_score(
                    task, agent, agent_workloads[agent.agent_id], context
                )
                agent_scores.append((agent, score))
            
            # Select best agent
            best_agent, best_score = max(agent_scores, key=lambda x: x[1])
            
            # Create assignment
            assignment = TaskAssignment(
                task=task,
                assigned_agent=best_agent,
                assignment_score=best_score,
                confidence=min(best_score, 0.95),
                reasoning=self._generate_assignment_reasoning(task, best_agent, best_score),
                estimated_completion=self._estimate_completion_time(
                    task, best_agent, agent_workloads[best_agent.agent_id]
                )
            )
            
            assignments.append(assignment)
            
            # Update agent workload
            agent_workloads[best_agent.agent_id] += task.estimated_hours
            
            logger.info(
                f"📋 Assigned task {task.id} to {best_agent.agent_id} "
                f"(score: {best_score:.2f})"
            )
        
        return assignments
    
    def _calculate_task_assignment_score(
        self, 
        task: Task, 
        agent: AgentProfile, 
        current_workload: float, 
        context: SelectionContext
    ) -> float:
        """Calculate score for assigning specific task to specific agent"""
        
        # Base compatibility score
        type_match = 1.0 if agent.agent_type == task.required_agent_type else 0.3
        
        # Workload impact
        projected_workload = current_workload + task.estimated_hours
        workload_factor = max(0.1, 1.0 - (projected_workload / agent.max_workload))
        
        # Priority alignment
        priority_factor = 1.0
        if task.priority == TaskPriority.CRITICAL:
            priority_factor = 1.2
        elif task.priority == TaskPriority.HIGH:
            priority_factor = 1.1
        
        # Skill relevance (extract from task description)
        skill_factor = self._calculate_skill_relevance(task, agent)
        
        score = type_match * workload_factor * priority_factor * skill_factor
        
        return min(score, 1.0)
    
    def _calculate_skill_relevance(self, task: Task, agent: AgentProfile) -> float:
        """Calculate how relevant agent's skills are to the task"""
        task_text = f"{task.title} {task.description}".lower()
        
        relevant_skills = 0
        total_skills = len(agent.capabilities)
        
        for capability in agent.capabilities:
            if capability.name.lower() in task_text:
                relevant_skills += capability.proficiency_level
        
        if total_skills == 0:
            return 0.5
        
        return min(relevant_skills / total_skills + 0.5, 1.0)
    
    def _generate_assignment_reasoning(
        self, task: Task, agent: AgentProfile, score: float
    ) -> List[str]:
        """Generate human-readable reasoning for assignment"""
        reasons = []
        
        if agent.agent_type == task.required_agent_type:
            reasons.append(f"Agent type matches required type ({task.required_agent_type})")
        
        if agent.get_availability() > 0.5:
            reasons.append(f"Agent has good availability ({agent.get_availability():.1%})")
        elif agent.get_availability() < 0.2:
            reasons.append("⚠️ Agent has limited availability")
        
        relevant_caps = [
            cap.name for cap in agent.capabilities
            if cap.name.lower() in task.description.lower() and cap.proficiency_level > 0.7
        ]
        if relevant_caps:
            reasons.append(f"High expertise in: {', '.join(relevant_caps)}")
        
        if score > 0.8:
            reasons.append("✅ Excellent fit for this task")
        elif score < 0.5:
            reasons.append("⚠️ Suboptimal assignment due to constraints")
        
        return reasons
    
    def _estimate_completion_time(
        self, task: Task, agent: AgentProfile, current_workload: float
    ) -> datetime:
        """Estimate when task will be completed"""
        # Simple estimation: current workload + task hours, assuming 8h/day
        days_delay = current_workload / 8.0
        task_duration = task.estimated_hours / 8.0
        
        start_date = datetime.now() + timedelta(days=days_delay)
        completion_date = start_date + timedelta(days=task_duration)
        
        return completion_date
    
    def _analyze_team_composition(
        self, 
        agents: List[AgentProfile], 
        assignments: List[TaskAssignment], 
        requirements: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float], List[str], List[str]]:
        """Analyze team composition quality"""
        
        # Calculate team score
        assignment_scores = [a.assignment_score for a in assignments]
        avg_assignment_score = sum(assignment_scores) / len(assignment_scores) if assignment_scores else 0
        
        team_utilization = sum(a.current_workload for a in agents) / sum(a.max_workload for a in agents)
        team_performance = sum(a.performance_history.get('last_month', 0.8) for a in agents) / len(agents)
        
        team_score = (avg_assignment_score + (1 - abs(team_utilization - 0.7)) + team_performance) / 3
        
        # Coverage analysis
        coverage = {}
        for agent_type, req_data in requirements.items():
            assigned_agents = [a for a in agents if a.agent_type == agent_type]
            total_capacity = sum(a.max_workload - a.current_workload for a in assigned_agents)
            required_capacity = req_data['total_hours']
            
            coverage[agent_type] = min(total_capacity / required_capacity, 1.0) if required_capacity > 0 else 1.0
        
        # Identify risks
        risks = []
        if team_utilization > 0.9:
            risks.append("High team utilization - risk of burnout")
        if len([c for c in coverage.values() if c < 0.8]) > 0:
            risks.append("Insufficient coverage for some task types")
        if len(agents) < 3:
            risks.append("Small team size - limited redundancy")
        
        overloaded_agents = [a for a in agents if a.current_workload > a.max_workload * 0.9]
        if overloaded_agents:
            risks.append(f"Overloaded agents: {[a.agent_id for a in overloaded_agents]}")
        
        # Generate recommendations
        recommendations = []
        if team_utilization < 0.5:
            recommendations.append("Consider reducing team size to optimize costs")
        elif team_utilization > 0.85:
            recommendations.append("Consider adding more agents or extending timeline")
        
        low_coverage = [t for t, c in coverage.items() if c < 0.7]
        if low_coverage:
            recommendations.append(f"Add more agents for: {', '.join(low_coverage)}")
        
        if team_performance < 0.8:
            recommendations.append("Consider replacing low-performing agents")
        
        return team_score, coverage, risks, recommendations
    
    def _calculate_project_timeline(
        self, assignments: List[TaskAssignment], dependencies: Dict[int, List[int]]
    ) -> Dict[str, Any]:
        """Calculate project timeline considering dependencies"""
        
        if not assignments:
            return {"total_duration_days": 0, "critical_path": [], "milestones": []}
        
        # Simple timeline calculation
        earliest_start = min(a.estimated_completion for a in assignments if a.estimated_completion)
        latest_end = max(a.estimated_completion for a in assignments if a.estimated_completion)
        
        total_duration = (latest_end - earliest_start).days
        
        # Identify critical path (simplified)
        critical_tasks = [a for a in assignments if a.task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]]
        critical_path = [t.task.id for t in critical_tasks]
        
        # Generate milestones
        milestones = []
        phases = ["Architecture & Design", "Implementation", "Testing", "Deployment"]
        phase_duration = total_duration / len(phases)
        
        for i, phase in enumerate(phases):
            milestone_date = earliest_start + timedelta(days=int((i + 1) * phase_duration))
            milestones.append({
                "phase": phase,
                "date": milestone_date.isoformat(),
                "day": int((i + 1) * phase_duration)
            })
        
        return {
            "total_duration_days": total_duration,
            "start_date": earliest_start.isoformat(),
            "end_date": latest_end.isoformat(),
            "critical_path": critical_path,
            "milestones": milestones
        }
    
    def _record_selection(
        self, 
        team_comp: TeamComposition,
        task_breakdown: Any,
        domain_context: DomainContext,
        selection_context: SelectionContext
    ):
        """Record selection for future learning"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "project_type": domain_context.project_type,
            "tech_stack": domain_context.tech_stack,
            "team_size": len(team_comp.selected_agents),
            "team_score": team_comp.team_score,
            "strategy_used": selection_context.strategy.value,
            "agent_types": [a.agent_type for a in team_comp.selected_agents],
            "total_tasks": len(team_comp.task_assignments)
        }
        self.selection_history.append(record)
        
        # Keep only last 100 records
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]

class PerformanceTracker:
    """Track agent performance for continuous learning"""
    
    def __init__(self):
        self.performance_data = {}
    
    def record_task_completion(
        self, agent_id: str, task: Task, 
        completion_time: float, quality_score: float
    ):
        """Record task completion metrics"""
        if agent_id not in self.performance_data:
            self.performance_data[agent_id] = []
        
        self.performance_data[agent_id].append({
            "task_type": task.task_type.value,
            "estimated_hours": task.estimated_hours,
            "actual_hours": completion_time,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, float]:
        """Get performance metrics for agent"""
        if agent_id not in self.performance_data:
            return {"accuracy": 0.8, "speed": 1.0, "quality": 0.8}
        
        records = self.performance_data[agent_id][-10:]  # Last 10 tasks
        
        # Calculate metrics
        accuracy_scores = []
        speed_ratios = []
        quality_scores = []
        
        for record in records:
            # Speed ratio (estimated/actual)
            if record["actual_hours"] > 0:
                speed_ratios.append(record["estimated_hours"] / record["actual_hours"])
            quality_scores.append(record["quality_score"])
        
        return {
            "speed": sum(speed_ratios) / len(speed_ratios) if speed_ratios else 1.0,
            "quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0.8,
            "consistency": 1.0 - (max(speed_ratios) - min(speed_ratios)) if len(speed_ratios) > 1 else 1.0
        }

# Demo and testing functions
async def demo_context_aware_agent_selection():
    """Demo the Context-Aware Agent Selection system"""
    print("🧠 Context-Aware Agent Selection Demo")
    print("=" * 60)
    
    # Initialize system
    selector = ContextAwareAgentSelector()
    selector.populate_demo_agents()
    
    # Test scenario 1: Full-stack web application
    print("\n🎯 Scenario 1: Full-stack Web Application")
    print("-" * 40)
    
    domain_context = DomainContext(
        tech_stack=["FastAPI", "React", "PostgreSQL", "Docker"],
        project_type="fullstack_web_app",
        current_phase="development"
    )
    
    selection_context = SelectionContext(
        project_priority=TaskPriority.HIGH,
        strategy=SelectionStrategy.BALANCED,
        max_team_size=6
    )
    
    project_description = "Create a comprehensive user management system with JWT authentication, role-based access control, React admin dashboard, and RESTful API endpoints"
    
    team_composition = await selector.intelligent_team_selection(
        project_description, domain_context, selection_context
    )
    
    print(f"\n✅ Team Selection Results:")
    print(f"   Team Score: {team_composition.team_score:.2f}")
    print(f"   Selected Agents: {len(team_composition.selected_agents)}")
    print(f"   Task Assignments: {len(team_composition.task_assignments)}")
    
    print(f"\n📋 Team Members:")
    for agent in team_composition.selected_agents:
        utilization = agent.current_workload / agent.max_workload
        print(f"   • {agent.agent_id} ({agent.agent_type})")
        print(f"     Expertise: {', '.join(agent.primary_expertise)}")
        print(f"     Utilization: {utilization:.1%}")
    
    print(f"\n📊 Coverage Analysis:")
    for agent_type, coverage in team_composition.coverage_analysis.items():
        print(f"   {agent_type}: {coverage:.1%} coverage")
    
    if team_composition.potential_risks:
        print(f"\n⚠️ Potential Risks:")
        for risk in team_composition.potential_risks:
            print(f"   - {risk}")
    
    if team_composition.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in team_composition.recommendations:
            print(f"   - {rec}")
    
    print(f"\n📅 Timeline:")
    timeline = team_composition.estimated_timeline
    if timeline:
        print(f"   Duration: {timeline.get('total_duration_days', 0)} days")
        if timeline.get('milestones'):
            print(f"   Milestones: {len(timeline['milestones'])}")
    
    print("\n" + "=" * 60)
    print("✅ Context-Aware Agent Selection Demo Completed!")

if __name__ == "__main__":
    print("🚀 Context-Aware Agent Selection & Task Assignment")
    print("Week 43 - Point 2 of 6 Critical AI Features")
    
    # Run demo
    asyncio.run(demo_context_aware_agent_selection())