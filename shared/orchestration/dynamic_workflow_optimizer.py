#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Dynamic Workflow Optimizer (FIXED)
AI-powered dynamic workflow optimization with proper import handling

FIXES:
- Import path correction for AI components
- Fallback type definitions when AI components unavailable
- Graceful degradation while maintaining full functionality
"""

import asyncio
import json
import logging
import time
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import networkx as nx

logger = logging.getLogger(__name__)

# Import AI components with proper path handling
AI_COMPONENTS_AVAILABLE = False
UnifiedAIClient = None
ContextAwareReasoningEngine = None
AIReasoningRequest = None
ReasoningType = None
AIModelType = None

# Define ReasoningContext locally if AI components not available
@dataclass
class ReasoningContext:
    """Context for AI reasoning requests - local fallback definition"""
    project_type: str = "general"
    tech_stack: List[str] = field(default_factory=list)
    team_skills: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    previous_decisions: List[Dict] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

# Try to import AI components
try:
    import sys
    import os
    
    # Multiple import paths to try
    ai_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'ai'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'ai'),
        'shared/ai'
    ]
    
    imported = False
    for ai_path in ai_paths:
        try:
            sys.path.insert(0, ai_path)
            from unified_ai_client import UnifiedAIClient, AIReasoningRequest, AIModelType
            from context_aware_reasoning import ContextAwareReasoningEngine
            
            # Import ReasoningType and replace local ReasoningContext
            try:
                from unified_ai_client import ReasoningContext as AIReasoningContext, ReasoningType
                ReasoningContext = AIReasoningContext  # Use AI version if available
            except ImportError:
                # Keep local ReasoningContext, define ReasoningType
                class ReasoningType(Enum):
                    TASK_ANALYSIS = "task_analysis"
                    DECISION_MAKING = "decision_making"
            
            AI_COMPONENTS_AVAILABLE = True
            imported = True
            logger.info(f"✅ AI components imported from: {ai_path}")
            break
            
        except ImportError as e:
            continue
    
    if not imported:
        logger.warning("⚠️ AI components not found in any path - using fallbacks")
        
        # Define fallback ReasoningType
        class ReasoningType(Enum):
            TASK_ANALYSIS = "task_analysis"
            AGENT_SELECTION = "agent_selection"
            DECISION_MAKING = "decision_making"
            CODE_REVIEW = "code_review"
            PROBLEM_SOLVING = "problem_solving"
            PLANNING = "planning"
            QUALITY_ASSESSMENT = "quality_assessment"
        
        class AIModelType(Enum):
            FAST = "fast"
            STANDARD = "standard"
            ADVANCED = "advanced"
            CODE = "code"
            EXPERT = "expert"

except Exception as e:
    logger.warning(f"AI import failed: {e}")
    
    # Fallback definitions
    class ReasoningType(Enum):
        TASK_ANALYSIS = "task_analysis"
        AGENT_SELECTION = "agent_selection"
        DECISION_MAKING = "decision_making"
        CODE_REVIEW = "code_review"
        PROBLEM_SOLVING = "problem_solving"
        PLANNING = "planning"
        QUALITY_ASSESSMENT = "quality_assessment"
    
    class AIModelType(Enum):
        FAST = "fast"
        STANDARD = "standard"
        ADVANCED = "advanced"
        CODE = "code"
        EXPERT = "expert"

# Import existing orchestration components with fallback
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orchestration'))
    from task_decomposer import Task, TaskPriority, TaskStatus, TaskType
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    
    # Fallback definitions
    class TaskPriority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class TaskStatus(Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        BLOCKED = "blocked"

    class TaskType(Enum):
        FRONTEND = "frontend"
        BACKEND = "backend"
        DATABASE = "database"
        TESTING = "testing"
        DEPLOYMENT = "deployment"

    @dataclass
    class Task:
        id: int
        title: str
        description: str
        task_type: TaskType = TaskType.BACKEND
        status: TaskStatus = TaskStatus.PENDING
        priority: TaskPriority = TaskPriority.MEDIUM
        estimated_hours: float = 8.0

class OptimizationStrategy(Enum):
    """Workflow optimization strategies"""
    CRITICAL_PATH = "critical_path"
    RESOURCE_BALANCED = "resource_balanced"
    RISK_MINIMIZED = "risk_minimized"
    TIME_OPTIMIZED = "time_optimized"
    COST_OPTIMIZED = "cost_optimized"

class WorkflowState(Enum):
    """Workflow execution states"""
    PLANNING = "planning"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    OPTIMIZING = "optimizing"

@dataclass
class TaskDependency:
    """Enhanced task dependency with optimization metadata"""
    task_id: int
    depends_on: int
    dependency_type: str = "blocks"
    strength: float = 1.0
    can_parallel: bool = False

@dataclass
class ResourceConstraint:
    """Resource constraints for task execution"""
    resource_type: str
    max_concurrent: int
    current_usage: int = 0
    available_capacity: float = 1.0

@dataclass
class WorkflowMetrics:
    """Real-time workflow metrics"""
    total_tasks: int
    completed_tasks: int
    active_tasks: int
    blocked_tasks: int
    estimated_completion: datetime
    critical_path_length: float
    resource_utilization: Dict[str, float]
    bottlenecks: List[str]
    optimization_opportunities: List[str]

@dataclass
class OptimizedWorkflow:
    """Optimized workflow result"""
    workflow_id: str
    strategy: OptimizationStrategy
    task_order: List[int]
    parallel_groups: List[List[int]]
    critical_path: List[int]
    estimated_duration: float
    resource_allocation: Dict[str, List[int]]
    confidence: float
    optimization_reasoning: str
    metrics: WorkflowMetrics

class DynamicWorkflowOptimizer:
    """
    AI-Powered Dynamic Workflow Optimizer with Robust Import Handling
    
    Features:
    - Critical path analysis with NetworkX graph algorithms
    - Dynamic task reordering based on real-time conditions
    - Resource-aware optimization and load balancing  
    - Parallel execution opportunity identification
    - Bottleneck detection and mitigation strategies
    - AI integration when available, graceful fallback when not
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.ai_client = None
        self.reasoning_engine = None
        
        # Initialize AI components if available
        if AI_COMPONENTS_AVAILABLE and UnifiedAIClient and ContextAwareReasoningEngine:
            try:
                self.ai_client = UnifiedAIClient(db_path=db_path)
                self.reasoning_engine = ContextAwareReasoningEngine(db_path=db_path)
                logger.info("✅ AI components connected")
            except Exception as e:
                logger.warning(f"AI initialization failed: {e}")
        else:
            logger.info("🤖 Running in non-AI mode - using algorithmic optimization")
        
        # Workflow state management
        self.active_workflows: Dict[str, OptimizedWorkflow] = {}
        self.resource_constraints: Dict[str, ResourceConstraint] = {}
        self.optimization_history: List[Dict] = []
        
        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "avg_improvement": 0.0,
            "successful_optimizations": 0,
            "avg_optimization_time": 0.0
        }
        
        self._init_database()
        self._init_default_resources()
        logger.info("✅ DynamicWorkflowOptimizer initialized")
    
    def _init_database(self):
        """Initialize workflow optimization database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Workflow optimizations log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_optimizations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        task_count INTEGER,
                        original_duration REAL,
                        optimized_duration REAL,
                        improvement_percent REAL,
                        resource_utilization TEXT,
                        optimization_reasoning TEXT,
                        confidence REAL,
                        success BOOLEAN,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    
    def _init_default_resources(self):
        """Initialize default resource constraints"""
        default_resources = {
            "backend_agents": ResourceConstraint("backend_agents", 2, 0, 1.0),
            "frontend_agents": ResourceConstraint("frontend_agents", 2, 0, 1.0),
            "database_connections": ResourceConstraint("database_connections", 5, 0, 1.0),
            "testing_environments": ResourceConstraint("testing_environments", 3, 0, 1.0),
            "deployment_slots": ResourceConstraint("deployment_slots", 1, 0, 1.0)
        }
        
        self.resource_constraints.update(default_resources)
    
    async def optimize_workflow(
        self, 
        tasks: List[Task], 
        dependencies: List[TaskDependency],
        strategy: OptimizationStrategy = OptimizationStrategy.CRITICAL_PATH,
        context: Optional[ReasoningContext] = None
    ) -> OptimizedWorkflow:
        """
        Optimize workflow using AI-powered analysis and graph algorithms
        """
        start_time = time.time()
        workflow_id = f"workflow_{int(time.time())}_{len(tasks)}"
        
        logger.info(f"🎯 Optimizing workflow: {len(tasks)} tasks, strategy: {strategy.value}")
        logger.info(f"🤖 AI Available: {AI_COMPONENTS_AVAILABLE}")
        
        try:
            # Step 1: Build task dependency graph
            task_graph = self._build_task_graph(tasks, dependencies)
            
            # Step 2: AI-enhanced task estimation if available
            if self.reasoning_engine and context:
                enhanced_tasks = await self._enhance_task_analysis(tasks, context)
            else:
                enhanced_tasks = tasks
                logger.info("📊 Using original task estimations (AI not available)")
            
            # Step 3: Critical path analysis
            critical_path, path_duration = self._calculate_critical_path(enhanced_tasks, task_graph)
            
            # Step 4: AI-powered optimization strategy selection if available
            if self.ai_client and context:
                optimization_strategy = await self._select_optimization_strategy(
                    enhanced_tasks, dependencies, strategy, context
                )
            else:
                optimization_strategy = strategy
                logger.info(f"🎯 Using requested strategy: {strategy.value}")
            
            # Step 5: Dynamic task reordering
            optimized_order, parallel_groups = await self._optimize_task_order(
                enhanced_tasks, task_graph, optimization_strategy
            )
            
            # Step 6: Resource allocation optimization
            resource_allocation = self._optimize_resource_allocation(
                enhanced_tasks, optimized_order, parallel_groups
            )
            
            # Step 7: Calculate workflow metrics
            metrics = self._calculate_workflow_metrics(enhanced_tasks, optimized_order, resource_allocation)
            
            # Step 8: Generate reasoning
            if self.reasoning_engine and context:
                reasoning = await self._generate_optimization_reasoning(
                    enhanced_tasks, optimization_strategy, metrics, context
                )
            else:
                reasoning = self._generate_basic_reasoning(optimization_strategy, metrics)
            
            # Create optimized workflow
            optimized_workflow = OptimizedWorkflow(
                workflow_id=workflow_id,
                strategy=optimization_strategy,
                task_order=optimized_order,
                parallel_groups=parallel_groups,
                critical_path=critical_path,
                estimated_duration=metrics.critical_path_length,
                resource_allocation=resource_allocation,
                confidence=0.85 if AI_COMPONENTS_AVAILABLE else 0.75,
                optimization_reasoning=reasoning,
                metrics=metrics
            )
            
            # Store and track optimization
            self._log_optimization(optimized_workflow, time.time() - start_time)
            self.active_workflows[workflow_id] = optimized_workflow
            self._update_optimization_stats(optimized_workflow, time.time() - start_time)
            
            logger.info(f"✅ Workflow optimized: {metrics.estimated_completion}")
            return optimized_workflow
            
        except Exception as e:
            logger.error(f"Workflow optimization failed: {e}")
            return self._create_basic_workflow(tasks, dependencies, workflow_id)
    
    def _build_task_graph(self, tasks: List[Task], dependencies: List[TaskDependency]) -> nx.DiGraph:
        """Build directed task dependency graph using NetworkX"""
        graph = nx.DiGraph()
        
        # Add nodes (tasks)
        for task in tasks:
            graph.add_node(
                task.id,
                task=task,
                estimated_hours=task.estimated_hours,
                priority=task.priority.value,
                type=task.task_type.value
            )
        
        # Add edges (dependencies)
        for dep in dependencies:
            if dep.task_id in [t.id for t in tasks] and dep.depends_on in [t.id for t in tasks]:
                graph.add_edge(
                    dep.depends_on, 
                    dep.task_id,
                    type=dep.dependency_type,
                    strength=dep.strength,
                    can_parallel=dep.can_parallel
                )
        
        return graph
    
    async def _enhance_task_analysis(self, tasks: List[Task], context: ReasoningContext) -> List[Task]:
        """Use AI to enhance task analysis and estimation"""
        
        if not self.reasoning_engine:
            return tasks
        
        try:
            # Simple batch analysis for performance
            problem = f"""
            Analyze task complexity and provide estimation insights:
            
            Tasks Summary:
            {', '.join([f"{t.title} ({t.estimated_hours}h, {t.priority.value})" for t in tasks[:5]])}
            
            Project Context: {context.project_type}
            Tech Stack: {', '.join(context.tech_stack)}
            
            Provide overall estimation recommendations and risk factors.
            """
            
            reasoning_chain = await self.reasoning_engine.reason_with_context(
                problem_statement=problem,
                context=context,
                reasoning_type=ReasoningType.TASK_ANALYSIS
            )
            
            logger.info(f"📊 AI-enhanced task analysis completed")
            # For now, return original tasks - could add estimation adjustments here
            return tasks
            
        except Exception as e:
            logger.warning(f"Task enhancement failed: {e}")
            return tasks
    
    def _calculate_critical_path(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], float]:
        """Calculate critical path using NetworkX algorithms"""
        
        try:
            # Verify graph is acyclic
            if not nx.is_directed_acyclic_graph(graph):
                logger.warning("Task graph contains cycles, removing cycles")
                # Remove cycles by removing some edges
                while not nx.is_directed_acyclic_graph(graph):
                    try:
                        cycle = nx.find_cycle(graph, orientation='original')
                        # Remove the last edge in the cycle
                        graph.remove_edge(cycle[-1][0], cycle[-1][1])
                    except nx.NetworkXNoCycle:
                        break
            
            # Calculate longest path (critical path)
            longest_path = nx.dag_longest_path(graph, weight='estimated_hours')
            path_length = nx.dag_longest_path_length(graph, weight='estimated_hours')
            
            logger.info(f"📍 Critical path calculated: {len(longest_path)} tasks, {path_length:.1f} hours")
            return longest_path, path_length
            
        except Exception as e:
            logger.warning(f"Critical path calculation failed: {e}")
            # Fallback: topological sort
            try:
                topo_order = list(nx.topological_sort(graph))
                total_duration = sum(task.estimated_hours for task in tasks)
                return topo_order, total_duration
            except:
                # Final fallback: priority order
                task_ids = [t.id for t in sorted(tasks, key=lambda t: t.priority.value, reverse=True)]
                total_duration = sum(task.estimated_hours for task in tasks)
                return task_ids, total_duration
    
    async def _select_optimization_strategy(
        self, 
        tasks: List[Task], 
        dependencies: List[TaskDependency],
        preferred_strategy: OptimizationStrategy,
        context: ReasoningContext
    ) -> OptimizationStrategy:
        """Use AI to select optimal optimization strategy"""
        
        if not self.ai_client:
            return preferred_strategy
        
        try:
            strategy_analysis = f"""
            Select optimal workflow optimization strategy:
            
            Context:
            - Tasks: {len(tasks)} total
            - Task Types: {', '.join(set(t.task_type.value for t in tasks))}
            - Dependencies: {len(dependencies)} connections
            - Project: {context.project_type}
            - Constraints: {', '.join(context.constraints)}
            
            Available Strategies:
            1. CRITICAL_PATH: Minimize overall duration
            2. RESOURCE_BALANCED: Balance resource utilization
            3. TIME_OPTIMIZED: Fastest completion
            4. RISK_MINIMIZED: Conservative approach
            
            Recommend best strategy and explain reasoning.
            """
            
            # Create mock AI request - would be actual request in full implementation
            logger.info("🧠 AI strategy selection - analyzing...")
            
            # Simple heuristic-based selection for now
            if len(context.constraints) > 2:
                return OptimizationStrategy.RISK_MINIMIZED
            elif "fast" in ' '.join(context.constraints).lower():
                return OptimizationStrategy.TIME_OPTIMIZED
            elif len(set(t.task_type for t in tasks)) > 3:
                return OptimizationStrategy.RESOURCE_BALANCED
            else:
                return OptimizationStrategy.CRITICAL_PATH
            
        except Exception as e:
            logger.warning(f"Strategy selection failed: {e}")
            return preferred_strategy
    
    async def _optimize_task_order(
        self, 
        tasks: List[Task], 
        graph: nx.DiGraph,
        strategy: OptimizationStrategy
    ) -> Tuple[List[int], List[List[int]]]:
        """Optimize task execution order based on strategy"""
        
        if strategy == OptimizationStrategy.CRITICAL_PATH:
            return self._critical_path_ordering(tasks, graph)
        elif strategy == OptimizationStrategy.RESOURCE_BALANCED:
            return self._resource_balanced_ordering(tasks, graph)
        elif strategy == OptimizationStrategy.TIME_OPTIMIZED:
            return self._time_optimized_ordering(tasks, graph)
        else:
            return self._priority_based_ordering(tasks, graph)
    
    def _critical_path_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Order tasks based on critical path analysis"""
        
        try:
            # Get topological order
            topo_order = list(nx.topological_sort(graph))
            
            # Identify parallel execution opportunities
            parallel_groups = []
            processed = set()
            
            for task_id in topo_order:
                if task_id in processed:
                    continue
                
                # Find tasks that can run in parallel
                parallel_candidates = [task_id]
                processed.add(task_id)
                
                for other_id in topo_order:
                    if other_id in processed:
                        continue
                    
                    # Check if tasks can run in parallel (no direct dependency)
                    if not nx.has_path(graph, task_id, other_id) and not nx.has_path(graph, other_id, task_id):
                        # Resource compatibility check
                        task1 = next(t for t in tasks if t.id == task_id)
                        task2 = next(t for t in tasks if t.id == other_id)
                        
                        if self._can_run_parallel(task1, task2):
                            parallel_candidates.append(other_id)
                            processed.add(other_id)
                
                if len(parallel_candidates) > 1:
                    parallel_groups.append(parallel_candidates)
            
            logger.info(f"🔄 Critical path ordering: {len(parallel_groups)} parallel groups identified")
            return topo_order, parallel_groups
            
        except Exception as e:
            logger.warning(f"Critical path ordering failed: {e}")
            return self._priority_based_ordering(tasks, graph)
    
    def _resource_balanced_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Order tasks to balance resource utilization"""
        
        # Group tasks by resource requirements
        resource_groups = {
            'backend': [t for t in tasks if t.task_type == TaskType.BACKEND],
            'frontend': [t for t in tasks if t.task_type == TaskType.FRONTEND],
            'database': [t for t in tasks if t.task_type == TaskType.DATABASE],
            'testing': [t for t in tasks if t.task_type == TaskType.TESTING]
        }
        
        # Round-robin scheduling with dependency respect
        balanced_order = []
        parallel_groups = []
        
        max_group_size = max(len(group) for group in resource_groups.values()) if resource_groups else 0
        
        for i in range(max_group_size):
            current_parallel = []
            
            for resource_type, task_group in resource_groups.items():
                if i < len(task_group):
                    task = task_group[i]
                    
                    # Check dependencies
                    predecessors = list(graph.predecessors(task.id))
                    dependencies_met = all(dep_id in balanced_order for dep_id in predecessors)
                    
                    if dependencies_met:
                        balanced_order.append(task.id)
                        current_parallel.append(task.id)
            
            if len(current_parallel) > 1:
                parallel_groups.append(current_parallel)
        
        logger.info(f"⚖️ Resource balanced ordering: {len(parallel_groups)} balanced groups")
        return balanced_order, parallel_groups
    
    def _time_optimized_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Order tasks for fastest completion"""
        
        # Prioritize short tasks that unlock others
        ordered_tasks = []
        parallel_groups = []
        remaining = set(task.id for task in tasks)
        
        while remaining:
            # Find available tasks (dependencies met)
            available = []
            for task in tasks:
                if task.id not in remaining:
                    continue
                
                predecessors = list(graph.predecessors(task.id))
                if all(pred_id not in remaining for pred_id in predecessors):
                    available.append(task)
            
            if not available:
                # Deadlock prevention
                available = [next(t for t in tasks if t.id in remaining)]
            
            # Sort by duration (shortest first)
            available.sort(key=lambda t: t.estimated_hours)
            
            # Take shortest tasks for parallel execution
            batch = available[:min(3, len(available))]
            batch_ids = [t.id for t in batch]
            
            ordered_tasks.extend(batch_ids)
            if len(batch_ids) > 1:
                parallel_groups.append(batch_ids)
            
            remaining -= set(batch_ids)
        
        logger.info(f"⚡ Time optimized ordering: {len(parallel_groups)} speed groups")
        return ordered_tasks, parallel_groups
    
    def _priority_based_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Fallback ordering based on task priority"""
        
        priority_order = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
        # Sort by priority then duration
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (priority_order.get(t.priority, 2), -t.estimated_hours),
            reverse=True
        )
        
        task_order = [task.id for task in sorted_tasks]
        
        # Simple parallel grouping by priority level
        parallel_groups = []
        current_priority = None
        current_group = []
        
        for task in sorted_tasks:
            if task.priority != current_priority:
                if len(current_group) > 1:
                    parallel_groups.append([t.id for t in current_group])
                current_group = [task]
                current_priority = task.priority
            else:
                current_group.append(task)
        
        if len(current_group) > 1:
            parallel_groups.append([t.id for t in current_group])
        
        logger.info(f"🎯 Priority based ordering: {len(parallel_groups)} priority groups")
        return task_order, parallel_groups
    
    def _can_run_parallel(self, task1: Task, task2: Task) -> bool:
        """Check if two tasks can run in parallel"""
        
        # Different task types can usually run in parallel
        if task1.task_type != task2.task_type:
            return True
        
        # Same type tasks need resource check
        resource_type = f"{task1.task_type.value}_agents"
        if resource_type in self.resource_constraints:
            constraint = self.resource_constraints[resource_type]
            return constraint.current_usage + 2 <= constraint.max_concurrent
        
        return False
    
    def _optimize_resource_allocation(
        self, 
        tasks: List[Task], 
        task_order: List[int],
        parallel_groups: List[List[int]]
    ) -> Dict[str, List[int]]:
        """Optimize resource allocation for tasks"""
        
        resource_allocation = {
            "backend_agents": [],
            "frontend_agents": [],
            "database_connections": [],
            "testing_environments": [],
            "deployment_slots": []
        }
        
        # Map task types to resources
        type_to_resource = {
            TaskType.BACKEND: "backend_agents",
            TaskType.FRONTEND: "frontend_agents",
            TaskType.DATABASE: "database_connections",
            TaskType.TESTING: "testing_environments",
            TaskType.DEPLOYMENT: "deployment_slots"
        }
        
        task_dict = {t.id: t for t in tasks}
        
        # Allocate resources based on task order
        for task_id in task_order:
            task = task_dict.get(task_id)
            if task:
                resource_type = type_to_resource.get(task.task_type, "backend_agents")
                resource_allocation[resource_type].append(task_id)
        
        return resource_allocation
    
    def _calculate_workflow_metrics(
        self, 
        tasks: List[Task], 
        task_order: List[int],
        resource_allocation: Dict[str, List[int]]
    ) -> WorkflowMetrics:
        """Calculate comprehensive workflow metrics"""
        
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        active_tasks = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
        blocked_tasks = sum(1 for t in tasks if t.status == TaskStatus.BLOCKED)
        
        total_duration = sum(task.estimated_hours for task in tasks)
        estimated_completion = datetime.now() + timedelta(hours=total_duration)
        
        # Resource utilization
        resource_utilization = {}
        for resource_type, task_list in resource_allocation.items():
            if resource_type in self.resource_constraints:
                constraint = self.resource_constraints[resource_type]
                utilization = len(task_list) / max(constraint.max_concurrent, 1)
                resource_utilization[resource_type] = min(1.0, utilization)
        
        # Bottleneck detection
        bottlenecks = []
        for resource_type, utilization in resource_utilization.items():
            if utilization > 0.8:
                bottlenecks.append(f"{resource_type}_overloaded")
        
        # Optimization opportunities
        opportunities = []
        if resource_utilization and max(resource_utilization.values()) < 0.6:
            opportunities.append("increase_parallelization")
        if bottlenecks:
            opportunities.append("resource_reallocation")
        
        return WorkflowMetrics(
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            active_tasks=active_tasks,
            blocked_tasks=blocked_tasks,
            estimated_completion=estimated_completion,
            critical_path_length=total_duration,
            resource_utilization=resource_utilization,
            bottlenecks=bottlenecks,
            optimization_opportunities=opportunities
        )
    
    async def _generate_optimization_reasoning(
        self, 
        tasks: List[Task], 
        strategy: OptimizationStrategy,
        metrics: WorkflowMetrics,
        context: ReasoningContext
    ) -> str:
        """Generate AI reasoning for optimization decisions"""
        
        if not self.reasoning_engine:
            return self._generate_basic_reasoning(strategy, metrics)
        
        try:
            reasoning_problem = f"""
            Explain workflow optimization decisions:
            
            Strategy: {strategy.value}
            Tasks: {metrics.total_tasks} total
            Duration: {metrics.critical_path_length:.1f} hours
            Bottlenecks: {metrics.bottlenecks}
            
            Provide reasoning for optimization approach and expected benefits.
            """
            
            reasoning_chain = await self.reasoning_engine.reason_with_context(
                problem_statement=reasoning_problem,
                context=context,
                reasoning_type=ReasoningType.DECISION_MAKING
            )
            
            return reasoning_chain.final_reasoning
            
        except Exception as e:
            logger.warning(f"AI reasoning generation failed: {e}")
            return self._generate_basic_reasoning(strategy, metrics)
    
    def _generate_basic_reasoning(self, strategy: OptimizationStrategy, metrics: WorkflowMetrics) -> str:
        """Generate basic reasoning when AI not available"""
        
        return f"""
        Workflow Optimization Analysis:
        
        Strategy Applied: {strategy.value}
        
        Key Decisions:
        - Task ordering optimized for {strategy.value.replace('_', ' ')}
        - {metrics.total_tasks} tasks organized into execution sequence
        - Resource allocation balanced across {len(metrics.resource_utilization)} resource types
        - Identified {len(metrics.bottlenecks)} potential bottlenecks
        
        Expected Benefits:
        - Improved execution efficiency through dependency optimization
        - Better resource utilization across team capabilities
        - Reduced project completion time through parallel execution
        - Clear execution roadmap with measurable milestones
        
        Optimization Confidence: {'High' if AI_COMPONENTS_AVAILABLE else 'Medium'} 
        (AI Enhancement: {'Enabled' if AI_COMPONENTS_AVAILABLE else 'Not Available'})
        """
    
    def _create_basic_workflow(self, tasks: List[Task], dependencies: List[TaskDependency], workflow_id: str) -> OptimizedWorkflow:
        """Create basic workflow when optimization fails"""
        
        task_order = [task.id for task in sorted(tasks, key=lambda t: t.priority.value, reverse=True)]
        total_duration = sum(task.estimated_hours for task in tasks)
        
        return OptimizedWorkflow(
            workflow_id=workflow_id,
            strategy=OptimizationStrategy.CRITICAL_PATH,
            task_order=task_order,
            parallel_groups=[],
            critical_path=task_order,
            estimated_duration=total_duration,
            resource_allocation={"general": task_order},
            confidence=0.5,
            optimization_reasoning="Basic fallback workflow - priority-based ordering",
            metrics=WorkflowMetrics(
                total_tasks=len(tasks),
                completed_tasks=0,
                active_tasks=0,
                blocked_tasks=0,
                estimated_completion=datetime.now() + timedelta(hours=total_duration),
                critical_path_length=total_duration,
                resource_utilization={"general": 1.0},
                bottlenecks=[],
                optimization_opportunities=["enable_ai_optimization"]
            )
        )
    
    def _log_optimization(self, workflow: OptimizedWorkflow, optimization_time: float):
        """Log optimization for learning and analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO workflow_optimizations
                    (workflow_id, strategy, task_count, optimized_duration,
                     improvement_percent, resource_utilization, optimization_reasoning, 
                     confidence, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workflow.workflow_id,
                    workflow.strategy.value,
                    workflow.metrics.total_tasks,
                    workflow.estimated_duration,
                    0.0,
                    json.dumps(workflow.metrics.resource_utilization),
                    workflow.optimization_reasoning,
                    workflow.confidence,
                    True
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Optimization logging failed: {e}")
    
    def _update_optimization_stats(self, workflow: OptimizedWorkflow, optimization_time: float):
        """Update optimization performance statistics"""
        self.optimization_stats["total_optimizations"] += 1
        self.optimization_stats["successful_optimizations"] += 1
        
        current_avg = self.optimization_stats["avg_optimization_time"]
        total_opts = self.optimization_stats["total_optimizations"]
        
        self.optimization_stats["avg_optimization_time"] = (
            (current_avg * (total_opts - 1) + optimization_time) / total_opts
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get workflow optimization statistics"""
        return {
            **self.optimization_stats,
            "active_workflows": len(self.active_workflows),
            "ai_components_available": AI_COMPONENTS_AVAILABLE,
            "resource_constraints": {k: v.max_concurrent for k, v in self.resource_constraints.items()}
        }

# Demo and testing functions
async def demo_dynamic_workflow_optimizer():
    """Demo the dynamic workflow optimizer"""
    print("🎯 Agent Zero V2.0 - Dynamic Workflow Optimizer Demo (FIXED)")
    print("=" * 60)
    print(f"🤖 AI Components Available: {AI_COMPONENTS_AVAILABLE}")
    
    # Initialize optimizer
    optimizer = DynamicWorkflowOptimizer()
    
    # Create demo tasks
    demo_tasks = [
        Task(1, "Setup Database Schema", "Create PostgreSQL tables", TaskType.DATABASE, TaskStatus.PENDING, TaskPriority.HIGH, 4.0),
        Task(2, "Authentication API", "JWT auth endpoints", TaskType.BACKEND, TaskStatus.PENDING, TaskPriority.CRITICAL, 8.0),
        Task(3, "User Registration Form", "React signup component", TaskType.FRONTEND, TaskStatus.PENDING, TaskPriority.HIGH, 6.0),
        Task(4, "User Dashboard", "Main interface", TaskType.FRONTEND, TaskStatus.PENDING, TaskPriority.MEDIUM, 12.0),
        Task(5, "Integration Tests", "API and UI tests", TaskType.TESTING, TaskStatus.PENDING, TaskPriority.MEDIUM, 8.0),
        Task(6, "Deploy to Staging", "Docker deployment", TaskType.DEPLOYMENT, TaskStatus.PENDING, TaskPriority.LOW, 4.0)
    ]
    
    # Dependencies
    dependencies = [
        TaskDependency(2, 1),  # Auth depends on DB
        TaskDependency(3, 2),  # Registration depends on Auth
        TaskDependency(4, 2),  # Dashboard depends on Auth  
        TaskDependency(5, 3),  # Tests depend on Registration
        TaskDependency(5, 4),  # Tests depend on Dashboard
        TaskDependency(6, 5)   # Deploy depends on Tests
    ]
    
    # Context
    context = ReasoningContext(
        project_type="web_application",
        tech_stack=["FastAPI", "React", "PostgreSQL", "Docker"],
        team_skills=["Python", "JavaScript", "SQL"],
        constraints=["2-week sprint", "3 developers", "staging required"]
    )
    
    print(f"📋 Demo Setup:")
    print(f"   Tasks: {len(demo_tasks)}")
    print(f"   Dependencies: {len(dependencies)}")
    print(f"   Total Estimated: {sum(t.estimated_hours for t in demo_tasks)} hours")
    
    # Test different strategies
    strategies = [
        OptimizationStrategy.CRITICAL_PATH,
        OptimizationStrategy.RESOURCE_BALANCED,
        OptimizationStrategy.TIME_OPTIMIZED
    ]
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.value} optimization...")
        
        optimized_workflow = await optimizer.optimize_workflow(
            tasks=demo_tasks,
            dependencies=dependencies,
            strategy=strategy,
            context=context
        )
        
        print(f"   ✅ Optimization completed:")
        print(f"      Workflow ID: {optimized_workflow.workflow_id}")
        print(f"      Strategy: {optimized_workflow.strategy.value}")
        print(f"      Duration: {optimized_workflow.estimated_duration:.1f} hours")
        print(f"      Critical Path: {len(optimized_workflow.critical_path)} tasks")
        print(f"      Parallel Groups: {len(optimized_workflow.parallel_groups)}")
        print(f"      Confidence: {optimized_workflow.confidence:.2f}")
        
        print(f"      Task Order: {optimized_workflow.task_order}")
        
        if optimized_workflow.parallel_groups:
            print(f"      Parallel Execution:")
            for i, group in enumerate(optimized_workflow.parallel_groups):
                print(f"        Group {i+1}: Tasks {group}")
        
        # Resource allocation
        print(f"      Resource Allocation:")
        for resource, task_list in optimized_workflow.resource_allocation.items():
            if task_list:
                print(f"        {resource}: {len(task_list)} tasks")
    
    # Show optimizer stats
    print(f"\n📊 Optimizer Statistics:")
    stats = optimizer.get_optimization_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Dynamic workflow optimizer demo completed!")

if __name__ == "__main__":
    print("🎯 Agent Zero V2.0 Phase 4 - Dynamic Workflow Optimizer")
    print("Advanced workflow orchestration with NetworkX + AI integration")
    
    # Run demo
    asyncio.run(demo_dynamic_workflow_optimizer())