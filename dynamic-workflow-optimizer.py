#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Production Workflow Orchestration
AI-powered dynamic workflow optimization with real-time monitoring

Priority 3.1: Dynamic Workflow Optimizer (1 SP)
- AI-powered task reordering and dependency optimization
- Critical path analysis and resource allocation
- Dynamic priority adjustment based on real-time conditions
- Integration with existing AI reasoning systems
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

# Import AI components
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from unified_ai_client import UnifiedAIClient, AIReasoningRequest, ReasoningContext, ReasoningType, AIModelType
    from context_aware_reasoning import ContextAwareReasoningEngine
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    AI_COMPONENTS_AVAILABLE = False
    print("⚠️ AI components not available")

# Import existing orchestration components
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

logger = logging.getLogger(__name__)

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
    dependency_type: str = "blocks"  # blocks, enables, enhances
    strength: float = 1.0  # Dependency strength (0.0-1.0)
    can_parallel: bool = False  # Can be executed in parallel with constraints

@dataclass
class ResourceConstraint:
    """Resource constraints for task execution"""
    resource_type: str  # agent, compute, database, etc.
    max_concurrent: int
    current_usage: int = 0
    available_capacity: float = 1.0  # 0.0-1.0

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
    task_order: List[int]  # Optimized execution order
    parallel_groups: List[List[int]]  # Tasks that can run in parallel
    critical_path: List[int]
    estimated_duration: float
    resource_allocation: Dict[str, List[int]]  # resource_type -> task_ids
    confidence: float
    optimization_reasoning: str
    metrics: WorkflowMetrics

class DynamicWorkflowOptimizer:
    """
    AI-Powered Dynamic Workflow Optimizer
    
    Features:
    - Critical path analysis with AI-enhanced estimation
    - Dynamic task reordering based on real-time conditions
    - Resource-aware optimization and load balancing
    - Parallel execution opportunity identification
    - Bottleneck detection and mitigation strategies
    - Integration with AI reasoning for complex decisions
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.ai_client = None
        self.reasoning_engine = None
        
        # Initialize AI components
        if AI_COMPONENTS_AVAILABLE:
            try:
                self.ai_client = UnifiedAIClient(db_path=db_path)
                self.reasoning_engine = ContextAwareReasoningEngine(db_path=db_path)
                logger.info("✅ AI components connected")
            except Exception as e:
                logger.warning(f"AI initialization failed: {e}")
        
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
                    resource_utilization TEXT,  -- JSON
                    optimization_reasoning TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Task execution history for learning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    task_type TEXT,
                    estimated_duration REAL,
                    actual_duration REAL,
                    success BOOLEAN,
                    agent_type TEXT,
                    resource_usage TEXT,  -- JSON
                    bottlenecks TEXT,  -- JSON
                    lessons_learned TEXT,
                    workflow_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Resource performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_type TEXT NOT NULL,
                    task_type TEXT,
                    utilization_rate REAL,
                    efficiency_score REAL,
                    bottleneck_factor REAL,
                    optimization_suggestions TEXT,  -- JSON
                    measurement_period TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
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
        
        try:
            # Step 1: Build task dependency graph
            task_graph = self._build_task_graph(tasks, dependencies)
            
            # Step 2: AI-enhanced task estimation and risk analysis
            enhanced_tasks = await self._enhance_task_analysis(tasks, context)
            
            # Step 3: Critical path analysis
            critical_path, path_duration = self._calculate_critical_path(enhanced_tasks, task_graph)
            
            # Step 4: AI-powered optimization strategy selection
            optimization_strategy = await self._select_optimization_strategy(
                enhanced_tasks, dependencies, strategy, context
            )
            
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
            
            # Step 8: AI reasoning for optimization decisions
            reasoning = await self._generate_optimization_reasoning(
                enhanced_tasks, optimization_strategy, metrics, context
            )
            
            # Create optimized workflow
            optimized_workflow = OptimizedWorkflow(
                workflow_id=workflow_id,
                strategy=optimization_strategy,
                task_order=optimized_order,
                parallel_groups=parallel_groups,
                critical_path=critical_path,
                estimated_duration=metrics.critical_path_length,
                resource_allocation=resource_allocation,
                confidence=0.85,  # Will be calibrated by AI
                optimization_reasoning=reasoning,
                metrics=metrics
            )
            
            # Store and track optimization
            self._log_optimization(optimized_workflow, time.time() - start_time)
            self.active_workflows[workflow_id] = optimized_workflow
            
            # Update performance statistics
            self._update_optimization_stats(optimized_workflow, time.time() - start_time)
            
            logger.info(f"✅ Workflow optimized: {metrics.estimated_completion}")
            return optimized_workflow
            
        except Exception as e:
            logger.error(f"Workflow optimization failed: {e}")
            # Return basic workflow without optimization
            return self._create_basic_workflow(tasks, dependencies, workflow_id)
    
    def _build_task_graph(self, tasks: List[Task], dependencies: List[TaskDependency]) -> nx.DiGraph:
        """Build directed task dependency graph"""
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
    
    async def _enhance_task_analysis(self, tasks: List[Task], context: Optional[ReasoningContext]) -> List[Task]:
        """Use AI to enhance task analysis and estimation"""
        
        if not self.reasoning_engine or not context:
            return tasks  # Return original tasks if AI not available
        
        enhanced_tasks = []
        
        for task in tasks:
            try:
                # Create AI reasoning request for task analysis
                problem = f"""
                Analyze and enhance this task estimation:
                Task: {task.title}
                Description: {task.description}
                Type: {task.task_type.value}
                Current Estimate: {task.estimated_hours} hours
                Priority: {task.priority.value}
                """
                
                # Use context-aware reasoning for better estimation
                reasoning_chain = await self.reasoning_engine.reason_with_context(
                    problem_statement=problem,
                    context=context,
                    reasoning_type=ReasoningType.TASK_ANALYSIS
                )
                
                # Extract enhanced estimation from reasoning
                enhanced_hours = self._extract_time_estimate(reasoning_chain.final_reasoning)
                
                # Create enhanced task
                enhanced_task = Task(
                    id=task.id,
                    title=task.title,
                    description=task.description,
                    task_type=task.task_type,
                    status=task.status,
                    priority=task.priority,
                    estimated_hours=enhanced_hours if enhanced_hours else task.estimated_hours
                )
                
                enhanced_tasks.append(enhanced_task)
                
            except Exception as e:
                logger.warning(f"Task enhancement failed for task {task.id}: {e}")
                enhanced_tasks.append(task)  # Use original if enhancement fails
        
        logger.info(f"📊 Enhanced {len(enhanced_tasks)} tasks with AI analysis")
        return enhanced_tasks
    
    def _extract_time_estimate(self, reasoning_text: str) -> Optional[float]:
        """Extract time estimate from AI reasoning text"""
        import re
        
        # Look for time estimates in hours
        patterns = [
            r'(\d+\.?\d*)\s*hours?',
            r'estimate[d]?[:\s]*(\d+\.?\d*)\s*h',
            r'duration[:\s]*(\d+\.?\d*)',
            r'time[:\s]*(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning_text.lower())
            if matches:
                try:
                    return float(matches[0])
                except:
                    continue
        
        return None
    
    def _calculate_critical_path(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], float]:
        """Calculate critical path using graph analysis"""
        
        try:
            # Find longest path (critical path) in DAG
            if nx.is_directed_acyclic_graph(graph):
                # Calculate longest path
                longest_path = nx.dag_longest_path(graph, weight='estimated_hours')
                path_length = nx.dag_longest_path_length(graph, weight='estimated_hours')
                
                return longest_path, path_length
            else:
                logger.warning("Task graph contains cycles, using topological approximation")
                # Fallback: use topological sort
                topo_order = list(nx.topological_sort(graph))
                total_duration = sum(task.estimated_hours for task in tasks)
                return topo_order, total_duration
                
        except Exception as e:
            logger.warning(f"Critical path calculation failed: {e}")
            # Fallback: priority-based ordering
            task_ids = [task.id for task in sorted(tasks, key=lambda t: t.priority.value, reverse=True)]
            total_duration = sum(task.estimated_hours for task in tasks)
            return task_ids, total_duration
    
    async def _select_optimization_strategy(
        self, 
        tasks: List[Task], 
        dependencies: List[TaskDependency],
        preferred_strategy: OptimizationStrategy,
        context: Optional[ReasoningContext]
    ) -> OptimizationStrategy:
        """Use AI to select optimal optimization strategy"""
        
        if not self.ai_client:
            return preferred_strategy
        
        try:
            strategy_analysis = f"""
            Select the optimal workflow optimization strategy:
            
            Tasks: {len(tasks)} total
            Task Types: {', '.join(set(t.task_type.value for t in tasks))}
            Dependencies: {len(dependencies)} connections
            Preferred Strategy: {preferred_strategy.value}
            
            Available Strategies:
            - CRITICAL_PATH: Minimize overall duration
            - RESOURCE_BALANCED: Balance resource utilization  
            - RISK_MINIMIZED: Reduce execution risks
            - TIME_OPTIMIZED: Fastest completion
            - COST_OPTIMIZED: Minimize resource costs
            
            Consider:
            - Project constraints and priorities
            - Resource availability and capacity
            - Risk tolerance and failure impact
            - Timeline requirements
            
            Recommend the best strategy with reasoning.
            """
            
            request = AIReasoningRequest(
                request_id=f"strategy_{int(time.time())}",
                reasoning_type=ReasoningType.DECISION_MAKING,
                prompt=strategy_analysis,
                context=context or ReasoningContext(),
                model_preference=AIModelType.ADVANCED
            )
            
            response = await self.ai_client.reason(request)
            
            # Extract strategy from response
            selected_strategy = self._extract_strategy(response.response_text)
            return selected_strategy if selected_strategy else preferred_strategy
            
        except Exception as e:
            logger.warning(f"Strategy selection failed: {e}")
            return preferred_strategy
    
    def _extract_strategy(self, reasoning_text: str) -> Optional[OptimizationStrategy]:
        """Extract optimization strategy from AI response"""
        
        text_lower = reasoning_text.lower()
        
        strategy_keywords = {
            OptimizationStrategy.CRITICAL_PATH: ['critical', 'path', 'duration', 'minimize overall'],
            OptimizationStrategy.RESOURCE_BALANCED: ['resource', 'balanced', 'utilization', 'capacity'],
            OptimizationStrategy.RISK_MINIMIZED: ['risk', 'minimize', 'safe', 'conservative'],
            OptimizationStrategy.TIME_OPTIMIZED: ['time', 'fastest', 'speed', 'quick'],
            OptimizationStrategy.COST_OPTIMIZED: ['cost', 'efficient', 'budget', 'economical']
        }
        
        strategy_scores = {}
        for strategy, keywords in strategy_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                strategy_scores[strategy] = score
        
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        
        return None
    
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
            # Default to topological sort with priority
            return self._priority_based_ordering(tasks, graph)
    
    def _critical_path_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Order tasks based on critical path analysis"""
        
        try:
            # Get topological order respecting dependencies
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
                        # Check resource compatibility
                        task1 = next(t for t in tasks if t.id == task_id)
                        task2 = next(t for t in tasks if t.id == other_id)
                        
                        if self._can_run_parallel(task1, task2):
                            parallel_candidates.append(other_id)
                            processed.add(other_id)
                
                if len(parallel_candidates) > 1:
                    parallel_groups.append(parallel_candidates)
            
            return topo_order, parallel_groups
            
        except Exception as e:
            logger.warning(f"Critical path ordering failed: {e}")
            # Fallback to simple priority ordering
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
        
        # Interleave tasks from different resource groups
        balanced_order = []
        parallel_groups = []
        
        # Simple round-robin scheduling
        max_group_size = max(len(group) for group in resource_groups.values())
        
        for i in range(max_group_size):
            current_parallel = []
            for resource_type, task_group in resource_groups.items():
                if i < len(task_group):
                    task_id = task_group[i].id
                    
                    # Check dependencies before adding
                    dependencies_met = all(
                        dep_id in balanced_order 
                        for dep_id in graph.predecessors(task_id)
                    )
                    
                    if dependencies_met:
                        balanced_order.append(task_id)
                        current_parallel.append(task_id)
            
            if len(current_parallel) > 1:
                parallel_groups.append(current_parallel)
        
        return balanced_order, parallel_groups
    
    def _time_optimized_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Order tasks for fastest completion"""
        
        # Prioritize short tasks that can run in parallel
        tasks_by_duration = sorted(tasks, key=lambda t: t.estimated_hours)
        
        optimized_order = []
        parallel_groups = []
        remaining_tasks = set(task.id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no pending dependencies
            available_tasks = []
            for task in tasks_by_duration:
                if task.id not in remaining_tasks:
                    continue
                    
                dependencies_met = all(
                    dep_id not in remaining_tasks
                    for dep_id in graph.predecessors(task.id)
                )
                
                if dependencies_met:
                    available_tasks.append(task)
            
            if not available_tasks:
                # Deadlock prevention: take first remaining task
                available_tasks = [next(t for t in tasks if t.id in remaining_tasks)]
            
            # Group available tasks by execution time for parallelization
            current_batch = available_tasks[:min(3, len(available_tasks))]  # Max 3 parallel
            batch_ids = [task.id for task in current_batch]
            
            optimized_order.extend(batch_ids)
            if len(batch_ids) > 1:
                parallel_groups.append(batch_ids)
            
            # Remove processed tasks
            remaining_tasks -= set(batch_ids)
        
        return optimized_order, parallel_groups
    
    def _priority_based_ordering(self, tasks: List[Task], graph: nx.DiGraph) -> Tuple[List[int], List[List[int]]]:
        """Fallback ordering based on task priority"""
        
        # Sort by priority, then by estimated duration
        priority_order = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
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
        
        return task_order, parallel_groups
    
    def _can_run_parallel(self, task1: Task, task2: Task) -> bool:
        """Check if two tasks can run in parallel"""
        
        # Different task types can usually run in parallel
        if task1.task_type != task2.task_type:
            return True
        
        # Same type tasks need resource availability check
        resource_type = f"{task1.task_type.value}_agents"
        if resource_type in self.resource_constraints:
            constraint = self.resource_constraints[resource_type]
            return constraint.current_usage + 2 <= constraint.max_concurrent
        
        # Conservative default
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
        
        # Allocate resources based on task order and parallel groups
        for task_id in task_order:
            task = next(t for t in tasks if t.id == task_id)
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
        
        # Estimate completion time
        total_duration = sum(task.estimated_hours for task in tasks)
        estimated_completion = datetime.now() + timedelta(hours=total_duration)
        
        # Calculate resource utilization
        resource_utilization = {}
        for resource_type, task_list in resource_allocation.items():
            if resource_type in self.resource_constraints:
                constraint = self.resource_constraints[resource_type]
                utilization = len(task_list) / max(constraint.max_concurrent, 1)
                resource_utilization[resource_type] = min(1.0, utilization)
        
        # Identify bottlenecks (simplified)
        bottlenecks = []
        for resource_type, utilization in resource_utilization.items():
            if utilization > 0.8:
                bottlenecks.append(f"{resource_type}_overloaded")
        
        # Optimization opportunities
        optimization_opportunities = []
        if max(resource_utilization.values()) < 0.6:
            optimization_opportunities.append("increase_parallelization")
        if len(bottlenecks) > 0:
            optimization_opportunities.append("resource_reallocation")
        
        return WorkflowMetrics(
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            active_tasks=active_tasks,
            blocked_tasks=blocked_tasks,
            estimated_completion=estimated_completion,
            critical_path_length=total_duration,
            resource_utilization=resource_utilization,
            bottlenecks=bottlenecks,
            optimization_opportunities=optimization_opportunities
        )
    
    async def _generate_optimization_reasoning(
        self, 
        tasks: List[Task], 
        strategy: OptimizationStrategy,
        metrics: WorkflowMetrics,
        context: Optional[ReasoningContext]
    ) -> str:
        """Generate AI reasoning for optimization decisions"""
        
        if not self.reasoning_engine:
            return f"Workflow optimized using {strategy.value} strategy with {metrics.total_tasks} tasks"
        
        try:
            reasoning_problem = f"""
            Explain the workflow optimization decisions made:
            
            Strategy: {strategy.value}
            Tasks: {metrics.total_tasks} total
            Estimated Duration: {metrics.critical_path_length:.1f} hours
            Resource Utilization: {metrics.resource_utilization}
            Bottlenecks: {metrics.bottlenecks}
            Optimization Opportunities: {metrics.optimization_opportunities}
            
            Provide clear reasoning for:
            1. Why this strategy was optimal
            2. How tasks were reordered and parallelized
            3. Resource allocation decisions
            4. Risk mitigation measures
            5. Expected benefits and outcomes
            """
            
            reasoning_chain = await self.reasoning_engine.reason_with_context(
                problem_statement=reasoning_problem,
                context=context or ReasoningContext(),
                reasoning_type=ReasoningType.DECISION_MAKING
            )
            
            return reasoning_chain.final_reasoning
            
        except Exception as e:
            logger.warning(f"Reasoning generation failed: {e}")
            return f"Workflow optimized using {strategy.value} strategy focusing on {metrics.optimization_opportunities}"
    
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
            optimization_reasoning="Basic workflow without AI optimization",
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
                    0.0,  # Will be calculated when actual results are available
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
        
        # Update average optimization time
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
    print("🎯 Agent Zero V2.0 - Dynamic Workflow Optimizer Demo")
    print("=" * 55)
    
    # Initialize optimizer
    optimizer = DynamicWorkflowOptimizer()
    
    # Create demo tasks
    demo_tasks = [
        Task(1, "Setup Database Schema", "Create PostgreSQL tables and indexes", TaskType.DATABASE, TaskStatus.PENDING, TaskPriority.HIGH, 4.0),
        Task(2, "Implement Authentication API", "JWT-based auth endpoints", TaskType.BACKEND, TaskStatus.PENDING, TaskPriority.CRITICAL, 8.0),
        Task(3, "Create User Registration Form", "React component for signup", TaskType.FRONTEND, TaskStatus.PENDING, TaskPriority.HIGH, 6.0),
        Task(4, "Setup User Dashboard", "Main user interface", TaskType.FRONTEND, TaskStatus.PENDING, TaskPriority.MEDIUM, 12.0),
        Task(5, "Write Integration Tests", "API and frontend tests", TaskType.TESTING, TaskStatus.PENDING, TaskPriority.MEDIUM, 8.0),
        Task(6, "Deploy to Staging", "Docker deployment setup", TaskType.DEPLOYMENT, TaskStatus.PENDING, TaskPriority.LOW, 4.0)
    ]
    
    # Create dependencies
    dependencies = [
        TaskDependency(2, 1),  # Auth depends on Database
        TaskDependency(3, 2),  # Registration depends on Auth
        TaskDependency(4, 2),  # Dashboard depends on Auth
        TaskDependency(5, 3),  # Tests depend on Registration
        TaskDependency(5, 4),  # Tests depend on Dashboard
        TaskDependency(6, 5)   # Deploy depends on Tests
    ]
    
    # Create context
    context = ReasoningContext(
        project_type="web_application",
        tech_stack=["FastAPI", "React", "PostgreSQL", "Docker"],
        team_skills=["Python", "JavaScript", "SQL"],
        constraints=["2-week timeline", "3 developers", "staging deployment required"]
    )
    
    print(f"📋 Demo Setup:")
    print(f"   Tasks: {len(demo_tasks)}")
    print(f"   Dependencies: {len(dependencies)}")
    print(f"   Total Estimated: {sum(t.estimated_hours for t in demo_tasks)} hours")
    
    # Test different optimization strategies
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
        print(f"      Estimated Duration: {optimized_workflow.estimated_duration:.1f} hours")
        print(f"      Critical Path: {len(optimized_workflow.critical_path)} tasks")
        print(f"      Parallel Groups: {len(optimized_workflow.parallel_groups)}")
        print(f"      Confidence: {optimized_workflow.confidence:.2f}")
        
        # Show task execution order
        print(f"      Execution Order: {optimized_workflow.task_order}")
        
        if optimized_workflow.parallel_groups:
            print(f"      Parallel Execution:")
            for i, group in enumerate(optimized_workflow.parallel_groups):
                print(f"        Group {i+1}: Tasks {group}")
        
        # Show resource allocation
        print(f"      Resource Allocation:")
        for resource, task_list in optimized_workflow.resource_allocation.items():
            if task_list:
                print(f"        {resource}: {len(task_list)} tasks")
    
    # Show optimizer statistics
    print(f"\n📊 Optimizer Statistics:")
    stats = optimizer.get_optimization_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Dynamic workflow optimizer demo completed!")

if __name__ == "__main__":
    print("🎯 Agent Zero V2.0 Phase 4 - Dynamic Workflow Optimizer")
    print("Testing advanced workflow orchestration...")
    
    # Run demo
    asyncio.run(demo_dynamic_workflow_optimizer())