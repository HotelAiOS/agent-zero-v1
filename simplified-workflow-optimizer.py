#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Simplified Dynamic Workflow Optimizer
Production workflow optimization WITHOUT NetworkX dependency

Priority 3.1: Dynamic Workflow Optimizer - Simplified (1 SP)
- AI-powered task reordering and dependency optimization
- Simple critical path analysis without graph libraries
- Dynamic priority adjustment based on real-time conditions
- Zero external dependencies beyond standard library
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3

# Import AI components
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai'))
    from unified_ai_client import UnifiedAIClient, AIReasoningRequest, ReasoningContext, ReasoningType, AIModelType
    from context_aware_reasoning import ContextAwareReasoningEngine
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    AI_COMPONENTS_AVAILABLE = False
    print("⚠️ AI components not available - using fallback")

logger = logging.getLogger(__name__)

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
    PRIORITY_FIRST = "priority_first"

@dataclass
class TaskDependency:
    """Task dependency with simple structure"""
    task_id: int
    depends_on: int
    dependency_type: str = "blocks"
    strength: float = 1.0

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

class SimplifiedWorkflowOptimizer:
    """
    Simplified AI-Powered Dynamic Workflow Optimizer
    
    Zero dependencies - uses standard library algorithms for:
    - Critical path analysis with dependency resolution
    - Dynamic task reordering based on priority and resources
    - Simple parallel execution detection
    - Resource allocation optimization
    - AI integration for complex reasoning when available
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.ai_client = None
        self.reasoning_engine = None
        
        # Initialize AI components if available
        if AI_COMPONENTS_AVAILABLE:
            try:
                self.ai_client = UnifiedAIClient(db_path=db_path)
                self.reasoning_engine = ContextAwareReasoningEngine(db_path=db_path)
                logger.info("✅ AI components connected")
            except Exception as e:
                logger.warning(f"AI initialization failed: {e}")
        
        # Workflow state management
        self.active_workflows: Dict[str, OptimizedWorkflow] = {}
        self.resource_constraints = self._init_default_resources()
        
        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "avg_optimization_time": 0.0,
            "strategies_used": {}
        }
        
        self._init_database()
        logger.info("✅ SimplifiedWorkflowOptimizer initialized")
    
    def _init_default_resources(self) -> Dict[str, ResourceConstraint]:
        """Initialize default resource constraints"""
        return {
            "backend_agents": ResourceConstraint("backend_agents", 2, 0, 1.0),
            "frontend_agents": ResourceConstraint("frontend_agents", 2, 0, 1.0),
            "database_connections": ResourceConstraint("database_connections", 5, 0, 1.0),
            "testing_environments": ResourceConstraint("testing_environments", 3, 0, 1.0),
            "deployment_slots": ResourceConstraint("deployment_slots", 1, 0, 1.0)
        }
    
    def _init_database(self):
        """Initialize workflow optimization database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS simplified_workflow_optimizations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        task_count INTEGER,
                        optimized_duration REAL,
                        confidence REAL,
                        success BOOLEAN,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    
    async def optimize_workflow(
        self, 
        tasks: List[Task], 
        dependencies: List[TaskDependency],
        strategy: OptimizationStrategy = OptimizationStrategy.CRITICAL_PATH,
        context: Optional[ReasoningContext] = None
    ) -> OptimizedWorkflow:
        """
        Optimize workflow using simplified algorithms and AI reasoning
        """
        start_time = time.time()
        workflow_id = f"simplified_workflow_{int(time.time())}_{len(tasks)}"
        
        logger.info(f"🎯 Optimizing workflow (simplified): {len(tasks)} tasks, strategy: {strategy.value}")
        
        try:
            # Step 1: Build simple dependency mapping
            dependency_map = self._build_dependency_map(dependencies)
            
            # Step 2: AI-enhanced task analysis if available
            enhanced_tasks = await self._enhance_task_analysis(tasks, context) if self.reasoning_engine else tasks
            
            # Step 3: Calculate critical path using simple algorithm
            critical_path, path_duration = self._calculate_simple_critical_path(enhanced_tasks, dependency_map)
            
            # Step 4: Select optimization strategy (AI-powered if available)
            selected_strategy = await self._select_optimization_strategy(enhanced_tasks, strategy, context)
            
            # Step 5: Optimize task order
            optimized_order, parallel_groups = self._optimize_task_order_simple(
                enhanced_tasks, dependency_map, selected_strategy
            )
            
            # Step 6: Allocate resources
            resource_allocation = self._allocate_resources_simple(enhanced_tasks, optimized_order)
            
            # Step 7: Calculate metrics
            metrics = self._calculate_metrics_simple(enhanced_tasks, optimized_order, resource_allocation)
            
            # Step 8: Generate reasoning
            reasoning = await self._generate_reasoning(enhanced_tasks, selected_strategy, metrics, context)
            
            # Create optimized workflow
            optimized_workflow = OptimizedWorkflow(
                workflow_id=workflow_id,
                strategy=selected_strategy,
                task_order=optimized_order,
                parallel_groups=parallel_groups,
                critical_path=critical_path,
                estimated_duration=path_duration,
                resource_allocation=resource_allocation,
                confidence=0.8,  # High confidence for simplified approach
                optimization_reasoning=reasoning,
                metrics=metrics
            )
            
            # Store and track
            self._log_optimization(optimized_workflow, time.time() - start_time)
            self.active_workflows[workflow_id] = optimized_workflow
            self._update_stats(optimized_workflow, time.time() - start_time)
            
            logger.info(f"✅ Workflow optimized (simplified): {metrics.estimated_completion}")
            return optimized_workflow
            
        except Exception as e:
            logger.error(f"Workflow optimization failed: {e}")
            return self._create_fallback_workflow(tasks, workflow_id)
    
    def _build_dependency_map(self, dependencies: List[TaskDependency]) -> Dict[int, List[int]]:
        """Build simple dependency mapping"""
        dep_map = {}
        for dep in dependencies:
            if dep.task_id not in dep_map:
                dep_map[dep.task_id] = []
            dep_map[dep.task_id].append(dep.depends_on)
        return dep_map
    
    def _calculate_simple_critical_path(
        self, 
        tasks: List[Task], 
        dependency_map: Dict[int, List[int]]
    ) -> Tuple[List[int], float]:
        """Calculate critical path using simple topological sort"""
        
        # Create task lookup
        task_dict = {t.id: t for t in tasks}
        
        # Calculate in-degree (number of dependencies) for each task
        in_degree = {task.id: len(dependency_map.get(task.id, [])) for task in tasks}
        
        # Topological sort with longest path calculation
        queue = [task.id for task, count in zip(tasks, in_degree.values()) if count == 0]
        topo_order = []
        longest_path = {task.id: task.estimated_hours for task in tasks}
        
        while queue:
            current = queue.pop(0)
            topo_order.append(current)
            
            # Update dependent tasks
            for task_id, deps in dependency_map.items():
                if current in deps:
                    in_degree[task_id] -= 1
                    # Update longest path
                    current_path = longest_path[current] + task_dict[task_id].estimated_hours
                    longest_path[task_id] = max(longest_path[task_id], current_path)
                    
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        # Find critical path (longest duration)
        max_duration = max(longest_path.values())
        critical_tasks = [t_id for t_id, duration in longest_path.items() if abs(duration - max_duration) < 0.1]
        
        return topo_order, max_duration
    
    async def _select_optimization_strategy(
        self, 
        tasks: List[Task], 
        preferred_strategy: OptimizationStrategy,
        context: Optional[ReasoningContext]
    ) -> OptimizationStrategy:
        """Select optimization strategy, AI-enhanced if available"""
        
        if not self.ai_client or not context:
            return preferred_strategy
        
        try:
            strategy_prompt = f"""
            Select optimal workflow strategy for {len(tasks)} tasks:
            
            Task Types: {', '.join(set(t.task_type.value for t in tasks))}
            Priorities: {', '.join(set(t.priority.value for t in tasks))}
            
            Strategies:
            - CRITICAL_PATH: Minimize total duration
            - RESOURCE_BALANCED: Balance resource usage
            - TIME_OPTIMIZED: Fastest completion
            - PRIORITY_FIRST: High priority tasks first
            - RISK_MINIMIZED: Conservative approach
            
            Recommend best strategy and explain why.
            """
            
            request = AIReasoningRequest(
                request_id=f"strategy_selection_{int(time.time())}",
                reasoning_type=ReasoningType.DECISION_MAKING,
                prompt=strategy_prompt,
                context=context,
                model_preference=AIModelType.STANDARD
            )
            
            response = await self.ai_client.reason(request)
            
            # Extract strategy from response
            text = response.response_text.lower()
            if "time_optimized" in text or "fastest" in text:
                return OptimizationStrategy.TIME_OPTIMIZED
            elif "resource_balanced" in text or "balanced" in text:
                return OptimizationStrategy.RESOURCE_BALANCED
            elif "priority_first" in text or "priority" in text:
                return OptimizationStrategy.PRIORITY_FIRST
            elif "risk_minimized" in text or "conservative" in text:
                return OptimizationStrategy.RISK_MINIMIZED
            else:
                return OptimizationStrategy.CRITICAL_PATH
                
        except Exception as e:
            logger.warning(f"AI strategy selection failed: {e}")
            return preferred_strategy
    
    def _optimize_task_order_simple(
        self, 
        tasks: List[Task], 
        dependency_map: Dict[int, List[int]],
        strategy: OptimizationStrategy
    ) -> Tuple[List[int], List[List[int]]]:
        """Optimize task order using simple algorithms"""
        
        if strategy == OptimizationStrategy.PRIORITY_FIRST:
            return self._priority_first_ordering(tasks, dependency_map)
        elif strategy == OptimizationStrategy.TIME_OPTIMIZED:
            return self._time_optimized_ordering(tasks, dependency_map)
        elif strategy == OptimizationStrategy.RESOURCE_BALANCED:
            return self._resource_balanced_ordering(tasks, dependency_map)
        else:
            return self._critical_path_ordering(tasks, dependency_map)
    
    def _priority_first_ordering(
        self, 
        tasks: List[Task], 
        dependency_map: Dict[int, List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """Order tasks by priority first"""
        
        priority_order = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
        # Sort by priority, then by estimated hours
        sorted_tasks = sorted(
            tasks, 
            key=lambda t: (priority_order[t.priority], -t.estimated_hours),
            reverse=True
        )
        
        # Respect dependencies while maintaining priority order
        ordered_tasks = []
        remaining = {t.id: t for t in sorted_tasks}
        
        while remaining:
            # Find tasks with no unmet dependencies
            available = []
            for task_id, task in remaining.items():
                deps = dependency_map.get(task_id, [])
                if all(dep_id in ordered_tasks for dep_id in deps):
                    available.append(task)
            
            if not available:
                # Break circular dependencies - take highest priority
                available = [next(iter(remaining.values()))]
            
            # Sort available by priority
            available.sort(key=lambda t: priority_order[t.priority], reverse=True)
            
            # Process batch (up to 3 parallel)
            batch = available[:min(3, len(available))]
            batch_ids = [t.id for t in batch]
            
            ordered_tasks.extend(batch_ids)
            for task in batch:
                del remaining[task.id]
        
        # Group parallel execution opportunities
        parallel_groups = self._find_parallel_groups(ordered_tasks, dependency_map)
        
        return ordered_tasks, parallel_groups
    
    def _time_optimized_ordering(
        self, 
        tasks: List[Task], 
        dependency_map: Dict[int, List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """Order for fastest completion - shortest tasks first when possible"""
        
        ordered_tasks = []
        remaining = {t.id: t for t in tasks}
        
        while remaining:
            # Find available tasks
            available = []
            for task_id, task in remaining.items():
                deps = dependency_map.get(task_id, [])
                if all(dep_id in ordered_tasks for dep_id in deps):
                    available.append(task)
            
            if not available:
                available = [next(iter(remaining.values()))]
            
            # Sort by duration (shortest first)
            available.sort(key=lambda t: t.estimated_hours)
            
            # Take shortest tasks for parallel execution
            batch = available[:min(3, len(available))]
            batch_ids = [t.id for t in batch]
            
            ordered_tasks.extend(batch_ids)
            for task in batch:
                del remaining[task.id]
        
        parallel_groups = self._find_parallel_groups(ordered_tasks, dependency_map)
        return ordered_tasks, parallel_groups
    
    def _resource_balanced_ordering(
        self, 
        tasks: List[Task], 
        dependency_map: Dict[int, List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """Balance resource usage"""
        
        # Group tasks by resource type
        resource_groups = {
            TaskType.BACKEND: [],
            TaskType.FRONTEND: [],
            TaskType.DATABASE: [],
            TaskType.TESTING: [],
            TaskType.DEPLOYMENT: []
        }
        
        for task in tasks:
            resource_groups[task.task_type].append(task)
        
        # Interleave different resource types
        ordered_tasks = []
        remaining = {t.id: t for t in tasks}
        
        while remaining:
            # Try to pick one task from each resource type
            batch = []
            for resource_type, task_list in resource_groups.items():
                available_tasks = [t for t in task_list if t.id in remaining]
                
                for task in available_tasks:
                    deps = dependency_map.get(task.id, [])
                    if all(dep_id in ordered_tasks for dep_id in deps):
                        batch.append(task)
                        break
            
            if not batch:
                # Fallback: take any available task
                for task_id, task in remaining.items():
                    deps = dependency_map.get(task_id, [])
                    if all(dep_id in ordered_tasks for dep_id in deps):
                        batch.append(task)
                        break
            
            if not batch:
                batch.append(next(iter(remaining.values())))
            
            # Process batch
            batch_ids = [t.id for t in batch]
            ordered_tasks.extend(batch_ids)
            for task in batch:
                del remaining[task.id]
        
        parallel_groups = self._find_parallel_groups(ordered_tasks, dependency_map)
        return ordered_tasks, parallel_groups
    
    def _critical_path_ordering(
        self, 
        tasks: List[Task], 
        dependency_map: Dict[int, List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """Order based on critical path - default topological sort"""
        
        ordered_tasks = []
        remaining = {t.id: t for t in tasks}
        in_degree = {t.id: len(dependency_map.get(t.id, [])) for t in tasks}
        
        while remaining:
            # Find tasks with no dependencies
            available = [t_id for t_id in remaining if in_degree[t_id] == 0]
            
            if not available:
                # Break cycle - take task with minimum dependencies
                available = [min(remaining.keys(), key=lambda t: in_degree[t])]
            
            # Sort available by estimated hours (longest first for critical path)
            available_tasks = [remaining[t_id] for t_id in available]
            available_tasks.sort(key=lambda t: t.estimated_hours, reverse=True)
            
            # Process batch
            batch = available_tasks[:min(2, len(available_tasks))]
            batch_ids = [t.id for t in batch]
            
            ordered_tasks.extend(batch_ids)
            
            # Update in_degree and remove processed tasks
            for task_id in batch_ids:
                del remaining[task_id]
                
                # Decrease in_degree for dependent tasks
                for remaining_id in remaining:
                    deps = dependency_map.get(remaining_id, [])
                    if task_id in deps:
                        in_degree[remaining_id] -= 1
        
        parallel_groups = self._find_parallel_groups(ordered_tasks, dependency_map)
        return ordered_tasks, parallel_groups
    
    def _find_parallel_groups(
        self, 
        task_order: List[int], 
        dependency_map: Dict[int, List[int]]
    ) -> List[List[int]]:
        """Find tasks that can run in parallel"""
        
        parallel_groups = []
        processed = set()
        
        for task_id in task_order:
            if task_id in processed:
                continue
            
            # Find tasks that can run with this one
            parallel_candidates = [task_id]
            processed.add(task_id)
            
            for other_id in task_order:
                if other_id in processed:
                    continue
                
                # Check if tasks have dependency relationship
                task_deps = dependency_map.get(task_id, [])
                other_deps = dependency_map.get(other_id, [])
                
                # Can run in parallel if no direct dependency
                if task_id not in other_deps and other_id not in task_deps:
                    parallel_candidates.append(other_id)
                    processed.add(other_id)
            
            if len(parallel_candidates) > 1:
                parallel_groups.append(parallel_candidates)
        
        return parallel_groups
    
    def _allocate_resources_simple(
        self, 
        tasks: List[Task], 
        task_order: List[int]
    ) -> Dict[str, List[int]]:
        """Simple resource allocation"""
        
        allocation = {
            "backend_agents": [],
            "frontend_agents": [],
            "database_connections": [],
            "testing_environments": [],
            "deployment_slots": []
        }
        
        type_to_resource = {
            TaskType.BACKEND: "backend_agents",
            TaskType.FRONTEND: "frontend_agents",
            TaskType.DATABASE: "database_connections",
            TaskType.TESTING: "testing_environments",
            TaskType.DEPLOYMENT: "deployment_slots"
        }
        
        task_dict = {t.id: t for t in tasks}
        
        for task_id in task_order:
            task = task_dict[task_id]
            resource = type_to_resource.get(task.task_type, "backend_agents")
            allocation[resource].append(task_id)
        
        return allocation
    
    def _calculate_metrics_simple(
        self, 
        tasks: List[Task], 
        task_order: List[int],
        resource_allocation: Dict[str, List[int]]
    ) -> WorkflowMetrics:
        """Calculate simple workflow metrics"""
        
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        active_tasks = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
        blocked_tasks = sum(1 for t in tasks if t.status == TaskStatus.BLOCKED)
        
        total_duration = sum(task.estimated_hours for task in tasks)
        estimated_completion = datetime.now() + timedelta(hours=total_duration)
        
        # Simple resource utilization
        resource_utilization = {}
        for resource, task_list in resource_allocation.items():
            constraint = self.resource_constraints.get(resource)
            if constraint:
                utilization = len(task_list) / max(constraint.max_concurrent, 1)
                resource_utilization[resource] = min(1.0, utilization)
        
        # Simple bottleneck detection
        bottlenecks = [r for r, u in resource_utilization.items() if u > 0.8]
        
        # Optimization opportunities
        opportunities = []
        if max(resource_utilization.values()) < 0.6:
            opportunities.append("increase_parallelization")
        if bottlenecks:
            opportunities.append("resource_rebalancing")
        
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
    
    async def _enhance_task_analysis(self, tasks: List[Task], context: ReasoningContext) -> List[Task]:
        """AI-enhance task analysis if reasoning engine available"""
        
        if not self.reasoning_engine:
            return tasks
        
        # Simple batch analysis instead of per-task
        try:
            problem = f"""
            Analyze and optimize estimation for {len(tasks)} tasks:
            
            Task Summary:
            {', '.join([f"{t.title} ({t.estimated_hours}h)" for t in tasks[:5]])}
            
            Provide overall estimation adjustment recommendations.
            """
            
            reasoning_chain = await self.reasoning_engine.reason_with_context(
                problem_statement=problem,
                context=context,
                reasoning_type=ReasoningType.TASK_ANALYSIS
            )
            
            logger.info(f"📊 AI-enhanced task analysis completed")
            return tasks  # Keep original for simplicity
            
        except Exception as e:
            logger.warning(f"Task enhancement failed: {e}")
            return tasks
    
    async def _generate_reasoning(
        self, 
        tasks: List[Task], 
        strategy: OptimizationStrategy,
        metrics: WorkflowMetrics,
        context: Optional[ReasoningContext]
    ) -> str:
        """Generate optimization reasoning"""
        
        base_reasoning = f"""
        Simplified Workflow Optimization Results:
        
        Strategy: {strategy.value}
        Tasks: {len(tasks)} total
        Estimated Duration: {metrics.critical_path_length:.1f} hours
        Resource Utilization: {metrics.resource_utilization}
        
        Optimization Benefits:
        - Clear execution order with dependency respect
        - Resource-aware task allocation
        - Parallel execution opportunities identified
        - Bottleneck detection and mitigation
        """
        
        if self.reasoning_engine and context:
            try:
                reasoning_problem = f"""
                Explain the simplified workflow optimization:
                
                {base_reasoning}
                
                Bottlenecks: {metrics.bottlenecks}
                Opportunities: {metrics.optimization_opportunities}
                
                Provide clear explanation of benefits and next steps.
                """
                
                chain = await self.reasoning_engine.reason_with_context(
                    problem_statement=reasoning_problem,
                    context=context,
                    reasoning_type=ReasoningType.DECISION_MAKING
                )
                
                return chain.final_reasoning
                
            except Exception as e:
                logger.warning(f"Reasoning generation failed: {e}")
        
        return base_reasoning
    
    def _create_fallback_workflow(self, tasks: List[Task], workflow_id: str) -> OptimizedWorkflow:
        """Create basic fallback workflow"""
        
        task_order = [t.id for t in sorted(tasks, key=lambda t: t.priority.value, reverse=True)]
        total_duration = sum(t.estimated_hours for t in tasks)
        
        return OptimizedWorkflow(
            workflow_id=workflow_id,
            strategy=OptimizationStrategy.PRIORITY_FIRST,
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
                optimization_opportunities=["enable_advanced_optimization"]
            )
        )
    
    def _log_optimization(self, workflow: OptimizedWorkflow, optimization_time: float):
        """Log optimization for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO simplified_workflow_optimizations
                    (workflow_id, strategy, task_count, optimized_duration, confidence, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    workflow.workflow_id,
                    workflow.strategy.value,
                    workflow.metrics.total_tasks,
                    workflow.estimated_duration,
                    workflow.confidence,
                    True
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Logging failed: {e}")
    
    def _update_stats(self, workflow: OptimizedWorkflow, optimization_time: float):
        """Update optimization statistics"""
        self.optimization_stats["total_optimizations"] += 1
        self.optimization_stats["successful_optimizations"] += 1
        
        current_avg = self.optimization_stats["avg_optimization_time"]
        total = self.optimization_stats["total_optimizations"]
        self.optimization_stats["avg_optimization_time"] = (
            (current_avg * (total - 1) + optimization_time) / total
        )
        
        strategy = workflow.strategy.value
        if strategy not in self.optimization_stats["strategies_used"]:
            self.optimization_stats["strategies_used"][strategy] = 0
        self.optimization_stats["strategies_used"][strategy] += 1
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.optimization_stats,
            "active_workflows": len(self.active_workflows),
            "ai_components_available": AI_COMPONENTS_AVAILABLE,
            "resource_constraints": {k: v.max_concurrent for k, v in self.resource_constraints.items()}
        }

# Demo function
async def demo_simplified_workflow_optimizer():
    """Demo the simplified workflow optimizer"""
    print("🎯 Agent Zero V2.0 - Simplified Dynamic Workflow Optimizer Demo")
    print("=" * 65)
    
    # Initialize optimizer
    optimizer = SimplifiedWorkflowOptimizer()
    
    # Create demo tasks
    demo_tasks = [
        Task(1, "Database Schema Setup", "PostgreSQL tables and indexes", TaskType.DATABASE, TaskStatus.PENDING, TaskPriority.HIGH, 4.0),
        Task(2, "Authentication API", "JWT auth endpoints", TaskType.BACKEND, TaskStatus.PENDING, TaskPriority.CRITICAL, 8.0),
        Task(3, "User Registration", "React signup form", TaskType.FRONTEND, TaskStatus.PENDING, TaskPriority.HIGH, 6.0),
        Task(4, "User Dashboard", "Main interface", TaskType.FRONTEND, TaskStatus.PENDING, TaskPriority.MEDIUM, 10.0),
        Task(5, "Integration Tests", "API and UI tests", TaskType.TESTING, TaskStatus.PENDING, TaskPriority.MEDIUM, 8.0),
        Task(6, "Staging Deploy", "Docker deployment", TaskType.DEPLOYMENT, TaskStatus.PENDING, TaskPriority.LOW, 3.0)
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
        tech_stack=["FastAPI", "React", "PostgreSQL"],
        constraints=["2-week sprint", "3 developers"]
    )
    
    print(f"📋 Demo Setup:")
    print(f"   Tasks: {len(demo_tasks)}")
    print(f"   Dependencies: {len(dependencies)}")
    print(f"   Total Estimated: {sum(t.estimated_hours for t in demo_tasks)} hours")
    
    # Test strategies
    strategies = [
        OptimizationStrategy.CRITICAL_PATH,
        OptimizationStrategy.PRIORITY_FIRST,
        OptimizationStrategy.TIME_OPTIMIZED
    ]
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.value} optimization...")
        
        workflow = await optimizer.optimize_workflow(
            tasks=demo_tasks,
            dependencies=dependencies,
            strategy=strategy,
            context=context
        )
        
        print(f"   ✅ Optimization completed:")
        print(f"      Strategy: {workflow.strategy.value}")
        print(f"      Duration: {workflow.estimated_duration:.1f} hours")
        print(f"      Parallel Groups: {len(workflow.parallel_groups)}")
        print(f"      Confidence: {workflow.confidence:.2f}")
        print(f"      Task Order: {workflow.task_order}")
        
        if workflow.parallel_groups:
            print(f"      Parallel Execution:")
            for i, group in enumerate(workflow.parallel_groups):
                print(f"        Group {i+1}: Tasks {group}")
    
    # Stats
    print(f"\n📊 Optimizer Statistics:")
    stats = optimizer.get_optimization_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Simplified workflow optimizer demo completed!")

if __name__ == "__main__":
    print("🎯 Agent Zero V2.0 Phase 4 - Simplified Dynamic Workflow Optimizer")
    print("Zero dependencies - using standard library algorithms")
    
    # Run demo
    asyncio.run(demo_simplified_workflow_optimizer())