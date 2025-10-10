# Hierarchical Task Planner - Core Module for Agent Zero V1
# Task: A0-17 V2.0 Development Plan - Start Hierarchical Task Planner (1 SP fundament)
# Timeline: 14:00-17:00
# Focus: Architektura systemu, base classes, testing framework

"""
Hierarchical Task Planner for Agent Zero V1
Advanced task decomposition and planning system

This module provides:
- Hierarchical task breakdown structure
- Intelligent task sequencing and dependencies
- Resource allocation and optimization
- Integration with existing Agent Zero components
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import networkx as nx
from collections import defaultdict, deque
import heapq

# Import existing components
try:
    import sys
    sys.path.insert(0, '.')
    from business_requirements_parser import BusinessRequirementsParser, IntentType, ComplexityLevel
    from simple_tracker import SimpleTracker
    from project_orchestrator import ProjectOrchestrator, Task, TaskStatus
except ImportError as e:
    logging.warning(f"Could not import existing components: {e}")
    # Fallback for testing
    class BusinessRequirementsParser:
        def parse_intent(self, text): return None
    class SimpleTracker:
        def track_event(self, event): pass

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    DEFERRED = 5

class TaskType(Enum):
    """Types of tasks in hierarchy"""
    EPIC = "epic"           # Large feature or project phase
    STORY = "story"         # User-facing functionality
    TASK = "task"           # Technical implementation
    SUBTASK = "subtask"     # Granular work items
    BUG_FIX = "bug_fix"     # Issue resolution
    RESEARCH = "research"   # Investigation or analysis

@dataclass
class TaskDependency:
    """Task dependency relationship"""
    predecessor_id: str
    successor_id: str
    dependency_type: str = "finish_to_start"  # finish_to_start, start_to_start, etc.
    lag_time: int = 0  # Minutes delay between tasks
    
@dataclass 
class ResourceRequirement:
    """Resource requirements for task execution"""
    agent_types: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_duration: int = 0  # minutes
    cpu_requirement: float = 1.0  # CPU units needed
    memory_requirement: int = 512  # MB memory needed
    storage_requirement: int = 100  # MB storage needed

@dataclass
class HierarchicalTask:
    """Enhanced task with hierarchical properties"""
    task_id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    business_value: int = 0  # 1-10 scale
    complexity_score: int = 0  # 1-10 scale
    risk_score: int = 0  # 1-10 scale
    
    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    level: int = 0  # Depth in hierarchy
    
    # Dependencies
    dependencies: List[TaskDependency] = field(default_factory=list)
    blocks: Set[str] = field(default_factory=set)  # Tasks this blocks
    
    # Resources
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    
    # Scheduling
    earliest_start: Optional[datetime] = None
    latest_start: Optional[datetime] = None
    earliest_finish: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    slack_time: int = 0  # minutes of scheduling flexibility
    
    # Execution
    status: TaskStatus = TaskStatus.PENDING
    assigned_agents: List[str] = field(default_factory=list)
    progress_percentage: int = 0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    acceptance_criteria: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

class TaskDecompositionEngine:
    """Intelligent task decomposition based on complexity and patterns"""
    
    def __init__(self):
        self.business_parser = BusinessRequirementsParser()
        self.logger = logging.getLogger(__name__)
        
        # Decomposition patterns for different intent types
        self.decomposition_patterns = {
            IntentType.CREATE: [
                "Analyze requirements",
                "Design architecture", 
                "Implement core functionality",
                "Add error handling",
                "Write tests",
                "Create documentation",
                "Deploy and validate"
            ],
            IntentType.ANALYZE: [
                "Gather data sources",
                "Clean and preprocess data",
                "Perform analysis",
                "Generate insights",
                "Create visualizations",
                "Document findings",
                "Present results"
            ],
            IntentType.OPTIMIZE: [
                "Profile current performance",
                "Identify bottlenecks",
                "Design optimizations",
                "Implement improvements",
                "Benchmark results",
                "Validate quality",
                "Document changes"
            ]
        }
        
        # Complexity-based task multiplication factors
        self.complexity_factors = {
            ComplexityLevel.SIMPLE: 1.0,
            ComplexityLevel.MODERATE: 1.5,
            ComplexityLevel.COMPLEX: 2.5,
            ComplexityLevel.ENTERPRISE: 4.0
        }
    
    def decompose_business_request(self, business_request: str, 
                                 max_depth: int = 3) -> List[HierarchicalTask]:
        """Decompose business request into hierarchical tasks"""
        
        # Parse business intent
        intent = self.business_parser.parse_intent(business_request)
        if not intent:
            # Fallback for simple decomposition
            return self._create_simple_task(business_request)
        
        # Create root epic task
        root_task = HierarchicalTask(
            task_id=f"epic_{int(time.time())}",
            name=f"Epic: {business_request[:50]}...",
            description=business_request,
            task_type=TaskType.EPIC,
            priority=self._determine_priority(intent),
            business_value=self._calculate_business_value(intent),
            complexity_score=self._calculate_complexity_score(intent),
            risk_score=self._calculate_risk_score(intent),
            level=0,
            created_at=datetime.now()
        )
        
        tasks = [root_task]
        
        # Decompose based on intent and complexity
        child_tasks = self._decompose_by_pattern(intent, max_depth - 1)
        
        for child_task in child_tasks:
            child_task.parent_id = root_task.task_id
            child_task.level = 1
            root_task.children_ids.add(child_task.task_id)
            tasks.append(child_task)
            
            # Further decompose complex tasks
            if (child_task.complexity_score > 5 and max_depth > 2):
                grandchild_tasks = self._decompose_task_further(child_task, max_depth - 2)
                for grandchild in grandchild_tasks:
                    grandchild.parent_id = child_task.task_id
                    grandchild.level = 2
                    child_task.children_ids.add(grandchild.task_id)
                    tasks.append(grandchild)
        
        return tasks
    
    def _create_simple_task(self, business_request: str) -> List[HierarchicalTask]:
        """Create simple task when parsing fails"""
        task = HierarchicalTask(
            task_id=f"task_{int(time.time())}",
            name=business_request[:100],
            description=business_request,
            task_type=TaskType.TASK,
            priority=TaskPriority.MEDIUM,
            business_value=5,
            complexity_score=3,
            risk_score=2,
            created_at=datetime.now()
        )
        return [task]
    
    def _decompose_by_pattern(self, intent, max_depth: int) -> List[HierarchicalTask]:
        """Decompose based on intent patterns"""
        
        pattern = self.decomposition_patterns.get(intent.primary_action, [
            "Analyze requirements",
            "Implement solution", 
            "Test and validate",
            "Document results"
        ])
        
        tasks = []
        complexity_factor = self.complexity_factors.get(intent.complexity, 1.0)
        
        for i, step_name in enumerate(pattern):
            task_id = f"story_{int(time.time())}_{i}"
            
            task = HierarchicalTask(
                task_id=task_id,
                name=f"{step_name} - {intent.primary_action.value}",
                description=f"{step_name} for: {intent.context.get('original_request', 'N/A')}",
                task_type=TaskType.STORY,
                priority=self._determine_priority(intent),
                business_value=max(1, int(8 - i)),  # Earlier steps more valuable
                complexity_score=max(1, int(3 * complexity_factor)),
                risk_score=self._calculate_risk_score(intent),
                created_at=datetime.now()
            )
            
            # Set dependencies (sequential by default)
            if i > 0:
                predecessor_id = f"story_{int(time.time())}_{i-1}"
                dependency = TaskDependency(
                    predecessor_id=tasks[i-1].task_id,
                    successor_id=task_id,
                    dependency_type="finish_to_start"
                )
                task.dependencies.append(dependency)
            
            tasks.append(task)
        
        return tasks
    
    def _decompose_task_further(self, parent_task: HierarchicalTask, max_depth: int) -> List[HierarchicalTask]:
        """Further decompose complex tasks into subtasks"""
        
        if max_depth < 1:
            return []
        
        subtasks = []
        
        # Generic subtask patterns based on task type
        if parent_task.task_type == TaskType.STORY:
            subtask_names = [
                "Setup and preparation",
                "Core implementation",
                "Integration testing",
                "Quality assurance",
                "Documentation"
            ]
        else:
            subtask_names = [
                "Analysis and planning",
                "Implementation",
                "Testing and validation"
            ]
        
        for i, subtask_name in enumerate(subtask_names):
            subtask = HierarchicalTask(
                task_id=f"subtask_{int(time.time())}_{i}",
                name=f"{subtask_name}",
                description=f"{subtask_name} for {parent_task.name}",
                task_type=TaskType.SUBTASK,
                priority=parent_task.priority,
                business_value=max(1, parent_task.business_value - 2),
                complexity_score=max(1, parent_task.complexity_score - 2),
                risk_score=parent_task.risk_score,
                level=parent_task.level + 1,
                created_at=datetime.now()
            )
            
            # Sequential dependencies for subtasks
            if i > 0:
                dependency = TaskDependency(
                    predecessor_id=subtasks[i-1].task_id,
                    successor_id=subtask.task_id,
                    dependency_type="finish_to_start"
                )
                subtask.dependencies.append(dependency)
            
            subtasks.append(subtask)
        
        return subtasks
    
    def _determine_priority(self, intent) -> TaskPriority:
        """Determine task priority based on intent"""
        
        if 'urgent' in intent.constraints:
            return TaskPriority.CRITICAL
        elif intent.complexity == ComplexityLevel.ENTERPRISE:
            return TaskPriority.HIGH
        elif intent.complexity == ComplexityLevel.COMPLEX:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
    
    def _calculate_business_value(self, intent) -> int:
        """Calculate business value score 1-10"""
        
        value = 5  # Base value
        
        # Intent type adjustments
        if intent.primary_action in [IntentType.CREATE, IntentType.GENERATE]:
            value += 2  # Creating new value
        elif intent.primary_action == IntentType.OPTIMIZE:
            value += 3  # High value optimization
        elif intent.primary_action == IntentType.ANALYZE:
            value += 1  # Insight generation
        
        # Complexity adjustments
        if intent.complexity == ComplexityLevel.ENTERPRISE:
            value += 2
        elif intent.complexity == ComplexityLevel.SIMPLE:
            value -= 1
        
        # Constraint adjustments
        if 'quality_requirement' in intent.constraints:
            value += 1
        
        return max(1, min(10, value))
    
    def _calculate_complexity_score(self, intent) -> int:
        """Calculate complexity score 1-10"""
        
        complexity_map = {
            ComplexityLevel.SIMPLE: 2,
            ComplexityLevel.MODERATE: 4,
            ComplexityLevel.COMPLEX: 7,
            ComplexityLevel.ENTERPRISE: 9
        }
        
        score = complexity_map.get(intent.complexity, 5)
        
        # Adjust based on entities
        score += min(len(intent.target_entities), 3)
        
        return max(1, min(10, score))
    
    def _calculate_risk_score(self, intent) -> int:
        """Calculate risk score 1-10"""
        
        risk = 3  # Base risk
        
        # Higher risk for complex tasks
        if intent.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
            risk += 2
        
        # Higher risk for creation tasks
        if intent.primary_action in [IntentType.CREATE, IntentType.DELETE]:
            risk += 2
        
        # Lower confidence = higher risk
        if intent.confidence < 0.7:
            risk += 2
        
        return max(1, min(10, risk))

class TaskScheduler:
    """Critical Path Method (CPM) scheduler for task optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def schedule_tasks(self, tasks: List[HierarchicalTask]) -> Dict[str, HierarchicalTask]:
        """Schedule tasks using Critical Path Method"""
        
        # Build dependency graph
        graph = self._build_dependency_graph(tasks)
        
        # Calculate earliest start/finish times (forward pass)
        self._forward_pass(tasks, graph)
        
        # Calculate latest start/finish times (backward pass) 
        self._backward_pass(tasks, graph)
        
        # Calculate slack time and identify critical path
        critical_path = self._calculate_slack_and_critical_path(tasks, graph)
        
        # Optimize resource allocation
        self._optimize_resource_allocation(tasks, critical_path)
        
        return {task.task_id: task for task in tasks}
    
    def _build_dependency_graph(self, tasks: List[HierarchicalTask]) -> nx.DiGraph:
        """Build directed graph of task dependencies"""
        
        graph = nx.DiGraph()
        
        # Add nodes
        for task in tasks:
            graph.add_node(task.task_id, task=task)
        
        # Add edges for dependencies
        for task in tasks:
            for dep in task.dependencies:
                if graph.has_node(dep.predecessor_id):
                    graph.add_edge(
                        dep.predecessor_id, 
                        dep.successor_id,
                        lag_time=dep.lag_time,
                        dependency_type=dep.dependency_type
                    )
        
        return graph
    
    def _forward_pass(self, tasks: List[HierarchicalTask], graph: nx.DiGraph):
        """Calculate earliest start and finish times"""
        
        # Initialize all tasks
        task_dict = {task.task_id: task for task in tasks}
        
        # Find tasks with no predecessors (start tasks)
        start_tasks = [
            task for task in tasks 
            if not task.dependencies or not any(
                graph.has_node(dep.predecessor_id) for dep in task.dependencies
            )
        ]
        
        # Set earliest start for start tasks
        project_start = datetime.now()
        for task in start_tasks:
            task.earliest_start = project_start
            task.earliest_finish = project_start + timedelta(
                minutes=task.resource_requirements.estimated_duration
            )
        
        # Process tasks in topological order
        try:
            for task_id in nx.topological_sort(graph):
                task = task_dict.get(task_id)
                if not task or task in start_tasks:
                    continue
                
                # Find latest finish time of all predecessors
                latest_predecessor_finish = project_start
                
                for dep in task.dependencies:
                    if dep.predecessor_id in task_dict:
                        pred_task = task_dict[dep.predecessor_id]
                        if pred_task.earliest_finish:
                            pred_finish_with_lag = pred_task.earliest_finish + timedelta(
                                minutes=dep.lag_time
                            )
                            latest_predecessor_finish = max(
                                latest_predecessor_finish,
                                pred_finish_with_lag
                            )
                
                task.earliest_start = latest_predecessor_finish
                task.earliest_finish = task.earliest_start + timedelta(
                    minutes=task.resource_requirements.estimated_duration
                )
        
        except nx.NetworkXError as e:
            self.logger.error(f"Circular dependency detected: {e}")
            # Handle circular dependencies by breaking them
            self._handle_circular_dependencies(graph, task_dict)
    
    def _backward_pass(self, tasks: List[HierarchicalTask], graph: nx.DiGraph):
        """Calculate latest start and finish times"""
        
        task_dict = {task.task_id: task for task in tasks}
        
        # Find end tasks (no successors)
        end_tasks = [
            task for task in tasks
            if not any(graph.has_edge(task.task_id, other_id) for other_id in graph.nodes())
        ]
        
        # Set latest finish for end tasks (same as earliest finish)
        for task in end_tasks:
            task.latest_finish = task.earliest_finish
            task.latest_start = task.latest_finish - timedelta(
                minutes=task.resource_requirements.estimated_duration
            )
        
        # Process tasks in reverse topological order
        try:
            for task_id in reversed(list(nx.topological_sort(graph))):
                task = task_dict.get(task_id)
                if not task or task in end_tasks:
                    continue
                
                # Find earliest start time of all successors
                earliest_successor_start = task.earliest_finish  # Default
                
                for successor_id in graph.successors(task_id):
                    successor = task_dict.get(successor_id)
                    if successor and successor.latest_start:
                        edge_data = graph.get_edge_data(task_id, successor_id)
                        lag_time = edge_data.get('lag_time', 0)
                        
                        successor_start_with_lag = successor.latest_start - timedelta(
                            minutes=lag_time
                        )
                        earliest_successor_start = min(
                            earliest_successor_start,
                            successor_start_with_lag
                        )
                
                task.latest_finish = earliest_successor_start
                task.latest_start = task.latest_finish - timedelta(
                    minutes=task.resource_requirements.estimated_duration
                )
        
        except nx.NetworkXError as e:
            self.logger.error(f"Error in backward pass: {e}")
    
    def _calculate_slack_and_critical_path(self, tasks: List[HierarchicalTask], 
                                         graph: nx.DiGraph) -> List[str]:
        """Calculate slack time and identify critical path"""
        
        critical_tasks = []
        
        for task in tasks:
            if task.earliest_start and task.latest_start:
                slack_delta = task.latest_start - task.earliest_start
                task.slack_time = int(slack_delta.total_seconds() / 60)
                
                # Tasks with zero slack are on critical path
                if task.slack_time <= 0:
                    critical_tasks.append(task.task_id)
        
        return critical_tasks
    
    def _optimize_resource_allocation(self, tasks: List[HierarchicalTask], 
                                    critical_path: List[str]):
        """Optimize resource allocation based on critical path"""
        
        # Prioritize critical path tasks
        for task in tasks:
            if task.task_id in critical_path:
                # Increase resource allocation for critical tasks
                task.resource_requirements.cpu_requirement *= 1.5
                task.resource_requirements.memory_requirement = int(
                    task.resource_requirements.memory_requirement * 1.5
                )
                
                # Assign best agents to critical tasks
                if not task.assigned_agents:
                    task.assigned_agents = self._get_optimal_agents(task)
    
    def _get_optimal_agents(self, task: HierarchicalTask) -> List[str]:
        """Get optimal agents for task based on requirements"""
        
        # Simplified agent selection based on task type and complexity
        if task.task_type == TaskType.EPIC:
            return ["project_manager", "architect"]
        elif task.complexity_score >= 8:
            return ["senior_developer", "specialist"]
        elif task.complexity_score >= 5:
            return ["developer", "tester"]
        else:
            return ["junior_developer"]
    
    def _handle_circular_dependencies(self, graph: nx.DiGraph, task_dict: Dict):
        """Handle circular dependencies by breaking cycles"""
        
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    # Break cycle by removing edge with lowest priority
                    min_priority_edge = None
                    min_priority = TaskPriority.CRITICAL.value
                    
                    for i in range(len(cycle)):
                        from_task = cycle[i]
                        to_task = cycle[(i + 1) % len(cycle)]
                        
                        if graph.has_edge(from_task, to_task):
                            task = task_dict.get(from_task)
                            if task and task.priority.value > min_priority:
                                min_priority = task.priority.value
                                min_priority_edge = (from_task, to_task)
                    
                    if min_priority_edge:
                        graph.remove_edge(*min_priority_edge)
                        self.logger.warning(f"Removed circular dependency: {min_priority_edge}")
        
        except Exception as e:
            self.logger.error(f"Error handling circular dependencies: {e}")

class HierarchicalTaskPlanner:
    """Main Hierarchical Task Planner class"""
    
    def __init__(self, integration_mode: bool = True):
        self.decomposition_engine = TaskDecompositionEngine()
        self.scheduler = TaskScheduler()
        self.logger = logging.getLogger(__name__)
        
        # Integration with existing components
        self.integration_mode = integration_mode
        if integration_mode:
            try:
                self.tracker = SimpleTracker()
                self.orchestrator = ProjectOrchestrator()
            except:
                self.tracker = None
                self.orchestrator = None
                self.logger.warning("Could not initialize integration components")
        
        # In-memory storage for tasks
        self.task_hierarchies: Dict[str, Dict[str, HierarchicalTask]] = {}
        self.execution_queue: List[Tuple[int, str, str]] = []  # (priority, hierarchy_id, task_id)
    
    async def create_task_hierarchy(self, hierarchy_id: str, business_requests: List[str],
                                  max_depth: int = 3) -> Dict[str, HierarchicalTask]:
        """Create hierarchical task breakdown from business requests"""
        
        all_tasks = []
        
        # Decompose each business request
        for i, request in enumerate(business_requests):
            self.logger.info(f"Decomposing request {i+1}/{len(business_requests)}: {request[:100]}...")
            
            request_tasks = self.decomposition_engine.decompose_business_request(
                request, max_depth
            )
            
            # Add unique prefix to avoid ID collisions
            for task in request_tasks:
                task.task_id = f"{hierarchy_id}_{i}_{task.task_id}"
                if task.parent_id:
                    task.parent_id = f"{hierarchy_id}_{i}_{task.parent_id}"
                
                # Update children IDs
                task.children_ids = {
                    f"{hierarchy_id}_{i}_{child_id}" for child_id in task.children_ids
                }
                
                # Update dependency IDs
                for dep in task.dependencies:
                    if not dep.predecessor_id.startswith(hierarchy_id):
                        dep.predecessor_id = f"{hierarchy_id}_{i}_{dep.predecessor_id}"
                    if not dep.successor_id.startswith(hierarchy_id):
                        dep.successor_id = f"{hierarchy_id}_{i}_{dep.successor_id}"
            
            all_tasks.extend(request_tasks)
        
        # Add inter-request dependencies if needed
        self._create_inter_request_dependencies(all_tasks, len(business_requests))
        
        # Schedule tasks
        scheduled_tasks = self.scheduler.schedule_tasks(all_tasks)
        
        # Store hierarchy
        self.task_hierarchies[hierarchy_id] = scheduled_tasks
        
        # Track in SimpleTracker if available
        if self.tracker:
            self.tracker.track_event({
                'type': 'task_hierarchy_created',
                'hierarchy_id': hierarchy_id,
                'total_tasks': len(scheduled_tasks),
                'max_depth': max_depth,
                'business_requests_count': len(business_requests)
            })
        
        self.logger.info(f"Created task hierarchy {hierarchy_id} with {len(scheduled_tasks)} tasks")
        
        return scheduled_tasks
    
    def _create_inter_request_dependencies(self, all_tasks: List[HierarchicalTask], 
                                         request_count: int):
        """Create dependencies between different business requests"""
        
        if request_count < 2:
            return
        
        # Group tasks by request (based on ID prefix)
        request_groups = defaultdict(list)
        for task in all_tasks:
            # Extract request index from task ID
            parts = task.task_id.split('_')
            if len(parts) >= 3:
                request_idx = parts[1]
                request_groups[request_idx].append(task)
        
        # Create sequential dependencies between request groups
        request_indices = sorted(request_groups.keys(), key=int)
        
        for i in range(1, len(request_indices)):
            prev_request = request_indices[i-1]
            curr_request = request_indices[i]
            
            # Find end tasks of previous request (no children)
            prev_end_tasks = [
                task for task in request_groups[prev_request]
                if not task.children_ids and task.task_type != TaskType.EPIC
            ]
            
            # Find start tasks of current request
            curr_start_tasks = [
                task for task in request_groups[curr_request]
                if not task.dependencies and task.task_type != TaskType.EPIC
            ]
            
            # Create dependencies
            if prev_end_tasks and curr_start_tasks:
                for curr_task in curr_start_tasks:
                    # Add dependency on the last end task of previous request
                    dependency = TaskDependency(
                        predecessor_id=prev_end_tasks[-1].task_id,
                        successor_id=curr_task.task_id,
                        dependency_type="finish_to_start",
                        lag_time=5  # 5 minute buffer between requests
                    )
                    curr_task.dependencies.append(dependency)
    
    def get_task_hierarchy(self, hierarchy_id: str) -> Optional[Dict[str, HierarchicalTask]]:
        """Get task hierarchy by ID"""
        return self.task_hierarchies.get(hierarchy_id)
    
    def get_task(self, hierarchy_id: str, task_id: str) -> Optional[HierarchicalTask]:
        """Get specific task from hierarchy"""
        hierarchy = self.get_task_hierarchy(hierarchy_id)
        if hierarchy:
            return hierarchy.get(task_id)
        return None
    
    def get_ready_tasks(self, hierarchy_id: str) -> List[HierarchicalTask]:
        """Get tasks that are ready to execute (no pending dependencies)"""
        
        hierarchy = self.get_task_hierarchy(hierarchy_id)
        if not hierarchy:
            return []
        
        ready_tasks = []
        
        for task in hierarchy.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_met = True
                for dep in task.dependencies:
                    dep_task = hierarchy.get(dep.predecessor_id)
                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    ready_tasks.append(task)
        
        # Sort by priority and business value
        ready_tasks.sort(key=lambda t: (t.priority.value, -t.business_value))
        
        return ready_tasks
    
    def get_critical_path(self, hierarchy_id: str) -> List[HierarchicalTask]:
        """Get tasks on the critical path"""
        
        hierarchy = self.get_task_hierarchy(hierarchy_id)
        if not hierarchy:
            return []
        
        critical_tasks = [
            task for task in hierarchy.values()
            if task.slack_time <= 0
        ]
        
        # Sort by earliest start time
        critical_tasks.sort(key=lambda t: t.earliest_start or datetime.now())
        
        return critical_tasks
    
    def update_task_progress(self, hierarchy_id: str, task_id: str, 
                           progress: int, status: Optional[TaskStatus] = None):
        """Update task progress and status"""
        
        task = self.get_task(hierarchy_id, task_id)
        if not task:
            return False
        
        old_status = task.status
        task.progress_percentage = max(0, min(100, progress))
        
        if status:
            task.status = status
            
            # Update timestamps
            if status == TaskStatus.RUNNING and old_status == TaskStatus.PENDING:
                task.started_at = datetime.now()
            elif status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now()
        
        # Track progress in SimpleTracker
        if self.tracker:
            self.tracker.track_event({
                'type': 'task_progress_updated',
                'hierarchy_id': hierarchy_id,
                'task_id': task_id,
                'progress': progress,
                'status': status.value if status else None,
                'old_status': old_status.value
            })
        
        return True
    
    def get_hierarchy_statistics(self, hierarchy_id: str) -> Optional[Dict]:
        """Get comprehensive statistics for task hierarchy"""
        
        hierarchy = self.get_task_hierarchy(hierarchy_id)
        if not hierarchy:
            return None
        
        tasks = list(hierarchy.values())
        
        # Status counts
        status_counts = defaultdict(int)
        for task in tasks:
            status_counts[task.status.value] += 1
        
        # Priority counts
        priority_counts = defaultdict(int)
        for task in tasks:
            priority_counts[task.priority.value] += 1
        
        # Type counts
        type_counts = defaultdict(int)
        for task in tasks:
            type_counts[task.task_type.value] += 1
        
        # Progress statistics
        total_progress = sum(task.progress_percentage for task in tasks)
        avg_progress = total_progress / len(tasks) if tasks else 0
        
        # Time and cost estimates
        total_estimated_duration = sum(
            task.resource_requirements.estimated_duration for task in tasks
        )
        total_estimated_cost = sum(
            task.resource_requirements.estimated_cost for task in tasks
        )
        
        # Critical path statistics
        critical_tasks = self.get_critical_path(hierarchy_id)
        critical_path_duration = sum(
            task.resource_requirements.estimated_duration for task in critical_tasks
        )
        
        return {
            'hierarchy_id': hierarchy_id,
            'total_tasks': len(tasks),
            'status_counts': dict(status_counts),
            'priority_counts': dict(priority_counts),
            'type_counts': dict(type_counts),
            'avg_progress': round(avg_progress, 1),
            'total_estimated_duration': total_estimated_duration,
            'total_estimated_cost': total_estimated_cost,
            'critical_path_tasks': len(critical_tasks),
            'critical_path_duration': critical_path_duration,
            'ready_tasks': len(self.get_ready_tasks(hierarchy_id))
        }
    
    def export_hierarchy_to_dict(self, hierarchy_id: str) -> Optional[Dict]:
        """Export task hierarchy to dictionary for serialization"""
        
        hierarchy = self.get_task_hierarchy(hierarchy_id)
        if not hierarchy:
            return None
        
        return {
            'hierarchy_id': hierarchy_id,
            'tasks': {
                task_id: {
                    'task_id': task.task_id,
                    'name': task.name,
                    'description': task.description,
                    'task_type': task.task_type.value,
                    'priority': task.priority.value,
                    'business_value': task.business_value,
                    'complexity_score': task.complexity_score,
                    'risk_score': task.risk_score,
                    'parent_id': task.parent_id,
                    'children_ids': list(task.children_ids),
                    'level': task.level,
                    'dependencies': [
                        {
                            'predecessor_id': dep.predecessor_id,
                            'successor_id': dep.successor_id,
                            'dependency_type': dep.dependency_type,
                            'lag_time': dep.lag_time
                        } for dep in task.dependencies
                    ],
                    'resource_requirements': asdict(task.resource_requirements),
                    'earliest_start': task.earliest_start.isoformat() if task.earliest_start else None,
                    'latest_start': task.latest_start.isoformat() if task.latest_start else None,
                    'earliest_finish': task.earliest_finish.isoformat() if task.earliest_finish else None,
                    'latest_finish': task.latest_finish.isoformat() if task.latest_finish else None,
                    'slack_time': task.slack_time,
                    'status': task.status.value,
                    'assigned_agents': task.assigned_agents,
                    'progress_percentage': task.progress_percentage,
                    'created_at': task.created_at.isoformat() if task.created_at else None,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'tags': list(task.tags),
                    'acceptance_criteria': task.acceptance_criteria,
                    'notes': task.notes
                } for task_id, task in hierarchy.items()
            },
            'statistics': self.get_hierarchy_statistics(hierarchy_id)
        }

# CLI interface for testing
async def main():
    """CLI interface for testing Hierarchical Task Planner"""
    
    planner = HierarchicalTaskPlanner()
    
    print("🎯 Agent Zero V1 - Hierarchical Task Planner")
    print("=" * 60)
    
    # Test business requests
    business_requests = [
        "Create a comprehensive user authentication system with JWT tokens and OAuth integration",
        "Analyze sales data from the last quarter and generate interactive dashboard with KPI metrics",
        "Build automated testing framework with unit tests, integration tests, and performance benchmarks",
        "Optimize database performance with query optimization, indexing strategy, and connection pooling"
    ]
    
    hierarchy_id = "test_hierarchy_001"
    
    print(f"\n📋 Creating task hierarchy for {len(business_requests)} business requests...")
    
    # Create task hierarchy
    start_time = time.time()
    tasks = await planner.create_task_hierarchy(
        hierarchy_id=hierarchy_id,
        business_requests=business_requests,
        max_depth=3
    )
    creation_time = time.time() - start_time
    
    print(f"✅ Created {len(tasks)} tasks in {creation_time:.2f} seconds")
    
    # Display statistics
    stats = planner.get_hierarchy_statistics(hierarchy_id)
    if stats:
        print(f"\n📊 Hierarchy Statistics:")
        print(f"   Total Tasks: {stats['total_tasks']}")
        print(f"   Task Types: {stats['type_counts']}")
        print(f"   Priority Distribution: {stats['priority_counts']}")
        print(f"   Estimated Duration: {stats['total_estimated_duration']} minutes")
        print(f"   Estimated Cost: ${stats['total_estimated_cost']:.3f}")
        print(f"   Critical Path: {stats['critical_path_tasks']} tasks, {stats['critical_path_duration']} minutes")
    
    # Show ready tasks
    ready_tasks = planner.get_ready_tasks(hierarchy_id)
    print(f"\n🚀 Ready to Execute ({len(ready_tasks)} tasks):")
    for i, task in enumerate(ready_tasks[:5]):  # Show first 5
        print(f"   {i+1}. [{task.priority.name}] {task.name}")
        print(f"      Type: {task.task_type.value}, Business Value: {task.business_value}")
    
    # Show critical path
    critical_path = planner.get_critical_path(hierarchy_id)
    print(f"\n⚡ Critical Path ({len(critical_path)} tasks):")
    for i, task in enumerate(critical_path[:5]):  # Show first 5
        print(f"   {i+1}. {task.name}")
        print(f"      Duration: {task.resource_requirements.estimated_duration} min, Slack: {task.slack_time} min")
    
    # Export hierarchy
    exported = planner.export_hierarchy_to_dict(hierarchy_id)
    if exported:
        print(f"\n💾 Hierarchy exported successfully")
        print(f"   JSON size: {len(json.dumps(exported))} characters")
    
    print(f"\n✅ Hierarchical Task Planner test completed successfully!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())