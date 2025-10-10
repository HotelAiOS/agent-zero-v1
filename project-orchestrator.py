# Project Orchestrator - Finalization Module for Agent Zero V1
# Task: A0-20 ProjectOrchestrator finale 10% (1 SP)
# Timeline: 11:00-12:00
# Focus: Lifecycle methods, state management, monitoring

"""
Project Orchestrator for Agent Zero V1
Advanced project lifecycle management with Kaizen integration

This module handles:
- Project lifecycle management
- State persistence and recovery  
- Real-time monitoring and metrics
- Integration with existing SimpleTracker and BusinessParser
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager

# Import existing components
try:
    import sys
    sys.path.insert(0, '.')
    from simple_tracker import SimpleTracker
    from business_requirements_parser import BusinessRequirementsParser, BusinessIntent, TechnicalSpec
    from feedback_loop_engine import FeedbackLoopEngine
except ImportError as e:
    logging.warning(f"Could not import existing components: {e}")
    # Fallback classes for testing
    class SimpleTracker:
        def track_event(self, event): pass
    class BusinessRequirementsParser:
        def parse_intent(self, text): return None
        def generate_technical_spec(self, intent): return None

class ProjectState(Enum):
    """Project lifecycle states"""
    CREATED = "created"
    PLANNING = "planning" 
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class TaskStatus(Enum):
    """Individual task states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ProjectMetrics:
    """Real-time project metrics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    estimated_duration: int = 0  # minutes
    actual_duration: int = 0     # minutes
    success_rate: float = 0.0
    avg_task_duration: float = 0.0
    last_updated: Optional[datetime] = None

@dataclass
class Task:
    """Individual task in project"""
    task_id: str
    business_request: str
    technical_spec: Optional[Dict] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: int = 0
    actual_duration: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    error_message: Optional[str] = None
    assigned_agents: List[str] = None
    dependencies: List[str] = None
    progress_percentage: int = 0

@dataclass
class Project:
    """Main project entity"""
    project_id: str
    name: str
    description: str
    state: ProjectState = ProjectState.CREATED
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: Dict[str, Task] = None
    metrics: ProjectMetrics = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = {}
        if self.metrics is None:
            self.metrics = ProjectMetrics()
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

class StateManager:
    """Persistent state management"""
    
    def __init__(self, db_path: str = "project_orchestrator.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for state persistence"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    state TEXT NOT NULL,
                    created_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT,
                    metrics TEXT
                )
            """)
            
            # Tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    business_request TEXT NOT NULL,
                    technical_spec TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    estimated_duration INTEGER,
                    actual_duration INTEGER,
                    estimated_cost REAL,
                    actual_cost REAL,
                    error_message TEXT,
                    assigned_agents TEXT,
                    dependencies TEXT,
                    progress_percentage INTEGER DEFAULT 0,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def save_project(self, project: Project):
        """Persist project to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO projects 
                (project_id, name, description, state, created_at, started_at, completed_at, metadata, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project.project_id,
                project.name,
                project.description,
                project.state.value,
                project.created_at,
                project.started_at,
                project.completed_at,
                json.dumps(project.metadata) if project.metadata else None,
                json.dumps(asdict(project.metrics)) if project.metrics else None
            ))
            
            # Save tasks
            for task in project.tasks.values():
                self.save_task(task, project.project_id, conn)
            
            conn.commit()
    
    def save_task(self, task: Task, project_id: str, conn=None):
        """Persist task to database"""
        if conn is None:
            with self.get_connection() as conn:
                self._save_task_internal(task, project_id, conn)
        else:
            self._save_task_internal(task, project_id, conn)
    
    def _save_task_internal(self, task: Task, project_id: str, conn):
        """Internal task save method"""
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, project_id, business_request, technical_spec, status, 
             created_at, started_at, completed_at, estimated_duration, actual_duration,
             estimated_cost, actual_cost, error_message, assigned_agents, dependencies, progress_percentage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            project_id,
            task.business_request,
            json.dumps(task.technical_spec) if task.technical_spec else None,
            task.status.value,
            task.created_at,
            task.started_at,
            task.completed_at,
            task.estimated_duration,
            task.actual_duration,
            task.estimated_cost,
            task.actual_cost,
            task.error_message,
            json.dumps(task.assigned_agents) if task.assigned_agents else None,
            json.dumps(task.dependencies) if task.dependencies else None,
            task.progress_percentage
        ))
        
        if conn.autocommit is False:
            conn.commit()
    
    def load_project(self, project_id: str) -> Optional[Project]:
        """Load project from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Load project
            cursor.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to project
            project = Project(
                project_id=row['project_id'],
                name=row['name'],
                description=row['description'],
                state=ProjectState(row['state']),
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                metrics=ProjectMetrics(**json.loads(row['metrics'])) if row['metrics'] else ProjectMetrics()
            )
            
            # Load tasks
            cursor.execute("SELECT * FROM tasks WHERE project_id = ?", (project_id,))
            task_rows = cursor.fetchall()
            
            for task_row in task_rows:
                task = Task(
                    task_id=task_row['task_id'],
                    business_request=task_row['business_request'],
                    technical_spec=json.loads(task_row['technical_spec']) if task_row['technical_spec'] else None,
                    status=TaskStatus(task_row['status']),
                    created_at=datetime.fromisoformat(task_row['created_at']) if task_row['created_at'] else None,
                    started_at=datetime.fromisoformat(task_row['started_at']) if task_row['started_at'] else None,
                    completed_at=datetime.fromisoformat(task_row['completed_at']) if task_row['completed_at'] else None,
                    estimated_duration=task_row['estimated_duration'],
                    actual_duration=task_row['actual_duration'],
                    estimated_cost=task_row['estimated_cost'],
                    actual_cost=task_row['actual_cost'],
                    error_message=task_row['error_message'],
                    assigned_agents=json.loads(task_row['assigned_agents']) if task_row['assigned_agents'] else [],
                    dependencies=json.loads(task_row['dependencies']) if task_row['dependencies'] else [],
                    progress_percentage=task_row['progress_percentage']
                )
                project.tasks[task.task_id] = task
            
            return project

class LifecycleManager:
    """Project lifecycle management"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
    
    async def transition_state(self, project: Project, new_state: ProjectState, 
                              reason: Optional[str] = None) -> bool:
        """Manage project state transitions"""
        
        old_state = project.state
        
        # Validate state transition
        if not self._is_valid_transition(old_state, new_state):
            self.logger.error(f"Invalid state transition from {old_state} to {new_state}")
            return False
        
        # Pre-transition hooks
        if not await self._pre_transition_hook(project, new_state):
            return False
        
        # Update timestamps
        now = datetime.now()
        if new_state == ProjectState.ACTIVE and project.started_at is None:
            project.started_at = now
        elif new_state in [ProjectState.COMPLETED, ProjectState.FAILED]:
            project.completed_at = now
        
        # Perform transition
        project.state = new_state
        
        # Update metadata
        if 'state_history' not in project.metadata:
            project.metadata['state_history'] = []
        
        project.metadata['state_history'].append({
            'from_state': old_state.value,
            'to_state': new_state.value,
            'timestamp': now.isoformat(),
            'reason': reason
        })
        
        # Post-transition hooks
        await self._post_transition_hook(project, old_state)
        
        # Persist changes
        self.orchestrator.state_manager.save_project(project)
        
        self.logger.info(f"Project {project.project_id} transitioned from {old_state} to {new_state}")
        return True
    
    def _is_valid_transition(self, from_state: ProjectState, to_state: ProjectState) -> bool:
        """Validate state transition rules"""
        valid_transitions = {
            ProjectState.CREATED: [ProjectState.PLANNING, ProjectState.ACTIVE, ProjectState.ARCHIVED],
            ProjectState.PLANNING: [ProjectState.ACTIVE, ProjectState.PAUSED, ProjectState.ARCHIVED],
            ProjectState.ACTIVE: [ProjectState.PAUSED, ProjectState.COMPLETED, ProjectState.FAILED],
            ProjectState.PAUSED: [ProjectState.ACTIVE, ProjectState.ARCHIVED, ProjectState.FAILED],
            ProjectState.COMPLETED: [ProjectState.ARCHIVED],
            ProjectState.FAILED: [ProjectState.PLANNING, ProjectState.ARCHIVED],
            ProjectState.ARCHIVED: []  # Terminal state
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    async def _pre_transition_hook(self, project: Project, new_state: ProjectState) -> bool:
        """Execute pre-transition checks and preparations"""
        
        if new_state == ProjectState.ACTIVE:
            # Ensure project has tasks before starting
            if not project.tasks:
                self.logger.error(f"Cannot start project {project.project_id} - no tasks defined")
                return False
        
        elif new_state == ProjectState.COMPLETED:
            # Check if all tasks are completed or skipped
            incomplete_tasks = [
                task for task in project.tasks.values() 
                if task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
            ]
            if incomplete_tasks:
                self.logger.warning(f"Completing project {project.project_id} with {len(incomplete_tasks)} incomplete tasks")
        
        return True
    
    async def _post_transition_hook(self, project: Project, old_state: ProjectState):
        """Execute post-transition actions"""
        
        if project.state == ProjectState.ACTIVE:
            # Start automatic task execution
            await self.orchestrator.start_task_execution(project.project_id)
        
        elif project.state in [ProjectState.COMPLETED, ProjectState.FAILED]:
            # Update final metrics
            self.orchestrator.metrics_monitor.calculate_final_metrics(project)
            
            # Trigger Kaizen learning
            if hasattr(self.orchestrator, 'feedback_engine'):
                await self.orchestrator.feedback_engine.process_project_completion(project)

class MetricsMonitor:
    """Real-time metrics monitoring"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self._monitoring = {}  # project_id -> monitoring_task
    
    async def start_monitoring(self, project_id: str):
        """Start real-time monitoring for project"""
        if project_id in self._monitoring:
            return  # Already monitoring
        
        async def monitor_loop():
            while project_id in self._monitoring:
                try:
                    project = self.orchestrator.get_project(project_id)
                    if not project or project.state not in [ProjectState.ACTIVE, ProjectState.PLANNING]:
                        break
                    
                    self.update_metrics(project)
                    self.orchestrator.state_manager.save_project(project)
                    
                    # Check for alerts
                    await self.check_alerts(project)
                    
                    await asyncio.sleep(30)  # Update every 30 seconds
                
                except Exception as e:
                    self.logger.error(f"Error in metrics monitoring for {project_id}: {e}")
                    await asyncio.sleep(60)  # Back off on error
        
        self._monitoring[project_id] = asyncio.create_task(monitor_loop())
        self.logger.info(f"Started metrics monitoring for project {project_id}")
    
    def stop_monitoring(self, project_id: str):
        """Stop monitoring for project"""
        if project_id in self._monitoring:
            self._monitoring[project_id].cancel()
            del self._monitoring[project_id]
            self.logger.info(f"Stopped metrics monitoring for project {project_id}")
    
    def update_metrics(self, project: Project):
        """Update project metrics"""
        metrics = project.metrics
        
        # Count tasks by status
        metrics.total_tasks = len(project.tasks)
        metrics.completed_tasks = sum(1 for task in project.tasks.values() 
                                    if task.status == TaskStatus.COMPLETED)
        metrics.failed_tasks = sum(1 for task in project.tasks.values() 
                                 if task.status == TaskStatus.FAILED)
        metrics.active_tasks = sum(1 for task in project.tasks.values() 
                                 if task.status == TaskStatus.RUNNING)
        
        # Calculate costs
        metrics.estimated_cost = sum(task.estimated_cost for task in project.tasks.values())
        metrics.actual_cost = sum(task.actual_cost for task in project.tasks.values())
        
        # Calculate duration
        metrics.estimated_duration = sum(task.estimated_duration for task in project.tasks.values())
        
        if project.started_at:
            if project.completed_at:
                metrics.actual_duration = int((project.completed_at - project.started_at).total_seconds() / 60)
            else:
                metrics.actual_duration = int((datetime.now() - project.started_at).total_seconds() / 60)
        
        # Calculate success rate
        if metrics.total_tasks > 0:
            metrics.success_rate = metrics.completed_tasks / metrics.total_tasks
        
        # Calculate average task duration
        completed_tasks = [task for task in project.tasks.values() 
                          if task.status == TaskStatus.COMPLETED and task.actual_duration > 0]
        if completed_tasks:
            metrics.avg_task_duration = sum(task.actual_duration for task in completed_tasks) / len(completed_tasks)
        
        metrics.last_updated = datetime.now()
    
    def calculate_final_metrics(self, project: Project):
        """Calculate final metrics when project completes"""
        self.update_metrics(project)
        
        # Additional final calculations
        metrics = project.metrics
        
        # Efficiency metrics
        if metrics.estimated_cost > 0:
            metrics.cost_efficiency = metrics.actual_cost / metrics.estimated_cost
        
        if metrics.estimated_duration > 0:
            metrics.time_efficiency = metrics.actual_duration / metrics.estimated_duration
        
        # Quality metrics  
        metrics.completion_rate = metrics.completed_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0
        metrics.failure_rate = metrics.failed_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0
    
    async def check_alerts(self, project: Project):
        """Check for alert conditions"""
        metrics = project.metrics
        alerts = []
        
        # Cost overrun alert
        if metrics.estimated_cost > 0 and metrics.actual_cost > metrics.estimated_cost * 1.2:
            alerts.append({
                'type': 'cost_overrun',
                'severity': 'warning',
                'message': f"Cost exceeded estimate by {((metrics.actual_cost / metrics.estimated_cost) - 1) * 100:.1f}%"
            })
        
        # Time overrun alert  
        if (metrics.estimated_duration > 0 and 
            metrics.actual_duration > metrics.estimated_duration * 1.2):
            alerts.append({
                'type': 'time_overrun',
                'severity': 'warning', 
                'message': f"Duration exceeded estimate by {((metrics.actual_duration / metrics.estimated_duration) - 1) * 100:.1f}%"
            })
        
        # High failure rate alert
        if metrics.total_tasks > 0 and metrics.failed_tasks / metrics.total_tasks > 0.3:
            alerts.append({
                'type': 'high_failure_rate',
                'severity': 'critical',
                'message': f"High failure rate: {(metrics.failed_tasks / metrics.total_tasks) * 100:.1f}%"
            })
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(project, alert)
    
    async def _process_alert(self, project: Project, alert: Dict):
        """Process and handle alerts"""
        self.logger.warning(f"Alert for project {project.project_id}: {alert['message']}")
        
        # Store alert in project metadata
        if 'alerts' not in project.metadata:
            project.metadata['alerts'] = []
        
        alert['timestamp'] = datetime.now().isoformat()
        project.metadata['alerts'].append(alert)
        
        # Trigger notifications if needed
        # This could integrate with external alerting systems

class ProjectOrchestrator:
    """Main Project Orchestrator class"""
    
    def __init__(self, db_path: str = "project_orchestrator.db"):
        self.state_manager = StateManager(db_path)
        self.lifecycle_manager = LifecycleManager(self)
        self.metrics_monitor = MetricsMonitor(self)
        
        # Integration with existing components
        try:
            self.tracker = SimpleTracker()
            self.business_parser = BusinessRequirementsParser()
            self.feedback_engine = getattr(FeedbackLoopEngine, '__call__', lambda: None)()
        except:
            self.tracker = None
            self.business_parser = None 
            self.feedback_engine = None
        
        self.logger = logging.getLogger(__name__)
        self._projects = {}  # In-memory cache
    
    async def create_project(self, name: str, description: str, 
                           business_requests: List[str] = None) -> str:
        """Create new project"""
        
        project_id = f"proj_{int(time.time())}"
        
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            created_at=datetime.now()
        )
        
        # Add initial tasks if provided
        if business_requests:
            for i, request in enumerate(business_requests):
                await self.add_task(project_id, request)
        
        # Cache and persist
        self._projects[project_id] = project
        self.state_manager.save_project(project)
        
        # Track in SimpleTracker
        if self.tracker:
            self.tracker.track_event({
                'type': 'project_created',
                'project_id': project_id,
                'name': name,
                'task_count': len(business_requests) if business_requests else 0
            })
        
        self.logger.info(f"Created project {project_id}: {name}")
        return project_id
    
    async def add_task(self, project_id: str, business_request: str) -> str:
        """Add task to project"""
        
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        task_id = f"{project_id}_task_{len(project.tasks) + 1}"
        
        # Parse business request if parser available
        technical_spec = None
        estimated_duration = 30  # Default 30 minutes
        estimated_cost = 0.05    # Default $0.05
        assigned_agents = ["orchestrator"]
        
        if self.business_parser:
            try:
                intent = self.business_parser.parse_intent(business_request)
                spec = self.business_parser.generate_technical_spec(intent)
                
                technical_spec = asdict(spec)
                estimated_duration = spec.estimated_time
                estimated_cost = spec.estimated_cost
                assigned_agents = spec.agent_sequence
                
            except Exception as e:
                self.logger.warning(f"Could not parse business request: {e}")
        
        task = Task(
            task_id=task_id,
            business_request=business_request,
            technical_spec=technical_spec,
            created_at=datetime.now(),
            estimated_duration=estimated_duration,
            estimated_cost=estimated_cost,
            assigned_agents=assigned_agents
        )
        
        project.tasks[task_id] = task
        
        # Update project metrics
        self.metrics_monitor.update_metrics(project)
        
        # Persist changes
        self.state_manager.save_project(project)
        
        self.logger.info(f"Added task {task_id} to project {project_id}")
        return task_id
    
    async def start_project(self, project_id: str) -> bool:
        """Start project execution"""
        
        project = self.get_project(project_id)
        if not project:
            return False
        
        success = await self.lifecycle_manager.transition_state(
            project, ProjectState.ACTIVE, "Manual start"
        )
        
        if success:
            await self.metrics_monitor.start_monitoring(project_id)
        
        return success
    
    async def start_task_execution(self, project_id: str):
        """Start executing tasks for active project"""
        
        project = self.get_project(project_id)
        if not project or project.state != ProjectState.ACTIVE:
            return
        
        # Find next pending task
        pending_tasks = [
            task for task in project.tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        
        if not pending_tasks:
            # No more tasks - check if project should be completed
            await self._check_project_completion(project)
            return
        
        # Start next task (simplified - in real implementation would use task scheduler)
        next_task = pending_tasks[0]
        await self.start_task(project_id, next_task.task_id)
    
    async def start_task(self, project_id: str, task_id: str) -> bool:
        """Start individual task"""
        
        project = self.get_project(project_id)
        if not project:
            return False
        
        task = project.tasks.get(task_id)
        if not task:
            return False
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.progress_percentage = 0
        
        # Persist changes
        self.state_manager.save_task(task, project_id)
        
        # Track in SimpleTracker
        if self.tracker:
            self.tracker.track_event({
                'type': 'task_started',
                'project_id': project_id,
                'task_id': task_id,
                'business_request': task.business_request
            })
        
        self.logger.info(f"Started task {task_id} in project {project_id}")
        
        # Simulate task execution (in real implementation would delegate to agents)
        asyncio.create_task(self._simulate_task_execution(project_id, task_id))
        
        return True
    
    async def _simulate_task_execution(self, project_id: str, task_id: str):
        """Simulate task execution - replace with real agent execution"""
        
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        
        if not task:
            return
        
        try:
            # Simulate progress updates
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(task.estimated_duration * 0.25 * 60 / 60)  # Scale to seconds
                task.progress_percentage = progress
                self.state_manager.save_task(task, project_id)
            
            # Mark as completed
            await self.complete_task(project_id, task_id, True, "Task completed successfully")
            
        except Exception as e:
            await self.complete_task(project_id, task_id, False, str(e))
    
    async def complete_task(self, project_id: str, task_id: str, 
                          success: bool, message: str = None):
        """Complete task execution"""
        
        project = self.get_project(project_id)
        task = project.tasks.get(task_id)
        
        if not task:
            return
        
        # Update task
        task.completed_at = datetime.now()
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.progress_percentage = 100 if success else task.progress_percentage
        
        if task.started_at:
            task.actual_duration = int((task.completed_at - task.started_at).total_seconds() / 60)
        
        if not success and message:
            task.error_message = message
        
        # Calculate actual cost (simplified)
        task.actual_cost = task.estimated_cost * (task.actual_duration / task.estimated_duration) if task.estimated_duration > 0 else task.estimated_cost
        
        # Persist changes
        self.state_manager.save_task(task, project_id)
        
        # Track in SimpleTracker
        if self.tracker:
            self.tracker.track_event({
                'type': 'task_completed',
                'project_id': project_id,
                'task_id': task_id,
                'success': success,
                'duration': task.actual_duration,
                'cost': task.actual_cost
            })
        
        self.logger.info(f"Task {task_id} {'completed' if success else 'failed'} in project {project_id}")
        
        # Check if project should be completed
        await self._check_project_completion(project)
        
        # Continue with next task
        if success and project.state == ProjectState.ACTIVE:
            await self.start_task_execution(project_id)
    
    async def _check_project_completion(self, project: Project):
        """Check if project should be marked as completed"""
        
        pending_tasks = [
            task for task in project.tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ]
        
        if not pending_tasks:
            # All tasks done - determine if completed or failed
            failed_tasks = [
                task for task in project.tasks.values()
                if task.status == TaskStatus.FAILED
            ]
            
            if not failed_tasks or len(failed_tasks) < len(project.tasks) * 0.5:
                # Success if no failures or less than 50% failed
                await self.lifecycle_manager.transition_state(
                    project, ProjectState.COMPLETED, "All tasks finished"
                )
            else:
                await self.lifecycle_manager.transition_state(
                    project, ProjectState.FAILED, "Too many failed tasks"
                )
            
            # Stop monitoring
            self.metrics_monitor.stop_monitoring(project.project_id)
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        
        # Check cache first
        if project_id in self._projects:
            return self._projects[project_id]
        
        # Load from database
        project = self.state_manager.load_project(project_id)
        if project:
            self._projects[project_id] = project
        
        return project
    
    def list_projects(self, state: Optional[ProjectState] = None) -> List[Project]:
        """List all projects, optionally filtered by state"""
        
        with self.state_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            if state:
                cursor.execute("SELECT project_id FROM projects WHERE state = ?", (state.value,))
            else:
                cursor.execute("SELECT project_id FROM projects")
            
            project_ids = [row[0] for row in cursor.fetchall()]
        
        projects = []
        for project_id in project_ids:
            project = self.get_project(project_id)
            if project:
                projects.append(project)
        
        return projects
    
    def get_project_status(self, project_id: str) -> Optional[Dict]:
        """Get comprehensive project status"""
        
        project = self.get_project(project_id)
        if not project:
            return None
        
        return {
            'project_id': project.project_id,
            'name': project.name,
            'description': project.description,
            'state': project.state.value,
            'created_at': project.created_at.isoformat() if project.created_at else None,
            'started_at': project.started_at.isoformat() if project.started_at else None,
            'completed_at': project.completed_at.isoformat() if project.completed_at else None,
            'metrics': asdict(project.metrics) if project.metrics else None,
            'task_count': len(project.tasks),
            'tasks': {
                task_id: {
                    'task_id': task.task_id,
                    'business_request': task.business_request,
                    'status': task.status.value,
                    'progress_percentage': task.progress_percentage,
                    'estimated_duration': task.estimated_duration,
                    'actual_duration': task.actual_duration,
                    'estimated_cost': task.estimated_cost,
                    'actual_cost': task.actual_cost
                } for task_id, task in project.tasks.items()
            }
        }

# CLI interface for testing
async def main():
    """CLI interface for testing Project Orchestrator"""
    
    orchestrator = ProjectOrchestrator()
    
    print("🎯 Agent Zero V1 - Project Orchestrator")
    print("=" * 50)
    
    # Create test project
    print("\n📋 Creating test project...")
    project_id = await orchestrator.create_project(
        "Test Multi-Agent Project",
        "Testing Project Orchestrator with multiple tasks",
        [
            "Create user authentication API with JWT tokens",
            "Analyze sales data from last quarter", 
            "Generate monthly report dashboard",
            "Optimize database performance queries"
        ]
    )
    
    print(f"✅ Created project: {project_id}")
    
    # Start project
    print(f"\n🚀 Starting project {project_id}...")
    success = await orchestrator.start_project(project_id)
    
    if success:
        print("✅ Project started successfully")
        
        # Monitor progress
        print("\n📊 Monitoring progress...")
        for i in range(10):
            await asyncio.sleep(2)
            status = orchestrator.get_project_status(project_id)
            
            if status:
                metrics = status['metrics']
                print(f"Progress: {metrics['completed_tasks']}/{metrics['total_tasks']} tasks completed")
                
                if status['state'] in ['completed', 'failed']:
                    print(f"🏁 Project {status['state']}")
                    break
        
        # Final status
        final_status = orchestrator.get_project_status(project_id)
        if final_status:
            print(f"\n📈 Final Status:")
            print(f"State: {final_status['state']}")
            print(f"Duration: {final_status['metrics']['actual_duration']} minutes")
            print(f"Cost: ${final_status['metrics']['actual_cost']:.3f}")
            print(f"Success Rate: {final_status['metrics']['success_rate']:.1%}")
    
    else:
        print("❌ Failed to start project")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())