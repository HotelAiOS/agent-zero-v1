#!/usr/bin/env python3
"""
🎯 Agent Zero V1 - Point 3: Dynamic Task Prioritization Implementation
====================================================================
Week 43 Critical Priority - Complete System Integration
Based on architectural analysis: 62% complete, Point 3 is next critical milestone
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import math

# FastAPI and async components
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dynamic_prioritization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DynamicTaskPrioritization")

# ================================
# DYNAMIC PRIORITY SYSTEM CORE
# ================================

class TaskPriority(Enum):
    """Dynamic task priority levels"""
    CRISIS = "CRISIS"                    # P0 - Immediate response required
    CRITICAL = "CRITICAL"                # P1 - Business critical, same day
    HIGH = "HIGH"                        # P2 - Important, within 24h
    MEDIUM = "MEDIUM"                    # P3 - Standard priority
    LOW = "LOW"                          # P4 - When resources available
    DEFERRED = "DEFERRED"                # P5 - Future consideration

class BusinessContext(Enum):
    """Business context for priority calculation"""
    REVENUE_IMPACT = "REVENUE_IMPACT"
    CUSTOMER_FACING = "CUSTOMER_FACING"
    SECURITY_CRITICAL = "SECURITY_CRITICAL"
    DEPENDENCY_BLOCKER = "DEPENDENCY_BLOCKER"
    REGULATORY_COMPLIANCE = "REGULATORY_COMPLIANCE"
    TECHNICAL_DEBT = "TECHNICAL_DEBT"
    INNOVATION = "INNOVATION"

class CrisisType(Enum):
    """Types of crisis situations"""
    SYSTEM_DOWN = "SYSTEM_DOWN"
    DATA_BREACH = "DATA_BREACH"
    CUSTOMER_ESCALATION = "CUSTOMER_ESCALATION"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"

@dataclass
class TaskMetrics:
    """Real-time task performance metrics"""
    execution_time_ms: float = 0.0
    success_rate: float = 1.0
    resource_usage: float = 0.0
    user_satisfaction: float = 1.0
    business_value_delivered: float = 0.0
    cost_efficiency: float = 1.0
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        # Weighted scoring algorithm
        weights = {
            'success_rate': 0.3,
            'user_satisfaction': 0.25,
            'business_value': 0.2,
            'cost_efficiency': 0.15,
            'execution_speed': 0.1  # Inverse of execution_time
        }
        
        # Normalize execution time (assume 1000ms is baseline)
        execution_score = max(0.0, 1.0 - (self.execution_time_ms / 1000.0))
        
        total_score = (
            weights['success_rate'] * self.success_rate +
            weights['user_satisfaction'] * self.user_satisfaction +
            weights['business_value'] * self.business_value_delivered +
            weights['cost_efficiency'] * self.cost_efficiency +
            weights['execution_speed'] * execution_score
        )
        
        return min(1.0, max(0.0, total_score))

@dataclass
class DynamicTask:
    """Enhanced task with dynamic prioritization capabilities"""
    id: str
    title: str
    description: str
    original_priority: TaskPriority
    current_priority: TaskPriority
    business_contexts: List[BusinessContext] = field(default_factory=list)
    
    # Time tracking
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    estimated_hours: float = 1.0
    
    # Dynamic attributes
    urgency_score: float = 0.5
    impact_score: float = 0.5
    effort_score: float = 0.5
    risk_score: float = 0.5
    
    # Assignment and tracking
    assigned_agent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)
    
    # Performance tracking
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    priority_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Crisis management
    is_crisis: bool = False
    crisis_type: Optional[CrisisType] = None
    escalation_level: int = 0
    
    def calculate_priority_score(self) -> float:
        """Calculate dynamic priority score based on multiple factors"""
        
        # Base score from current priority
        priority_values = {
            TaskPriority.CRISIS: 1.0,
            TaskPriority.CRITICAL: 0.9,
            TaskPriority.HIGH: 0.7,
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.LOW: 0.3,
            TaskPriority.DEFERRED: 0.1
        }
        
        base_score = priority_values.get(self.current_priority, 0.5)
        
        # Time decay factor (increases urgency as deadline approaches)
        time_factor = self._calculate_time_factor()
        
        # Business context multiplier
        context_multiplier = self._calculate_context_multiplier()
        
        # Dependency factor (higher if blocking others)
        dependency_factor = self._calculate_dependency_factor()
        
        # Crisis escalation
        crisis_factor = 2.0 if self.is_crisis else 1.0
        
        # Performance history factor
        performance_factor = 0.8 + (self.metrics.calculate_performance_score() * 0.4)
        
        # Final score calculation
        priority_score = (
            base_score * 
            time_factor * 
            context_multiplier * 
            dependency_factor * 
            crisis_factor * 
            performance_factor
        )
        
        return min(10.0, max(0.1, priority_score))
    
    def _calculate_time_factor(self) -> float:
        """Calculate time-based urgency factor"""
        if not self.deadline:
            return 1.0
            
        now = datetime.now()
        time_remaining = (self.deadline - now).total_seconds()
        
        if time_remaining <= 0:
            return 3.0  # Overdue - highest urgency
        elif time_remaining <= 3600:  # 1 hour
            return 2.5
        elif time_remaining <= 86400:  # 1 day
            return 2.0
        elif time_remaining <= 604800:  # 1 week
            return 1.5
        else:
            return 1.0
    
    def _calculate_context_multiplier(self) -> float:
        """Calculate business context impact multiplier"""
        if not self.business_contexts:
            return 1.0
            
        context_weights = {
            BusinessContext.REVENUE_IMPACT: 1.8,
            BusinessContext.SECURITY_CRITICAL: 1.7,
            BusinessContext.CUSTOMER_FACING: 1.5,
            BusinessContext.DEPENDENCY_BLOCKER: 1.6,
            BusinessContext.REGULATORY_COMPLIANCE: 1.4,
            BusinessContext.TECHNICAL_DEBT: 0.9,
            BusinessContext.INNOVATION: 1.1
        }
        
        # Take maximum multiplier from all contexts
        max_multiplier = max(
            context_weights.get(context, 1.0) 
            for context in self.business_contexts
        )
        
        return max_multiplier
    
    def _calculate_dependency_factor(self) -> float:
        """Calculate impact based on task dependencies"""
        # If this task is blocking others, increase priority
        if len(self.dependencies) > 0:
            return 1.0 + (len(self.dependencies) * 0.2)
        
        # If this task is blocked, decrease priority slightly
        if len(self.blocked_by) > 0:
            return max(0.5, 1.0 - (len(self.blocked_by) * 0.1))
        
        return 1.0
    
    def update_priority(self, new_priority: TaskPriority, reason: str) -> None:
        """Update task priority with audit trail"""
        old_priority = self.current_priority
        self.current_priority = new_priority
        
        # Record priority change
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "old_priority": old_priority.value,
            "new_priority": new_priority.value,
            "reason": reason,
            "priority_score": self.calculate_priority_score()
        }
        
        self.priority_changes.append(change_record)
        
        logger.info(f"Task {self.id} priority updated: {old_priority.value} → {new_priority.value} ({reason})")
    
    def escalate_to_crisis(self, crisis_type: CrisisType, reason: str) -> None:
        """Escalate task to crisis status"""
        self.is_crisis = True
        self.crisis_type = crisis_type
        self.escalation_level += 1
        self.update_priority(TaskPriority.CRISIS, f"Crisis escalation: {reason}")
        
        logger.warning(f"Task {self.id} escalated to CRISIS: {crisis_type.value} - {reason}")

# ================================
# DYNAMIC PRIORITY ENGINE
# ================================

class DynamicPriorityEngine:
    """Advanced dynamic task prioritization engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tasks: Dict[str, DynamicTask] = {}
        self.priority_queue: List[str] = []
        self.agent_workloads: Dict[str, float] = {}
        
        # Configuration
        self.rebalance_interval = 300  # 5 minutes
        self.crisis_detection_threshold = 0.8
        self.performance_window = 3600  # 1 hour performance tracking
        
        # Initialize database
        self._init_database()
        
        # Background tasks will be started via start_background_tasks()
        
    def _init_database(self):
        """Initialize SQLite database for task prioritization"""
        
        self.db_path = "dynamic_prioritization.db"
        
        with sqlite3.connect(self.db_path) as conn:
            # Tasks table with dynamic fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dynamic_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    original_priority TEXT,
                    current_priority TEXT,
                    business_contexts TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    deadline DATETIME,
                    estimated_hours REAL,
                    
                    urgency_score REAL DEFAULT 0.5,
                    impact_score REAL DEFAULT 0.5,
                    effort_score REAL DEFAULT 0.5,
                    risk_score REAL DEFAULT 0.5,
                    
                    assigned_agent_id TEXT,
                    dependencies TEXT,
                    blocked_by TEXT,
                    
                    is_crisis BOOLEAN DEFAULT 0,
                    crisis_type TEXT,
                    escalation_level INTEGER DEFAULT 0,
                    
                    metrics_json TEXT,
                    priority_changes_json TEXT
                )
            """)
            
            # Priority history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS priority_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    old_priority TEXT,
                    new_priority TEXT,
                    reason TEXT,
                    priority_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES dynamic_tasks(id)
                )
            """)
            
            # Crisis events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crisis_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    crisis_type TEXT NOT NULL,
                    description TEXT,
                    severity_level INTEGER,
                    resolved BOOLEAN DEFAULT 0,
                    response_time_seconds REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    execution_time_ms REAL,
                    success_rate REAL,
                    resource_usage REAL,
                    user_satisfaction REAL,
                    business_value_delivered REAL,
                    cost_efficiency REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        self.logger.info("📊 Dynamic prioritization database initialized")
    
    async def start_background_tasks(self):
        """Start background processing tasks"""
        asyncio.create_task(self._priority_rebalancing_loop())
        asyncio.create_task(self._crisis_detection_loop())
        self.logger.info("🚀 Background tasks started")
    
    async def add_task(self, task: DynamicTask) -> str:
        """Add new task to dynamic prioritization system"""
        
        self.tasks[task.id] = task
        
        # Store in database
        await self._store_task(task)
        
        # Update priority queue
        await self._update_priority_queue()
        
        # Check for immediate crisis conditions
        await self._check_crisis_conditions(task)
        
        self.logger.info(f"Task {task.id} added with priority {task.current_priority.value}")
        
        return task.id
    
    async def update_task_metrics(self, task_id: str, metrics: TaskMetrics) -> bool:
        """Update task performance metrics"""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_metrics = task.metrics
        task.metrics = metrics
        
        # Store metrics in database
        await self._store_performance_metrics(task_id, metrics)
        
        # Check if priority needs adjustment based on performance
        await self._adjust_priority_based_on_performance(task, old_metrics)
        
        self.logger.debug(f"Task {task_id} metrics updated - performance score: {metrics.calculate_performance_score():.2f}")
        
        return True
    
    async def get_prioritized_task_queue(self, agent_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get prioritized task queue for agent or system"""
        
        # Update priority queue first
        await self._update_priority_queue()
        
        # Filter by agent if specified
        if agent_id:
            filtered_tasks = [
                task_id for task_id in self.priority_queue
                if self.tasks[task_id].assigned_agent_id == agent_id
            ]
        else:
            filtered_tasks = self.priority_queue[:limit]
        
        # Build response with priority information
        prioritized_queue = []
        
        for task_id in filtered_tasks[:limit]:
            task = self.tasks[task_id]
            
            prioritized_queue.append({
                "id": task.id,
                "title": task.title,
                "current_priority": task.current_priority.value,
                "priority_score": task.calculate_priority_score(),
                "assigned_agent_id": task.assigned_agent_id,
                "deadline": task.deadline.isoformat() if task.deadline else None,
                "estimated_hours": task.estimated_hours,
                "is_crisis": task.is_crisis,
                "business_contexts": [ctx.value for ctx in task.business_contexts],
                "performance_score": task.metrics.calculate_performance_score(),
                "dependencies_count": len(task.dependencies),
                "blocked_by_count": len(task.blocked_by)
            })
        
        return prioritized_queue
    
    async def trigger_crisis_response(self, crisis_type: CrisisType, description: str, affected_tasks: List[str] = None) -> Dict[str, Any]:
        """Trigger crisis response protocol"""
        
        crisis_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.warning(f"Crisis response triggered: {crisis_type.value} - {description}")
        
        # Record crisis event
        await self._record_crisis_event(crisis_type, description, affected_tasks)
        
        # Automatic crisis actions
        response_actions = []
        
        if crisis_type == CrisisType.SYSTEM_DOWN:
            response_actions.extend([
                "Escalate all system-critical tasks to CRISIS priority",
                "Reassign non-critical tasks to reduce system load",
                "Activate backup systems if available"
            ])
            
        elif crisis_type == CrisisType.PERFORMANCE_DEGRADATION:
            response_actions.extend([
                "Prioritize performance optimization tasks",
                "Defer low-priority resource-intensive operations",
                "Scale up resource allocation"
            ])
            
        elif crisis_type == CrisisType.CUSTOMER_ESCALATION:
            response_actions.extend([
                "Escalate customer-facing tasks to HIGH priority",
                "Assign best available agents to customer issues",
                "Activate communication protocols"
            ])
        
        # Execute automated responses
        automated_responses = await self._execute_automated_crisis_response(crisis_type, affected_tasks)
        
        response_time = time.time() - start_time
        
        return {
            "crisis_id": crisis_id,
            "crisis_type": crisis_type.value,
            "description": description,
            "response_time_seconds": response_time,
            "automated_actions": automated_responses,
            "recommended_actions": response_actions,
            "affected_tasks_count": len(affected_tasks) if affected_tasks else 0,
            "status": "active"
        }
    
    async def _update_priority_queue(self):
        """Update master priority queue based on dynamic scores"""
        
        # Calculate current priority scores for all tasks
        task_scores = [
            (task_id, task.calculate_priority_score())
            for task_id, task in self.tasks.items()
        ]
        
        # Sort by priority score (highest first)
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update priority queue
        self.priority_queue = [task_id for task_id, _ in task_scores]
        
        self.logger.debug(f"Priority queue updated with {len(self.priority_queue)} tasks")
    
    async def _priority_rebalancing_loop(self):
        """Background process for periodic priority rebalancing"""
        
        while True:
            try:
                await asyncio.sleep(self.rebalance_interval)
                
                self.logger.debug("Starting priority rebalancing cycle")
                
                rebalanced_count = 0
                
                for task_id, task in self.tasks.items():
                    old_priority = task.current_priority
                    
                    # Recalculate optimal priority based on current conditions
                    optimal_priority = await self._calculate_optimal_priority(task)
                    
                    if optimal_priority != old_priority:
                        task.update_priority(optimal_priority, "Automatic rebalancing")
                        rebalanced_count += 1
                
                # Update priority queue
                await self._update_priority_queue()
                
                if rebalanced_count > 0:
                    self.logger.info(f"Priority rebalancing complete: {rebalanced_count} tasks updated")
                
            except Exception as e:
                self.logger.error(f"Priority rebalancing error: {e}")
    
    async def _crisis_detection_loop(self):
        """Background process for crisis detection"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Detect potential crisis situations
                crisis_indicators = await self._detect_crisis_indicators()
                
                for indicator in crisis_indicators:
                    await self.trigger_crisis_response(
                        indicator["crisis_type"],
                        indicator["description"],
                        indicator.get("affected_tasks", [])
                    )
                
            except Exception as e:
                self.logger.error(f"Crisis detection error: {e}")
    
    # Placeholder methods for database operations and advanced logic
    async def _store_task(self, task: DynamicTask):
        """Store task in database"""
        # Implementation for database storage
        pass
    
    async def _store_performance_metrics(self, task_id: str, metrics: TaskMetrics):
        """Store performance metrics in database"""
        # Implementation for metrics storage
        pass
    
    async def _check_crisis_conditions(self, task: DynamicTask):
        """Check for crisis conditions on new task"""
        # Implementation for crisis detection
        pass
    
    async def _adjust_priority_based_on_performance(self, task: DynamicTask, old_metrics: TaskMetrics):
        """Adjust priority based on performance changes"""
        # Implementation for performance-based priority adjustment
        pass
    
    async def _record_crisis_event(self, crisis_type: CrisisType, description: str, affected_tasks: List[str] = None):
        """Record crisis event in database"""
        # Implementation for crisis event recording
        pass
    
    async def _execute_automated_crisis_response(self, crisis_type: CrisisType, affected_tasks: List[str] = None):
        """Execute automated crisis response actions"""
        # Implementation for automated crisis response
        return []
    
    async def _detect_crisis_indicators(self) -> List[Dict[str, Any]]:
        """Detect potential crisis situations"""
        # Implementation for crisis detection
        return []
    
    async def _calculate_optimal_priority(self, task: DynamicTask) -> TaskPriority:
        """Calculate optimal priority for task"""
        score = task.calculate_priority_score()
        
        if score >= 8.0:
            return TaskPriority.CRISIS
        elif score >= 6.0:
            return TaskPriority.CRITICAL
        elif score >= 4.0:
            return TaskPriority.HIGH
        elif score >= 2.0:
            return TaskPriority.MEDIUM
        elif score >= 1.0:
            return TaskPriority.LOW
        else:
            return TaskPriority.DEFERRED

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Agent Zero V1 - Dynamic Task Prioritization",
    description="Point 3: Dynamic Task Prioritization & Crisis Response System", 
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize priority engine
priority_engine = DynamicPriorityEngine()

@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    await priority_engine.start_background_tasks()

@app.get("/")
async def priority_system_root():
    """Dynamic prioritization system information"""
    return {
        "system": "Agent Zero V1 - Dynamic Task Prioritization",
        "version": "1.0.0", 
        "status": "OPERATIONAL",
        "description": "Point 3: Real-time dynamic task prioritization with crisis response",
        "capabilities": [
            "Real-time priority adjustment",
            "Crisis detection and response", 
            "Workload balancing",
            "Performance-based optimization",
            "Business context integration"
        ],
        "endpoints": {
            "priority_queue": "GET /api/v1/priority/queue",
            "add_task": "POST /api/v1/priority/tasks",
            "crisis_response": "POST /api/v1/priority/crisis",
            "system_metrics": "GET /api/v1/priority/metrics"
        }
    }

@app.get("/api/v1/priority/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(priority_engine.tasks),
        "version": "1.0.0"
    }

@app.get("/api/v1/priority/queue")
async def get_priority_queue(agent_id: str = None, limit: int = 50):
    """Get current prioritized task queue"""
    
    queue = await priority_engine.get_prioritized_task_queue(agent_id, limit)
    
    return {
        "status": "success",
        "priority_queue": queue,
        "queue_size": len(queue),
        "agent_filter": agent_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/priority/tasks")
async def add_priority_task(task_data: dict):
    """Add new task to dynamic prioritization system"""
    
    # Create dynamic task from request
    task = DynamicTask(
        id=str(uuid.uuid4()),
        title=task_data.get("title", "Untitled Task"),
        description=task_data.get("description", ""),
        original_priority=TaskPriority(task_data.get("priority", "MEDIUM")),
        current_priority=TaskPriority(task_data.get("priority", "MEDIUM")),
        estimated_hours=task_data.get("estimated_hours", 1.0)
    )
    
    # Add business contexts if provided
    if "business_contexts" in task_data:
        task.business_contexts = [
            BusinessContext(ctx) for ctx in task_data["business_contexts"]
        ]
    
    # Set deadline if provided
    if "deadline" in task_data:
        task.deadline = datetime.fromisoformat(task_data["deadline"])
    
    task_id = await priority_engine.add_task(task)
    
    return {
        "status": "success",
        "task_id": task_id,
        "initial_priority": task.current_priority.value,
        "priority_score": task.calculate_priority_score(),
        "message": f"Task added with {task.current_priority.value} priority"
    }

@app.post("/api/v1/priority/crisis")
async def trigger_crisis(crisis_data: dict):
    """Trigger crisis response protocol"""
    
    crisis_type = CrisisType(crisis_data.get("crisis_type", "SYSTEM_DOWN"))
    description = crisis_data.get("description", "Crisis detected")
    affected_tasks = crisis_data.get("affected_tasks", [])
    
    response = await priority_engine.trigger_crisis_response(
        crisis_type, description, affected_tasks
    )
    
    return {
        "status": "crisis_response_activated",
        "crisis_response": response,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/priority/metrics")
async def get_system_metrics():
    """Get system performance and prioritization metrics"""
    
    return {
        "status": "success",
        "metrics": {
            "total_tasks": len(priority_engine.tasks),
            "crisis_tasks": len([t for t in priority_engine.tasks.values() if t.is_crisis]),
            "agent_workloads": priority_engine.agent_workloads,
            "priority_distribution": {
                priority.value: len([t for t in priority_engine.tasks.values() if t.current_priority == priority])
                for priority in TaskPriority
            }
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("🎯 Starting Agent Zero V1 - Point 3: Dynamic Task Prioritization...")
    logger.info("🚀 Real-time priority adjustment and crisis response system")
    logger.info("📊 System ready on port 8003")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        workers=1,
        log_level="info",
        reload=False
    )