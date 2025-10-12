# intelligence_v2/interfaces.py
"""
Agent Zero V1 - Intelligence V2.0 Data Contracts
Standardized interfaces for all Point 3-6 components with full backward compatibility
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid

# === CORE ENUMS ===

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AgentCapability(Enum):
    PYTHON = "python"
    FASTAPI = "fastapi"
    ANALYSIS = "analysis"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    AI_ML = "ai_ml"
    DEVOPS = "devops"

class BusinessContext(Enum):
    REVENUE_CRITICAL = "revenue_critical"
    SECURITY_CRITICAL = "security_critical"
    CUSTOMER_FACING = "customer_facing"
    COMPLIANCE_REQUIRED = "compliance_required"
    INTERNAL_TOOLS = "internal_tools"

class CrisisType(Enum):
    SYSTEM_DOWN = "system_down"
    DATA_BREACH = "data_breach"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CUSTOMER_ESCALATION = "customer_escalation"
    SECURITY_INCIDENT = "security_incident"
    REGULATORY_VIOLATION = "regulatory_violation"

# === CORE DATA STRUCTURES ===

@dataclass
class Task:
    """Enhanced Task model with full V2.0 Intelligence support"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    business_contexts: List[BusinessContext] = field(default_factory=list)
    
    # Resource estimation
    estimated_hours: float = 0.0
    complexity_score: float = 0.5  # 0.0-1.0
    
    # Temporal constraints
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies and relationships
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    blocks: List[str] = field(default_factory=list)       # Task IDs this task blocks
    
    # Agent assignment
    assigned_agent_id: Optional[str] = None
    preferred_capabilities: List[AgentCapability] = field(default_factory=list)
    
    # Performance tracking
    actual_hours: Optional[float] = None
    success_score: Optional[float] = None
    cost_usd: Optional[float] = None
    
    # Metadata and context
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'business_contexts': [bc.value for bc in self.business_contexts],
            'estimated_hours': self.estimated_hours,
            'complexity_score': self.complexity_score,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'dependencies': self.dependencies,
            'assigned_agent_id': self.assigned_agent_id,
            'preferred_capabilities': [cap.value for cap in self.preferred_capabilities],
            'metadata': self.metadata
        }

@dataclass 
class AgentProfile:
    """Agent profile with capabilities and performance metrics"""
    id: str
    name: str
    capabilities: List[AgentCapability]
    current_workload: float = 0.0  # Hours per week
    max_capacity: float = 40.0     # Max hours per week
    performance_score: float = 0.8  # 0.0-1.0
    success_rate: float = 0.85     # 0.0-1.0
    avg_completion_time: float = 1.0  # Multiplier of estimated time
    cost_per_hour: float = 50.0    # USD per hour
    availability: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PriorityDecision:
    """Result of priority calculation with full justification"""
    task_id: str
    calculated_priority: TaskPriority
    priority_score: float  # 0.0-1.0
    confidence: float      # 0.0-1.0
    
    # Scoring breakdown
    base_priority_score: float
    urgency_multiplier: float
    business_context_multiplier: float
    dependency_impact_score: float
    resource_availability_score: float
    
    # Decision reasoning
    reasoning: str
    factors_considered: List[str]
    risk_assessment: str
    recommended_action: str
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.now)
    calculation_duration_ms: int = 0
    
@dataclass
class ReassignmentDecision:
    """Agent reassignment recommendation with cost analysis"""
    task_id: str
    current_agent_id: Optional[str]
    recommended_agent_id: str
    reassignment_reason: str
    confidence: float
    
    # Cost analysis
    transition_cost: float     # Hours lost in handoff
    efficiency_gain: float     # Expected improvement
    timeline_impact: int       # Days saved/lost
    
    # Agent analysis
    current_agent_utilization: float
    target_agent_utilization: float
    capability_match_score: float
    
    # Decision metadata
    recommended_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None

@dataclass
class PredictiveOutcome:
    """Prediction for task execution outcomes"""
    task_id: str
    predicted_success_probability: float
    predicted_completion_time: float  # Hours
    predicted_cost: float            # USD
    
    # Risk factors
    risk_factors: List[str]
    confidence_intervals: Dict[str, tuple]  # metric -> (low, high)
    
    # Resource predictions
    optimal_agent_id: str
    required_capabilities: List[AgentCapability]
    estimated_complexity: float
    
    # Business impact
    business_value_score: float
    roi_estimate: float
    strategic_importance: str
    
    # Prediction metadata
    model_version: str
    predicted_at: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0

@dataclass
class FeedbackItem:
    """Structured feedback for continuous learning"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    feedback_type: str  # "completion", "quality", "prediction_accuracy"
    
    # Actual vs predicted
    actual_outcome: Dict[str, Any]
    predicted_outcome: Optional[Dict[str, Any]] = None
    
    # Feedback content
    rating: int  # 1-5 scale
    comment: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    # Performance metrics
    actual_duration: Optional[float] = None
    actual_cost: Optional[float] = None
    quality_score: Optional[float] = None
    
    # Context
    provided_by: str  # Agent ID or "system"
    provided_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringSnapshot:
    """Real-time system monitoring snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # System metrics
    active_tasks: int
    pending_tasks: int
    completed_tasks_last_hour: int
    
    # Agent metrics
    total_agents: int
    active_agents: int
    average_utilization: float
    
    # Performance metrics  
    average_task_completion_time: float
    success_rate_last_24h: float
    cost_efficiency_score: float
    
    # Intelligence metrics
    prioritization_accuracy: float
    prediction_accuracy: float
    learning_rate: float
    
    # System health
    system_load: float
    response_time_ms: float
    error_rate: float
    
    # Business metrics
    tasks_by_priority: Dict[str, int]
    revenue_impact_active: float
    cost_savings_achieved: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

# === COMPATIBILITY TYPES ===

# Legacy compatibility types for existing Point 3 service
LegacyTaskDict = Dict[str, Union[str, int, float, List[str]]]
LegacyAgentDict = Dict[str, Union[str, int, float]]
LegacyPriorityResult = Dict[str, Union[str, float]]

# === UTILITY FUNCTIONS ===

def task_from_legacy_dict(data: LegacyTaskDict) -> Task:
    """Convert legacy task dictionary to Task dataclass"""
    return Task(
        id=data.get('id', str(uuid.uuid4())),
        title=data.get('title', ''),
        description=data.get('description', ''),
        status=TaskStatus(data.get('status', 'pending')),
        priority=TaskPriority(data.get('priority', 'medium')),
        estimated_hours=data.get('estimated_hours', 0.0),
        business_contexts=[BusinessContext(bc) for bc in data.get('business_contexts', [])],
        dependencies=data.get('dependencies', []),
        assigned_agent_id=data.get('assigned_agent_id')
    )

def task_to_legacy_dict(task: Task) -> LegacyTaskDict:
    """Convert Task dataclass to legacy dictionary format"""
    return {
        'id': task.id,
        'title': task.title,
        'description': task.description,
        'status': task.status.value,
        'priority': task.priority.value,
        'estimated_hours': task.estimated_hours,
        'business_contexts': [bc.value for bc in task.business_contexts],
        'dependencies': task.dependencies,
        'assigned_agent_id': task.assigned_agent_id
    }

def create_monitoring_snapshot_from_simple_tracker(tracker_data: Dict[str, Any]) -> MonitoringSnapshot:
    """Create monitoring snapshot from SimpleTracker data"""
    return MonitoringSnapshot(
        active_tasks=tracker_data.get('active_tasks', 0),
        pending_tasks=tracker_data.get('pending_tasks', 0),
        completed_tasks_last_hour=tracker_data.get('completed_last_hour', 0),
        total_agents=tracker_data.get('total_agents', 0),
        active_agents=tracker_data.get('active_agents', 0),
        average_utilization=tracker_data.get('avg_utilization', 0.0),
        success_rate_last_24h=tracker_data.get('success_rate_24h', 0.0),
        system_load=tracker_data.get('system_load', 0.0),
        response_time_ms=tracker_data.get('response_time_ms', 0),
        metadata=tracker_data.get('metadata', {})
    )