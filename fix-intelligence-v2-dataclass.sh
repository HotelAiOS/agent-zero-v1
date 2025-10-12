#!/bin/bash

# Agent Zero V1 - Intelligence V2.0 HOTFIX
# Naprawia błąd dataclass w interfaces.py

echo "🔧 Intelligence V2.0 Hotfix - Naprawa błędu dataclass"
echo "===================================================="

# Backup existing file
if [ -f "intelligence_v2/interfaces.py" ]; then
    cp intelligence_v2/interfaces.py intelligence_v2/interfaces.py.backup
    echo "✅ Backup created: intelligence_v2/interfaces.py.backup"
fi

# Create corrected interfaces.py
cat > intelligence_v2/interfaces.py << 'EOF'
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
    # Required fields first
    title: str
    description: str
    
    # Optional fields with defaults
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    business_contexts: List[BusinessContext] = field(default_factory=list)
    estimated_hours: float = 0.0
    complexity_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None
    preferred_capabilities: List[AgentCapability] = field(default_factory=list)
    actual_hours: Optional[float] = None
    success_score: Optional[float] = None
    cost_usd: Optional[float] = None
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
    # Required fields
    id: str
    name: str
    capabilities: List[AgentCapability]
    
    # Optional fields with defaults
    current_workload: float = 0.0
    max_capacity: float = 40.0
    performance_score: float = 0.8
    success_rate: float = 0.85
    avg_completion_time: float = 1.0
    cost_per_hour: float = 50.0
    availability: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PriorityDecision:
    """Result of priority calculation with full justification"""
    # Required fields
    task_id: str
    calculated_priority: TaskPriority
    priority_score: float
    confidence: float
    base_priority_score: float
    urgency_multiplier: float
    business_context_multiplier: float
    dependency_impact_score: float
    resource_availability_score: float
    reasoning: str
    factors_considered: List[str]
    risk_assessment: str
    recommended_action: str
    
    # Optional fields with defaults
    calculated_at: datetime = field(default_factory=datetime.now)
    calculation_duration_ms: int = 0

@dataclass
class ReassignmentDecision:
    """Agent reassignment recommendation with cost analysis"""
    # Required fields
    task_id: str
    recommended_agent_id: str
    reassignment_reason: str
    confidence: float
    transition_cost: float
    efficiency_gain: float
    timeline_impact: int
    current_agent_utilization: float
    target_agent_utilization: float
    capability_match_score: float
    
    # Optional fields with defaults
    current_agent_id: Optional[str] = None
    recommended_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None

@dataclass
class PredictiveOutcome:
    """Prediction for task execution outcomes"""
    # Required fields
    task_id: str
    predicted_success_probability: float
    predicted_completion_time: float
    predicted_cost: float
    optimal_agent_id: str
    required_capabilities: List[AgentCapability]
    estimated_complexity: float
    business_value_score: float
    roi_estimate: float
    strategic_importance: str
    model_version: str
    
    # Optional fields with defaults
    risk_factors: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)
    predicted_at: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0

@dataclass
class FeedbackItem:
    """Structured feedback for continuous learning"""
    # Required fields
    task_id: str
    feedback_type: str
    actual_outcome: Dict[str, Any]
    rating: int
    provided_by: str
    
    # Optional fields with defaults
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    predicted_outcome: Optional[Dict[str, Any]] = None
    comment: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    actual_duration: Optional[float] = None
    actual_cost: Optional[float] = None
    quality_score: Optional[float] = None
    provided_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringSnapshot:
    """Real-time system monitoring snapshot"""
    # Required fields
    active_tasks: int
    pending_tasks: int
    completed_tasks_last_hour: int
    total_agents: int
    active_agents: int
    average_utilization: float
    average_task_completion_time: float
    success_rate_last_24h: float
    cost_efficiency_score: float
    prioritization_accuracy: float
    prediction_accuracy: float
    learning_rate: float
    system_load: float
    response_time_ms: float
    error_rate: float
    tasks_by_priority: Dict[str, int]
    revenue_impact_active: float
    cost_savings_achieved: float
    
    # Optional fields with defaults
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

# === FACTORY FUNCTIONS ===

def create_simple_task(title: str, description: str = "", priority: str = "medium") -> Task:
    """Create a simple task with minimal required fields"""
    return Task(
        title=title,
        description=description,
        priority=TaskPriority(priority)
    )

def create_agent_profile(agent_id: str, name: str, capabilities: List[str]) -> AgentProfile:
    """Create an agent profile with basic information"""
    return AgentProfile(
        id=agent_id,
        name=name,
        capabilities=[AgentCapability(cap) for cap in capabilities]
    )

def create_monitoring_snapshot_minimal(active_tasks: int = 0, total_agents: int = 0) -> MonitoringSnapshot:
    """Create a minimal monitoring snapshot for testing"""
    return MonitoringSnapshot(
        active_tasks=active_tasks,
        pending_tasks=0,
        completed_tasks_last_hour=0,
        total_agents=total_agents,
        active_agents=total_agents,
        average_utilization=0.5,
        average_task_completion_time=1.0,
        success_rate_last_24h=0.9,
        cost_efficiency_score=0.8,
        prioritization_accuracy=0.85,
        prediction_accuracy=0.8,
        learning_rate=0.1,
        system_load=0.3,
        response_time_ms=150,
        error_rate=0.05,
        tasks_by_priority={'high': 1, 'medium': 2, 'low': 1},
        revenue_impact_active=1000.0,
        cost_savings_achieved=500.0
    )
EOF

echo "✅ Fixed intelligence_v2/interfaces.py"

# Test the corrected interfaces
echo "🔧 Testing corrected interfaces..."

python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from intelligence_v2.interfaces import Task, TaskPriority, create_simple_task
    
    # Test creating a simple task
    task = create_simple_task('Test Task', 'Test Description', 'high')
    print(f'✅ Created task: {task.title} with priority {task.priority.value}')
    
    # Test dataclass functionality
    task_dict = task.to_dict()
    print(f'✅ Task serialization works: {len(task_dict)} fields')
    
    print('✅ All interfaces working correctly!')
    
except Exception as e:
    print(f'❌ Interface test failed: {e}')
    exit(1)
" || {
    echo "❌ Interface test still failing"
    exit 1
}

# Test full import chain
echo "🔧 Testing full import chain..."

python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from intelligence_v2.prioritization import DynamicTaskPrioritizer
    from intelligence_v2.interfaces import Task, TaskPriority
    
    # Test creating prioritizer
    prioritizer = DynamicTaskPrioritizer()
    print('✅ DynamicTaskPrioritizer created successfully')
    
    # Test creating task
    task = Task(title='Test Task', description='Test Description')
    print('✅ Task creation works')
    
    print('✅ Full Intelligence V2.0 import chain working!')
    
except Exception as e:
    print(f'❌ Full import test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 HOTFIX SUCCESS!"
    echo "=================="
    echo "✅ Dataclass syntax errors fixed"
    echo "✅ All interfaces working correctly"
    echo "✅ Import chain functional"
    echo ""
    echo "Intelligence V2.0 is now ready to start!"
    echo ""
    echo "Next steps:"
    echo "1. python3 intelligence-v2-main.py"
    echo "2. Test: curl http://localhost:8012/health"
    echo "3. Use: curl http://localhost:8012/api/v2/intelligence/prioritize"
    exit 0
else
    echo "❌ Hotfix failed - check the error above"
    exit 1
fi