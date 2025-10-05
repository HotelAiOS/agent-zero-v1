"""
Orchestration Module
System orkiestracji projektów, zadań i zespołów agentów
"""

from .planner import IntelligentPlanner, ProjectPlan
from .team_formation import TeamFormationEngine, Team
from .task_decomposer import TaskDecomposer, Task, TaskType
from .dependency_graph import DependencyGraph, DependencyType
from .quality_gates import QualityGateManager, QualityGate, GateStatus
from .scheduler import TaskScheduler, ScheduleStrategy

__all__ = [
    'IntelligentPlanner',
    'ProjectPlan',
    'TeamFormationEngine',
    'Team',
    'TaskDecomposer',
    'Task',
    'TaskType',
    'DependencyGraph',
    'DependencyType',
    'QualityGateManager',
    'QualityGate',
    'GateStatus',
    'TaskScheduler',
    'ScheduleStrategy'
]

__version__ = '1.0.0'
