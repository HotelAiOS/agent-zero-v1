# intelligence_v2/__init__.py
"""
Agent Zero V1 - Intelligence V2.0 Package
Unified Point 3-6 Intelligence Layer with backward compatibility

This package consolidates existing Point 3 functionality with new Point 4-6 features
while maintaining 100% compatibility with existing services.
"""

__version__ = "2.0.0"
__author__ = "Agent Zero V1 Team"

# Core exports
from .interfaces import (
    Task, AgentProfile, PriorityDecision, ReassignmentDecision,
    PredictiveOutcome, FeedbackItem, MonitoringSnapshot
)

from .prioritization import DynamicTaskPrioritizer
from .predictive_planning import PredictiveResourcePlanner  
from .adaptive_learning import AdaptiveLearningEngine
from .realtime_monitoring import RealtimeMonitoringEngine
from .orchestrator import IntelligenceOrchestrator

# Compatibility exports for existing Point 3 service
from .legacy_compatibility import Point3CompatibilityWrapper

__all__ = [
    'Task', 'AgentProfile', 'PriorityDecision', 'ReassignmentDecision',
    'PredictiveOutcome', 'FeedbackItem', 'MonitoringSnapshot',
    'DynamicTaskPrioritizer', 'PredictiveResourcePlanner',
    'AdaptiveLearningEngine', 'RealtimeMonitoringEngine', 
    'IntelligenceOrchestrator', 'Point3CompatibilityWrapper'
]