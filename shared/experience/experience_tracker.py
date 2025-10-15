"""
Stub Experience Tracker for testing purposes.
"""

from typing import Any, Optional, Dict

class ExperienceTracker:
    """
    Stub Experience Tracker for testing.
    """
    def __init__(self):
        pass
        
    async def record_task_outcome(self, task_id: str, agent_id: str, success: bool, **kwargs: Any) -> None:
        pass
        
    async def get_agent_performance_stats(self, agent_id: str) -> Dict[str, Any]:
        return {"success_rate": 0.5, "avg_latency_ms": 500.0}
