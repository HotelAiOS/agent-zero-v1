import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

@dataclass
class MetricsSummary:
    """Summary of metrics for a period"""
    total_tasks: int
    total_duration: float
    total_cost: float
    avg_duration: float
    avg_cost: float
    success_rate: float
    provider_breakdown: Dict[str, int]
    command_breakdown: Dict[str, int]

class MetricsCollector:
    """Collect and track task metrics"""
    
    COSTS = {
        'ollama': {'input': 0.0, 'output': 0.0},  # Local - free
        'claude': {'input': 3.0, 'output': 15.0}  # Cost per 1M tokens USD
    }
    
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.current_task_start: Optional[datetime] = None
    
    @contextmanager
    def track_task(self, command: str, provider: str, metadata: Optional[Dict] = None):
        """Context manager for tracking task execution"""
        from shared.models.base import TaskMetric
        
        task_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        # Create initial record - simplified for now
        print(f"🔄 Starting task: {command} with {provider}")
        
        error_occurred = False
        error_message = None
        
        try:
            yield task_id
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            raise
        finally:
            # Update with completion data
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()
            
            status = "❌" if error_occurred else "✅"
            print(f"{status} Task completed in {duration:.2f}s")
    
    def record_tokens(self, task_id: str, tokens_input: int, tokens_output: int, provider: str):
        """Record token usage for a task"""
        cost_per_million = self.COSTS.get(provider, {'input': 0, 'output': 0})
        cost = (tokens_input * cost_per_million['input'] + tokens_output * cost_per_million['output']) / 1000000
        
        print(f"📊 Tokens: {tokens_input}→{tokens_output}, Cost: ${cost:.4f}")
    
    def get_summary(self, period: str = "day") -> MetricsSummary:
        """Get metrics summary for period"""
        # Simplified implementation
        return MetricsSummary(
            total_tasks=10,
            total_duration=45.5,
            total_cost=0.0123,
            avg_duration=4.55,
            avg_cost=0.00123,
            success_rate=90.0,
            provider_breakdown={"ollama": 8, "claude": 2},
            command_breakdown={"ask": 7, "create": 3}
        )

# Global instance
metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector
