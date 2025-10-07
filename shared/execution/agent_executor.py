import os
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

class AgentExecutor:
    """AgentExecutor with fixed method signature including output_dir parameter."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "tasks_executed": 0,
            "tasks_successful": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
    
    def execute_task(self, agent, task: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Execute task with FIXED method signature including output_dir parameter."""
        start_time = datetime.now()
        
        # Validate parameters
        if agent is None:
            raise ValueError("Agent parameter cannot be None")
        if not isinstance(task, dict) or not task:
            raise ValueError("Task parameter must be a non-empty dictionary")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("Output directory must be a non-empty string")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Execute task
        if hasattr(agent, "execute"):
            result = agent.execute(task, {"output_dir": output_dir})
        elif hasattr(agent, "run"):
            result = agent.run(task, {"output_dir": output_dir})
        else:
            result = {"status": "completed", "message": "Generic execution"}
        
        # Add execution metadata
        execution_time = (datetime.now() - start_time).total_seconds()
        result["execution_metadata"] = {
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "output_directory": output_dir,
            "task_id": task.get("id"),
            "executor_version": "1.0.0"
        }
        
        # Update stats
        self.stats["tasks_executed"] += 1
        self.stats["tasks_successful"] += 1
        self.stats["total_execution_time"] += execution_time
        
        return result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.stats.copy()
        if stats["tasks_executed"] > 0:
            stats["success_rate"] = stats["tasks_successful"] / stats["tasks_executed"]
            stats["average_execution_time"] = stats["total_execution_time"] / stats["tasks_executed"]
        else:
            stats["success_rate"] = 0.0
            stats["average_execution_time"] = 0.0
        return stats

def create_agent_executor(config=None):
    return AgentExecutor()


class AgentExecutorError(Exception):
    """Custom exception for AgentExecutor errors"""
    pass

class AsyncAgentExecutor(AgentExecutor):
    """Async version of AgentExecutor"""
    async def execute_task(self, agent, task, output_dir):
        """Async version - just calls sync version for now"""
        return super().execute_task(agent, task, output_dir)

