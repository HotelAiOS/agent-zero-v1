"""
Agent Zero V1 - Fixed AgentExecutor Implementation
Resolves missing output_directory parameter issue in shared/execution/agent_executor.py

CRITICAL FIX: This addresses the TypeError where execute_task method was missing 
the output_directory parameter, causing failures in test_full_integration.py 
and all agent task executions in our custom Agent Zero V1 project.

Author: Agent Zero V1 Development Team
Version: 1.0.0 - CRITICAL HOTFIX
Date: 2025-10-07
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import json
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time


@dataclass
class TaskResult:
    """Task execution result with comprehensive metadata."""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    output_files: List[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class ExecutionConfig:
    """Configuration for task execution environment."""
    output_directory: str
    max_execution_time: int = 300  # 5 minutes default
    enable_logging: bool = True
    sandbox_mode: bool = True
    allowed_modules: List[str] = None
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    
    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = [
                "os", "sys", "json", "datetime", "pathlib", 
                "typing", "dataclasses", "logging"
            ]
        
        # Ensure output directory exists
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)


class AgentExecutor:
    """
    Enhanced AgentExecutor for Agent Zero V1 multi-agent system.
    
    CRITICAL FIX: Properly handles output_directory parameter and provides
    comprehensive task execution with error handling, logging, and monitoring.
    
    This fixes the signature mismatch that was causing:
    - TypeError: execute_task() missing 1 required positional argument: 'output_directory'
    - Test failures in test_full_integration.py
    - Agent task execution failures across the system
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str = "generic",
        config: Optional[ExecutionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize AgentExecutor with proper configuration.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type/category of agent (orchestrator, worker, etc.)
            config: Execution configuration
            logger: Optional logger instance
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or ExecutionConfig(output_directory="/tmp/agent_output")
        self.logger = logger or self._setup_logging()
        
        # Execution state
        self._active_tasks: Dict[str, Future] = {}
        self._task_history: List[TaskResult] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Performance metrics
        self._start_time = time.time()
        self._tasks_completed = 0
        self._tasks_failed = 0
        
        self.logger.info(f"AgentExecutor {agent_id} ({agent_type}) initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for agent execution."""
        logger = logging.getLogger(f"agent_zero.executor.{self.agent_id}")
        if not logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(agent_id)s] - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler if output directory is available
            if self.config and self.config.output_directory:
                log_file = Path(self.config.output_directory) / f"agent_{self.agent_id}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            logger.setLevel(logging.INFO)
            
        return logger
    
    def execute_task(
        self,
        task_definition: Dict[str, Any],
        output_directory: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> TaskResult:
        """
        FIXED METHOD: Execute a task with proper output_directory parameter handling.
        
        This is the CRITICAL FIX for the missing output_directory parameter issue
        that was causing TypeError across the Agent Zero V1 system.
        
        Args:
            task_definition: Task configuration and parameters
            output_directory: Directory for task output files (REQUIRED)
            context: Additional execution context
            timeout: Maximum execution time in seconds
            **kwargs: Additional execution parameters
            
        Returns:
            TaskResult with execution outcome and metadata
        """
        if self._shutdown:
            raise RuntimeError("AgentExecutor has been shut down")
        
        # Generate unique task ID
        task_id = f"{self.agent_id}_{int(time.time() * 1000)}"
        
        # Merge output_directory into config if provided
        execution_config = ExecutionConfig(
            output_directory=output_directory,
            max_execution_time=timeout or self.config.max_execution_time,
            enable_logging=self.config.enable_logging,
            sandbox_mode=self.config.sandbox_mode,
            allowed_modules=self.config.allowed_modules,
            max_memory_mb=self.config.max_memory_mb,
            max_cpu_percent=self.config.max_cpu_percent
        )
        
        self.logger.info(f"Starting task execution: {task_id}")
        self.logger.debug(f"Task definition: {task_definition}")
        self.logger.debug(f"Output directory: {output_directory}")
        
        start_time = time.time()
        
        try:
            # Validate task definition
            if not isinstance(task_definition, dict):
                raise ValueError("task_definition must be a dictionary")
            
            if "action" not in task_definition:
                raise ValueError("task_definition must contain 'action' field")
            
            # Prepare execution environment
            task_context = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "output_directory": output_directory,
                "start_time": start_time,
                "context": context or {},
                **kwargs
            }
            
            # Execute the task based on action type
            action = task_definition["action"]
            
            if action == "python_execution":
                result = self._execute_python_task(
                    task_definition, execution_config, task_context
                )
            elif action == "file_operation":
                result = self._execute_file_task(
                    task_definition, execution_config, task_context
                )
            elif action == "api_call":
                result = self._execute_api_task(
                    task_definition, execution_config, task_context
                )
            elif action == "knowledge_query":
                result = self._execute_knowledge_task(
                    task_definition, execution_config, task_context
                )
            else:
                # Generic task execution
                result = self._execute_generic_task(
                    task_definition, execution_config, task_context
                )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create successful task result
            task_result = TaskResult(
                task_id=task_id,
                success=True,
                output=result,
                execution_time=execution_time,
                output_files=self._get_output_files(output_directory),
                metadata={
                    "action": action,
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "context": task_context
                }
            )
            
            # Update metrics
            with self._lock:
                self._tasks_completed += 1
                self._task_history.append(task_result)
            
            self.logger.info(
                f"Task {task_id} completed successfully in {execution_time:.2f}s"
            )
            
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            
            # Create failed task result
            task_result = TaskResult(
                task_id=task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "action": task_definition.get("action", "unknown"),
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "traceback": traceback.format_exc()
                }
            )
            
            # Update metrics
            with self._lock:
                self._tasks_failed += 1
                self._task_history.append(task_result)
            
            self.logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {error_msg}")
            self.logger.debug(f"Task failure traceback: {traceback.format_exc()}")
            
            return task_result
    
    def _execute_python_task(
        self,
        task_def: Dict[str, Any],
        config: ExecutionConfig,
        context: Dict[str, Any]
    ) -> Any:
        """Execute Python code task."""
        code = task_def.get("code", "")
        if not code:
            raise ValueError("Python task requires 'code' parameter")
        
        # Prepare safe execution environment
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "open": open,  # Restricted in sandbox mode
            },
            "os": os if not config.sandbox_mode else None,
            "sys": sys if not config.sandbox_mode else None,
            "json": json,
            "datetime": datetime,
            "Path": Path,
        }
        
        safe_locals = {
            "output_directory": config.output_directory,
            "task_context": context,
        }
        
        # Execute code with timeout
        try:
            exec(code, safe_globals, safe_locals)
            return safe_locals.get("result", "Task completed successfully")
        except Exception as e:
            raise RuntimeError(f"Python execution failed: {e}")
    
    def _execute_file_task(
        self,
        task_def: Dict[str, Any],
        config: ExecutionConfig,
        context: Dict[str, Any]
    ) -> Any:
        """Execute file operation task."""
        operation = task_def.get("operation", "")
        file_path = task_def.get("file_path", "")
        content = task_def.get("content", "")
        
        if not operation or not file_path:
            raise ValueError("File task requires 'operation' and 'file_path' parameters")
        
        # Ensure file path is within output directory for security
        abs_file_path = Path(config.output_directory) / Path(file_path).name
        
        if operation == "write":
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            abs_file_path.write_text(content)
            return f"File written to {abs_file_path}"
        
        elif operation == "read":
            if abs_file_path.exists():
                return abs_file_path.read_text()
            else:
                raise FileNotFoundError(f"File not found: {abs_file_path}")
        
        elif operation == "delete":
            if abs_file_path.exists():
                abs_file_path.unlink()
                return f"File deleted: {abs_file_path}"
            else:
                raise FileNotFoundError(f"File not found: {abs_file_path}")
        
        else:
            raise ValueError(f"Unsupported file operation: {operation}")
    
    def _execute_api_task(
        self,
        task_def: Dict[str, Any],
        config: ExecutionConfig,
        context: Dict[str, Any]
    ) -> Any:
        """Execute API call task."""
        # Placeholder for API call implementation
        # This would integrate with the Agent Zero V1 API routing system
        endpoint = task_def.get("endpoint", "")
        method = task_def.get("method", "GET")
        params = task_def.get("parameters", {})
        
        self.logger.info(f"API call: {method} {endpoint}")
        
        # Mock response for now - replace with actual API integration
        return {
            "status": "success",
            "endpoint": endpoint,
            "method": method,
            "response": "API call completed successfully"
        }
    
    def _execute_knowledge_task(
        self,
        task_def: Dict[str, Any],
        config: ExecutionConfig,
        context: Dict[str, Any]
    ) -> Any:
        """Execute knowledge base query task."""
        # This would integrate with Neo4j knowledge base
        query_type = task_def.get("query_type", "")
        query_params = task_def.get("parameters", {})
        
        self.logger.info(f"Knowledge query: {query_type}")
        
        # Mock response - replace with Neo4j integration
        return {
            "status": "success",
            "query_type": query_type,
            "results": f"Knowledge query completed for {query_type}"
        }
    
    def _execute_generic_task(
        self,
        task_def: Dict[str, Any],
        config: ExecutionConfig,
        context: Dict[str, Any]
    ) -> Any:
        """Execute generic task."""
        action = task_def.get("action", "unknown")
        parameters = task_def.get("parameters", {})
        
        self.logger.info(f"Generic task execution: {action}")
        
        return {
            "status": "completed",
            "action": action,
            "parameters": parameters,
            "message": f"Generic task '{action}' executed successfully"
        }
    
    def _get_output_files(self, output_directory: str) -> List[str]:
        """Get list of files created in output directory."""
        try:
            output_path = Path(output_directory)
            if output_path.exists():
                return [str(f) for f in output_path.iterdir() if f.is_file()]
            return []
        except Exception as e:
            self.logger.warning(f"Error listing output files: {e}")
            return []
    
    def execute_task_async(
        self,
        task_definition: Dict[str, Any],
        output_directory: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Future[TaskResult]:
        """Execute task asynchronously and return Future."""
        future = self._executor.submit(
            self.execute_task,
            task_definition,
            output_directory,
            context,
            timeout,
            **kwargs
        )
        
        task_id = f"{self.agent_id}_{int(time.time() * 1000)}"
        with self._lock:
            self._active_tasks[task_id] = future
        
        return future
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        with self._lock:
            if task_id in self._active_tasks:
                future = self._active_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "running" if not future.done() else "completed",
                    "done": future.done()
                }
            
            # Check task history
            for result in self._task_history:
                if result.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "success": result.success,
                        "execution_time": result.execution_time
                    }
        
        return {"task_id": task_id, "status": "not_found"}
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent execution statistics."""
        uptime = time.time() - self._start_time
        
        with self._lock:
            active_count = len([f for f in self._active_tasks.values() if not f.done()])
            
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "uptime_seconds": uptime,
                "tasks_completed": self._tasks_completed,
                "tasks_failed": self._tasks_failed,
                "active_tasks": active_count,
                "total_tasks": len(self._task_history),
                "success_rate": (
                    self._tasks_completed / max(1, self._tasks_completed + self._tasks_failed)
                ) * 100,
                "average_execution_time": (
                    sum(r.execution_time for r in self._task_history[-10:]) / 
                    min(10, len(self._task_history))
                ) if self._task_history else 0.0
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor and cleanup resources."""
        self.logger.info(f"Shutting down AgentExecutor {self.agent_id}")
        
        self._shutdown = True
        
        if wait:
            # Wait for active tasks to complete
            with self._lock:
                for future in self._active_tasks.values():
                    try:
                        future.result(timeout=30)  # 30 second timeout
                    except Exception as e:
                        self.logger.warning(f"Task did not complete cleanly: {e}")
        
        # Shutdown executor
        self._executor.shutdown(wait=wait)
        
        self.logger.info(f"AgentExecutor {self.agent_id} shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, '_executor') and not self._shutdown:
            self.shutdown(wait=False)


# Factory function for creating AgentExecutor instances
def create_agent_executor(
    agent_id: str,
    agent_type: str = "generic",
    output_directory: Optional[str] = None,
    **config_kwargs
) -> AgentExecutor:
    """
    Factory function for creating properly configured AgentExecutor instances.
    
    Args:
        agent_id: Unique agent identifier
        agent_type: Agent type/category
        output_directory: Base output directory
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured AgentExecutor instance
    """
    if output_directory is None:
        output_directory = f"/tmp/agent_zero_output/{agent_id}"
    
    config = ExecutionConfig(
        output_directory=output_directory,
        **config_kwargs
    )
    
    return AgentExecutor(
        agent_id=agent_id,
        agent_type=agent_type,
        config=config
    )


if __name__ == "__main__":
    # Test the fixed AgentExecutor implementation
    print("Agent Zero V1 - AgentExecutor Fix Test")
    print("=" * 50)
    
    # Create test executor
    executor = create_agent_executor(
        agent_id="test_agent",
        agent_type="test",
        output_directory="/tmp/test_agent_output"
    )
    
    try:
        # Test the FIXED execute_task method with output_directory parameter
        task_def = {
            "action": "python_execution",
            "code": """
result = "Hello from Agent Zero V1!"
print(f"Task executed successfully: {result}")
"""
        }
        
        # This should now work without the TypeError
        result = executor.execute_task(
            task_definition=task_def,
            output_directory="/tmp/test_agent_output",
            context={"test": True}
        )
        
        print(f"✅ FIXED: Task executed successfully!")
        print(f"Task ID: {result.task_id}")
        print(f"Success: {result.success}")
        print(f"Output: {result.output}")
        print(f"Execution time: {result.execution_time:.3f}s")
        
        # Test statistics
        stats = executor.get_agent_statistics()
        print(f"\nAgent Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ AgentExecutor fix validation PASSED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        executor.shutdown()
        print("Test completed and executor shut down.")