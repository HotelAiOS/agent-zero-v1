"""
Agent Executor - FIXED VERSION
Agent Zero V1 - Critical Fix A0-6

Fixes:
- Standardized execute_task() method signature
- Proper type hints and validation
- Enhanced error handling
- Async support
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Context for task execution"""
    task_id: str
    agent_type: str
    input_data: Dict[str, Any]
    workspace_dir: str
    timeout: Optional[int] = 300  # 5 minutes default
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        return self.status == TaskStatus.FAILED


class AgentExecutor:
    """
    Enhanced Agent Executor with standardized interface

    FIXED: Method signature standardization for AI interface compatibility
    """

    def __init__(self, agent_registry: Optional[Dict] = None):
        """
        Initialize executor with agent registry

        Args:
            agent_registry: Dictionary mapping agent types to implementations
        """
        self.agent_registry = agent_registry or {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        logger.info("AgentExecutor initialized")

    async def execute_task(
        self,
        context: ExecutionContext,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """
        Execute agent task with standardized interface

        Args:
            context: Execution context with task details
            callback: Optional callback for progress updates

        Returns:
            ExecutionResult with task outcome

        Raises:
            ValueError: If agent type is not registered
            TimeoutError: If execution exceeds timeout
        """
        import time
        start_time = time.time()

        logger.info(f"Executing task {context.task_id} with agent {context.agent_type}")

        # Validate agent type
        if context.agent_type not in self.agent_registry:
            error_msg = f"Agent type '{context.agent_type}' not registered"
            logger.error(error_msg)
            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        try:
            # Get agent implementation
            agent = self.agent_registry[context.agent_type]

            # Execute with timeout
            task = asyncio.create_task(
                self._execute_with_timeout(agent, context, callback)
            )
            self.active_tasks[context.task_id] = task

            result = await task
            result.execution_time = time.time() - start_time

            logger.info(
                f"✅ Task {context.task_id} completed in {result.execution_time:.2f}s"
            )
            return result

        except asyncio.TimeoutError:
            error_msg = f"Task execution timed out after {context.timeout}s"
            logger.error(error_msg)
            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        finally:
            # Cleanup
            if context.task_id in self.active_tasks:
                del self.active_tasks[context.task_id]

    async def _execute_with_timeout(
        self,
        agent: Any,
        context: ExecutionContext,
        callback: Optional[Callable]
    ) -> ExecutionResult:
        """Execute agent task with timeout"""

        # Call progress callback if provided
        if callback:
            callback(context.task_id, TaskStatus.RUNNING)

        try:
            # Execute agent
            if asyncio.iscoroutinefunction(agent.execute):
                output = await asyncio.wait_for(
                    agent.execute(context.input_data, context.workspace_dir),
                    timeout=context.timeout
                )
            else:
                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        agent.execute,
                        context.input_data,
                        context.workspace_dir
                    ),
                    timeout=context.timeout
                )

            # Success
            if callback:
                callback(context.task_id, TaskStatus.COMPLETED)

            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.COMPLETED,
                output=output
            )

        except Exception as e:
            if callback:
                callback(context.task_id, TaskStatus.FAILED)
            raise

    def register_agent(self, agent_type: str, agent_instance: Any) -> None:
        """Register agent implementation"""
        self.agent_registry[agent_type] = agent_instance
        logger.info(f"Registered agent: {agent_type}")

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running task"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            logger.info(f"Cancelled task: {task_id}")
            return True
        return False

    def get_active_tasks(self) -> list:
        """Get list of active task IDs"""
        return list(self.active_tasks.keys())


# Backward compatibility wrapper
class LegacyAgentExecutor:
    """Wrapper for legacy code compatibility"""

    def __init__(self):
        self.executor = AgentExecutor()

    async def execute_agent_task(
        self,
        agent: Any,
        task: Dict[str, Any],
        output_dir: str
    ) -> Any:
        """Legacy method signature - converts to new interface"""

        context = ExecutionContext(
            task_id=task.get("id", "legacy_task"),
            agent_type=task.get("type", "unknown"),
            input_data=task,
            workspace_dir=output_dir
        )

        # Register agent if not already registered
        if context.agent_type not in self.executor.agent_registry:
            self.executor.register_agent(context.agent_type, agent)

        result = await self.executor.execute_task(context)

        if not result.is_success:
            raise RuntimeError(result.error)

        return result.output


# Example usage
if __name__ == "__main__":
    async def test_executor():
        # Mock agent
        class MockAgent:
            async def execute(self, input_data, workspace_dir):
                await asyncio.sleep(0.5)
                return {"result": "success", "data": input_data}

        executor = AgentExecutor()
        executor.register_agent("test_agent", MockAgent())

        context = ExecutionContext(
            task_id="test-001",
            agent_type="test_agent",
            input_data={"test": "data"},
            workspace_dir="/tmp/workspace"
        )

        result = await executor.execute_task(context)
        print(f"Result: {result}")
        print(f"Success: {result.is_success}")

    asyncio.run(test_executor())
