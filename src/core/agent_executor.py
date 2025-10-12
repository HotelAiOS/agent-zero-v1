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
    timeout: Optional[int] = 300
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


class AgentExecutor:
    """Enhanced Agent Executor with standardized interface"""

    def __init__(self, agent_registry: Optional[Dict] = None):
        self.agent_registry = agent_registry or {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        logger.info("AgentExecutor initialized")

    async def execute_task(
        self,
        context: ExecutionContext,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute agent task with standardized interface"""
        import time
        start_time = time.time()

        logger.info(f"Executing task {context.task_id} with agent {context.agent_type}")

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
            agent = self.agent_registry[context.agent_type]

            task = asyncio.create_task(
                self._execute_with_timeout(agent, context, callback)
            )
            self.active_tasks[context.task_id] = task

            result = await task
            result.execution_time = time.time() - start_time

            logger.info(f"✅ Task {context.task_id} completed in {result.execution_time:.2f}s")
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
            if context.task_id in self.active_tasks:
                del self.active_tasks[context.task_id]

    async def _execute_with_timeout(
        self,
        agent: Any,
        context: ExecutionContext,
        callback: Optional[Callable]
    ) -> ExecutionResult:
        """Execute agent task with timeout"""

        if callback:
            callback(context.task_id, TaskStatus.RUNNING)

        try:
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
