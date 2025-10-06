import asyncio
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, AsyncIterator
from dataclasses import dataclass

from shared.monitoring.livemonitor import LiveMonitor
from agent_factory.factory import AgentInstance
from orchestration.task_decomposer import Task
from llm.llm_factory import LLMFactory

# Verbose logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('./agent_execution.log')]
)
logger = logging.getLogger('AgentExecutor')

@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict
    output: Optional[str] = None
    success: bool = False

class AgentExecutor:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.tool_calls_executed: List[ToolCall] = []
        self.live_monitor: LiveMonitor = None

    async def execute_task(
        self,
        agent: AgentInstance,
        task: Task,
        output_dir: Path
    ):
        from execution.project_orchestrator import TaskResult, TaskStatus

        start_time = datetime.now()
        logger.info(f"🚀 EXECUTING TASK: {task.id}")
        logger.info(f"   Agent Type: {agent.agent_type}")
        logger.info(f"   Description: {task.description}")

        # Ensure live_monitor is set
        if self.live_monitor is None:
            self.live_monitor = LiveMonitor()

        prompt = self._build_task_prompt(agent, task)
        task_dir = output_dir / self._sanitize_filename(task.name)
        task_dir.mkdir(exist_ok=True, parents=True)

        full_output = ""
        artifacts: List[str] = []

        logger.debug(f"📤 Prompt snippet: {prompt[:100]}...")
        async for chunk in self._stream_llm_response(agent, prompt):
            full_output += chunk
            await self.live_monitor.stream_token(chunk)

        logger.info(f"📝 Output length: {len(full_output)} chars")

        # Tool calls
        tool_calls = self._extract_tool_calls(full_output)
        for tc in tool_calls:
            logger.info(f"🔧 Tool call: {tc.tool_name}")
            await self._execute_tool_call(tc, task_dir)
            if tc.success and tc.output:
                artifacts.append(tc.output)

        # Extract code if no tool calls
        if not tool_calls:
            code_artifacts = await self._extract_and_save_code(full_output, task_dir, task)
            artifacts.extend(code_artifacts)

        # Validation
        valid = self._validate_output(full_output, task)
        if not valid:
            logger.warning("⚠️ Output validation warning")

        duration = (datetime.now() - start_time).total_seconds()
        result = TaskResult(
            task_id=task.id,
            task_name=task.name,
            status=TaskStatus.COMPLETED,
            agent_id=agent.id,
            agent_type=agent.agent_type,
            output=full_output,
            artifacts=artifacts,
            duration_seconds=duration
        )

        with open(task_dir / "output.txt", 'w', encoding='utf-8') as f:
            f.write(full_output)

        logger.info(f"✅ Completed {task.id} in {duration:.2f}s, artifacts={len(artifacts)}")
        return result

    # ... pozostałe metody bez zmian ...
