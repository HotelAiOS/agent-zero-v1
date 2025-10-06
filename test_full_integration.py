#!/usr/bin/env python3
import asyncio, sys
sys.path.insert(0, '/home/ianua/projects/agent-zero-v1')

from shared.orchestration.task_decomposer import TaskDecomposer
from shared.orchestration.team_builder import TeamBuilder
from shared.execution.agent_executor import AgentExecutor
from shared.execution.code_generator import CodeGenerator
from shared.monitoring.livemonitor import LiveMonitor, AgentUpdate, AgentStatus

class FullyIntegratedOrchestrator:
    def __init__(self):
        from shared.llm.llm_factory import LLMFactory
        from shared.agent_factory.factory import AgentFactory

        self.llm_factory = LLMFactory()
        self.agent_factory = AgentFactory()
        self.task_decomposer = TaskDecomposer()
        self.team_builder = TeamBuilder()
        self.agent_executor = AgentExecutor(self.llm_factory)
        self.code_generator = CodeGenerator()
        self.interactive_control = LiveMonitor()

    async def execute_project_full_integration(self, requirements, project_name):
        print(f"🚀 Starting: {project_name}")
        live_monitor = self.interactive_control
        live_monitor.should_stop = False

        monitor_task = asyncio.create_task(
            live_monitor.start_monitoring_live(project_name)
        )

        tasks = await self.task_decomposer.decompose_project(requirements)
        for i, task in enumerate(tasks):
            update = AgentUpdate(
                agent_id=f"agent_{i}",
                agent_type=getattr(task, 'agent_type','backend'),
                task_id=getattr(task,'id',f"task_{i}"),
                project_name=project_name,
                status=AgentStatus.EXECUTING,
                thinking_text="Starting task",
                progress_percent=0.0
            )
            await live_monitor.send_agent_update(update)
            result = await self.agent_executor.execute_task(
                AgentInstance(id=str(i), agent_type=update.agent_type),
                task,
                Path("./output")
            )
            await live_monitor.stream_token(f"\nCompleted {task.id}\n")
        live_monitor.should_stop = True
        await monitor_task

if __name__ == "__main__":
    asyncio.run(FullyIntegratedOrchestrator().execute_project_full_integration(
        "Create simple calculator", "simple_calc"
    ))
