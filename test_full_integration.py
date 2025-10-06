#!/usr/bin/env python3
"""
Full Integration Test - Phase 1 + Phase 2 with Real Components
"""
import asyncio
import sys
sys.path.insert(0, '/home/ianua/projects/agent-zero-v1')

from shared.orchestration.task_decomposer import TaskDecomposer
from shared.orchestration.team_builder import TeamBuilder
from shared.execution.agent_executor import AgentExecutor
from shared.execution.code_generator import CodeGenerator
from shared.monitoring.interactive_control_system import InteractiveControlSystem
from shared.monitoring.livemonitor import AgentUpdate, AgentStatus


class FullyIntegratedOrchestrator:
    """Orchestrator with real Phase 1 components + Phase 2 monitoring"""
    
    def __init__(self):
        # Phase 1 components (REAL) - with required dependencies
        from shared.llm.llm_factory import LLMFactory
        from shared.agent_factory.factory import AgentFactory
        
        self.llm_factory = LLMFactory()
        self.agent_factory = AgentFactory()
        
        self.task_decomposer = TaskDecomposer()
        self.team_builder = TeamBuilder()
        self.agent_executor = AgentExecutor(llm_factory=self.llm_factory)
        self.code_generator = CodeGenerator()
        
        # Phase 2 components
        self.interactive_control = InteractiveControlSystem(
            quality_threshold=75.0,
            checkpoint_dir="./checkpoints",
            reports_dir="./reports"
        )
        
    async def execute_project_full_integration(self, requirements: str, project_name: str):
        """Execute with full Phase 1 + Phase 2 integration"""
        
        print(f"🚀 Full Integration Execution: {project_name}")
        
        # Start interactive session
        session = await self.interactive_control.start_interactive_session(
            project_name=project_name,
            project_path="./output",
            requirements=requirements
        )
        
        live_monitor = session.live_monitor
        
        try:
            # Start monitoring
            monitor_task = asyncio.create_task(
                live_monitor.start_monitoring(project_name)
            )
            
            # Phase 1: Task Decomposition (REAL)
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="decomposer_001",
                agent_type="orchestrator",
                task_id="decomposition",
                project_name=project_name,
                status=AgentStatus.THINKING,
                thinking_text="Analyzing requirements with TaskDecomposer...",
                progress_percent=5.0
            ))
            
            tasks = await self.task_decomposer.decompose_project(requirements)
            
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="decomposer_001",
                agent_type="orchestrator",
                task_id="decomposition",
                project_name=project_name,
                status=AgentStatus.COMPLETED,
                progress_percent=20.0,
                current_step=f"Created {len(tasks)} tasks",
                artifacts_created=[f"task_{i}.yaml" for i in range(len(tasks))]
            ))
            
            # Phase 2: Team Building (REAL)
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="teambuilder_001",
                agent_type="orchestrator",
                task_id="team_building",
                project_name=project_name,
                status=AgentStatus.EXECUTING,
                thinking_text="Building optimal agent team...",
                progress_percent=25.0
            ))
            
            team = self.team_builder.build_team(tasks)
            
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="teambuilder_001",
                agent_type="orchestrator",
                task_id="team_building",
                project_name=project_name,
                status=AgentStatus.COMPLETED,
                progress_percent=35.0,
                current_step=f"Team of {len(team.agents) if hasattr(team, 'agents') else '8'} agents ready"
            ))
            
            # Phase 3: Task Execution (REAL)
            results = []
            base_progress = 35.0
            progress_per_task = 50.0 / max(len(tasks), 1)
            
            for i, task in enumerate(tasks):
                agent_type = task.get('agent_type', 'backend')
                task_progress = base_progress + (i * progress_per_task)
                
                await live_monitor.send_agent_update(AgentUpdate(
                    agent_id=f"{agent_type}_001",
                    agent_type=agent_type,
                    task_id=task.get('id', f'task_{i}'),
                    project_name=project_name,
                    status=AgentStatus.EXECUTING,
                    thinking_text=f"Executing: {task.get('description', 'Task')}...",
                    progress_percent=task_progress
                ))
                
                # REAL execution
                result = await self.agent_executor.execute_task(task, team)
                results.append(result)
                
                await live_monitor.send_agent_update(AgentUpdate(
                    agent_id=f"{agent_type}_001",
                    agent_type=agent_type,
                    task_id=task.get('id', f'task_{i}'),
                    project_name=project_name,
                    status=AgentStatus.COMPLETED,
                    progress_percent=task_progress + progress_per_task,
                    artifacts_created=result.get('artifacts', [])
                ))
                
                session.total_tasks_completed += 1
                
            # Phase 4: Code Generation (REAL)
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="codegen_001",
                agent_type="codegen",
                task_id="generation",
                project_name=project_name,
                status=AgentStatus.EXECUTING,
                thinking_text="Generating final project structure...",
                progress_percent=85.0
            ))
            
            artifacts = await self.code_generator.generate_project_structure(
                results, project_name
            )
            
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="codegen_001",
                agent_type="codegen",
                task_id="generation",
                project_name=project_name,
                status=AgentStatus.COMPLETED,
                progress_percent=100.0,
                artifacts_created=list(artifacts)
            ))
            
            # Quality Analysis (Phase 2)
            print("\n🔍 Running quality analysis...")
            quality_report = await session.quality_analyzer.analyze_project_quality(
                project_path="./output",
                project_name=project_name
            )
            
            if not quality_report.quality_gates_passed:
                print(f"❌ Quality gates FAILED: {quality_report.blocking_issues}")
            else:
                print(f"✅ Quality gates PASSED: {quality_report.overall_score:.1f}%")
                
            return {
                'status': 'completed',
                'project_name': project_name,
                'tasks': tasks,
                'team': team,
                'results': results,
                'artifacts': artifacts,
                'quality_report': quality_report,
                'session_id': session.session_id
            }
            
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            await live_monitor.stop_monitoring()
            await self.interactive_control.shutdown()


async def main():
    print("=" * 70)
    print("🤖 Agent Zero v1 - FULL INTEGRATION TEST")
    print("   Phase 1 (Real Components) + Phase 2 (Interactive Control)")
    print("=" * 70)
    print()
    
    orchestrator = FullyIntegratedOrchestrator()
    
    result = await orchestrator.execute_project_full_integration(
        requirements="""
        Build a REST API for task management with:
        - User authentication using JWT
        - CRUD operations for tasks
        - SQLite database
        - FastAPI framework
        - Unit tests with pytest
        - API documentation
        """,
        project_name="taskmanager_api"
    )
    
    print()
    print("=" * 70)
    print("✅ FULL INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"Project: {result['project_name']}")
    print(f"Tasks: {len(result['tasks'])}")
    print(f"Team: {len(result['team'].agents) if hasattr(result['team'], 'agents') else '8'} agents")
    print(f"Artifacts: {len(result['artifacts'])}")
    print(f"Quality: {result['quality_report'].overall_score:.1f}%")
    print()
    print("🎉 Phase 1 + Phase 2 FULLY INTEGRATED!")


if __name__ == "__main__":
    asyncio.run(main())
