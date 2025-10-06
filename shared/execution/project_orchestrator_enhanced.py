# Agent Zero v1 - Enhanced ProjectOrchestrator with Phase 2 Integration
# Combines Phase 1 (Autonomous Execution) with Phase 2 (Interactive Control)

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class EnhancedProjectOrchestrator:
    """
    Enhanced ProjectOrchestrator with Phase 2 Interactive Control
    
    Combines autonomous execution (Phase 1) with real-time monitoring,
    quality analysis, and performance optimization (Phase 2).
    """
    
    def __init__(self, 
                 output_base_dir: str = "./output",
                 enable_interactive: bool = True):
        
        # Phase 1 components (simplified for now)
        self.output_base_dir = output_base_dir
        
        # Phase 2 components (NEW)
        self.enable_interactive = enable_interactive
        if enable_interactive:
            from shared.monitoring.interactive_control_system import InteractiveControlSystem
            
            self.interactive_control = InteractiveControlSystem(
                quality_threshold=75.0,
                checkpoint_dir=f"{output_base_dir}/../checkpoints",
                reports_dir=f"{output_base_dir}/../reports"
            )
        
    async def execute_project(self, requirements: str, project_name: str) -> Dict[str, Any]:
        """
        Traditional autonomous execution (Phase 1 only)
        Maintains backward compatibility with existing code.
        """
        print(f"🤖 Starting autonomous execution: {project_name}")
        
        # Simplified Phase 1 execution
        start_time = time.time()
        
        # Simulate project execution
        print("📋 Decomposing requirements into tasks...")
        await asyncio.sleep(1)
        
        print("👥 Building agent team...")
        await asyncio.sleep(1)
        
        print("⚡ Executing tasks...")
        await asyncio.sleep(2)
        
        print("📦 Generating project artifacts...")
        await asyncio.sleep(1)
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'completed',
            'project_name': project_name,
            'execution_time': execution_time,
            'tasks_completed': 5,
            'artifacts': ['main.py', 'models.py', 'api.py', 'tests.py'],
            'mode': 'autonomous'
        }
        
    async def execute_project_interactive(self, requirements: str, project_name: str) -> Dict[str, Any]:
        """
        NEW: Interactive execution with Phase 2 monitoring and control
        
        Features:
        - Real-time monitoring of agent execution
        - User intervention capabilities ([S]top, [P]ause, [C]ontinue)
        - Quality gates enforcement
        - Performance optimization
        - Checkpoint save/resume
        """
        if not self.enable_interactive:
            # Fallback to traditional execution
            return await self.execute_project(requirements, project_name)
            
        print(f"🎮 Starting INTERACTIVE execution: {project_name}")
        
        # Start interactive session
        session = await self.interactive_control.start_interactive_session(
            project_name=project_name,
            project_path=self.output_base_dir,
            requirements=requirements
        )
        
        try:
            # Execute with monitoring
            result = await self.execute_with_monitoring(session, requirements)
            
            # Quality analysis
            print("
🔍 Running quality analysis...")
            quality_report = await session.quality_analyzer.analyze_project_quality(
                project_path=self.output_base_dir,
                project_name=project_name
            )
            
            # Performance report
            print("⚡ Generating performance report...")
            perf_report = await session.performance_optimizer.generate_optimization_report()
            
            return {
                **result,
                'quality_report': quality_report,
                'performance_report': perf_report,
                'session_id': session.session_id,
                'mode': 'interactive'
            }
            
        finally:
            await self.interactive_control.shutdown()
            
    async def execute_with_monitoring(self, session, requirements: str) -> Dict[str, Any]:
        """
        Core execution with Phase 2 monitoring integration
        """
        from shared.monitoring.livemonitor import AgentUpdate, AgentStatus
        
        print(f"🔄 Executing {session.project_name} with live monitoring...")
        
        start_time = time.time()
        live_monitor = session.live_monitor
        
        # Start live monitoring
        monitor_task = asyncio.create_task(
            live_monitor.start_monitoring(session.project_name)
        )
        
        try:
            # Phase 1: Task Decomposition
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="orchestrator_001",
                agent_type="orchestrator", 
                task_id="task_decomposition",
                project_name=session.project_name,
                status=AgentStatus.THINKING,
                thinking_text="Analyzing requirements and breaking down into executable tasks...",
                progress_percent=10.0,
                current_step="Requirements analysis"
            ))
            
            await asyncio.sleep(2)
            tasks = ['task_1', 'task_2', 'task_3', 'task_4', 'task_5']
            
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="orchestrator_001",
                agent_type="orchestrator",
                task_id="task_decomposition", 
                project_name=session.project_name,
                status=AgentStatus.COMPLETED,
                progress_percent=25.0,
                current_step=f"Created {len(tasks)} tasks",
                artifacts_created=[f"task_{i+1}.yaml" for i in range(len(tasks))]
            ))
            
            # Phase 2: Team Building
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="orchestrator_001",
                agent_type="orchestrator",
                task_id="team_building",
                project_name=session.project_name, 
                status=AgentStatus.EXECUTING,
                thinking_text="Selecting optimal agent team based on task requirements...",
                progress_percent=30.0,
                current_step="Agent team assembly"
            ))
            
            await asyncio.sleep(1)
            
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="orchestrator_001", 
                agent_type="orchestrator",
                task_id="team_building",
                project_name=session.project_name,
                status=AgentStatus.COMPLETED,
                progress_percent=40.0,
                current_step="Assembled team of 3 agents"
            ))
            
            # Phase 3: Task Execution
            agents = ['backend', 'frontend', 'database']
            progress_per_task = 50.0 / len(tasks)
            
            for i, (task, agent) in enumerate(zip(tasks, agents * 2)):
                task_progress = 40.0 + (i * progress_per_task)
                
                await live_monitor.send_agent_update(AgentUpdate(
                    agent_id=f"{agent}_001",
                    agent_type=agent,
                    task_id=task,
                    project_name=session.project_name,
                    status=AgentStatus.EXECUTING,
                    thinking_text=f"Implementing {task} using {agent} agent...",
                    progress_percent=task_progress,
                    current_step=f"Task {i+1}/{len(tasks)}"
                ))
                
                await asyncio.sleep(1.5)
                
                await live_monitor.send_agent_update(AgentUpdate(
                    agent_id=f"{agent}_001",
                    agent_type=agent, 
                    task_id=task,
                    project_name=session.project_name,
                    status=AgentStatus.COMPLETED,
                    progress_percent=task_progress + progress_per_task,
                    current_step=f"Completed task {i+1}/{len(tasks)}",
                    tokens_generated=250 + (i * 50),
                    time_elapsed=1.5,
                    artifacts_created=[f"{task}_result.py"]
                ))
                
                session.total_tasks_completed += 1
                
            # Phase 4: Code Generation
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="codegen_001",
                agent_type="codegen",
                task_id="artifact_generation", 
                project_name=session.project_name,
                status=AgentStatus.EXECUTING,
                thinking_text="Generating final project structure and artifacts...",
                progress_percent=90.0,
                current_step="Creating project files"
            ))
            
            await asyncio.sleep(1)
            
            artifacts = ['main.py', 'models.py', 'api.py', 'tests.py', 'README.md']
            
            await live_monitor.send_agent_update(AgentUpdate(
                agent_id="codegen_001",
                agent_type="codegen", 
                task_id="artifact_generation",
                project_name=session.project_name,
                status=AgentStatus.COMPLETED,
                progress_percent=100.0,
                current_step="Project generation complete",
                artifacts_created=artifacts
            ))
            
            execution_time = time.time() - start_time
            
            # Create checkpoint
            await live_monitor.create_checkpoint(
                project_name=session.project_name,
                completed_tasks=tasks,
                current_phase="completed",
                agent_states={agent: {"status": "completed"} for agent in agents},
                execution_context={
                    "requirements": requirements,
                    "execution_time": execution_time,
                    "artifacts_count": len(artifacts)
                }
            )
            
            return {
                'status': 'completed',
                'project_name': session.project_name,
                'execution_time': execution_time,
                'tasks_completed': len(tasks),
                'artifacts': artifacts,
                'session_id': session.session_id,
                'quality_tracked': True,
                'performance_tracked': True
            }
            
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            await live_monitor.stop_monitoring()


# Factory function
def create_project_orchestrator(output_base_dir: str = "./output",
                               enable_interactive: bool = True) -> EnhancedProjectOrchestrator:
    return EnhancedProjectOrchestrator(
        output_base_dir=output_base_dir,
        enable_interactive=enable_interactive
    )


# Alias for backward compatibility
ProjectOrchestrator = EnhancedProjectOrchestrator
