"""
Project Orchestrator - Entry point for executing complete projects.

Responsibilities:
- Execute projects end-to-end
- Manage execution phases (sequential/parallel)
- Handle task dependencies
- Coordinate multiple agents
- Save results to output directory
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from orchestration.task_decomposer import Task
from agent_factory.factory import AgentInstance


class PhaseStatus(Enum):
    """Status fazy wykonania."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(Enum):
    """Status zadania."""
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Wynik wykonania pojedynczego zadania."""
    task_id: str
    task_name: str
    status: TaskStatus
    agent_id: str
    agent_type: str
    output: Optional[str] = None
    artifacts: List[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class PhaseResult:
    """Wynik wykonania fazy."""
    phase_number: int
    phase_name: str
    status: PhaseStatus
    task_results: List[TaskResult]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0


@dataclass
class ProjectResult:
    """Kompletny wynik wykonania projektu."""
    project_name: str
    requirements: str
    status: str
    phase_results: List[PhaseResult]
    total_duration_seconds: float
    output_directory: str
    start_time: datetime
    end_time: datetime
    summary: Dict[str, any] = None
    
    def __post_init__(self):
        if self.summary is None:
            self.summary = {}


class ProjectOrchestrator:
    """
    Orkiestrator projektów - zarządza pełnym cyklem wykonania projektu.
    
    Proces wykonania:
    1. Dekomponuje wymagania na zadania (TaskDecomposer)
    2. Buduje zespół agentów (TeamBuilder)
    3. Wykonuje zadania fazami z obsługą zależności
    4. Zbiera wyniki i generuje artefakty
    5. Zapisuje wszystko do katalogu wyjściowego
    """
    
    def __init__(
        self,
        task_decomposer,
        team_builder,
        agent_executor,
        output_base_dir: str = "./output"
    ):
        """
        Args:
            task_decomposer: TaskDecomposer instance
            team_builder: TeamBuilder instance
            agent_executor: AgentExecutor instance
            output_base_dir: Bazowy katalog dla wyników projektów
        """
        self.task_decomposer = task_decomposer
        self.team_builder = team_builder
        self.agent_executor = agent_executor
        self.output_base_dir = Path(output_base_dir)
        
        self.task_results: Dict[str, TaskResult] = {}
        self.completed_tasks: set = set()
        
    async def execute_project(
        self,
        requirements: str,
        project_name: Optional[str] = None
    ) -> ProjectResult:
        """
        Wykonaj kompletny projekt od wymagań do działającego kodu.
        
        Args:
            requirements: Wymagania biznesowe projektu
            project_name: Nazwa projektu (opcjonalna, wygenerowana jeśli None)
            
        Returns:
            ProjectResult z kompletnymi wynikami
        """
        start_time = datetime.now()
        
        if project_name is None:
            project_name = f"project_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🚀 === STARTING PROJECT: {project_name} ===")
        print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 Requirements:\n{requirements}\n")
        
        output_dir = self.output_base_dir / project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("📊 Phase 0: Task Decomposition...")
            tasks, phases = await self.task_decomposer.decompose(requirements)
            print(f"✅ Created {len(tasks)} tasks in {len(phases)} execution phases\n")
            
            print("👥 Phase 0: Team Building...")
            team = await self.team_builder.build_team(tasks)
            print(f"✅ Assembled team of {len(team)} agents\n")
            
            self._save_project_plan(output_dir, requirements, tasks, phases, team)
            
            phase_results = []
            for phase_num, phase_tasks in phases.items():
                phase_result = await self.execute_phase(
                    phase_num=phase_num,
                    phase_tasks=phase_tasks,
                    team=team,
                    output_dir=output_dir
                )
                phase_results.append(phase_result)
                
                if phase_result.status == PhaseStatus.FAILED:
                    print(f"❌ Phase {phase_num} failed, aborting project")
                    break
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            all_completed = all(
                pr.status == PhaseStatus.COMPLETED for pr in phase_results
            )
            status = "COMPLETED" if all_completed else "FAILED"
            
            project_result = ProjectResult(
                project_name=project_name,
                requirements=requirements,
                status=status,
                phase_results=phase_results,
                total_duration_seconds=duration,
                output_directory=str(output_dir),
                start_time=start_time,
                end_time=end_time,
                summary=self._generate_summary(phase_results)
            )
            
            self._save_final_report(output_dir, project_result)
            
            print(f"\n{'🎉' if status == 'COMPLETED' else '⚠️'} === PROJECT {status} ===")
            print(f"⏱️  Total duration: {duration:.2f}s")
            print(f"📁 Output directory: {output_dir}")
            
            return project_result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n❌ PROJECT FAILED: {str(e)}")
            
            return ProjectResult(
                project_name=project_name,
                requirements=requirements,
                status="FAILED",
                phase_results=[],
                total_duration_seconds=duration,
                output_directory=str(output_dir),
                start_time=start_time,
                end_time=end_time,
                summary={"error": str(e)}
            )
    
    async def execute_phase(
        self,
        phase_num: int,
        phase_tasks: List[Task],
        team: List[AgentInstance],
        output_dir: Path
    ) -> PhaseResult:
        """
        Wykonaj pojedynczą fazę z wieloma zadaniami.
        
        Zadania w jednej fazie mogą być wykonywane równolegle,
        ponieważ nie mają między sobą zależności.
        """
        start_time = datetime.now()
        phase_name = f"Phase {phase_num}"
        
        print(f"\n⚙️  === EXECUTING {phase_name} ===")
        print(f"📋 Tasks: {len(phase_tasks)}")
        
        phase_dir = output_dir / f"phase_{phase_num}"
        phase_dir.mkdir(exist_ok=True)
        
        for task in phase_tasks:
            if not await self._wait_for_dependencies(task):
                print(f"❌ Task {task.name} dependencies not met")
                return PhaseResult(
                    phase_number=phase_num,
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    task_results=[],
                    start_time=start_time,
                    end_time=datetime.now()
                )
        
        task_coroutines = [
            self._execute_task_with_agent(task, team, phase_dir)
            for task in phase_tasks
        ]
        
        task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                print(f"❌ Task {phase_tasks[i].name} failed: {str(result)}")
                valid_results.append(TaskResult(
                    task_id=phase_tasks[i].id,
                    task_name=phase_tasks[i].name,
                    status=TaskStatus.FAILED,
                    agent_id="unknown",
                    agent_type="unknown",
                    error=str(result)
                ))
            else:
                valid_results.append(result)
                if result.status == TaskStatus.COMPLETED:
                    self.completed_tasks.add(result.task_id)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        all_completed = all(r.status == TaskStatus.COMPLETED for r in valid_results)
        status = PhaseStatus.COMPLETED if all_completed else PhaseStatus.FAILED
        
        phase_result = PhaseResult(
            phase_number=phase_num,
            phase_name=phase_name,
            status=status,
            task_results=valid_results,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration
        )
        
        print(f"✅ {phase_name} {status.value} in {duration:.2f}s")
        
        return phase_result
    
    async def _execute_task_with_agent(
        self,
        task: Task,
        team: List[AgentInstance],
        output_dir: Path
    ) -> TaskResult:
        """Wykonaj pojedyncze zadanie z przypisanym agentem."""
        agent = next((a for a in team if a.id == task.assigned_agent_id), None)
        
        if agent is None:
            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                agent_id="unknown",
                agent_type="unknown",
                error="No agent assigned to this task"
            )
        
        print(f"  🔧 [{agent.agent_type}] {task.name}...")
        
        try:
            result = await self.agent_executor.execute_task(
                agent=agent,
                task=task,
                output_dir=output_dir
            )
            
            self.task_results[task.id] = result
            
            status_icon = "✅" if result.status == TaskStatus.COMPLETED else "❌"
            print(f"  {status_icon} {task.name} - {result.duration_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"  ❌ {task.name} failed: {str(e)}")
            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                agent_id=agent.id,
                agent_type=agent.agent_type,
                error=str(e)
            )
    
    async def _wait_for_dependencies(self, task: Task) -> bool:
        """Sprawdź czy wszystkie zależności zadania zostały ukończone."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def _save_project_plan(
        self,
        output_dir: Path,
        requirements: str,
        tasks: List[Task],
        phases: Dict[int, List[Task]],
        team: List[AgentInstance]
    ):
        """Zapisz plan projektu do pliku JSON."""
        plan = {
            "requirements": requirements,
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "required_capabilities": t.required_capabilities,
                    "dependencies": t.dependencies,
                    "estimated_duration": t.estimated_duration,
                    "assigned_agent": t.assigned_agent_id
                }
                for t in tasks
            ],
            "phases": {
                str(phase_num): [t.id for t in phase_tasks]
                for phase_num, phase_tasks in phases.items()
            },
            "team": [
                {
                    "id": a.id,
                    "type": a.agent_type,
                    "capabilities": a.capabilities,
                    "model": a.model
                }
                for a in team
            ]
        }
        
        plan_file = output_dir / "project_plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Project plan saved: {plan_file}")
    
    def _generate_summary(self, phase_results: List[PhaseResult]) -> Dict:
        """Generuj podsumowanie projektu."""
        total_tasks = sum(len(pr.task_results) for pr in phase_results)
        completed_tasks = sum(
            len([tr for tr in pr.task_results if tr.status == TaskStatus.COMPLETED])
            for pr in phase_results
        )
        failed_tasks = total_tasks - completed_tasks
        
        total_artifacts = sum(
            len(tr.artifacts)
            for pr in phase_results
            for tr in pr.task_results
        )
        
        return {
            "total_phases": len(phase_results),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": f"{(completed_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%",
            "total_artifacts": total_artifacts
        }
    
    def _save_final_report(self, output_dir: Path, result: ProjectResult):
        """Zapisz końcowy raport projektu."""
        report_file = output_dir / "final_report.json"
        
        report_data = {
            "project_name": result.project_name,
            "status": result.status,
            "requirements": result.requirements,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "total_duration_seconds": result.total_duration_seconds,
            "summary": result.summary,
            "phases": [
                {
                    "phase_number": pr.phase_number,
                    "phase_name": pr.phase_name,
                    "status": pr.status.value,
                    "duration_seconds": pr.duration_seconds,
                    "tasks": [
                        {
                            "task_id": tr.task_id,
                            "task_name": tr.task_name,
                            "status": tr.status.value,
                            "agent_type": tr.agent_type,
                            "duration_seconds": tr.duration_seconds,
                            "artifacts": tr.artifacts,
                            "error": tr.error
                        }
                        for tr in pr.task_results
                    ]
                }
                for pr in result.phase_results
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Final report saved: {report_file}")
