"""
Task Scheduler
Harmonogramowanie i scheduling zadań dla zespołów agentów
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .task_decomposer import Task, TaskStatus, TaskPriority
from .dependency_graph import DependencyGraph
from .team_formation import Team, TeamMember

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScheduleStrategy(Enum):
    """Strategie schedulowania"""
    CRITICAL_PATH = "critical_path"  # Najpierw critical path
    PRIORITY_FIRST = "priority_first"  # Najpierw high priority
    FIFO = "fifo"  # First In First Out
    LOAD_BALANCED = "load_balanced"  # Równomierne obciążenie
    PARALLEL_MAX = "parallel_max"  # Maksymalna równoległość


@dataclass
class ScheduledTask:
    """Zadanie z harmonogramem"""
    task: Task
    scheduled_start: datetime
    scheduled_end: datetime
    assigned_agent: str
    dependencies_met: bool = False
    
    def is_overdue(self) -> bool:
        """Czy zadanie jest opóźnione"""
        if self.task.status != TaskStatus.COMPLETED:
            return datetime.now() > self.scheduled_end
        return False


class TaskScheduler:
    """
    Scheduler zadań
    Tworzy harmonogramy wykonania zadań przez zespoły agentów
    """
    
    def __init__(self, strategy: ScheduleStrategy = ScheduleStrategy.LOAD_BALANCED):
        self.strategy = strategy
        self.schedule: Dict[str, ScheduledTask] = {}
        logger.info(f"TaskScheduler zainicjalizowany (strategia: {strategy.value})")
    
    def create_schedule(
        self,
        tasks: List[Task],
        team: Team,
        dependency_graph: DependencyGraph,
        start_date: Optional[datetime] = None
    ) -> List[ScheduledTask]:
        """
        Utwórz harmonogram wykonania zadań
        
        Args:
            tasks: Lista zadań do zaplanowania
            team: Zespół agentów
            dependency_graph: Graf zależności
            start_date: Data rozpoczęcia (domyślnie teraz)
        
        Returns:
            Lista ScheduledTask
        """
        if start_date is None:
            start_date = datetime.now()
        
        # Wybierz strategię
        if self.strategy == ScheduleStrategy.CRITICAL_PATH:
            return self._schedule_critical_path(tasks, team, dependency_graph, start_date)
        elif self.strategy == ScheduleStrategy.PRIORITY_FIRST:
            return self._schedule_priority_first(tasks, team, dependency_graph, start_date)
        elif self.strategy == ScheduleStrategy.LOAD_BALANCED:
            return self._schedule_load_balanced(tasks, team, dependency_graph, start_date)
        elif self.strategy == ScheduleStrategy.PARALLEL_MAX:
            return self._schedule_parallel_max(tasks, team, dependency_graph, start_date)
        else:
            return self._schedule_fifo(tasks, team, dependency_graph, start_date)
    
    def _schedule_load_balanced(
        self,
        tasks: List[Task],
        team: Team,
        dependency_graph: DependencyGraph,
        start_date: datetime
    ) -> List[ScheduledTask]:
        """Strategia load balanced - równomierne obciążenie"""
        scheduled = []
        completed_tasks = []
        agent_availability: Dict[str, datetime] = {
            m.agent_id: start_date for m in team.members
        }
        
        # Sortowanie topologiczne dla zależności
        task_order = dependency_graph.topological_sort()
        if not task_order:
            logger.error("Nie można utworzyć harmonogramu - cykl w zależnościach")
            return []
        
        task_map = {t.task_id: t for t in tasks}
        
        for task_id in task_order:
            if task_id not in task_map:
                continue
            
            task = task_map[task_id]
            
            # Znajdź agenta odpowiedniego typu z najwcześniejszą dostępnością
            suitable_agents = [
                m for m in team.members 
                if m.agent_type == task.required_agent_type
            ]
            
            if not suitable_agents:
                logger.warning(f"Brak agenta dla zadania {task.title}")
                continue
            
            # Agent z najwcześniejszą dostępnością
            chosen_agent = min(
                suitable_agents,
                key=lambda a: agent_availability[a.agent_id]
            )
            
            # Oblicz start date - max(agent_availability, dependencies_end)
            agent_available = agent_availability[chosen_agent.agent_id]
            
            # Sprawdź kiedy dependencies się kończą
            dep_end = start_date
            for dep in task.depends_on:
                if dep in self.schedule:
                    dep_end = max(dep_end, self.schedule[dep].scheduled_end)
            
            task_start = max(agent_available, dep_end)
            task_end = task_start + timedelta(hours=task.estimated_hours)
            
            # Utwórz scheduled task
            scheduled_task = ScheduledTask(
                task=task,
                scheduled_start=task_start,
                scheduled_end=task_end,
                assigned_agent=chosen_agent.agent_id,
                dependencies_met=len(task.depends_on) == 0 or 
                                all(d in completed_tasks for d in task.depends_on)
            )
            
            scheduled.append(scheduled_task)
            self.schedule[task.task_id] = scheduled_task
            
            # Aktualizuj dostępność agenta
            agent_availability[chosen_agent.agent_id] = task_end
            
            # Symuluj completion
            completed_tasks.append(task.task_id)
        
        logger.info(
            f"Utworzono harmonogram: {len(scheduled)} zadań, "
            f"koniec: {max(s.scheduled_end for s in scheduled) if scheduled else start_date}"
        )
        
        return scheduled
    
    def _schedule_critical_path(
        self,
        tasks: List[Task],
        team: Team,
        dependency_graph: DependencyGraph,
        start_date: datetime
    ) -> List[ScheduledTask]:
        """Strategia critical path - najpierw zadania z critical path"""
        # Znajdź critical path
        critical_path, _ = dependency_graph.get_critical_path(tasks)
        
        # Podziel na critical i non-critical
        task_map = {t.task_id: t for t in tasks}
        critical_tasks = [task_map[tid] for tid in critical_path if tid in task_map]
        non_critical = [t for t in tasks if t.task_id not in critical_path]
        
        # Zaplanuj critical path first
        ordered_tasks = critical_tasks + non_critical
        
        # Użyj load balanced dla uporządkowanej listy
        return self._schedule_load_balanced(ordered_tasks, team, dependency_graph, start_date)
    
    def _schedule_priority_first(
        self,
        tasks: List[Task],
        team: Team,
        dependency_graph: DependencyGraph,
        start_date: datetime
    ) -> List[ScheduledTask]:
        """Strategia priority first - najpierw high priority"""
        # Sortuj po priorytecie
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        return self._schedule_load_balanced(sorted_tasks, team, dependency_graph, start_date)
    
    def _schedule_parallel_max(
        self,
        tasks: List[Task],
        team: Team,
        dependency_graph: DependencyGraph,
        start_date: datetime
    ) -> List[ScheduledTask]:
        """Strategia parallel max - maksymalna równoległość"""
        scheduled = []
        completed = []
        current_time = start_date
        
        task_map = {t.task_id: t for t in tasks}
        
        while len(completed) < len(tasks):
            # Znajdź zadania ready do wykonania (dependencies met)
            ready_tasks = dependency_graph.get_independent_tasks(completed)
            ready_task_objects = [
                task_map[tid] for tid in ready_tasks 
                if tid in task_map
            ]
            
            if not ready_task_objects:
                break
            
            # Przypisz wszystkie ready tasks do dostępnych agentów
            for task in ready_task_objects:
                suitable_agents = [
                    m for m in team.members 
                    if m.agent_type == task.required_agent_type
                ]
                
                if suitable_agents:
                    chosen_agent = suitable_agents[0]
                    
                    task_end = current_time + timedelta(hours=task.estimated_hours)
                    
                    scheduled_task = ScheduledTask(
                        task=task,
                        scheduled_start=current_time,
                        scheduled_end=task_end,
                        assigned_agent=chosen_agent.agent_id,
                        dependencies_met=True
                    )
                    
                    scheduled.append(scheduled_task)
                    self.schedule[task.task_id] = scheduled_task
                    completed.append(task.task_id)
            
            # Przesuń czas do najkrótszego zadania
            if ready_task_objects:
                min_duration = min(t.estimated_hours for t in ready_task_objects)
                current_time += timedelta(hours=min_duration)
        
        logger.info(f"Harmonogram parallel max: {len(scheduled)} zadań")
        return scheduled
    
    def _schedule_fifo(
        self,
        tasks: List[Task],
        team: Team,
        dependency_graph: DependencyGraph,
        start_date: datetime
    ) -> List[ScheduledTask]:
        """Strategia FIFO - w kolejności dodania"""
        return self._schedule_load_balanced(tasks, team, dependency_graph, start_date)
    
    def get_schedule_summary(self) -> Dict[str, any]:
        """Podsumowanie harmonogramu"""
        if not self.schedule:
            return {'total_tasks': 0}
        
        scheduled_tasks = list(self.schedule.values())
        
        earliest_start = min(s.scheduled_start for s in scheduled_tasks)
        latest_end = max(s.scheduled_end for s in scheduled_tasks)
        total_duration = (latest_end - earliest_start).total_seconds() / 3600  # hours
        
        # Grupuj po agentach
        by_agent = {}
        for st in scheduled_tasks:
            agent = st.assigned_agent
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(st)
        
        return {
            'total_tasks': len(scheduled_tasks),
            'start_date': earliest_start.isoformat(),
            'end_date': latest_end.isoformat(),
            'total_duration_hours': total_duration,
            'total_duration_days': total_duration / 8,  # 8h work days
            'agents_count': len(by_agent),
            'tasks_per_agent': {
                agent: len(tasks) for agent, tasks in by_agent.items()
            },
            'strategy': self.strategy.value
        }
    
    def get_overdue_tasks(self) -> List[ScheduledTask]:
        """Zwróć zadania opóźnione"""
        overdue = [
            st for st in self.schedule.values()
            if st.is_overdue()
        ]
        
        if overdue:
            logger.warning(f"⚠️  {len(overdue)} zadań opóźnionych")
        
        return overdue
    
    def get_upcoming_tasks(self, hours: int = 24) -> List[ScheduledTask]:
        """Zwróć zadania do wykonania w najbliższych X godzinach"""
        now = datetime.now()
        threshold = now + timedelta(hours=hours)
        
        upcoming = [
            st for st in self.schedule.values()
            if st.scheduled_start <= threshold and st.task.status == TaskStatus.PENDING
        ]
        
        return sorted(upcoming, key=lambda st: st.scheduled_start)


def create_scheduler(strategy: ScheduleStrategy = ScheduleStrategy.LOAD_BALANCED) -> TaskScheduler:
    """Utwórz TaskScheduler"""
    return TaskScheduler(strategy)

