"""
Intelligent Planner
Główny system planowania projektów - integruje wszystkie komponenty orchestration
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

from .task_decomposer import TaskDecomposer, Task, TaskType
from .dependency_graph import DependencyGraph
from .team_formation import TeamFormationEngine, Team
from .quality_gates import QualityGateManager, QualityGate
from .scheduler import TaskScheduler, ScheduleStrategy, ScheduledTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProjectPlan:
    """Kompletny plan projektu"""
    project_id: str
    project_name: str
    project_type: str
    
    # Components
    tasks: List[Task]
    team: Team
    schedule: List[ScheduledTask]
    quality_gates: Dict[str, QualityGate]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration_hours: float = 0.0
    estimated_duration_days: float = 0.0
    total_cost_estimate: float = 0.0
    
    # Status
    status: str = "planned"
    progress: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Zwróć podsumowanie planu"""
        return {
            'project_id': self.project_id,
            'project_name': self.project_name,
            'project_type': self.project_type,
            'status': self.status,
            'progress': self.progress,
            'total_tasks': len(self.tasks),
            'team_size': len(self.team.members),
            'estimated_duration_hours': self.estimated_duration_hours,
            'estimated_duration_days': self.estimated_duration_days,
            'quality_gates': len(self.quality_gates),
            'created_at': self.created_at.isoformat()
        }


class IntelligentPlanner:
    """
    Intelligent Project Planner
    Orkiestruje wszystkie komponenty do tworzenia kompletnych planów projektów
    """
    
    def __init__(self):
        self.task_decomposer = TaskDecomposer()
        self.team_formation = TeamFormationEngine()
        self.quality_gate_manager = QualityGateManager()
        self.quality_gate_manager.define_standard_gates()
        
        self.plans: Dict[str, ProjectPlan] = {}
        
        logger.info("✅ IntelligentPlanner zainicjalizowany")
    
    def create_project_plan(
        self,
        project_name: str,
        project_type: str,
        business_requirements: List[str],
        schedule_strategy: ScheduleStrategy = ScheduleStrategy.LOAD_BALANCED,
        start_date: Optional[datetime] = None
    ) -> ProjectPlan:
        """
        Utwórz kompletny plan projektu
        
        Args:
            project_name: Nazwa projektu
            project_type: Typ projektu ('fullstack_web_app', 'api_backend', etc.)
            business_requirements: Lista wymagań biznesowych
            schedule_strategy: Strategia schedulowania
            start_date: Data rozpoczęcia
        
        Returns:
            ProjectPlan
        """
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎯 Tworzenie planu projektu: {project_name}")
        logger.info(f"   Typ: {project_type}")
        logger.info(f"   Wymagania: {len(business_requirements)}")
        
        # 1. Dekompozycja na zadania
        logger.info("📋 Dekompozycja na zadania techniczne...")
        tasks = self.task_decomposer.decompose_project(
            project_type,
            business_requirements
        )
        
        if not tasks:
            logger.error("Nie udało się zdekomponować projektu")
            return None
        
        logger.info(f"   ✅ Utworzono {len(tasks)} zadań")
        
        # 2. Budowa grafu zależności
        logger.info("🔗 Budowa grafu zależności...")
        dependency_graph = DependencyGraph()
        dependency_graph.build_from_tasks(tasks)
        
        # Sprawdź cykle
        has_cycle, cycle = dependency_graph.has_cycle()
        if has_cycle:
            logger.error(f"❌ Wykryto cykl w zależnościach: {cycle}")
            return None
        
        logger.info("   ✅ Graf zależności zbudowany (brak cykli)")
        
        # 3. Formowanie zespołu
        logger.info("👥 Formowanie zespołu...")
        team = self.team_formation.form_team(project_id, tasks)
        logger.info(f"   ✅ Zespół: {len(team.members)} członków")
        
        # 4. Przypisanie zadań do zespołu
        logger.info("📌 Przypisywanie zadań do agentów...")
        self.team_formation.assign_tasks_to_team(team, tasks)
        logger.info("   ✅ Zadania przypisane")
        
        # 5. Utworzenie harmonogramu
        logger.info(f"📅 Tworzenie harmonogramu ({schedule_strategy.value})...")
        scheduler = TaskScheduler(schedule_strategy)
        schedule = scheduler.create_schedule(tasks, team, dependency_graph, start_date)
        logger.info(f"   ✅ Harmonogram: {len(schedule)} zadań")
        
        # 6. Szacowanie czasu i kosztu
        schedule_summary = scheduler.get_schedule_summary()
        estimated_hours = schedule_summary.get('total_duration_hours', 0)
        estimated_days = schedule_summary.get('total_duration_days', 0)
        
        # Prosty cost estimate: 100 PLN/h * liczba agentów * czas
        estimated_cost = estimated_hours * 100 * len(team.members)
        
        logger.info(f"   ⏱️  Szacowany czas: {estimated_hours:.1f}h ({estimated_days:.1f} dni)")
        logger.info(f"   💰 Szacowany koszt: {estimated_cost:.2f} PLN")
        
        # 7. Setup quality gates
        logger.info("🚦 Konfiguracja quality gates...")
        quality_gates = {
            gid: gate for gid, gate in self.quality_gate_manager.gates.items()
        }
        logger.info(f"   ✅ {len(quality_gates)} quality gates")
        
        # 8. Utworzenie ProjectPlan
        plan = ProjectPlan(
            project_id=project_id,
            project_name=project_name,
            project_type=project_type,
            tasks=tasks,
            team=team,
            schedule=schedule,
            quality_gates=quality_gates,
            estimated_duration_hours=estimated_hours,
            estimated_duration_days=estimated_days,
            total_cost_estimate=estimated_cost
        )
        
        self.plans[project_id] = plan
        
        logger.info("="*70)
        logger.info(f"✅ Plan projektu '{project_name}' utworzony pomyślnie!")
        logger.info(f"   Project ID: {project_id}")
        logger.info(f"   Zadania: {len(tasks)}")
        logger.info(f"   Zespół: {len(team.members)} agentów")
        logger.info(f"   Czas: {estimated_days:.1f} dni")
        logger.info(f"   Koszt: {estimated_cost:,.2f} PLN")
        logger.info("="*70)
        
        return plan
    
    def get_plan(self, project_id: str) -> Optional[ProjectPlan]:
        """Pobierz plan projektu"""
        return self.plans.get(project_id)
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """Lista wszystkich planów"""
        return [plan.get_summary() for plan in self.plans.values()]


def create_planner() -> IntelligentPlanner:
    """Utwórz IntelligentPlanner"""
    return IntelligentPlanner()
