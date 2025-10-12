"""
Team Formation Engine
Automatyczne tworzenie zespołów agentów do projektów
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .task_decomposer import Task, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeamMember:
    """Członek zespołu"""
    agent_id: str
    agent_type: str
    role: str
    capabilities: List[str] = field(default_factory=list)
    current_workload: float = 0.0  # Godziny przypisanych zadań
    max_workload: float = 40.0  # Max godziny na tydzień


@dataclass
class Team:
    """Zespół agentów do projektu"""
    team_id: str
    project_id: str
    members: List[TeamMember]
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_member_by_type(self, agent_type: str) -> Optional[TeamMember]:
        """Znajdź członka zespołu po typie agenta"""
        for member in self.members:
            if member.agent_type == agent_type:
                return member
        return None
    
    def get_available_members(self, max_workload_threshold: float = 0.8) -> List[TeamMember]:
        """Znajdź członków z dostępną przepustowością"""
        available = []
        for member in self.members:
            utilization = member.current_workload / member.max_workload
            if utilization < max_workload_threshold:
                available.append(member)
        return available
    
    def get_team_utilization(self) -> float:
        """Oblicz średnie wykorzystanie zespołu (0.0 - 1.0)"""
        if not self.members:
            return 0.0
        
        total_utilization = sum(
            m.current_workload / m.max_workload 
            for m in self.members
        )
        return total_utilization / len(self.members)


class TeamFormationEngine:
    """
    Silnik formowania zespołów
    Automatycznie dobiera agentów do projektów na podstawie wymagań
    """
    
    def __init__(self):
        self.teams: Dict[str, Team] = {}
        self.agent_pool: Dict[str, List[str]] = {}  # {agent_type: [agent_ids]}
        logger.info("TeamFormationEngine zainicjalizowany")
    
    def register_agent_pool(self, agent_type: str, agent_ids: List[str]):
        """Zarejestruj pulę dostępnych agentów"""
        self.agent_pool[agent_type] = agent_ids
        logger.info(f"Zarejestrowano {len(agent_ids)} agentów typu {agent_type}")
    
    def analyze_project_requirements(
        self,
        tasks: List[Task]
    ) -> Dict[str, Dict[str, any]]:
        """
        Analizuj wymagania projektu - jakie role są potrzebne
        
        Returns:
            Dict {agent_type: {count: int, total_hours: float, tasks: List[Task]}}
        """
        requirements = {}
        
        for task in tasks:
            agent_type = task.required_agent_type
            
            if agent_type not in requirements:
                requirements[agent_type] = {
                    'count': 0,
                    'total_hours': 0.0,
                    'tasks': []
                }
            
            requirements[agent_type]['total_hours'] += task.estimated_hours
            requirements[agent_type]['tasks'].append(task)
        
        # Oblicz ile agentów każdego typu jest potrzebne
        # Zakładamy 40h/tydzień na agenta
        for agent_type, data in requirements.items():
            weeks_of_work = data['total_hours'] / 40.0
            # Zaokrąglij w górę, minimum 1
            data['count'] = max(1, int(weeks_of_work + 0.5))
        
        logger.info(f"Wymagania projektu: {len(requirements)} typów agentów")
        for agent_type, data in requirements.items():
            logger.info(
                f"  {agent_type}: {data['count']} agent(ów), "
                f"{data['total_hours']:.1f}h pracy"
            )
        
        return requirements
    
    def form_team(
        self,
        project_id: str,
        tasks: List[Task],
        team_id: Optional[str] = None,
        force_minimal: bool = False
    ) -> Team:
        """
        Utwórz zespół dla projektu
        
        Args:
            project_id: ID projektu
            tasks: Lista zadań projektu
            team_id: Opcjonalne ID zespołu
            force_minimal: Czy użyć minimalnego zespołu (1 agent na typ)
        
        Returns:
            Utworzony Team
        """
        if team_id is None:
            team_id = f"team_{project_id}"
        
        requirements = self.analyze_project_requirements(tasks)
        members = []
        
        for agent_type, data in requirements.items():
            needed_count = 1 if force_minimal else data['count']
            
            # Sprawdź czy mamy dostępnych agentów tego typu
            available_agents = self.agent_pool.get(agent_type, [])
            
            if not available_agents:
                logger.warning(
                    f"Brak dostępnych agentów typu {agent_type}! "
                    f"Potrzeba: {needed_count}"
                )
                continue
            
            # Przypisz agentów
            for i in range(min(needed_count, len(available_agents))):
                agent_id = available_agents[i]
                
                member = TeamMember(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    role=self._get_role_name(agent_type),
                    capabilities=[agent_type],
                    current_workload=0.0,
                    max_workload=40.0
                )
                members.append(member)
        
        team = Team(
            team_id=team_id,
            project_id=project_id,
            members=members
        )
        
        self.teams[team_id] = team
        
        logger.info(
            f"✅ Utworzono zespół {team_id}: {len(members)} członków"
        )
        for member in members:
            logger.info(f"   - {member.agent_id} ({member.role})")
        
        return team
    
    def _get_role_name(self, agent_type: str) -> str:
        """Zmapuj typ agenta na nazwę roli"""
        role_mapping = {
            'architect': 'Architekt Systemu',
            'backend': 'Backend Developer',
            'frontend': 'Frontend Developer',
            'database': 'Database Expert',
            'tester': 'QA Tester',
            'devops': 'DevOps Engineer',
            'security': 'Security Auditor',
            'performance': 'Performance Engineer'
        }
        return role_mapping.get(agent_type, agent_type.title())
    
    def assign_tasks_to_team(
        self,
        team: Team,
        tasks: List[Task]
    ) -> Dict[str, List[Task]]:
        """
        Przypisz zadania do członków zespołu
        
        Returns:
            Dict {agent_id: [assigned_tasks]}
        """
        assignments = {member.agent_id: [] for member in team.members}
        
        # Grupuj zadania po typie agenta
        tasks_by_type = {}
        for task in tasks:
            agent_type = task.required_agent_type
            if agent_type not in tasks_by_type:
                tasks_by_type[agent_type] = []
            tasks_by_type[agent_type].append(task)
        
        # Przypisz zadania do agentów danego typu
        for agent_type, type_tasks in tasks_by_type.items():
            # Znajdź wszystkich agentów tego typu w zespole
            type_members = [m for m in team.members if m.agent_type == agent_type]
            
            if not type_members:
                logger.warning(
                    f"Brak agenta typu {agent_type} w zespole! "
                    f"Zadania nieprzypisan: {len(type_tasks)}"
                )
                continue
            
            # Load balancing - równomiernie rozdziel zadania
            sorted_tasks = sorted(type_tasks, key=lambda t: t.estimated_hours, reverse=True)
            
            # Przypisuj do agenta z najmniejszym obciążeniem
            for task in sorted_tasks:
                least_loaded = min(type_members, key=lambda m: m.current_workload)
                
                assignments[least_loaded.agent_id].append(task)
                least_loaded.current_workload += task.estimated_hours
                task.assigned_agent = least_loaded.agent_id
        
        logger.info(f"Przypisano zadania w zespole {team.team_id}")
        for agent_id, assigned in assignments.items():
            if assigned:
                total_hours = sum(t.estimated_hours for t in assigned)
                logger.info(
                    f"   {agent_id}: {len(assigned)} zadań, {total_hours:.1f}h"
                )
        
        return assignments
    
    def optimize_team_composition(
        self,
        team: Team,
        tasks: List[Task]
    ) -> Team:
        """
        Optymalizuj skład zespołu - dodaj/usuń agentów based on workload
        """
        requirements = self.analyze_project_requirements(tasks)
        
        # Sprawdź czy są overloaded members
        for member in team.members:
            if member.current_workload > member.max_workload * 1.5:
                logger.warning(
                    f"Agent {member.agent_id} przeciążony: "
                    f"{member.current_workload:.1f}h > {member.max_workload * 1.5:.1f}h"
                )
                
                # Spróbuj dodać kolejnego agenta tego typu
                available = self.agent_pool.get(member.agent_type, [])
                existing_ids = {m.agent_id for m in team.members}
                new_agents = [a for a in available if a not in existing_ids]
                
                if new_agents:
                    new_member = TeamMember(
                        agent_id=new_agents[0],
                        agent_type=member.agent_type,
                        role=member.role,
                        capabilities=member.capabilities.copy()
                    )
                    team.members.append(new_member)
                    logger.info(f"➕ Dodano {new_member.agent_id} do zespołu")
        
        return team
    
    def get_team_status(self, team_id: str) -> Optional[Dict[str, any]]:
        """Pobierz status zespołu"""
        if team_id not in self.teams:
            return None
        
        team = self.teams[team_id]
        
        return {
            'team_id': team.team_id,
            'project_id': team.project_id,
            'members_count': len(team.members),
            'members': [
                {
                    'agent_id': m.agent_id,
                    'role': m.role,
                    'workload': m.current_workload,
                    'utilization': m.current_workload / m.max_workload
                }
                for m in team.members
            ],
            'team_utilization': team.get_team_utilization(),
            'created_at': team.created_at.isoformat()
        }
    
    def disband_team(self, team_id: str):
        """Rozwiąż zespół po zakończeniu projektu"""
        if team_id in self.teams:
            team = self.teams[team_id]
            logger.info(
                f"Rozwiązano zespół {team_id} "
                f"({len(team.members)} członków wraca do puli)"
            )
            del self.teams[team_id]


def create_team_formation_engine() -> TeamFormationEngine:
    """Utwórz TeamFormationEngine"""
    return TeamFormationEngine()
