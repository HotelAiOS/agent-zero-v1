"""
Team Builder - Dobiera odpowiednich agentów do zadań
Integracja z AgentFactory i CapabilityMatcher
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path

# Import z agent_factory
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_factory.factory import AgentFactory, AgentTemplate
from agent_factory.capabilities import CapabilityMatcher, AgentCapability, TechStack, SkillLevel
from agent_factory.lifecycle import AgentLifecycleManager, AgentInstance, AgentState

# Import Task z task_decomposer
from .task_decomposer import Task, TaskPriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TeamMember:
    """Członek zespołu przypisany do zadania"""
    agent_instance: AgentInstance
    assigned_tasks: List[str] = field(default_factory=list)
    workload: float = 0.0
    
    def add_task(self, task: Task):
        """Przypisz zadanie do członka zespołu"""
        self.assigned_tasks.append(task.task_id)
        self.workload += task.estimated_duration_hours
    
    def can_handle_task(self, task: Task) -> bool:
        """Sprawdź czy agent może obsłużyć zadanie"""
        return self.agent_instance.agent_type == task.agent_type


@dataclass
class TeamComposition:
    """Skład zespołu dla projektu"""
    project_id: str
    members: Dict[str, TeamMember] = field(default_factory=dict)
    task_assignments: Dict[str, str] = field(default_factory=dict)
    
    def add_member(self, member: TeamMember):
        """Dodaj członka do zespołu"""
        self.members[member.agent_instance.agent_id] = member
        logger.info(f"✅ Added {member.agent_instance.agent_type} to team: {member.agent_instance.agent_id}")
    
    def assign_task(self, task: Task, agent_id: str):
        """Przypisz zadanie do agenta"""
        if agent_id in self.members:
            self.members[agent_id].add_task(task)
            self.task_assignments[task.task_id] = agent_id
            task.assigned_agent_id = agent_id
            logger.info(f"📋 Assigned {task.task_id} to {agent_id}")
        else:
            logger.error(f"Agent {agent_id} not in team!")
    
    def get_agent_for_task(self, task_id: str) -> Optional[str]:
        """Zwróć agent_id dla zadania"""
        return self.task_assignments.get(task_id)
    
    def get_team_summary(self) -> Dict:
        """Podsumowanie zespołu"""
        return {
            'project_id': self.project_id,
            'team_size': len(self.members),
            'total_tasks': len(self.task_assignments),
            'agents': [
                {
                    'agent_id': member.agent_instance.agent_id,
                    'agent_type': member.agent_instance.agent_type,
                    'assigned_tasks': len(member.assigned_tasks),
                    'workload_hours': member.workload
                }
                for member in self.members.values()
            ]
        }


class TeamBuilder:
    """
    Budowanie zespołu agentów dla projektu
    Dopasowuje agentów do zadań na podstawie capabilities
    """
    
    def __init__(
        self,
        factory: Optional[AgentFactory] = None,
        capability_matcher: Optional[CapabilityMatcher] = None,
        lifecycle_manager: Optional[AgentLifecycleManager] = None
    ):
        """
        Args:
            factory: AgentFactory do tworzenia nowych agentów
            capability_matcher: CapabilityMatcher do dopasowywania
            lifecycle_manager: AgentLifecycleManager do zarządzania agentami
        """
        self.factory = factory or AgentFactory()
        self.capability_matcher = capability_matcher or CapabilityMatcher()
        self.lifecycle_manager = lifecycle_manager or AgentLifecycleManager()
        
        # Cache dostępnych typów agentów
        self.available_agent_types = set()
        for template_name in self.factory.templates.keys():
            self.available_agent_types.add(template_name)
        
        logger.info(f"✅ TeamBuilder initialized with {len(self.available_agent_types)} agent types")
    
    def build_team(self, tasks: List[Task], project_id: str = "default") -> TeamComposition:
        """
        Zbuduj zespół dla listy zadań
        
        Args:
            tasks: Lista zadań do wykonania
            project_id: ID projektu
        
        Returns:
            TeamComposition z przypisanymi agentami
        """
        logger.info(f"🏗️ Building team for project: {project_id}")
        logger.info(f"📋 Tasks to assign: {len(tasks)}")
        
        team = TeamComposition(project_id=project_id)
        
        # 1. Określ potrzebne typy agentów
        required_agent_types = self._identify_required_agents(tasks)
        logger.info(f"🤖 Required agent types: {required_agent_types}")
        
        # 2. Utwórz agentów dla każdego typu
        for agent_type in required_agent_types:
            agent = self._create_or_get_agent(agent_type)
            if agent:
                member = TeamMember(agent_instance=agent)
                team.add_member(member)
        
        # 3. Przypisz zadania do agentów
        self._assign_tasks_to_team(tasks, team)
        
        # 4. Podsumowanie
        summary = team.get_team_summary()
        logger.info(f"✅ Team built: {summary['team_size']} agents, {summary['total_tasks']} tasks assigned")
        
        return team
    
    def _identify_required_agents(self, tasks: List[Task]) -> Set[str]:
        """Zidentyfikuj wymagane typy agentów na podstawie zadań"""
        required = set()
        for task in tasks:
            if task.agent_type in self.available_agent_types:
                required.add(task.agent_type)
            else:
                logger.warning(f"Unknown agent type: {task.agent_type}")
        
        return required
    
    def _create_or_get_agent(self, agent_type: str) -> Optional[AgentInstance]:
        """Utwórz nowego agenta danego typu lub pobierz istniejącego"""
        
        # Sprawdź czy mamy już agenta tego typu w stanie READY lub IDLE
        existing_agents = self.lifecycle_manager.get_agents_by_state(AgentState.READY)
        existing_agents += self.lifecycle_manager.get_agents_by_state(AgentState.IDLE)
        
        for agent in existing_agents:
            if agent.agent_type == agent_type:
                logger.info(f"♻️ Reusing existing {agent_type}: {agent.agent_id}")
                return agent
        
        # Jeśli nie ma - utwórz nowego
        try:
            agent = self.factory.create_agent(agent_type)
            logger.info(f"✨ Created new {agent_type}: {agent.agent_id}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}")
            return None
    
    def _assign_tasks_to_team(self, tasks: List[Task], team: TeamComposition):
        """Przypisz zadania do członków zespołu"""
        
        # Sortuj zadania według priorytetu (najwyższy najpierw)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        for task in sorted_tasks:
            # Znajdź agenta odpowiedniego typu
            assigned = False
            
            for member in team.members.values():
                if member.can_handle_task(task):
                    team.assign_task(task, member.agent_instance.agent_id)
                    assigned = True
                    break
            
            if not assigned:
                logger.warning(f"⚠️ No agent available for task {task.task_id} (type: {task.agent_type})")
    
    def match_agent_to_task(
        self, 
        task: Task, 
        available_agents: List[AgentInstance]
    ) -> Optional[AgentInstance]:
        """
        Dopasuj najlepszego agenta do zadania
        
        Args:
            task: Zadanie do wykonania
            available_agents: Lista dostępnych agentów
        
        Returns:
            Najlepiej dopasowany agent lub None
        """
        candidates = []
        
        for agent in available_agents:
            if agent.agent_type != task.agent_type:
                continue
            
            if agent.state not in [AgentState.READY, AgentState.IDLE]:
                continue
            
            score = self._calculate_match_score(task, agent)
            candidates.append((agent, score))
        
        if not candidates:
            return None
        
        # Sortuj według score (najwyższy najpierw)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_score = candidates[0]
        
        logger.info(f"🎯 Best match for {task.task_id}: {best_agent.agent_id} (score: {best_score:.2f})")
        return best_agent
    
    def _calculate_match_score(self, task: Task, agent: AgentInstance) -> float:
        """
        Oblicz score dopasowania agenta do zadania
        
        Factors:
        - Agent type match (base score)
        - Tech stack overlap
        - Agent workload (prefer less busy agents)
        - Agent success rate
        """
        score = 0.0
        
        # Base score dla matching type
        if agent.agent_type == task.agent_type:
            score += 10.0
        
        # Tech stack overlap (jeśli agent ma template z capabilities)
        if hasattr(agent, 'template') and agent.template:
            template_techs = set()
            for cap in agent.template.capabilities:
                if 'technologies' in cap:
                    template_techs.update(cap.get('technologies', []))
            
            task_techs = set(task.tech_stack)
            overlap = len(template_techs.intersection(task_techs))
            score += overlap * 2.0
        
        # Workload penalty (preferuj mniej zajętych agentów)
        tasks_completed = agent.metrics.tasks_completed
        if tasks_completed > 0:
            score += min(tasks_completed * 0.5, 5.0)
        
        # Success rate bonus
        total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
        if total_tasks > 0:
            success_rate = agent.metrics.tasks_completed / total_tasks
            score += success_rate * 3.0
        
        return score
    
    def validate_team_capabilities(
        self, 
        team: TeamComposition, 
        required_capabilities: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Sprawdź czy zespół ma wszystkie wymagane capabilities
        
        Returns:
            (is_valid, missing_capabilities)
        """
        team_capabilities = set()
        
        for member in team.members.values():
            agent = member.agent_instance
            if hasattr(agent, 'template') and agent.template:
                for cap in agent.template.capabilities:
                    if 'name' in cap:
                        team_capabilities.add(cap['name'])
        
        missing = [cap for cap in required_capabilities if cap not in team_capabilities]
        
        is_valid = len(missing) == 0
        
        if is_valid:
            logger.info("✅ Team has all required capabilities")
        else:
            logger.warning(f"⚠️ Missing capabilities: {missing}")
        
        return is_valid, missing
