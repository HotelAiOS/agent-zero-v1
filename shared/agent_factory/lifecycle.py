"""
Agent Lifecycle Manager
Zarządzanie cyklem życia agentów
"""

from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Stany cyklu życia agenta"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentMetrics:
    """Metryki wydajności agenta"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    average_response_time: float = 0.0
    uptime_seconds: float = 0.0
    error_count: int = 0
    last_active: Optional[datetime] = None
    
    def update_response_time(self, new_time: float):
        """Aktualizuj średni czas odpowiedzi"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            self.average_response_time = new_time
        else:
            self.average_response_time = (
                (self.average_response_time * (total_tasks - 1) + new_time) / total_tasks
            )


@dataclass
class AgentInstance:
    """Instancja agenta w systemie"""
    agent_id: str
    agent_type: str
    state: AgentState
    created_at: datetime = field(default_factory=datetime.now)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    current_task: Optional[str] = None
    error_message: Optional[str] = None
    
    def mark_active(self):
        """Oznacz agenta jako aktywnego"""
        self.metrics.last_active = datetime.now()


class AgentLifecycleManager:
    """Manager cyklu życia agentów"""
    
    def __init__(self):
        self.agents: Dict[str, AgentInstance] = {}
        self.state_transitions: Dict[AgentState, List[AgentState]] = {
            AgentState.CREATED: [AgentState.INITIALIZING],
            AgentState.INITIALIZING: [AgentState.READY, AgentState.ERROR],
            AgentState.READY: [AgentState.BUSY, AgentState.IDLE, AgentState.PAUSED],
            AgentState.BUSY: [AgentState.IDLE, AgentState.READY, AgentState.ERROR],
            AgentState.IDLE: [AgentState.BUSY, AgentState.PAUSED, AgentState.TERMINATED],
            AgentState.PAUSED: [AgentState.READY, AgentState.TERMINATED],
            AgentState.ERROR: [AgentState.READY, AgentState.TERMINATED],
            AgentState.TERMINATED: []
        }
        logger.info("AgentLifecycleManager zainicjalizowany")
    
    def create_agent(
        self, 
        agent_id: str, 
        agent_type: str
    ) -> AgentInstance:
        """Utwórz nową instancję agenta"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} już istnieje")
            return self.agents[agent_id]
        
        agent = AgentInstance(
            agent_id=agent_id,
            agent_type=agent_type,
            state=AgentState.CREATED
        )
        self.agents[agent_id] = agent
        logger.info(f"Utworzono agenta: {agent_id} (typ: {agent_type})")
        return agent
    
    def transition_state(
        self,
        agent_id: str,
        new_state: AgentState,
        error_message: Optional[str] = None
    ) -> bool:
        """Zmień stan agenta"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} nie istnieje")
            return False
        
        agent = self.agents[agent_id]
        current_state = agent.state
        
        # Sprawdź czy transition jest dozwolony
        if new_state not in self.state_transitions[current_state]:
            logger.error(
                f"Niedozwolona zmiana stanu: {current_state.value} -> {new_state.value}"
            )
            return False
        
        # Wykonaj transition
        agent.state = new_state
        if new_state == AgentState.ERROR:
            agent.error_message = error_message
            agent.metrics.error_count += 1
        
        logger.info(
            f"Agent {agent_id}: {current_state.value} -> {new_state.value}"
        )
        return True
    
    def assign_task(self, agent_id: str, task_id: str) -> bool:
        """Przypisz zadanie agentowi"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} nie istnieje")
            return False
        
        agent = self.agents[agent_id]
        
        if agent.state not in [AgentState.READY, AgentState.IDLE]:
            logger.warning(
                f"Agent {agent_id} nie jest gotowy (stan: {agent.state.value})"
            )
            return False
        
        agent.current_task = task_id
        agent.mark_active()
        self.transition_state(agent_id, AgentState.BUSY)
        
        logger.info(f"Przypisano zadanie {task_id} agentowi {agent_id}")
        return True
    
    def complete_task(
        self,
        agent_id: str,
        success: bool = True,
        tokens_used: int = 0,
        response_time: float = 0.0
    ):
        """Oznacz zadanie jako zakończone"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        if success:
            agent.metrics.tasks_completed += 1
        else:
            agent.metrics.tasks_failed += 1
        
        agent.metrics.total_tokens_used += tokens_used
        agent.metrics.update_response_time(response_time)
        agent.current_task = None
        agent.mark_active()
        
        self.transition_state(agent_id, AgentState.IDLE)
        
        status = "sukces" if success else "błąd"
        logger.info(
            f"Agent {agent_id} zakończył zadanie ({status}, "
            f"tokens: {tokens_used}, czas: {response_time:.2f}s)"
        )
    
    def get_available_agents(self, agent_type: Optional[str] = None) -> List[str]:
        """Pobierz listę dostępnych agentów"""
        available = []
        
        for agent_id, agent in self.agents.items():
            if agent.state in [AgentState.READY, AgentState.IDLE]:
                if agent_type is None or agent.agent_type == agent_type:
                    available.append(agent_id)
        
        logger.info(
            f"Dostępnych agentów: {len(available)}" +
            (f" (typ: {agent_type})" if agent_type else "")
        )
        return available
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Pobierz metryki agenta"""
        if agent_id not in self.agents:
            return None
        return self.agents[agent_id].metrics
    
    def get_system_health(self) -> Dict:
        """Pobierz status całego systemu"""
        total = len(self.agents)
        if total == 0:
            return {"status": "no_agents", "details": {}}
        
        state_counts = {}
        for state in AgentState:
            state_counts[state.value] = sum(
                1 for a in self.agents.values() if a.state == state
            )
        
        total_tasks = sum(a.metrics.tasks_completed for a in self.agents.values())
        total_errors = sum(a.metrics.error_count for a in self.agents.values())
        
        health = {
            "status": "healthy" if state_counts.get("error", 0) == 0 else "degraded",
            "total_agents": total,
            "state_distribution": state_counts,
            "total_tasks_completed": total_tasks,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_tasks, 1)
        }
        
        logger.info(f"System health: {health['status']}")
        return health
    
    def terminate_agent(self, agent_id: str):
        """Zakończ działanie agenta"""
        if agent_id not in self.agents:
            return
        
        self.transition_state(agent_id, AgentState.TERMINATED)
        logger.info(f"Agent {agent_id} został zakończony")
    
    def pause_agent(self, agent_id: str):
        """Wstrzymaj agenta"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        if agent.state in [AgentState.READY, AgentState.IDLE]:
            self.transition_state(agent_id, AgentState.PAUSED)
            logger.info(f"Agent {agent_id} został wstrzymany")
    
    def resume_agent(self, agent_id: str):
        """Wznów działanie agenta"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        if agent.state == AgentState.PAUSED:
            self.transition_state(agent_id, AgentState.READY)
            logger.info(f"Agent {agent_id} został wznowiony")
