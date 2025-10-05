"""
Agent Lifecycle Manager
Zarządzanie cyklem życia agentów z integracją messaging
"""

from enum import Enum
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
import time
import sys
from pathlib import Path

# Import dla LLM
sys.path.append(str(Path(__file__).parent.parent))
from llm.prompt_builder import PromptBuilder, PromptContext
from llm.response_parser import ResponseParser

# Import dla messaging
from messaging import AgentCommunicator, Message, MessageType

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
    messages_sent: int = 0
    messages_received: int = 0
    
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
    
    # Pola dla LLM
    llm_client: Any = None
    template: Any = None
    
    # NOWE: Komunikator dla messaging
    communicator: Optional[AgentCommunicator] = None
    
    def mark_active(self):
        """Oznacz agenta jako aktywnego"""
        self.metrics.last_active = datetime.now()
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wykonaj zadanie używając LLM
        
        Args:
            task: Dict z opisem zadania:
                - description: str - opis zadania
                - name: str - nazwa zadania (opcjonalny)
                - context: Dict[str, Any] - kontekst (opcjonalny)
                - tech_stack: List[str] - stack technologiczny (opcjonalny)
                - requirements: List[str] - wymagania (opcjonalny)
        
        Returns:
            Dict z wynikiem:
                - success: bool
                - output: str - wygenerowany kod/wynik
                - raw_response: str - pełna odpowiedź LLM
                - error: str - komunikat błędu (jeśli failed)
                - tokens_used: int
                - response_time: float
        """
        if self.llm_client is None:
            logger.error(f"Agent {self.agent_id}: brak LLM client")
            return {
                'success': False,
                'error': 'LLM client not initialized',
                'output': None
            }
        
        start_time = time.time()
        
        try:
            # Zbuduj context dla promptu
            context = PromptContext(
                agent_type=self.agent_type,
                task_name=task.get('name', 'Task'),
                task_description=task.get('description', ''),
                tech_stack=task.get('tech_stack', []),
                requirements=task.get('requirements', []),
                context=task.get('context', {})
            )
            
            # Zbuduj prompt
            messages = PromptBuilder.build_task_prompt(context)
            
            logger.info(f"Agent {self.agent_id}: wykonuje zadanie...")
            
            # Wywołaj LLM
            response = self.llm_client.chat(
                messages=messages,
                agent_type=self.agent_type
            )
            
            # Parsuj odpowiedź
            response_text = response.get('message', {}).get('content', '')
            code = ResponseParser.extract_first_code_block(response_text)
            
            # Oblicz metryki
            response_time = time.time() - start_time
            tokens_used = response.get('eval_count', 0) + response.get('prompt_eval_count', 0)
            
            result = {
                'success': True,
                'output': code or response_text,
                'raw_response': response_text,
                'tokens_used': tokens_used,
                'response_time': response_time
            }
            
            logger.info(
                f"Agent {self.agent_id}: zadanie wykonane "
                f"({tokens_used} tokens, {response_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Agent {self.agent_id}: błąd wykonania zadania: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'output': None,
                'response_time': response_time,
                'tokens_used': 0
            }
    
    # === NOWE: METODY KOMUNIKACJI ===
    
    def send_message(
        self,
        recipient_id: str,
        subject: str,
        content: str,
        message_type: MessageType = MessageType.NOTIFICATION,
        payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Wyślij wiadomość do innego agenta
        
        Args:
            recipient_id: ID odbiorcy
            subject: Temat
            content: Treść
            message_type: Typ wiadomości
            payload: Dodatkowe dane
        
        Returns:
            True jeśli wysłano
        """
        if self.communicator is None:
            logger.error(f"Agent {self.agent_id}: brak communicatora")
            return False
        
        success = self.communicator.send_direct(
            recipient_id=recipient_id,
            subject=subject,
            content=content,
            message_type=message_type,
            payload=payload or {}
        )
        
        if success:
            self.metrics.messages_sent += 1
            self.mark_active()
        
        return success
    
    def broadcast(
        self,
        subject: str,
        content: str,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Wyślij broadcast do wszystkich agentów
        
        Args:
            subject: Temat
            content: Treść
            project_id: Opcjonalnie ogranicz do projektu
        
        Returns:
            True jeśli wysłano
        """
        if self.communicator is None:
            logger.error(f"Agent {self.agent_id}: brak communicatora")
            return False
        
        success = self.communicator.broadcast(
            subject=subject,
            content=content,
            project_id=project_id
        )
        
        if success:
            self.metrics.messages_sent += 1
            self.mark_active()
        
        return success
    
    def start_listening(self):
        """Rozpocznij nasłuchiwanie na wiadomości"""
        if self.communicator is None:
            logger.error(f"Agent {self.agent_id}: brak communicatora")
            return False
        
        # Handler dla odbieranych wiadomości
        def handle_message(msg: Message):
            self.metrics.messages_received += 1
            self.mark_active()
            logger.info(
                f"Agent {self.agent_id} otrzymał: {msg.message_type.value} "
                f"od {msg.sender_id}"
            )
            # Tu możesz dodać custom logikę przetwarzania
        
        self.communicator.on_message(handle_message)
        self.communicator.start_listening(block=False)
        
        logger.info(f"Agent {self.agent_id}: nasłuchuje na wiadomości")
        return True
    
    def stop_listening(self):
        """Zatrzymaj nasłuchiwanie"""
        if self.communicator:
            self.communicator.stop_listening()


class AgentLifecycleManager:
    """Manager cyklu życia agentów"""
    
    def __init__(self, enable_messaging: bool = True):
        """
        Initialize lifecycle manager
        
        Args:
            enable_messaging: Czy włączyć komunikację przez RabbitMQ
        """
        self.agents: Dict[str, AgentInstance] = {}
        self.enable_messaging = enable_messaging
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
        logger.info(
            f"AgentLifecycleManager zainicjalizowany "
            f"(messaging: {'enabled' if enable_messaging else 'disabled'})"
        )
    
    def create_agent(
        self, 
        agent_id: str, 
        agent_type: str,
        enable_messaging: Optional[bool] = None
    ) -> AgentInstance:
        """
        Utwórz nową instancję agenta
        
        Args:
            agent_id: ID agenta
            agent_type: Typ agenta
            enable_messaging: Czy włączyć messaging (None = użyj domyślnego)
        
        Returns:
            AgentInstance
        """
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} już istnieje")
            return self.agents[agent_id]
        
        agent = AgentInstance(
            agent_id=agent_id,
            agent_type=agent_type,
            state=AgentState.CREATED
        )
        
        # Dodaj communicator jeśli messaging włączony
        use_messaging = enable_messaging if enable_messaging is not None else self.enable_messaging
        if use_messaging:
            try:
                agent.communicator = AgentCommunicator(
                    agent_id=agent_id,
                    auto_connect=True
                )
                logger.info(f"Agent {agent_id}: communicator zainicjalizowany")
            except Exception as e:
                logger.warning(f"Agent {agent_id}: nie udało się zainicjalizować communicatora: {e}")
        
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
        total_messages = sum(
            a.metrics.messages_sent + a.metrics.messages_received 
            for a in self.agents.values()
        )
        
        health = {
            "status": "healthy" if state_counts.get("error", 0) == 0 else "degraded",
            "total_agents": total,
            "state_distribution": state_counts,
            "total_tasks_completed": total_tasks,
            "total_errors": total_errors,
            "total_messages": total_messages,
            "error_rate": total_errors / max(total_tasks, 1)
        }
        
        logger.info(f"System health: {health['status']}")
        return health
    
    def terminate_agent(self, agent_id: str):
        """Zakończ działanie agenta"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # Zatrzymaj messaging
        if agent.communicator:
            agent.stop_listening()
            agent.communicator.disconnect()
        
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
