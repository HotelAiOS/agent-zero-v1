"""
Base Protocol
Bazowa klasa dla wszystkich protokołów komunikacji między agentami
"""

from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Typy protokołów"""
    CODE_REVIEW = "code_review"
    PROBLEM_SOLVING = "problem_solving"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    ESCALATION = "escalation"
    CONSENSUS = "consensus"
    CONSULTATION = "consultation"


class ProtocolStatus(Enum):
    """Status protokołu"""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    WAITING_RESPONSE = "waiting_response"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProtocolMessage:
    """Wiadomość w protokole"""
    message_id: str
    from_agent: str
    to_agent: Optional[str]  # None = broadcast
    timestamp: datetime
    content: Dict[str, Any]
    protocol_type: ProtocolType
    parent_message_id: Optional[str] = None
    requires_response: bool = False
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"msg_{uuid.uuid4().hex[:8]}"


class BaseProtocol(ABC):
    """
    Bazowa klasa protokołu
    Wszystkie protokoły dziedziczą z tej klasy
    """
    
    def __init__(self, protocol_type: ProtocolType):
        self.protocol_type = protocol_type
        self.protocol_id = f"proto_{uuid.uuid4().hex[:8]}"
        self.status = ProtocolStatus.INITIATED
        self.messages: List[ProtocolMessage] = []
        self.participants: List[str] = []
        self.initiated_by: Optional[str] = None
        self.initiated_at: datetime = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        
        logger.info(f"Protocol {protocol_type.value} ({self.protocol_id}) initialized")
    
    @abstractmethod
    def initiate(self, initiator: str, context: Dict[str, Any]) -> bool:
        """
        Inicjuj protokół
        
        Args:
            initiator: ID agenta inicjującego
            context: Kontekst protokołu
        
        Returns:
            True jeśli pomyślnie zainicjowano
        """
        pass
    
    @abstractmethod
    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """
        Przetwórz wiadomość w protokole
        
        Args:
            message: ProtocolMessage do przetworzenia
        
        Returns:
            Opcjonalna odpowiedź (ProtocolMessage)
        """
        pass
    
    @abstractmethod
    def complete(self) -> Dict[str, Any]:
        """
        Zakończ protokół i zwróć wynik
        
        Returns:
            Wynik protokołu
        """
        pass
    
    def send_message(
        self,
        from_agent: str,
        to_agent: Optional[str],
        content: Dict[str, Any],
        requires_response: bool = False,
        parent_id: Optional[str] = None
    ) -> ProtocolMessage:
        """Wyślij wiadomość w ramach protokołu"""
        message = ProtocolMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            from_agent=from_agent,
            to_agent=to_agent,
            timestamp=datetime.now(),
            content=content,
            protocol_type=self.protocol_type,
            parent_message_id=parent_id,
            requires_response=requires_response
        )
        
        self.messages.append(message)
        
        if to_agent:
            logger.info(f"Message {message.message_id}: {from_agent} -> {to_agent}")
        else:
            logger.info(f"Broadcast {message.message_id}: {from_agent} -> ALL")
        
        return message
    
    def broadcast_message(
        self,
        from_agent: str,
        content: Dict[str, Any]
    ) -> ProtocolMessage:
        """Broadcast wiadomości do wszystkich uczestników"""
        return self.send_message(from_agent, None, content, requires_response=False)
    
    def add_participant(self, agent_id: str):
        """Dodaj uczestnika do protokołu"""
        if agent_id not in self.participants:
            self.participants.append(agent_id)
            logger.info(f"Agent {agent_id} joined protocol {self.protocol_id}")
    
    def remove_participant(self, agent_id: str):
        """Usuń uczestnika z protokołu"""
        if agent_id in self.participants:
            self.participants.remove(agent_id)
            logger.info(f"Agent {agent_id} left protocol {self.protocol_id}")
    
    def get_messages_from(self, agent_id: str) -> List[ProtocolMessage]:
        """Pobierz wszystkie wiadomości od danego agenta"""
        return [m for m in self.messages if m.from_agent == agent_id]
    
    def get_messages_to(self, agent_id: str) -> List[ProtocolMessage]:
        """Pobierz wszystkie wiadomości do danego agenta"""
        return [m for m in self.messages if m.to_agent == agent_id]
    
    def get_conversation(self, agent1: str, agent2: str) -> List[ProtocolMessage]:
        """Pobierz konwersację między dwoma agentami"""
        return [
            m for m in self.messages
            if (m.from_agent == agent1 and m.to_agent == agent2) or
               (m.from_agent == agent2 and m.to_agent == agent1)
        ]
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Pobierz podsumowanie statusu protokołu"""
        duration = None
        if self.completed_at:
            duration = (self.completed_at - self.initiated_at).total_seconds()
        
        return {
            'protocol_id': self.protocol_id,
            'protocol_type': self.protocol_type.value,
            'status': self.status.value,
            'initiated_by': self.initiated_by,
            'initiated_at': self.initiated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': duration,
            'participants': self.participants,
            'message_count': len(self.messages),
            'result': self.result
        }
    
    def cancel(self, reason: str):
        """Anuluj protokół"""
        self.status = ProtocolStatus.CANCELLED
        self.completed_at = datetime.now()
        self.result = {'cancelled': True, 'reason': reason}
        logger.warning(f"Protocol {self.protocol_id} cancelled: {reason}")
    
    def fail(self, error: str):
        """Oznacz protokół jako nieudany"""
        self.status = ProtocolStatus.FAILED
        self.completed_at = datetime.now()
        self.result = {'failed': True, 'error': error}
        logger.error(f"Protocol {self.protocol_id} failed: {error}")
