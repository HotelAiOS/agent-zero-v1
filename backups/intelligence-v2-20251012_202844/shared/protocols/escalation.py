"""
Escalation Protocol
Protokół eskalacji problemów do wyższego poziomu (senior agent, human)
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base_protocol import BaseProtocol, ProtocolMessage, ProtocolStatus, ProtocolType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Poziomy eskalacji"""
    LEVEL_1 = 1  # Team lead
    LEVEL_2 = 2  # Senior agent / Architect
    LEVEL_3 = 3  # Human supervisor
    LEVEL_4 = 4  # Management / Product owner


class EscalationReason(Enum):
    """Powody eskalacji"""
    BLOCKED = "blocked"  # Zablokowany, nie może kontynuować
    CRITICAL_DECISION = "critical_decision"  # Krytyczna decyzja
    CONFLICTING_REQUIREMENTS = "conflicting_requirements"  # Sprzeczne wymagania
    TECHNICAL_LIMITATION = "technical_limitation"  # Ograniczenie techniczne
    SECURITY_CONCERN = "security_concern"  # Obawy o bezpieczeństwo
    BUDGET_EXCEEDED = "budget_exceeded"  # Przekroczony budżet
    DEADLINE_RISK = "deadline_risk"  # Ryzyko niedotrzymania deadline
    QUALITY_ISSUE = "quality_issue"  # Problem jakości
    EXTERNAL_DEPENDENCY = "external_dependency"  # Zależność zewnętrzna


@dataclass
class EscalationTicket:
    """Ticket eskalacji"""
    ticket_id: str
    escalated_by: str
    escalation_level: EscalationLevel
    reason: EscalationReason
    title: str
    description: str
    context: Dict[str, Any]
    urgency: str  # critical, high, medium, low
    created_at: datetime = field(default_factory=datetime.now)
    
    # Assignment
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    
    # Resolution
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Tracking
    escalation_path: List[str] = field(default_factory=list)  # Historia eskalacji
    related_tasks: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)


class EscalationProtocol(BaseProtocol):
    """
    Protokół Eskalacji
    Agenci eskalują problemy do wyższego poziomu gdy:
    - Są zablokowane
    - Potrzebują krytycznej decyzji
    - Napotykają problemy poza ich zakresem
    """
    
    def __init__(self):
        super().__init__(ProtocolType.ESCALATION)
        self.tickets: List[EscalationTicket] = []
        self.escalation_hierarchy: Dict[int, List[str]] = {
            1: [],  # Team leads
            2: [],  # Senior agents
            3: [],  # Human supervisors
            4: []   # Management
        }
        self.auto_escalate_after_hours: int = 24
        self.max_escalation_level: int = 3
    
    def initiate(
        self,
        initiator: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Inicjuj eskalację
        
        Context powinien zawierać:
        - title: str
        - description: str
        - reason: str (EscalationReason)
        - urgency: str
        - current_level: int
        - hierarchy: Dict[int, List[str]] - kto obsługuje który poziom
        """
        self.initiated_by = initiator
        self.add_participant(initiator)
        
        # Setup hierarchy
        hierarchy = context.get('hierarchy', {})
        for level, agents in hierarchy.items():
            self.escalation_hierarchy[int(level)] = agents
            for agent in agents:
                self.add_participant(agent)
        
        self.auto_escalate_after_hours = context.get('auto_escalate_hours', 24)
        self.max_escalation_level = context.get('max_level', 3)
        
        self.status = ProtocolStatus.IN_PROGRESS
        
        logger.info(f"Escalation Protocol initiated by {initiator}")
        return True
    
    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Przetwórz wiadomość w eskalacji"""
        action = message.content.get('action')
        
        if action == 'escalate':
            return self._handle_escalation(message)
        elif action == 'assign':
            return self._handle_assignment(message)
        elif action == 'resolve':
            return self._handle_resolution(message)
        elif action == 'escalate_further':
            return self._handle_further_escalation(message)
        
        return None
    
    def _handle_escalation(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż nową eskalację"""
        content = message.content
        
        # Określ poziom eskalacji
        initial_level = EscalationLevel(content.get('level', 1))
        
        ticket = EscalationTicket(
            ticket_id=f"esc_{len(self.tickets) + 1}",
            escalated_by=message.from_agent,
            escalation_level=initial_level,
            reason=EscalationReason[content.get('reason', 'BLOCKED')],
            title=content.get('title', ''),
            description=content.get('description', ''),
            context=content.get('context', {}),
            urgency=content.get('urgency', 'medium'),
            related_tasks=content.get('related_tasks', [])
        )
        
        ticket.escalation_path.append(f"L{initial_level.value}: {message.from_agent}")
        
        self.tickets.append(ticket)
        
        logger.info(
            f"Escalation created: {ticket.title} "
            f"(Level {ticket.escalation_level.value}, {ticket.reason.value})"
        )
        
        # Znajdź odpowiednią osobę do assignment
        handlers = self.escalation_hierarchy.get(initial_level.value, [])
        if handlers:
            # Auto-assign do pierwszego dostępnego
            ticket.assigned_to = handlers[0]
            ticket.assigned_at = datetime.now()
            
            # Wyślij notification
            return self.send_message(
                from_agent='system',
                to_agent=ticket.assigned_to,
                content={
                    'action': 'escalation_assigned',
                    'ticket_id': ticket.ticket_id,
                    'title': ticket.title,
                    'urgency': ticket.urgency,
                    'escalated_by': ticket.escalated_by
                },
                requires_response=True
            )
        else:
            logger.warning(f"No handlers for level {initial_level.value}")
        
        return None
    
    def _handle_assignment(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż przypisanie ticketu"""
        ticket_id = message.content.get('ticket_id')
        assigned_to = message.content.get('assigned_to')
        
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                ticket.assigned_to = assigned_to
                ticket.assigned_at = datetime.now()
                logger.info(f"Ticket {ticket_id} assigned to {assigned_to}")
                break
        
        return None
    
    def _handle_resolution(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż rozwiązanie eskalacji"""
        ticket_id = message.content.get('ticket_id')
        resolution = message.content.get('resolution')
        
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                ticket.resolved = True
                ticket.resolution = resolution
                ticket.resolved_by = message.from_agent
                ticket.resolved_at = datetime.now()
                
                logger.info(
                    f"Escalation {ticket_id} resolved by {message.from_agent}"
                )
                
                # Notify oryginalnemu agentowi
                return self.send_message(
                    from_agent=message.from_agent,
                    to_agent=ticket.escalated_by,
                    content={
                        'action': 'escalation_resolved',
                        'ticket_id': ticket.ticket_id,
                        'resolution': resolution
                    }
                )
        
        return None
    
    def _handle_further_escalation(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż dalszą eskalację do wyższego poziomu"""
        ticket_id = message.content.get('ticket_id')
        reason = message.content.get('reason', 'Needs higher authority')
        
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                current_level = ticket.escalation_level.value
                
                if current_level >= self.max_escalation_level:
                    logger.warning(
                        f"Ticket {ticket_id} already at max level {current_level}"
                    )
                    return None
                
                # Eskaluj o poziom wyżej
                new_level = EscalationLevel(current_level + 1)
                ticket.escalation_level = new_level
                ticket.escalation_path.append(
                    f"L{new_level.value}: escalated by {message.from_agent} - {reason}"
                )
                
                # Przypisz do nowego poziomu
                handlers = self.escalation_hierarchy.get(new_level.value, [])
                if handlers:
                    ticket.assigned_to = handlers[0]
                    ticket.assigned_at = datetime.now()
                    
                    logger.info(
                        f"Ticket {ticket_id} escalated to Level {new_level.value}"
                    )
                    
                    return self.send_message(
                        from_agent='system',
                        to_agent=ticket.assigned_to,
                        content={
                            'action': 'escalation_assigned',
                            'ticket_id': ticket.ticket_id,
                            'title': ticket.title,
                            'level': new_level.value,
                            'escalation_reason': reason
                        },
                        requires_response=True
                    )
        
        return None
    
    def escalate_issue(
        self,
        agent_id: str,
        title: str,
        description: str,
        reason: EscalationReason,
        urgency: str = 'medium',
        level: int = 1
    ) -> EscalationTicket:
        """Wygodna metoda do eskalacji"""
        message = ProtocolMessage(
            message_id=f"msg_{len(self.messages)}",
            from_agent=agent_id,
            to_agent=None,
            timestamp=datetime.now(),
            content={
                'action': 'escalate',
                'title': title,
                'description': description,
                'reason': reason.name,
                'urgency': urgency,
                'level': level
            },
            protocol_type=self.protocol_type
        )
        
        self._handle_escalation(message)
        return self.tickets[-1]
    
    def complete(self) -> Dict[str, Any]:
        """Zakończ eskalację i zwróć wynik"""
        self.status = ProtocolStatus.COMPLETED
        self.completed_at = datetime.now()
        
        resolved_count = sum(1 for t in self.tickets if t.resolved)
        unresolved_count = len(self.tickets) - resolved_count
        
        # Statystyki po poziomach
        by_level = {}
        for ticket in self.tickets:
            level = ticket.escalation_level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        # Statystyki po powodach
        by_reason = {}
        for ticket in self.tickets:
            reason = ticket.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        self.result = {
            'total_escalations': len(self.tickets),
            'resolved': resolved_count,
            'unresolved': unresolved_count,
            'by_level': by_level,
            'by_reason': by_reason,
            'critical_unresolved': [
                {
                    'id': t.ticket_id,
                    'title': t.title,
                    'urgency': t.urgency,
                    'level': t.escalation_level.value
                }
                for t in self.tickets
                if not t.resolved and t.urgency == 'critical'
            ]
        }
        
        logger.info(f"Escalation Protocol completed: {self.result}")
        return self.result
    
    def get_unresolved_tickets(self, urgency: Optional[str] = None) -> List[EscalationTicket]:
        """Pobierz nierozwiązane tickety"""
        tickets = [t for t in self.tickets if not t.resolved]
        
        if urgency:
            tickets = [t for t in tickets if t.urgency == urgency]
        
        return sorted(tickets, key=lambda t: t.created_at)
    
    def get_overdue_tickets(self) -> List[EscalationTicket]:
        """Pobierz opóźnione tickety (> auto_escalate_after_hours)"""
        threshold = datetime.now().timestamp() - (self.auto_escalate_after_hours * 3600)
        
        return [
            t for t in self.tickets
            if not t.resolved and t.created_at.timestamp() < threshold
        ]


def create_escalation() -> EscalationProtocol:
    """Utwórz Escalation Protocol"""
    return EscalationProtocol()
