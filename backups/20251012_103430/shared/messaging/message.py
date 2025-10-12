"""
Message Structure
Struktura wiadomości dla komunikacji między agentami
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid


class MessagePriority(Enum):
    """Priorytety wiadomości"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageType(Enum):
    """Typy wiadomości"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUESTION = "question"
    ANSWER = "answer"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    CODE_REVIEW_REQUEST = "code_review_request"
    CODE_REVIEW_RESPONSE = "code_review_response"
    COLLABORATION = "collaboration"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


@dataclass
class Message:
    """
    Wiadomość między agentami
    Zgodnie z PDF: każda wiadomość ma priorytet i kontekst
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.NOTIFICATION
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Nadawca i odbiorca
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None = broadcast
    
    # Treść
    subject: str = ""
    content: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Kontekst
    project_id: Optional[str] = None
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    reply_to: Optional[str] = None  # ID wiadomości na którą odpowiada
    
    # Routing
    routing_key: str = ""
    exchange: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuj do dict dla JSON serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'subject': self.subject,
            'content': self.content,
            'payload': self.payload,
            'project_id': self.project_id,
            'task_id': self.task_id,
            'conversation_id': self.conversation_id,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'requires_response': self.requires_response,
            'reply_to': self.reply_to,
            'routing_key': self.routing_key,
            'exchange': self.exchange
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Utwórz Message z dict"""
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=MessageType(data.get('message_type', 'notification')),
            priority=MessagePriority(data.get('priority', 2)),
            sender_id=data.get('sender_id', ''),
            recipient_id=data.get('recipient_id'),
            subject=data.get('subject', ''),
            content=data.get('content', ''),
            payload=data.get('payload', {}),
            project_id=data.get('project_id'),
            task_id=data.get('task_id'),
            conversation_id=data.get('conversation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            requires_response=data.get('requires_response', False),
            reply_to=data.get('reply_to'),
            routing_key=data.get('routing_key', ''),
            exchange=data.get('exchange', '')
        )
    
    def is_broadcast(self) -> bool:
        """Czy to wiadomość broadcast?"""
        return self.recipient_id is None
    
    def is_direct(self) -> bool:
        """Czy to wiadomość bezpośrednia?"""
        return self.recipient_id is not None
    
    def is_expired(self) -> bool:
        """Czy wiadomość wygasła?"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


def create_task_request(
    sender_id: str,
    recipient_id: str,
    task_description: str,
    task_id: str,
    project_id: str,
    priority: MessagePriority = MessagePriority.NORMAL,
    payload: Optional[Dict[str, Any]] = None
) -> Message:
    """Pomocnicza funkcja: utwórz task request"""
    return Message(
        message_type=MessageType.TASK_REQUEST,
        priority=priority,
        sender_id=sender_id,
        recipient_id=recipient_id,
        subject=f"Task Request: {task_id}",
        content=task_description,
        payload=payload or {},
        project_id=project_id,
        task_id=task_id,
        requires_response=True
    )


def create_broadcast(
    sender_id: str,
    subject: str,
    content: str,
    project_id: Optional[str] = None,
    priority: MessagePriority = MessagePriority.NORMAL
) -> Message:
    """Pomocnicza funkcja: utwórz broadcast"""
    return Message(
        message_type=MessageType.BROADCAST,
        priority=priority,
        sender_id=sender_id,
        recipient_id=None,  # Broadcast
        subject=subject,
        content=content,
        project_id=project_id,
        requires_response=False
    )


def create_code_review_request(
    sender_id: str,
    recipient_id: str,
    code: str,
    task_id: str,
    project_id: str,
    language: str = "python"
) -> Message:
    """Pomocnicza funkcja: code review request"""
    return Message(
        message_type=MessageType.CODE_REVIEW_REQUEST,
        priority=MessagePriority.HIGH,
        sender_id=sender_id,
        recipient_id=recipient_id,
        subject=f"Code Review Request: {task_id}",
        content=code,
        payload={'language': language},
        project_id=project_id,
        task_id=task_id,
        requires_response=True
    )
