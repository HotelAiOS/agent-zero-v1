from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """Role wiadomości"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    """Pojedyncza wiadomość w czacie"""
    id: Optional[str] = None
    session_id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tokens: int = 0
    metadata: dict = Field(default_factory=dict)

class ChatSession(BaseModel):
    """Sesja czatu"""
    id: str
    user_id: str
    title: Optional[str] = "New Chat"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    total_tokens: int = 0
    metadata: dict = Field(default_factory=dict)

class ChatRequest(BaseModel):
    """Żądanie czatu"""
    session_id: Optional[str] = None
    user_id: str
    message: str
    task_type: str = "chat"
    model_preference: Optional[str] = None

class ChatResponse(BaseModel):
    """Odpowiedź czatu"""
    session_id: str
    message_id: str
    content: str
    tokens: int
    latency_ms: float
    model_used: str

class SessionListResponse(BaseModel):
    """Lista sesji"""
    sessions: List[ChatSession]
    total: int

class SessionHistoryResponse(BaseModel):
    """Historia sesji"""
    session: ChatSession
    messages: List[Message]
