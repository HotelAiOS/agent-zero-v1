from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

class Provider(str, Enum):
    """Dostępni dostawcy AI"""
    OLLAMA = "ollama"
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"

class TaskType(str, Enum):
    """Typy zadań"""
    CODE = "code"
    CHAT = "chat"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"

class GenerateRequest(BaseModel):
    """Żądanie generacji"""
    prompt: str = Field(..., description="Prompt dla modelu AI")
    task_type: TaskType = Field(default=TaskType.CHAT, description="Typ zadania")
    provider: Optional[Provider] = Field(default=None, description="Wybrany dostawca (opcjonalnie)")
    model: Optional[str] = Field(default=None, description="Nazwa modelu (opcjonalnie)")
    max_tokens: int = Field(default=4096, description="Maksymalna liczba tokenów")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura (0-2)")
    stream: bool = Field(default=False, description="Czy streamować odpowiedź")

class GenerateResponse(BaseModel):
    """Odpowiedź generacji"""
    response: str = Field(..., description="Wygenerowana odpowiedź")
    provider: Provider = Field(..., description="Użyty dostawca")
    model: str = Field(..., description="Użyty model")
    tokens: int = Field(..., description="Liczba użytych tokenów")
    latency_ms: float = Field(..., description="Opóźnienie w milisekundach")

class HealthResponse(BaseModel):
    """Status zdrowia serwisu"""
    status: str
    version: str
    providers: dict[str, bool]
