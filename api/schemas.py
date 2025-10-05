"""
API Schemas
Pydantic models dla request/response
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Project Schemas
class ProjectCreate(BaseModel):
    """Schema tworzenia projektu"""
    project_name: str = Field(..., min_length=3, max_length=200)
    project_type: str = Field(..., description="fullstack_web_app, api_backend, etc.")
    business_requirements: List[str] = Field(..., min_items=1)
    schedule_strategy: Optional[str] = Field("load_balanced", description="critical_path, load_balanced, parallel, sequential")


class ProjectResponse(BaseModel):
    """Schema odpowiedzi projektu"""
    project_id: str
    project_name: str
    project_type: str
    status: str
    progress: float
    estimated_duration_days: Optional[float]
    estimated_cost: Optional[float]
    quality_score: Optional[float]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ProjectUpdate(BaseModel):
    """Schema update projektu"""
    status: Optional[str] = None
    progress: Optional[float] = None
    quality_score: Optional[float] = None


# Agent Schemas
class AgentResponse(BaseModel):
    """Schema odpowiedzi agenta"""
    agent_id: str
    agent_type: str
    agent_name: Optional[str]
    status: str
    tasks_completed: int
    avg_quality_score: float
    success_rate: float
    
    class Config:
        from_attributes = True


class AgentPerformanceUpdate(BaseModel):
    """Schema update performance agenta"""
    tasks_completed: Optional[int] = None
    avg_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)


# Task Schemas
class TaskCreate(BaseModel):
    """Schema tworzenia zadania"""
    task_name: str = Field(..., min_length=3)
    description: Optional[str] = None
    task_type: str
    complexity: int = Field(5, ge=1, le=10)
    priority: int = Field(3, ge=1, le=5)
    estimated_hours: float = Field(..., gt=0)


class TaskResponse(BaseModel):
    """Schema odpowiedzi zadania"""
    task_id: str
    task_name: str
    task_type: str
    status: str
    complexity: int
    priority: int
    estimated_hours: float
    actual_hours: Optional[float]
    quality_score: Optional[float]
    
    class Config:
        from_attributes = True


class TaskUpdate(BaseModel):
    """Schema update zadania"""
    status: Optional[str] = None
    actual_hours: Optional[float] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)


# Protocol Schemas
class ProtocolCreate(BaseModel):
    """Schema tworzenia protokołu"""
    protocol_type: str = Field(..., description="code_review, consensus, problem_solving, etc.")
    context: Dict[str, Any]


class ProtocolResponse(BaseModel):
    """Schema odpowiedzi protokołu"""
    protocol_id: str
    protocol_type: str
    initiator: str
    status: str
    participants: List[str]
    messages_count: int
    
    class Config:
        from_attributes = True


# System Schemas
class SystemStatusResponse(BaseModel):
    """Schema statusu systemu"""
    active_projects: int
    completed_projects: int
    active_protocols: int
    total_agents: int
    agent_types_available: int
    patterns_detected: int
    antipatterns_known: int
    cache_keys: int


class HealthResponse(BaseModel):
    """Schema health check"""
    status: str
    database: bool
    cache: bool
    timestamp: datetime


# Analysis Schemas
class AnalysisResponse(BaseModel):
    """Schema analizy post-mortem"""
    analysis_id: str
    project_status: str
    quality_score: float
    planned_duration: float
    actual_duration: float
    insights_count: int
    lessons_learned: List[str]
    
    class Config:
        from_attributes = True


# Execution Schemas
class ProjectExecutionRequest(BaseModel):
    """Schema wykonania projektu"""
    auto_advance: bool = Field(True, description="Auto przejście przez fazy")
    perform_post_mortem: bool = Field(True, description="Czy wykonać post-mortem")


class ProjectExecutionResponse(BaseModel):
    """Schema wyniku wykonania"""
    project_id: str
    status: str
    total_duration_hours: float
    average_quality: float
    phases_completed: int


# Recommendation Schemas
class RecommendationResponse(BaseModel):
    """Schema rekomendacji"""
    recommendation_id: str
    category: str
    title: str
    description: str
    confidence: float
    expected_impact: str
    action_items: List[str]
    
    class Config:
        from_attributes = True


# WebSocket Schemas
class WebSocketMessage(BaseModel):
    """Schema wiadomości WebSocket"""
    type: str = Field(..., description="project_update, agent_status, protocol_event, etc.")
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Error Schema
class ErrorResponse(BaseModel):
    """Schema błędu"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
