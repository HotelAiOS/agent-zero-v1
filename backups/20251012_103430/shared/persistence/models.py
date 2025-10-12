"""
Database Models
SQLAlchemy models dla Agent Zero
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, 
    Text, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from .database import Base


class ProjectModel(Base):
    """Model projektu"""
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String(100), unique=True, index=True, nullable=False)
    project_name = Column(String(200), nullable=False)
    project_type = Column(String(50), nullable=False)
    
    # Status
    status = Column(String(50), default='planned')
    progress = Column(Float, default=0.0)
    
    # Business requirements
    business_requirements = Column(JSON)
    
    # Planning
    estimated_duration_days = Column(Float)
    estimated_cost = Column(Float)
    actual_duration_days = Column(Float)
    actual_cost = Column(Float)
    
    # Quality
    quality_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agents = relationship("AgentModel", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("TaskModel", back_populates="project", cascade="all, delete-orphan")
    protocols = relationship("ProtocolModel", back_populates="project", cascade="all, delete-orphan")
    analyses = relationship("AnalysisModel", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project {self.project_name} ({self.status})>"


class AgentModel(Base):
    """Model agenta"""
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(100), unique=True, index=True, nullable=False)
    agent_type = Column(String(50), nullable=False)
    agent_name = Column(String(200))
    
    # Configuration
    model = Column(String(100))
    capabilities = Column(JSON)
    specializations = Column(JSON)
    
    # Status
    status = Column(String(50), default='created')
    
    # Performance
    tasks_completed = Column(Integer, default=0)
    avg_quality_score = Column(Float, default=0.0)
    success_rate = Column(Float, default=1.0)
    
    # Project association
    project_id = Column(Integer, ForeignKey('projects.id'))
    project = relationship("ProjectModel", back_populates="agents")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tasks = relationship("TaskModel", back_populates="assigned_agent")
    
    def __repr__(self):
        return f"<Agent {self.agent_id} ({self.agent_type})>"


class TaskModel(Base):
    """Model zadania"""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), unique=True, index=True, nullable=False)
    task_name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Classification
    task_type = Column(String(50))
    complexity = Column(Integer, default=5)
    priority = Column(Integer, default=3)
    
    # Execution
    status = Column(String(50), default='pending')
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    
    # Quality
    quality_score = Column(Float)
    
    # Dependencies
    dependencies = Column(JSON)  # List of task_ids
    
    # Project & Agent association
    project_id = Column(Integer, ForeignKey('projects.id'))
    agent_id = Column(Integer, ForeignKey('agents.id'))
    
    project = relationship("ProjectModel", back_populates="tasks")
    assigned_agent = relationship("AgentModel", back_populates="tasks")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Task {self.task_name} ({self.status})>"


class ProtocolModel(Base):
    """Model protokołu komunikacji"""
    __tablename__ = 'protocols'
    
    id = Column(Integer, primary_key=True, index=True)
    protocol_id = Column(String(100), unique=True, index=True, nullable=False)
    protocol_type = Column(String(50), nullable=False)
    
    # Context
    initiator = Column(String(100))
    participants = Column(JSON)  # List of agent_ids
    context = Column(JSON)
    
    # Status
    status = Column(String(50), default='initiated')
    
    # Results
    result = Column(JSON)
    messages_count = Column(Integer, default=0)
    
    # Project association
    project_id = Column(Integer, ForeignKey('projects.id'))
    project = relationship("ProjectModel", back_populates="protocols")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Protocol {self.protocol_type} ({self.status})>"


class AnalysisModel(Base):
    """Model analizy post-mortem"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Results
    project_status = Column(String(50))
    quality_score = Column(Float)
    
    # Metrics
    planned_duration = Column(Float)
    actual_duration = Column(Float)
    planned_cost = Column(Float)
    actual_cost = Column(Float)
    
    # Insights
    insights = Column(JSON)
    what_went_well = Column(JSON)
    what_went_wrong = Column(JSON)
    lessons_learned = Column(JSON)
    action_items = Column(JSON)
    
    # Project association
    project_id = Column(Integer, ForeignKey('projects.id'))
    project = relationship("ProjectModel", back_populates="analyses")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Analysis {self.analysis_id} (quality: {self.quality_score})>"


class PatternModel(Base):
    """Model wzorca"""
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_id = Column(String(100), unique=True, index=True, nullable=False)
    pattern_type = Column(String(50), nullable=False)
    
    # Details
    name = Column(String(200), nullable=False)
    description = Column(Text)
    confidence = Column(Float)
    
    # Statistics
    occurrences = Column(Integer, default=1)
    success_rate = Column(Float)
    projects = Column(JSON)  # List of project_ids
    
    # Characteristics
    conditions = Column(JSON)
    outcomes = Column(JSON)
    correlated_patterns = Column(JSON)
    
    # Timestamps
    discovered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Pattern {self.name} ({self.pattern_type})>"


class AntiPatternModel(Base):
    """Model anty-wzorca"""
    __tablename__ = 'antipatterns'
    
    id = Column(Integer, primary_key=True, index=True)
    antipattern_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Details
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    severity = Column(String(50), nullable=False)
    description = Column(Text)
    
    # Impact
    impact_description = Column(Text)
    avg_negative_impact = Column(Float)
    
    # Detection
    detected_in = Column(JSON)  # List of project_ids
    detection_count = Column(Integer, default=0)
    
    # Remediation
    remediation_steps = Column(JSON)
    estimated_fix_hours = Column(Float)
    
    # Timestamps
    first_detected = Column(DateTime, default=datetime.utcnow)
    last_detected = Column(DateTime)
    
    def __repr__(self):
        return f"<AntiPattern {self.name} ({self.severity})>"


class RecommendationModel(Base):
    """Model rekomendacji"""
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(String(100), unique=True, index=True, nullable=False)
    category = Column(String(50), nullable=False)
    
    # Details
    title = Column(String(200), nullable=False)
    description = Column(Text)
    rationale = Column(Text)
    
    # Confidence & Impact
    confidence = Column(Float)
    expected_impact = Column(String(50))
    
    # Implementation
    action_items = Column(JSON)
    estimated_effort_hours = Column(Float)
    
    # Evidence
    based_on_projects = Column(JSON)
    success_rate_with = Column(Float)
    success_rate_without = Column(Float)
    
    # Status
    implemented = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Recommendation {self.title} ({self.category})>"
