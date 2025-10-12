from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Text, Boolean
from datetime import datetime

class TaskMetric(BaseModel):
    """Metryki wykonania zadań"""
    __tablename__ = "task_metrics"
    
    task_id = Column(String(100), unique=True, nullable=False, index=True)
    command = Column(String(50), nullable=False, index=True)  # ask, create, orchestra
    provider = Column(String(20), nullable=False)  # ollama, claude
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Cost
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    
    # Quality
    user_rating = Column(Integer, nullable=True)  # 1-5
    was_successful = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Context
    prompt_length = Column(Integer, default=0)
    response_length = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)
