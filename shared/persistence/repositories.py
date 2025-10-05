"""
Repositories
Data Access Layer - Repository pattern dla każdego modelu
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from .models import (
    ProjectModel,
    AgentModel,
    TaskModel,
    ProtocolModel,
    AnalysisModel,
    PatternModel,
    AntiPatternModel,
    RecommendationModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRepository:
    """Bazowe repository"""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """Utwórz nowy rekord"""
        obj = self.model_class(**kwargs)
        self.session.add(obj)
        self.session.commit()
        self.session.refresh(obj)
        return obj
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """Pobierz po ID"""
        return self.session.query(self.model_class).filter_by(id=id).first()
    
    def get_all(self, limit: int = 100) -> List[Any]:
        """Pobierz wszystkie (z limitem)"""
        return self.session.query(self.model_class).limit(limit).all()
    
    def update(self, id: int, **kwargs) -> Optional[Any]:
        """Zaktualizuj rekord"""
        obj = self.get_by_id(id)
        if obj:
            for key, value in kwargs.items():
                setattr(obj, key, value)
            self.session.commit()
            self.session.refresh(obj)
        return obj
    
    def delete(self, id: int) -> bool:
        """Usuń rekord"""
        obj = self.get_by_id(id)
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False


class ProjectRepository(BaseRepository):
    """Repository dla projektów"""
    
    def __init__(self, session: Session):
        super().__init__(session, ProjectModel)
    
    def get_by_project_id(self, project_id: str) -> Optional[ProjectModel]:
        """Pobierz po project_id"""
        return self.session.query(ProjectModel).filter_by(project_id=project_id).first()
    
    def get_active(self) -> List[ProjectModel]:
        """Pobierz aktywne projekty"""
        return self.session.query(ProjectModel).filter(
            ProjectModel.status.in_(['planned', 'in_progress'])
        ).all()
    
    def get_completed(self, limit: int = 50) -> List[ProjectModel]:
        """Pobierz zakończone projekty"""
        return self.session.query(ProjectModel).filter_by(
            status='completed'
        ).order_by(ProjectModel.completed_at.desc()).limit(limit).all()
    
    def update_status(self, project_id: str, status: str, progress: float = None) -> Optional[ProjectModel]:
        """Zaktualizuj status projektu"""
        project = self.get_by_project_id(project_id)
        if project:
            project.status = status
            if progress is not None:
                project.progress = progress
            if status == 'completed':
                project.completed_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(project)
        return project
    
    def get_statistics(self) -> Dict[str, Any]:
        """Pobierz statystyki projektów"""
        total = self.session.query(ProjectModel).count()
        completed = self.session.query(ProjectModel).filter_by(status='completed').count()
        active = self.session.query(ProjectModel).filter(
            ProjectModel.status.in_(['planned', 'in_progress'])
        ).count()
        
        avg_quality = self.session.query(ProjectModel).filter(
            ProjectModel.quality_score.isnot(None)
        ).with_entities(ProjectModel.quality_score).all()
        avg_quality = sum(q[0] for q in avg_quality) / len(avg_quality) if avg_quality else 0.0
        
        return {
            'total': total,
            'completed': completed,
            'active': active,
            'avg_quality_score': avg_quality
        }


class AgentRepository(BaseRepository):
    """Repository dla agentów"""
    
    def __init__(self, session: Session):
        super().__init__(session, AgentModel)
    
    def get_by_agent_id(self, agent_id: str) -> Optional[AgentModel]:
        """Pobierz po agent_id"""
        return self.session.query(AgentModel).filter_by(agent_id=agent_id).first()
    
    def get_by_type(self, agent_type: str) -> List[AgentModel]:
        """Pobierz wszystkich agentów danego typu"""
        return self.session.query(AgentModel).filter_by(agent_type=agent_type).all()
    
    def get_by_project(self, project_id: int) -> List[AgentModel]:
        """Pobierz agentów w projekcie"""
        return self.session.query(AgentModel).filter_by(project_id=project_id).all()
    
    def update_performance(
        self,
        agent_id: str,
        tasks_completed: int = None,
        avg_quality: float = None,
        success_rate: float = None
    ) -> Optional[AgentModel]:
        """Zaktualizuj performance metrics"""
        agent = self.get_by_agent_id(agent_id)
        if agent:
            if tasks_completed is not None:
                agent.tasks_completed = tasks_completed
            if avg_quality is not None:
                agent.avg_quality_score = avg_quality
            if success_rate is not None:
                agent.success_rate = success_rate
            self.session.commit()
            self.session.refresh(agent)
        return agent


class TaskRepository(BaseRepository):
    """Repository dla zadań"""
    
    def __init__(self, session: Session):
        super().__init__(session, TaskModel)
    
    def get_by_task_id(self, task_id: str) -> Optional[TaskModel]:
        """Pobierz po task_id"""
        return self.session.query(TaskModel).filter_by(task_id=task_id).first()
    
    def get_by_project(self, project_id: int) -> List[TaskModel]:
        """Pobierz zadania projektu"""
        return self.session.query(TaskModel).filter_by(project_id=project_id).all()
    
    def get_by_agent(self, agent_id: int) -> List[TaskModel]:
        """Pobierz zadania agenta"""
        return self.session.query(TaskModel).filter_by(agent_id=agent_id).all()
    
    def get_by_status(self, status: str, project_id: int = None) -> List[TaskModel]:
        """Pobierz zadania po statusie"""
        query = self.session.query(TaskModel).filter_by(status=status)
        if project_id:
            query = query.filter_by(project_id=project_id)
        return query.all()
    
    def update_status(
        self,
        task_id: str,
        status: str,
        quality_score: float = None
    ) -> Optional[TaskModel]:
        """Zaktualizuj status zadania"""
        task = self.get_by_task_id(task_id)
        if task:
            task.status = status
            if quality_score is not None:
                task.quality_score = quality_score
            if status == 'completed':
                task.completed_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(task)
        return task


class ProtocolRepository(BaseRepository):
    """Repository dla protokołów"""
    
    def __init__(self, session: Session):
        super().__init__(session, ProtocolModel)
    
    def get_by_protocol_id(self, protocol_id: str) -> Optional[ProtocolModel]:
        """Pobierz po protocol_id"""
        return self.session.query(ProtocolModel).filter_by(protocol_id=protocol_id).first()
    
    def get_by_project(self, project_id: int) -> List[ProtocolModel]:
        """Pobierz protokoły projektu"""
        return self.session.query(ProtocolModel).filter_by(project_id=project_id).all()
    
    def get_by_type(self, protocol_type: str) -> List[ProtocolModel]:
        """Pobierz protokoły po typie"""
        return self.session.query(ProtocolModel).filter_by(protocol_type=protocol_type).all()
    
    def get_active(self) -> List[ProtocolModel]:
        """Pobierz aktywne protokoły"""
        return self.session.query(ProtocolModel).filter(
            ProtocolModel.status.in_(['initiated', 'in_progress', 'waiting_response'])
        ).all()


class AnalysisRepository(BaseRepository):
    """Repository dla analiz"""
    
    def __init__(self, session: Session):
        super().__init__(session, AnalysisModel)
    
    def get_by_analysis_id(self, analysis_id: str) -> Optional[AnalysisModel]:
        """Pobierz po analysis_id"""
        return self.session.query(AnalysisModel).filter_by(analysis_id=analysis_id).first()
    
    def get_by_project(self, project_id: int) -> Optional[AnalysisModel]:
        """Pobierz analizę projektu"""
        return self.session.query(AnalysisModel).filter_by(project_id=project_id).first()
    
    def get_recent(self, limit: int = 10) -> List[AnalysisModel]:
        """Pobierz ostatnie analizy"""
        return self.session.query(AnalysisModel).order_by(
            AnalysisModel.created_at.desc()
        ).limit(limit).all()


class PatternRepository(BaseRepository):
    """Repository dla wzorców"""
    
    def __init__(self, session: Session):
        super().__init__(session, PatternModel)
    
    def get_by_pattern_id(self, pattern_id: str) -> Optional[PatternModel]:
        """Pobierz po pattern_id"""
        return self.session.query(PatternModel).filter_by(pattern_id=pattern_id).first()
    
    def get_by_type(self, pattern_type: str) -> List[PatternModel]:
        """Pobierz wzorce po typie"""
        return self.session.query(PatternModel).filter_by(pattern_type=pattern_type).all()
    
    def get_high_confidence(self, min_confidence: float = 0.7) -> List[PatternModel]:
        """Pobierz wzorce z wysokim confidence"""
        return self.session.query(PatternModel).filter(
            PatternModel.confidence >= min_confidence
        ).order_by(PatternModel.confidence.desc()).all()


def create_repositories(session: Session) -> Dict[str, BaseRepository]:
    """Utwórz wszystkie repositories"""
    return {
        'projects': ProjectRepository(session),
        'agents': AgentRepository(session),
        'tasks': TaskRepository(session),
        'protocols': ProtocolRepository(session),
        'analyses': AnalysisRepository(session),
        'patterns': PatternRepository(session)
    }
