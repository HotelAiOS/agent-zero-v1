"""
Projects API Routes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import json

from persistence import get_db_session, ProjectRepository
from core import AgentZeroCore, ProjectManager
from orchestration import ScheduleStrategy
from api.dependencies import get_core_engine, get_project_manager
from api.schemas import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectExecutionRequest,
    ProjectExecutionResponse
)

router = APIRouter()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    core: AgentZeroCore = Depends(get_core_engine),
    db: Session = Depends(get_db_session)
):
    """
    Utwórz nowy projekt
    
    - **project_name**: Nazwa projektu
    - **project_type**: Typ (fullstack_web_app, api_backend, etc.)
    - **business_requirements**: Lista wymagań biznesowych
    - **schedule_strategy**: Strategia schedulowania
    """
    try:
        # Utwórz projekt w core engine
        schedule_strategy = ScheduleStrategy[project.schedule_strategy.upper()]
        
        execution = core.create_project(
            project_name=project.project_name,
            project_type=project.project_type,
            business_requirements=project.business_requirements,
            schedule_strategy=schedule_strategy
        )
        
        # Zapisz do bazy
        repo = ProjectRepository(db)
        db_project = repo.create(
            project_id=execution.project_id,
            project_name=execution.project_name,
            project_type=project.project_type,
            status=execution.status,
            progress=execution.progress,
            business_requirements=json.dumps(project.business_requirements),
            estimated_duration_days=execution.plan.estimated_duration_days if execution.plan else None,
            estimated_cost=execution.plan.total_cost_estimate if execution.plan else None
        )
        
        return db_project
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    limit: int = 100,
    db: Session = Depends(get_db_session)
):
    """Lista wszystkich projektów"""
    repo = ProjectRepository(db)
    projects = repo.get_all(limit=limit)
    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: Session = Depends(get_db_session)
):
    """Pobierz projekt po ID"""
    repo = ProjectRepository(db)
    project = repo.get_by_project_id(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    update: ProjectUpdate,
    db: Session = Depends(get_db_session)
):
    """Zaktualizuj projekt"""
    repo = ProjectRepository(db)
    
    if update.status:
        project = repo.update_status(project_id, update.status, update.progress)
    else:
        project = repo.get_by_project_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if update.quality_score is not None:
            project.quality_score = update.quality_score
            db.commit()
            db.refresh(project)
    
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    db: Session = Depends(get_db_session)
):
    """Usuń projekt"""
    repo = ProjectRepository(db)
    project = repo.get_by_project_id(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    repo.delete(project.id)
    return None


@router.post("/{project_id}/execute", response_model=ProjectExecutionResponse)
async def execute_project(
    project_id: str,
    request: ProjectExecutionRequest,
    core: AgentZeroCore = Depends(get_core_engine),
    pm: ProjectManager = Depends(get_project_manager),
    db: Session = Depends(get_db_session)
):
    """
    Wykonaj projekt przez zespół agentów
    
    - **auto_advance**: Czy automatycznie przejść przez wszystkie fazy
    - **perform_post_mortem**: Czy wykonać analizę post-mortem
    """
    # Pobierz projekt z core engine
    if project_id not in core.active_projects:
        raise HTTPException(status_code=404, detail="Project not found in active projects")
    
    execution = core.active_projects[project_id]
    
    # Wykonaj projekt
    result = pm.execute_project(execution, auto_advance=request.auto_advance)
    
    # Post-mortem
    if request.perform_post_mortem:
        core.complete_project(project_id, perform_post_mortem=True)
    
    # Update w bazie
    repo = ProjectRepository(db)
    repo.update_status(project_id, 'completed', progress=1.0)
    
    return {
        "project_id": project_id,
        "status": "completed",
        "total_duration_hours": result['total_duration_hours'],
        "average_quality": result['average_quality'],
        "phases_completed": result['phases_completed']
    }


@router.get("/{project_id}/status")
async def get_project_status(
    project_id: str,
    core: AgentZeroCore = Depends(get_core_engine)
):
    """Pobierz szczegółowy status projektu"""
    status = core.get_project_status(project_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return status
