"""
Tasks API Routes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))

from fastapi import APIRouter, Depends, HTTPException, status as http_status
from sqlalchemy.orm import Session
from typing import List

from persistence import get_db_session, TaskRepository
from api.schemas import TaskCreate, TaskResponse, TaskUpdate

router = APIRouter()


@router.post("/", response_model=TaskResponse, status_code=http_status.HTTP_201_CREATED)
async def create_task(
    task: TaskCreate,
    project_id: int,
    agent_id: int = None,
    db: Session = Depends(get_db_session)
):
    """
    Utwórz nowe zadanie
    
    - **task_name**: Nazwa zadania
    - **task_type**: Typ (design, implementation, testing, etc.)
    - **complexity**: Poziom złożoności (1-10)
    - **priority**: Priorytet (1-5)
    - **estimated_hours**: Szacowany czas
    """
    repo = TaskRepository(db)
    
    db_task = repo.create(
        task_id=f"task_{task.task_name.lower().replace(' ', '_')}",
        task_name=task.task_name,
        description=task.description,
        task_type=task.task_type,
        complexity=task.complexity,
        priority=task.priority,
        estimated_hours=task.estimated_hours,
        project_id=project_id,
        agent_id=agent_id
    )
    
    return db_task


@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    project_id: int = None,
    agent_id: int = None,
    status: str = None,
    limit: int = 100,
    db: Session = Depends(get_db_session)
):
    """
    Lista zadań z filtrami
    
    - **project_id**: Filtruj po projekcie
    - **agent_id**: Filtruj po agencie
    - **status**: Filtruj po statusie (pending, in_progress, completed)
    """
    repo = TaskRepository(db)
    
    if status:
        tasks = repo.get_by_status(status, project_id)
    elif project_id:
        tasks = repo.get_by_project(project_id)
    elif agent_id:
        tasks = repo.get_by_agent(agent_id)
    else:
        tasks = repo.get_all(limit=limit)
    
    return tasks


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: Session = Depends(get_db_session)
):
    """Pobierz zadanie po ID"""
    repo = TaskRepository(db)
    task = repo.get_by_task_id(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    update: TaskUpdate,
    db: Session = Depends(get_db_session)
):
    """
    Zaktualizuj zadanie
    
    - **status**: Nowy status (pending, in_progress, completed, failed)
    - **actual_hours**: Faktyczny czas wykonania
    - **quality_score**: Ocena jakości (0.0-1.0)
    """
    repo = TaskRepository(db)
    
    task = repo.update_status(
        task_id=task_id,
        status=update.status if update.status else None,
        quality_score=update.quality_score
    )
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if update.actual_hours:
        task.actual_hours = update.actual_hours
        db.commit()
        db.refresh(task)
    
    return task
