"""
Agents API Routes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from persistence import get_db_session, AgentRepository
from api.schemas import AgentResponse, AgentPerformanceUpdate

router = APIRouter()


@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    project_id: int = None,
    agent_type: str = None,
    limit: int = 100,
    db: Session = Depends(get_db_session)
):
    """
    Lista agentów z filtrami
    
    - **project_id**: Filtruj po projekcie
    - **agent_type**: Filtruj po typie (architect, backend, etc.)
    - **limit**: Maksymalna liczba wyników
    """
    repo = AgentRepository(db)
    
    if project_id:
        agents = repo.get_by_project(project_id)
    elif agent_type:
        agents = repo.get_by_type(agent_type)
    else:
        agents = repo.get_all(limit=limit)
    
    return agents


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: Session = Depends(get_db_session)
):
    """Pobierz agenta po ID"""
    repo = AgentRepository(db)
    agent = repo.get_by_agent_id(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent


@router.patch("/{agent_id}/performance", response_model=AgentResponse)
async def update_agent_performance(
    agent_id: str,
    update: AgentPerformanceUpdate,
    db: Session = Depends(get_db_session)
):
    """
    Zaktualizuj performance metrics agenta
    
    - **tasks_completed**: Liczba ukończonych zadań
    - **avg_quality_score**: Średnia jakość (0.0-1.0)
    - **success_rate**: Wskaźnik sukcesu (0.0-1.0)
    """
    repo = AgentRepository(db)
    
    agent = repo.update_performance(
        agent_id=agent_id,
        tasks_completed=update.tasks_completed,
        avg_quality=update.avg_quality_score,
        success_rate=update.success_rate
    )
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent
