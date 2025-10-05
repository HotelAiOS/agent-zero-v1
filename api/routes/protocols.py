"""
Protocols API Routes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))

from fastapi import APIRouter, Depends, HTTPException, status as http_status
from sqlalchemy.orm import Session
from typing import List
import json

from persistence import get_db_session, ProtocolRepository
from core import AgentZeroCore
from api.dependencies import get_core_engine
from api.schemas import ProtocolCreate, ProtocolResponse

router = APIRouter()


@router.post("/", response_model=ProtocolResponse, status_code=http_status.HTTP_201_CREATED)
async def create_protocol(
    protocol: ProtocolCreate,
    project_id: str,
    core: AgentZeroCore = Depends(get_core_engine),
    db: Session = Depends(get_db_session)
):
    """
    Utwórz nowy protokół komunikacji
    
    - **protocol_type**: Typ (code_review, consensus, problem_solving, etc.)
    - **context**: Kontekst protokołu (JSON)
    """
    try:
        # Utwórz protokół w core engine
        proto = core.start_protocol(
            project_id=project_id,
            protocol_type=protocol.protocol_type,
            context=protocol.context
        )
        
        if not proto:
            raise HTTPException(status_code=400, detail="Failed to create protocol")
        
        # Zapisz do bazy
        repo = ProtocolRepository(db)
        db_protocol = repo.create(
            protocol_id=proto.protocol_id,
            protocol_type=proto.protocol_type.value,
            initiator=proto.initiated_by,
            participants=json.dumps(list(proto.participants)),
            status=proto.status.value,
            context=json.dumps(protocol.context),
            messages_count=len(proto.messages),
            project_id=int(project_id.split('_')[1]) if '_' in project_id else None
        )
        
        return db_protocol
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ProtocolResponse])
async def list_protocols(
    project_id: int = None,
    protocol_type: str = None,
    active_only: bool = False,
    limit: int = 100,
    db: Session = Depends(get_db_session)
):
    """
    Lista protokołów z filtrami
    
    - **project_id**: Filtruj po projekcie
    - **protocol_type**: Filtruj po typie
    - **active_only**: Tylko aktywne protokoły
    """
    repo = ProtocolRepository(db)
    
    if active_only:
        protocols = repo.get_active()
    elif project_id:
        protocols = repo.get_by_project(project_id)
    elif protocol_type:
        protocols = repo.get_by_type(protocol_type)
    else:
        protocols = repo.get_all(limit=limit)
    
    return protocols


@router.get("/{protocol_id}", response_model=ProtocolResponse)
async def get_protocol(
    protocol_id: str,
    db: Session = Depends(get_db_session)
):
    """Pobierz protokół po ID"""
    repo = ProtocolRepository(db)
    protocol = repo.get_by_protocol_id(protocol_id)
    
    if not protocol:
        raise HTTPException(status_code=404, detail="Protocol not found")
    
    return protocol
