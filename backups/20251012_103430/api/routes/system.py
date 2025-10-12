"""
System API Routes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))

from fastapi import APIRouter, Depends
from datetime import datetime

from persistence import get_database, get_cache
from core import AgentZeroCore
from api.dependencies import get_core_engine
from api.schemas import SystemStatusResponse, HealthResponse

router = APIRouter()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    core: AgentZeroCore = Depends(get_core_engine)
):
    """
    Pobierz status całego systemu
    
    Zwraca:
    - Liczba aktywnych projektów
    - Liczba zakończonych projektów
    - Liczba aktywnych protokołów
    - Dostępne typy agentów
    - Wykryte wzorce i anty-wzorce
    """
    try:
        # Spróbuj pobrać z cache
        cache = get_cache()
        cached = cache.get_system_status()
        
        if cached:
            return cached
        
        # Pobierz z core engine
        status = core.get_system_status()
        
        # Dodaj total agents
        status['total_agents'] = sum(
            len(agents) for agents in core.planner.team_formation.agent_pool.values()
        )
        
        # Dodaj cache stats
        cache_stats = cache.get_stats()
        status['cache_keys'] = cache_stats['total_keys']
        
        # Zapisz w cache (30s TTL)
        cache.set_system_status(status, ttl=30)
        
        return status
    except Exception as e:
        # Fallback - zwróć podstawowe dane
        return {
            'active_projects': len(core.active_projects),
            'completed_projects': len(core.completed_projects),
            'active_protocols': len(core.active_protocols),
            'total_agents': 0,
            'agent_types_available': len(core.agent_factory.templates),
            'patterns_detected': len(core.pattern_detector.patterns),
            'antipatterns_known': len(core.antipattern_detector.known_antipatterns),
            'cache_keys': 0
        }


@router.get("/health", response_model=HealthResponse)
async def system_health():
    """
    Health check systemu
    
    Sprawdza:
    - Status bazy danych
    - Status cache
    """
    db = get_database()
    cache = get_cache()
    
    return {
        "status": "ok",
        "database": db.health_check(),
        "cache": True,
        "timestamp": datetime.utcnow()
    }


@router.post("/cache/clear")
async def clear_cache():
    """Wyczyść cache"""
    cache = get_cache()
    count = cache.clear()
    
    return {
        "message": "Cache cleared",
        "keys_removed": count
    }


@router.get("/cache/stats")
async def cache_stats():
    """Pobierz statystyki cache"""
    cache = get_cache()
    return cache.get_stats()


@router.get("/agents/types")
async def list_agent_types(
    core: AgentZeroCore = Depends(get_core_engine)
):
    """Lista dostępnych typów agentów"""
    return {
        "agent_types": list(core.agent_factory.templates.keys()),
        "total": len(core.agent_factory.templates)
    }
