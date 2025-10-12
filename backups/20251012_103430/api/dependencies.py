"""
API Dependencies
FastAPI dependencies - unika circular imports
"""

from fastapi import HTTPException

# Global instances
core_engine = None
project_manager = None


def set_core_engine(engine):
    """Set global core engine"""
    global core_engine
    core_engine = engine


def set_project_manager(pm):
    """Set global project manager"""
    global project_manager
    project_manager = pm


def get_core_engine():
    """Get core engine instance"""
    if core_engine is None:
        raise HTTPException(status_code=503, detail="Core engine not initialized")
    return core_engine


def get_project_manager():
    """Get project manager instance"""
    if project_manager is None:
        raise HTTPException(status_code=503, detail="Project manager not initialized")
    return project_manager
