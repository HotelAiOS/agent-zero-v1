"""
Integration Test for Execution Engine (Phase 1)

Tests:
- ProjectOrchestrator
- AgentExecutor
- CodeGenerator
- End-to-end project execution
"""

import asyncio
import sys
from pathlib import Path

# Dodaj shared do PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "shared"))

from execution.project_orchestrator import ProjectOrchestrator
from execution.agent_executor import AgentExecutor
from execution.code_generator import CodeGenerator
from orchestration.task_decomposer import TaskDecomposer
from orchestration.team_builder import TeamBuilder
from agent_factory.factory import AgentFactory
from agent_factory.lifecycle import AgentLifecycleManager
from llm.llm_factory import LLMFactory
from knowledge.neo4j_client import Neo4jClient


async def test_execution_engine():
    """Test pełnego cyklu wykonania projektu."""
    print("=" * 80)
    print("🧪 EXECUTION ENGINE INTEGRATION TEST")
    print("=" * 80)
    
    # Konfiguracja
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "neo4j123"
    
    # Wymagania testowe - prosty projekt
    requirements = """
    Create a simple REST API for a task management system.
    
    Requirements:
    - Python FastAPI backend
    - SQLite database
    - CRUD operations for tasks
    - Basic error handling
    
    Tasks should have:
    - ID (auto-generated)
    - Title (string)
    - Description (text)
    - Status (todo, in_progress, done)
    - Created date
    """
    
    try:
        # 1. Inicjalizacja komponentów
        print("\n📦 Initializing components...")
        
        # Neo4j
        neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
        await neo4j_client.connect()
        print("  ✅ Neo4j connected")
        
        # LLM Factory
        llm_factory = LLMFactory()
        print("  ✅ LLM Factory initialized")
