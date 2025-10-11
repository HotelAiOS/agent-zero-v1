"""
Agent Registry - Service Discovery dla Multi-Agent System

Rejestr wszystkich agentów w systemie. Umożliwia:
- Rejestrację agentów przy starcie
- Znajdowanie agentów po capabilities
- Znajdowanie agentów po typie
- Śledzenie statusu agentów (online/busy/offline)
"""
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """
    Informacje o agencie w systemie.
    
    Attributes:
        agent_id: Unikalny identyfikator agenta (np. "backend_001")
        agent_type: Typ agenta (np. "backend", "frontend", "devops")
        capabilities: Lista umiejętności (np. ["python", "fastapi", "docker"])
        status: Status agenta ("online", "busy", "offline")
        last_seen: Ostatnia aktywność agenta
        queue_name: Nazwa kolejki RabbitMQ agenta
    """
    agent_id: str
    agent_type: str
    capabilities: List[str]
    status: str
    last_seen: datetime
    queue_name: str


class AgentRegistry:
    """
    Rejestr wszystkich agentów w systemie.
    
    Singleton przechowujący informacje o wszystkich aktywnych agentach.
    Thread-safe dzięki asyncio.Lock.
    """
    
    def __init__(self):
        """Inicjalizacja pustego rejestru."""
        self._agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()
        logger.info("📖 Agent Registry initialized")
        
    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """
        Zarejestruj nowego agenta w systemie.
        
        Args:
            agent_info: Kompletne informacje o agencie
            
        Returns:
            True jeśli rejestracja się powiodła
        """
        async with self._lock:
            self._agents[agent_info.agent_id] = agent_info
            logger.info(
                f"✅ Agent registered: {agent_info.agent_id} "
                f"({agent_info.agent_type}) - "
                f"capabilities: {agent_info.capabilities}"
            )
            return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Wyrejestruj agenta z systemu (np. przy shutdown).
        
        Args:
            agent_id: ID agenta do wyrejestrowania
            
        Returns:
            True jeśli agent został znaleziony i usunięty
        """
        async with self._lock:
            if agent_id in self._agents:
                agent_type = self._agents[agent_id].agent_type
                del self._agents[agent_id]
                logger.info(f"👋 Agent unregistered: {agent_id} ({agent_type})")
                return True
            else:
                logger.warning(f"⚠️  Agent not found for unregister: {agent_id}")
                return False
    
    async def find_agent_by_capability(self, capability: str) -> Optional[AgentInfo]:
        """
        Znajdź pierwszego dostępnego agenta z daną umiejętnością.
        
        Args:
            capability: Nazwa umiejętności (np. "python", "react", "docker")
            
        Returns:
            AgentInfo pierwszego znalezionego agenta lub None
        """
        async with self._lock:
            for agent in self._agents.values():
                if capability in agent.capabilities and agent.status == "online":
                    logger.debug(f"🔍 Found agent with '{capability}': {agent.agent_id}")
                    return agent
            
            logger.warning(f"⚠️  No online agent found with capability: {capability}")
            return None
    
    async def find_agents_by_type(self, agent_type: str) -> List[AgentInfo]:
        """
        Znajdź wszystkich online agentów danego typu.
        
        Args:
            agent_type: Typ agenta (np. "backend", "frontend")
            
        Returns:
            Lista AgentInfo dla wszystkich znalezionych agentów
        """
        async with self._lock:
            agents = [
                agent for agent in self._agents.values() 
                if agent.agent_type == agent_type and agent.status == "online"
            ]
            logger.debug(f"🔍 Found {len(agents)} agents of type '{agent_type}'")
            return agents
    
    async def get_all_agents(self) -> List[AgentInfo]:
        """
        Pobierz listę wszystkich agentów (niezależnie od statusu).
        
        Returns:
            Lista wszystkich AgentInfo w rejestrze
        """
        async with self._lock:
            return list(self._agents.values())
    
    async def update_agent_status(self, agent_id: str, status: str) -> bool:
        """
        Zaktualizuj status agenta i timestamp.
        
        Args:
            agent_id: ID agenta
            status: Nowy status ("online", "busy", "offline")
            
        Returns:
            True jeśli agent został znaleziony i zaktualizowany
        """
        async with self._lock:
            if agent_id in self._agents:
                old_status = self._agents[agent_id].status
                self._agents[agent_id].status = status
                self._agents[agent_id].last_seen = datetime.now()
                logger.info(
                    f"🔄 Agent {agent_id} status: {old_status} → {status}"
                )
                return True
            else:
                logger.warning(f"⚠️  Agent not found for status update: {agent_id}")
                return False
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Pobierz informacje o konkretnym agencie.
        
        Args:
            agent_id: ID agenta
            
        Returns:
            AgentInfo lub None jeśli nie znaleziono
        """
        async with self._lock:
            return self._agents.get(agent_id)
    
    async def get_stats(self) -> Dict[str, int]:
        """
        Pobierz statystyki rejestru.
        
        Returns:
            Słownik ze statystykami (total, online, busy, offline)
        """
        async with self._lock:
            stats = {
                "total": len(self._agents),
                "online": sum(1 for a in self._agents.values() if a.status == "online"),
                "busy": sum(1 for a in self._agents.values() if a.status == "busy"),
                "offline": sum(1 for a in self._agents.values() if a.status == "offline")
            }
            return stats


# Singleton instance - jedna instancja dla całego systemu
agent_registry = AgentRegistry()
