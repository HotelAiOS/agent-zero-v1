"""
Intelligent Agent - Agent z AI capabilities i komunikacją RabbitMQ

Agent który:
- Rejestruje się w Agent Registry
- Komunikuje się przez RabbitMQ
- Ma handlers dla różnych typów wiadomości
- Może delegować taski do innych agentów
- Może broadcast'ować wiadomości
- Używa AI Brain do generowania kodu
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from messagebus import message_bus
from agent_registry import agent_registry, AgentInfo
from ai_agent_brain import ai_agent_brain

logger = logging.getLogger(__name__)


class IntelligentAgent:
    """
    Agent z AI capabilities i komunikacją RabbitMQ.
    
    Example:
        >>> agent = IntelligentAgent(
        ...     agent_id="backend_001",
        ...     agent_type="backend",
        ...     capabilities=["python", "fastapi"]
        ... )
        >>> await agent.start()
        >>> await agent.send_to_agent("frontend_001", "task", {"work": "data"})
        >>> await agent.stop()
    """
    
    def __init__(
        self, 
        agent_id: str,
        agent_type: str,
        capabilities: list[str],
        ai_brain=None
    ):
        """
        Inicjalizacja agenta.
        
        Args:
            agent_id: Unikalny ID agenta (np. "backend_001")
            agent_type: Typ agenta (np. "backend", "frontend")
            capabilities: Lista umiejętności (np. ["python", "fastapi"])
            ai_brain: Opcjonalny AI Brain (domyślnie ai_agent_brain)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        
        # Użyj przekazanego ai_brain LUB domyślnego
        self.ai_brain = ai_brain if ai_brain else ai_agent_brain
        
        self.status = "offline"
        
        # Handlers dla różnych typów wiadomości
        self.message_handlers: Dict[str, Callable] = {}
        
        logger.info(f"🤖 Agent created: {agent_id} ({agent_type})")
        
    async def start(self):
        """
        Uruchom agenta - połącz z RabbitMQ i zarejestruj w registry.
        
        Proces:
        1. Połącz z RabbitMQ
        2. Zarejestruj w Agent Registry
        3. Subscribe do własnej kolejki
        4. Zmień status na "online"
        """
        logger.info(f"🚀 Starting agent {self.agent_id}...")
        
        # 1. Połącz z RabbitMQ
        await message_bus.connect()
        logger.info(f"   ✅ Connected to RabbitMQ")
        
        # 2. Zarejestruj w registry
        agent_info = AgentInfo(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            status="online",
            last_seen=datetime.now(),
            queue_name=f"agent_{self.agent_type}_{self.agent_id}"
        )
        await agent_registry.register_agent(agent_info)
        logger.info(f"   ✅ Registered in Agent Registry")
        
        # 3. Subscribe do własnej kolejki
        routing_key = f"agent.{self.agent_type}.{self.agent_id}.#"
        await message_bus.subscribe(
            routing_key=routing_key,
            handler=self._handle_message
        )
        logger.info(f"   ✅ Subscribed to {routing_key}")
        
        # 4. Status online
        self.status = "online"
        logger.info(f"✅ Agent {self.agent_id} is now ONLINE!\n")
        
    async def stop(self):
        """
        Zatrzymaj agenta - wyrejestruj i rozłącz.
        
        Proces:
        1. Zmień status na "offline"
        2. Wyrejestruj z Agent Registry
        3. Rozłącz z RabbitMQ
        """
        logger.info(f"🛑 Stopping agent {self.agent_id}...")
        
        self.status = "offline"
        await agent_registry.unregister_agent(self.agent_id)
        await message_bus.close()
        
        logger.info(f"✅ Agent {self.agent_id} stopped\n")
        
    def register_handler(self, message_type: str, handler: Callable):
        """
        Zarejestruj handler dla typu wiadomości.
        
        Args:
            message_type: Typ wiadomości (np. "task_request", "collaboration")
            handler: Async funkcja która obsłuży wiadomość
            
        Example:
            >>> async def handle_task(msg):
            ...     print(f"Got task: {msg['data']}")
            >>> agent.register_handler("task_request", handle_task)
        """
        self.message_handlers[message_type] = handler
        logger.info(f"📝 Handler registered for '{message_type}'")
        
    async def _handle_message(self, message: Dict[str, Any]):
        """
        Wewnętrzna funkcja - obsłuż przychodzącą wiadomość.
        
        Args:
            message: Wiadomość z RabbitMQ
        """
        msg_type = message.get("type")
        from_agent = message.get("from", "unknown")
        
        logger.info(f"📨 {self.agent_id} received '{msg_type}' from {from_agent}")
        
        # Znajdź handler dla tego typu
        if msg_type in self.message_handlers:
            try:
                await self.message_handlers[msg_type](message)
            except Exception as e:
                logger.error(f"❌ Error handling {msg_type}: {e}")
        else:
            logger.warning(f"⚠️  No handler for message type: {msg_type}")
    
    async def send_to_agent(
        self, 
        target_agent_id: str, 
        message_type: str, 
        data: Dict[str, Any]
    ) -> bool:
        """
        Wyślij wiadomość do konkretnego agenta.
        
        Args:
            target_agent_id: ID agenta docelowego (np. "frontend_001")
            message_type: Typ wiadomości (np. "task_request")
            data: Dane wiadomości (dowolny dict)
            
        Returns:
            True jeśli wysłano, False jeśli agent nie znaleziony
            
        Example:
            >>> await agent.send_to_agent(
            ...     "backend_001",
            ...     "task_request",
            ...     {"description": "Create API endpoint"}
            ... )
        """
        # Znajdź target agent w registry
        target = await agent_registry.get_agent_by_id(target_agent_id)
        
        if not target:
            logger.error(f"❌ Agent {target_agent_id} not found in registry")
            return False
        
        # Routing key: agent.backend.backend_001.task_request
        routing_key = f"agent.{target.agent_type}.{target_agent_id}.{message_type}"
        
        # Przygotuj wiadomość
        message = {
            "type": message_type,
            "from": self.agent_id,
            "to": target_agent_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Wyślij przez RabbitMQ
        await message_bus.publish(routing_key, message)
        
        logger.info(f"📤 {self.agent_id} → {target_agent_id}: {message_type}")
        return True
    
    async def broadcast(self, message_type: str, data: Dict[str, Any]):
        """
        Broadcast wiadomość do wszystkich agentów danego typu.
        
        Args:
            message_type: Typ wiadomości
            data: Dane do wysłania
            
        Example:
            >>> await agent.broadcast("status_update", {"status": "busy"})
        """
        routing_key = f"agent.{self.agent_type}.broadcast.{message_type}"
        
        message = {
            "type": message_type,
            "from": self.agent_id,
            "broadcast": True,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await message_bus.publish(routing_key, message)
        logger.info(f"📢 {self.agent_id} broadcast: {message_type}")
        
    async def delegate_task(
        self, 
        capability: str, 
        task: Dict[str, Any]
    ) -> Optional[str]:
        """
        Deleguj task do agenta który ma daną capability.
        
        Args:
            capability: Wymagana umiejętność (np. "python", "react")
            task: Opis taska do wykonania
            
        Returns:
            ID agenta który dostał task, lub None jeśli nie znaleziono
            
        Example:
            >>> agent_id = await agent.delegate_task(
            ...     "api_development",
            ...     {"description": "Create user auth API"}
            ... )
        """
        # Znajdź agenta z capability
        target_agent = await agent_registry.find_agent_by_capability(capability)
        
        if not target_agent:
            logger.error(f"❌ No agent found with capability: {capability}")
            return None
        
        # Wyślij task
        await self.send_to_agent(
            target_agent_id=target_agent.agent_id,
            message_type="task_delegation",
            data=task
        )
        
        logger.info(f"✅ Task delegated to {target_agent.agent_id}")
        return target_agent.agent_id
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Pobierz status agenta.
        
        Returns:
            Dict ze statusem agenta
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "capabilities": self.capabilities,
            "handlers": list(self.message_handlers.keys())
        }
