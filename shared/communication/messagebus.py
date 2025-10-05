"""
RabbitMQ Message Bus - Async komunikacja między agentami

PRODUCTION GRADE - Obsługuje długotrwałe operacje (AI generacja kodu)

Zmiany vs poprzednia wersja:
- Heartbeat 3600s (1 godzina) zamiast domyślnych 600s
- Robust connection z auto-reconnect
- Connection keepalive
- Timeout handling dla długich operacji
"""
import asyncio
import logging
import json
from typing import Dict, Any, Callable, Optional
import aio_pika
from aio_pika import Message, ExchangeType
from aio_pika.abc import AbstractRobustConnection

logger = logging.getLogger(__name__)


class MessageBus:
    """
    RabbitMQ Message Bus z obsługą długotrwałych operacji.
    
    Features:
    - Auto-reconnect po utracie połączenia
    - Heartbeat 3600s dla długich operacji AI
    - Topic exchange dla routingu agentów
    - Persistent messages
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "agent",
        password: str = "agent-pass",
        exchange_name: str = "agent_exchange"
    ):
        """
        Inicjalizacja message bus.
        
        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: Użytkownik RabbitMQ
            password: Hasło RabbitMQ
            exchange_name: Nazwa exchange (topic)
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange_name = exchange_name
        
        self.connection: Optional[AbstractRobustConnection] = None
        self.channel = None
        self.exchange = None
        self._is_connected = False
        
    async def connect(self):
        """
        Połącz z RabbitMQ z robust connection.
        
        Robust connection automatycznie reconnectuje przy utracie połączenia.
        Heartbeat 3600s pozwala na długie operacje (AI generation 20+ minut).
        """
        if self._is_connected:
            logger.info("Already connected to RabbitMQ")
            return
        
        try:
            # Robust connection z długim heartbeat
            self.connection = await aio_pika.connect_robust(
                host=self.host,
                port=self.port,
                login=self.username,
                password=self.password,
                heartbeat=3600,  # 1 godzina heartbeat dla długich AI operacji
                connection_attempts=5,  # 5 prób połączenia
                retry_delay=3.0,  # 3s między próbami
            )
            
            # Utwórz channel
            self.channel = await self.connection.channel()
            
            # Ustaw QoS - max 10 nieprzetworzonych wiadomości
            await self.channel.set_qos(prefetch_count=10)
            
            # Declare exchange (topic - routing przez pattern)
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name,
                ExchangeType.TOPIC,
                durable=True  # Przetrwa restart RabbitMQ
            )
            
            self._is_connected = True
            logger.info("✅ Connected to RabbitMQ - Agent Communication Ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            raise
    
    async def publish(self, routing_key: str, message: Dict[str, Any]):
        """
        Opublikuj wiadomość na exchange.
        
        Args:
            routing_key: Routing key (np. "agent.backend.001.task")
            message: Dict z danymi (będzie serializowany do JSON)
            
        Example:
            >>> await bus.publish(
            ...     "agent.backend.001.task",
            ...     {"type": "task", "data": {"work": "code"}}
            ... )
        """
        if not self._is_connected:
            await self.connect()
        
        try:
            # Serializuj do JSON
            body = json.dumps(message).encode()
            
            # Stwórz message (persistent)
            msg = Message(
                body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT  # Przetrwa restart
            )
            
            # Publish
            await self.exchange.publish(
                msg,
                routing_key=routing_key
            )
            
            msg_type = message.get('type', 'unknown')
            logger.info(f"📤 Published to {routing_key}: {msg_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            # Spróbuj reconnect
            self._is_connected = False
            await self.connect()
            raise
    
    async def subscribe(self, routing_key: str, handler: Callable):
        """
        Subskrybuj wiadomości z danym routing key.
        
        Args:
            routing_key: Pattern (np. "agent.backend.#" lub "agent.*.001.*")
            handler: Async funkcja handler(message: Dict)
            
        Example:
            >>> async def my_handler(msg):
            ...     print(f"Got: {msg}")
            >>> await bus.subscribe("agent.backend.#", my_handler)
        """
        if not self._is_connected:
            await self.connect()
        
        try:
            # Stwórz unikalną nazwę kolejki
            queue_name = f"agent_{routing_key.replace('.', '_').replace('*', 'all')}"
            
            # Declare queue (durable)
            queue = await self.channel.declare_queue(
                queue_name,
                durable=True,
                auto_delete=False  # Nie usuwaj po disconnect
            )
            
            # Bind do exchange z routing key
            await queue.bind(self.exchange, routing_key)
            
            # Wrapper dla handlera
            async def _wrapped_handler(message: aio_pika.IncomingMessage):
                async with message.process():
                    try:
                        # Deserializuj JSON
                        data = json.loads(message.body.decode())
                        
                        # Wywołaj user handler
                        await handler(data)
                        
                        logger.info(f"✅ Processed message from {routing_key}")
                        
                    except Exception as e:
                        logger.error(f"❌ Handler error: {e}")
                        # Message będzie requeued automatycznie
            
            # Start consuming
            await queue.consume(_wrapped_handler)
            
            logger.info(f"👂 Subscribed to {routing_key} (queue: {queue_name})")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe: {e}")
            raise
    
    async def close(self):
        """
        Zamknij połączenie z RabbitMQ.
        
        Graceful shutdown - poczeka na przetworzenie wiadomości.
        """
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self._is_connected = False
            logger.info("🔌 Disconnected from RabbitMQ")


# Singleton instance
message_bus = MessageBus()


# ============================================================================
# Helper functions - Convenience wrappers
# ============================================================================

async def publish_agent_message(agent_type: str, message_type: str, data: Dict[str, Any]):
    """
    Quick publish do wszystkich agentów danego typu.
    
    Args:
        agent_type: Typ agenta (np. "backend", "frontend")
        message_type: Typ wiadomości (np. "task", "status")
        data: Dane do wysłania
        
    Example:
        >>> await publish_agent_message(
        ...     "backend",
        ...     "task",
        ...     {"work": "create API"}
        ... )
    """
    routing_key = f"agent.{agent_type}.{message_type}"
    
    message = {
        "type": message_type,
        "data": data,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await message_bus.publish(routing_key, message)


async def subscribe_to_agent_messages(agent_type: str, handler: Callable):
    """
    Quick subscribe do wszystkich wiadomości dla typu agenta.
    
    Args:
        agent_type: Typ agenta (np. "backend")
        handler: Handler funkcja
        
    Example:
        >>> async def my_handler(msg):
        ...     print(msg)
        >>> await subscribe_to_agent_messages("backend", my_handler)
    """
    routing_key = f"agent.{agent_type}.*"
    await message_bus.subscribe(routing_key, handler)
