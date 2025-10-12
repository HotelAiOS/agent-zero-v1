"""
Bus wiadomości RabbitMQ
Obsługa automatycznych retry i reconnect - odporność na timeouty i zamknięcia kanału.
"""
import asyncio
import logging
import json
from typing import Dict, Any, Callable, Optional
import aio_pika
from aio_pika import Message, ExchangeType
from aio_pika.abc import AbstractRobustConnection
import aiormq

logger = logging.getLogger(__name__)

class MessageBus:
    """
    Komunikacja RabbitMQ z automatycznym retry, reconnect i obsługą długich operacji.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "agent",
        password: str = "agent-pass",
        exchange_name: str = "agent_exchange"
    ):
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
        Połączenie do RabbitMQ. Automatycznie próbuje połączenie do skutku.
        Heartbeat 3600s – obsługa długich operacji AI.
        """
        if self._is_connected:
            logger.info("Już połączone z RabbitMQ")
            return

        for attempt in range(5):
            try:
                self.connection = await aio_pika.connect_robust(
                    host=self.host,
                    port=self.port,
                    login=self.username,
                    password=self.password,
                    heartbeat=3600,
                    connection_attempts=5,
                    retry_delay=3.0,
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=10)
                self.exchange = await self.channel.declare_exchange(
                    self.exchange_name,
                    ExchangeType.TOPIC,
                    durable=True
                )
                self._is_connected = True
                logger.info("✅ Połączono z RabbitMQ")
                return
            except Exception as e:
                logger.error(f"❌ Próba {attempt+1}/5: Błąd połączenia RabbitMQ: {e}")
                await asyncio.sleep(2)
        raise RuntimeError("Nie udało się połączyć z RabbitMQ po 5 próbach")

    async def publish(self, routing_key: str, message: Dict[str, Any]):
        """
        Publikacja wiadomości z retry/reconnect. Obsługa zamkniętych kanałów!
        """
        max_retries = 5
        attempt = 0

        while attempt < max_retries:
            if not self._is_connected:
                await self.connect()
            try:
                body = json.dumps(message).encode()
                msg = Message(
                    body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                )
                await self.exchange.publish(
                    msg,
                    routing_key
                )
                msg_type = message.get('type', 'unknown')
                logger.info(f"📤 Published to {routing_key}: {msg_type}")
                return
            except aiormq.exceptions.ChannelInvalidStateError:
                logger.warning("⚠️ Channel zamknięty: reconnect i retry")
                self._is_connected = False
                await self.connect()
                attempt += 1
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"❌ Błąd publish: {e}")
                attempt += 1
                await asyncio.sleep(1)
        raise RuntimeError("Publikacja RabbitMQ nieudana po wielu próbach")

    async def subscribe(self, routing_key: str, handler: Callable):
        """
        Subskrypcja do patternu – obsługa reconnect przy błędzie.
        """
        if not self._is_connected:
            await self.connect()

        queue_name = f"agent_{routing_key.replace('.', '_').replace('*', 'all')}"

        def _is_channel_open():
            return self.channel and not self.channel.is_closed

        for attempt in range(3):
            try:
                queue = await self.channel.declare_queue(
                    queue_name,
                    durable=True,
                    auto_delete=False
                )
                await queue.bind(self.exchange, routing_key)

                async def _wrapped_handler(message: aio_pika.IncomingMessage):
                    try:
                        async with message.process():
                            data = json.loads(message.body.decode())
                            try:
                                await handler(data)
                                logger.info(f"✅ Wiadomość obsłużona: {routing_key}")
                            except Exception as eh:
                                logger.error(f"❌ Błąd handlera: {eh}")
                    except aiormq.exceptions.ChannelInvalidStateError:
                        logger.warning("⚠️ Channel zamknięty w czasie konsumpcji, reconnect")
                        self._is_connected = False
                        await self.connect()
                        await asyncio.sleep(1)
                await queue.consume(_wrapped_handler)
                logger.info(f"👂 Subskrypcja aktywna: {routing_key} (kolejka: {queue_name})")
                return
            except aiormq.exceptions.ChannelInvalidStateError:
                logger.warning("⚠️ Channel zamknięty przy subskrypcji, reconnect")
                self._is_connected = False
                await self.connect()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"❌ Błąd subskrypcji: {e}")
                await asyncio.sleep(1)
        raise RuntimeError("Subskrypcja do RabbitMQ nieudana po kilku próbach")

    async def close(self):
        """
        Zamykaj połączenie (graceful shutdown).
        """
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            self._is_connected = False
            logger.info("🔌 Rozłączono RabbitMQ")


# Singleton
message_bus = MessageBus()

# Helpery (nie zmieniane)
async def publish_agent_message(agent_type: str, message_type: str, data: Dict[str, Any]):
    routing_key = f"agent.{agent_type}.{message_type}"
    message = {
        "type": message_type,
        "data": data,
        "timestamp": asyncio.get_event_loop().time()
    }
    await message_bus.publish(routing_key, message)

async def subscribe_to_agent_messages(agent_type: str, handler: Callable):
    routing_key = f"agent.{agent_type}.*"
    await message_bus.subscribe(routing_key, handler)
