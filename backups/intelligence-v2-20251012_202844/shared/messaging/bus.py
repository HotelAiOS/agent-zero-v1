import pika
import logging
from typing import Optional, Callable
from dataclasses import dataclass

# Verbose logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('./agent_communications.log')]
)
logger = logging.getLogger('MessageBus')

@dataclass
class BusConfig:
    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    heartbeat: int = 600
    connection_timeout: int = 30

class MessageBus:
    def __init__(self, config: Optional[BusConfig] = None):
        self.config = config or BusConfig()
        self.connection = None
        self.channel = None
        self._is_connected = False
        logger.info(f"MessageBus initialized (host={self.config.host})")

    # connect/disconnect unchanged...

    async def publish(self, message):
        logger.info(f"📨 AGENT → QUEUE: {message.sender_id} → {message.recipient_id}")
        logger.debug(f"   Type: {message.message_type}")
        logger.debug(f"   Payload: {message.payload}")
        # ... existing publish code ...
        logger.info("✓ Message sent")

    async def subscribe(self, queue_name: str, callback: Callable):
        logger.info(f"📥 Listening to queue: {queue_name}")
        # ... existing subscribe loop ...
        async for msg in self.channel.iterator():
            logger.info(f"🔔 Received on {queue_name}")
            logger.debug(f"   From: {msg.sender_id}")
            logger.debug(f"   Content: {msg.payload}")
            await callback(msg)
