import asyncio
import aio_pika
import json
import os
import logging
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MessageBus:
    """Asynchronous message bus for agent-to-agent communication"""
    
    def __init__(self):
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self._is_connected = False

    async def connect(self):
        """Connect to RabbitMQ"""
        if self._is_connected:
            return
            
        rabbitmq_url = os.getenv(
            "RABBITMQ_URL", 
            "amqp://agent:agent-pass@localhost:5672/"
        )
        
        try:
            self.connection = await aio_pika.connect_robust(rabbitmq_url)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=100)
            
            # Exchange for agent communication
            self.exchange = await self.channel.declare_exchange(
                "agent_communication", 
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            self._is_connected = True
            logger.info("✅ Connected to RabbitMQ - Agent Communication Ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            raise

    async def publish(self, routing_key: str, message: Dict[str, Any]):
        """Send message to other agents"""
        if not self._is_connected:
            await self.connect()
            
        try:
            body = json.dumps(message, default=str).encode()
            msg = aio_pika.Message(
                body, 
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"timestamp": str(asyncio.get_event_loop().time())}
            )
            
            await self.exchange.publish(msg, routing_key=routing_key)
            logger.info(f"📤 Published to {routing_key}: {message.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            raise

    async def subscribe(self, routing_key: str, handler: Callable):
        """Subscribe to messages from other agents"""
        if not self._is_connected:
            await self.connect()
            
        try:
            # Create valid queue name (sanitize routing key)
            queue_name = f"agent_{routing_key.replace('.', '_').replace('*', 'all')}"
            queue = await self.channel.declare_queue(queue_name, durable=True)
            await queue.bind(self.exchange, routing_key)
            
            async def message_wrapper(message: aio_pika.IncomingMessage):
                try:
                    data = json.loads(message.body.decode())
                    await handler(data)
                    await message.ack()
                    logger.info(f"✅ Processed message from {routing_key}")
                except Exception as e:
                    logger.error(f"❌ Handler error for {routing_key}: {e}")
                    await message.nack()
            
            await queue.consume(message_wrapper)
            logger.info(f"👂 Subscribed to {routing_key} (queue: {queue_name})")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {routing_key}: {e}")
            raise

    async def close(self):
        """Close connection"""
        if self.connection:
            await self.connection.close()
            self._is_connected = False
            logger.info("🔌 Disconnected from RabbitMQ")

# Global instance for easy import
message_bus = MessageBus()

# Convenience functions
async def publish_agent_message(agent_type: str, message_type: str, data: Dict[str, Any]):
    """Quick publish to agent type"""
    routing_key = f"agent.{agent_type}.{message_type}"
    message = {
        "type": message_type,
        "data": data,
        "timestamp": asyncio.get_event_loop().time()
    }
    await message_bus.publish(routing_key, message)

async def subscribe_to_agent_messages(agent_type: str, handler: Callable):
    """Quick subscribe to all messages for agent type"""
    routing_key = f"agent.{agent_type}.*"
    await message_bus.subscribe(routing_key, handler)
