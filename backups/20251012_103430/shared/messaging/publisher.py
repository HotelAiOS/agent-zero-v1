"""
Message Publisher
Publikowanie wiadomości do RabbitMQ
"""

import json
import logging
from typing import Optional
from .bus import MessageBus
from .message import Message, MessagePriority
import pika

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagePublisher:
    """
    Publisher wiadomości
    Wysyła wiadomości do exchange z odpowiednim routingiem
    """
    
    def __init__(self, bus: MessageBus):
        """
        Initialize publisher
        
        Args:
            bus: MessageBus instance
        """
        self.bus = bus
        logger.info("MessagePublisher zainicjalizowany")
    
    def publish(
        self,
        message: Message,
        exchange: str = "agent_exchange",
        routing_key: Optional[str] = None
    ) -> bool:
        """
        Publikuj wiadomość
        
        Args:
            message: Message do wysłania
            exchange: Nazwa exchange
            routing_key: Klucz routingu (auto-generate jeśli None)
        
        Returns:
            True jeśli sukces
        """
        if not self.bus.is_connected():
            logger.error("Brak połączenia z RabbitMQ")
            return False
        
        try:
            # Auto-generate routing key jeśli nie podano
            if routing_key is None:
                routing_key = self._generate_routing_key(message)
            
            # Update message routing info
            message.exchange = exchange
            message.routing_key = routing_key
            
            # Serialize message
            body = json.dumps(message.to_dict())
            
            # Ustaw właściwości wiadomości
            properties = pika.BasicProperties(
                delivery_mode=2,  # Persistent
                priority=message.priority.value,
                content_type='application/json',
                message_id=message.message_id,
                timestamp=int(message.timestamp.timestamp()),
                expiration=str(int((message.expires_at - message.timestamp).total_seconds() * 1000)) if message.expires_at else None
            )
            
            # Publikuj
            self.bus.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=body,
                properties=properties
            )
            
            logger.info(
                f"📤 Wysłano: {message.message_type.value} "
                f"({message.sender_id} → {message.recipient_id or 'broadcast'}) "
                f"[{routing_key}]"
            )
            return True
            
        except Exception as e:
            logger.error(f"Błąd publikacji: {e}")
            return False
    
    def _generate_routing_key(self, message: Message) -> str:
        """
        Generuj routing key na podstawie wiadomości
        
        Format: message_type.priority.recipient
        Przykład: task_request.high.backend_1
        """
        priority_map = {
            MessagePriority.LOW: "low",
            MessagePriority.NORMAL: "normal",
            MessagePriority.HIGH: "high",
            MessagePriority.URGENT: "urgent"
        }
        
        parts = [
            message.message_type.value,
            priority_map[message.priority]
        ]
        
        # Dodaj recipient jeśli nie broadcast
        if message.recipient_id:
            parts.append(message.recipient_id)
        else:
            parts.append("broadcast")
        
        return ".".join(parts)
    
    def publish_direct(
        self,
        message: Message,
        recipient_id: str
    ) -> bool:
        """
        Publikuj wiadomość bezpośrednią do konkretnego agenta
        
        Args:
            message: Message do wysłania
            recipient_id: ID odbiorcy
        
        Returns:
            True jeśli sukces
        """
        message.recipient_id = recipient_id
        routing_key = f"direct.{recipient_id}"
        return self.publish(message, routing_key=routing_key)
    
    def publish_broadcast(
        self,
        message: Message,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Publikuj broadcast do wszystkich agentów
        
        Args:
            message: Message do wysłania
            project_id: Opcjonalnie ogranicz do projektu
        
        Returns:
            True jeśli sukces
        """
        message.recipient_id = None
        
        if project_id:
            routing_key = f"broadcast.project.{project_id}"
        else:
            routing_key = "broadcast.all"
        
        return self.publish(message, routing_key=routing_key)
    
    def publish_to_team(
        self,
        message: Message,
        team_name: str
    ) -> bool:
        """
        Publikuj do zespołu agentów
        
        Args:
            message: Message do wysłania
            team_name: Nazwa zespołu (np. "backend_team")
        
        Returns:
            True jeśli sukces
        """
        routing_key = f"team.{team_name}"
        return self.publish(message, routing_key=routing_key)
