"""
Agent Communication
High-level wrapper dla komunikacji między agentami
"""

import logging
from typing import Callable, Optional, Dict, Any
from .bus import MessageBus, BusConfig
from .publisher import MessagePublisher
from .consumer import MessageConsumer
from .message import (
    Message, MessageType, MessagePriority,
    create_task_request, create_broadcast, create_code_review_request
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentCommunicator:
    """
    High-level API dla komunikacji agenta
    Łączy Publisher + Consumer + Message Bus
    """
    
    def __init__(
        self,
        agent_id: str,
        bus_config: Optional[BusConfig] = None,
        auto_connect: bool = True
    ):
        """
        Initialize communicator
        
        Args:
            agent_id: ID agenta
            bus_config: Konfiguracja RabbitMQ
            auto_connect: Czy automatycznie połączyć z RabbitMQ
        """
        self.agent_id = agent_id
        config = bus_config or BusConfig()
        
        # ZMIANA: Osobne połączenia dla Publisher i Consumer (thread safety)
        self.bus_publish = MessageBus(config)
        self.bus_consume = MessageBus(config)
        
        self.publisher = MessagePublisher(self.bus_publish)
        self.consumer = MessageConsumer(self.bus_consume, agent_id)
        
        if auto_connect:
            self.connect()
        
        logger.info(f"AgentCommunicator zainicjalizowany dla {agent_id}")
    
    def connect(self) -> bool:
        """
        Połącz z RabbitMQ i skonfiguruj exchange/kolejki
        
        Returns:
            True jeśli sukces
        """
        # Połącz oba połączenia
        if not self.bus_publish.connect():
            return False
        if not self.bus_consume.connect():
            return False
        
        # Zadeklaruj exchange (wystarczy raz)
        self.bus_publish.declare_exchange("agent_exchange", "topic", durable=True)
        
        # Skonfiguruj kolejkę dla agenta
        self.consumer.setup_queue(
            exchange="agent_exchange",
            routing_keys=[
                f"direct.{self.agent_id}",
                "broadcast.all",
                f"broadcast.project.*",
                "*.urgent.*",
                f"team.*",
            ]
        )
        
        logger.info(f"✅ {self.agent_id} połączony z message bus")
        return True
    
    def disconnect(self):
        """Rozłącz z RabbitMQ"""
        self.consumer.stop_consuming()
        self.bus_consume.disconnect()
        self.bus_publish.disconnect()
        logger.info(f"{self.agent_id} rozłączony")
    
    # === WYSYŁANIE WIADOMOŚCI ===
    
    def send_direct(
        self,
        recipient_id: str,
        subject: str,
        content: str,
        message_type: MessageType = MessageType.NOTIFICATION,
        priority: MessagePriority = MessagePriority.NORMAL,
        payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Wyślij wiadomość bezpośrednią do innego agenta
        
        Args:
            recipient_id: ID odbiorcy
            subject: Temat
            content: Treść
            message_type: Typ wiadomości
            priority: Priorytet
            payload: Dodatkowe dane
        
        Returns:
            True jeśli wysłano
        """
        message = Message(
            message_type=message_type,
            priority=priority,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            subject=subject,
            content=content,
            payload=payload or {}
        )
        
        return self.publisher.publish_direct(message, recipient_id)
    
    def broadcast(
        self,
        subject: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Wyślij broadcast do wszystkich agentów
        
        Args:
            subject: Temat
            content: Treść
            priority: Priorytet
            project_id: Opcjonalnie ogranicz do projektu
        
        Returns:
            True jeśli wysłano
        """
        message = create_broadcast(
            sender_id=self.agent_id,
            subject=subject,
            content=content,
            project_id=project_id,
            priority=priority
        )
        
        return self.publisher.publish_broadcast(message, project_id)
    
    def request_task(
        self,
        recipient_id: str,
        task_description: str,
        task_id: str,
        project_id: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Wyślij request do wykonania zadania
        
        Args:
            recipient_id: ID agenta który ma wykonać
            task_description: Opis zadania
            task_id: ID zadania
            project_id: ID projektu
            priority: Priorytet
            payload: Dodatkowe dane
        
        Returns:
            True jeśli wysłano
        """
        message = create_task_request(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            task_description=task_description,
            task_id=task_id,
            project_id=project_id,
            priority=priority,
            payload=payload
        )
        
        return self.publisher.publish_direct(message, recipient_id)
    
    def request_code_review(
        self,
        recipient_id: str,
        code: str,
        task_id: str,
        project_id: str,
        language: str = "python"
    ) -> bool:
        """
        Poproś o code review
        
        Args:
            recipient_id: ID reviewera
            code: Kod do review
            task_id: ID zadania
            project_id: ID projektu
            language: Język programowania
        
        Returns:
            True jeśli wysłano
        """
        message = create_code_review_request(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            code=code,
            task_id=task_id,
            project_id=project_id,
            language=language
        )
        
        return self.publisher.publish_direct(message, recipient_id)
    
    def send_to_team(
        self,
        team_name: str,
        subject: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Wyślij wiadomość do zespołu
        
        Args:
            team_name: Nazwa zespołu
            subject: Temat
            content: Treść
            priority: Priorytet
        
        Returns:
            True jeśli wysłano
        """
        message = Message(
            message_type=MessageType.NOTIFICATION,
            priority=priority,
            sender_id=self.agent_id,
            subject=subject,
            content=content
        )
        
        return self.publisher.publish_to_team(message, team_name)
    
    # === ODBIERANIE WIADOMOŚCI ===
    
    def on_message(self, handler: Callable[[Message], None]):
        """
        Zarejestruj handler do przetwarzania wiadomości
        
        Args:
            handler: Funkcja callback(message)
        """
        self.consumer.register_handler(handler)
    
    def start_listening(self, block: bool = False):
        """
        Rozpocznij nasłuchiwanie na wiadomości
        
        Args:
            block: Czy blokować wątek
        """
        self.consumer.start_consuming(block=block)
    
    def stop_listening(self):
        """Zatrzymaj nasłuchiwanie"""
        self.consumer.stop_consuming()
    
    def is_listening(self) -> bool:
        """Czy nasłuchuje?"""
        return self.consumer.is_consuming()
    
    def __enter__(self):
        """Context manager: connect"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: disconnect"""
        self.disconnect()
