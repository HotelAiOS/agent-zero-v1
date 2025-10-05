"""
Message Consumer
Odbieranie i przetwarzanie wiadomości z RabbitMQ
"""

import json
import logging
from typing import Callable, Optional, List
from .bus import MessageBus
from .message import Message
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageConsumer:
    """
    Consumer wiadomości
    Nasłuchuje na wiadomości i wywołuje handlery
    """
    
    def __init__(self, bus: MessageBus, agent_id: str):
        """
        Initialize consumer
        
        Args:
            bus: MessageBus instance
            agent_id: ID agenta (dla dedykowanej kolejki)
        """
        self.bus = bus
        self.agent_id = agent_id
        self.queue_name = f"queue.{agent_id}"
        self.handlers: List[Callable[[Message], None]] = []
        self._consuming = False
        self._consumer_thread: Optional[threading.Thread] = None
        
        logger.info(f"MessageConsumer zainicjalizowany dla {agent_id}")
    
    def register_handler(self, handler: Callable[[Message], None]):
        """
        Zarejestruj handler do przetwarzania wiadomości
        
        Args:
            handler: Funkcja(message) -> None
        """
        self.handlers.append(handler)
        logger.info(f"Zarejestrowano handler: {handler.__name__}")
    
    def setup_queue(
        self,
        exchange: str = "agent_exchange",
        routing_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Skonfiguruj kolejkę i binding
        
        Args:
            exchange: Nazwa exchange
            routing_keys: Lista routing keys do nasłuchiwania
        
        Returns:
            True jeśli sukces
        """
        if not self.bus.is_connected():
            logger.error("Brak połączenia z RabbitMQ")
            return False
        
        try:
            # Zadeklaruj kolejkę dla agenta
            self.bus.declare_queue(
                self.queue_name,
                durable=True,
                exclusive=False,
                auto_delete=False
            )
            
            # Default routing keys
            if routing_keys is None:
                routing_keys = [
                    f"direct.{self.agent_id}",      # Bezpośrednie do tego agenta
                    "broadcast.all",                 # Wszystkie broadcasty
                    f"broadcast.project.*",          # Broadcasty projektowe
                    "*.urgent.*",                    # Wszystkie urgent
                ]
            
            # Bind kolejkę z exchange dla każdego routing key
            for routing_key in routing_keys:
                self.bus.bind_queue(
                    self.queue_name,
                    exchange,
                    routing_key
                )
            
            logger.info(f"✅ Kolejka {self.queue_name} skonfigurowana z {len(routing_keys)} routing keys")
            return True
            
        except Exception as e:
            logger.error(f"Błąd konfiguracji kolejki: {e}")
            return False
    
    def _process_message(self, ch, method, properties, body):
        """
        Callback do przetwarzania wiadomości
        Wywoływany przez pika gdy nadejdzie wiadomość
        """
        try:
            # Deserialize
            data = json.loads(body)
            message = Message.from_dict(data)
            
            logger.info(
                f"📥 Odebrano: {message.message_type.value} "
                f"od {message.sender_id} "
                f"[{method.routing_key}]"
            )
            
            # Wywołaj wszystkie handlery
            for handler in self.handlers:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Błąd w handlerze {handler.__name__}: {e}")
            
            # Acknowledge
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania wiadomości: {e}")
            # Reject message
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def start_consuming(self, block: bool = True):
        """
        Rozpocznij nasłuchiwanie na wiadomości
        
        Args:
            block: Czy blokować wątek (True) czy uruchomić w tle (False)
        """
        if not self.bus.is_connected():
            logger.error("Brak połączenia z RabbitMQ")
            return
        
        if self._consuming:
            logger.warning("Consumer już działa")
            return
        
        try:
            # Ustaw QoS - ile wiadomości prefetchować
            self.bus.channel.basic_qos(prefetch_count=1)
            
            # Zacznij konsumować
            self.bus.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self._process_message
            )
            
            self._consuming = True
            logger.info(f"🎧 Consumer nasłuchuje na {self.queue_name}...")
            
            if block:
                # Blokuj i nasłuchuj
                self.bus.channel.start_consuming()
            else:
                # Uruchom w osobnym wątku
                self._consumer_thread = threading.Thread(
                    target=self.bus.channel.start_consuming,
                    daemon=True
                )
                self._consumer_thread.start()
                logger.info("Consumer działa w tle")
                
        except KeyboardInterrupt:
            logger.info("Zatrzymano consumer (Ctrl+C)")
            self.stop_consuming()
        except Exception as e:
            logger.error(f"Błąd consumer: {e}")
            self._consuming = False
    
    def stop_consuming(self):
        """Zatrzymaj nasłuchiwanie"""
        if not self._consuming:
            return
        
        try:
            if self.bus.channel:
                self.bus.channel.stop_consuming()
            self._consuming = False
            logger.info("Consumer zatrzymany")
        except Exception as e:
            logger.error(f"Błąd zatrzymywania consumer: {e}")
    
    def is_consuming(self) -> bool:
        """Sprawdź czy consumer działa"""
        return self._consuming
