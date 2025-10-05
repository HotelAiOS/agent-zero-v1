"""
RabbitMQ Message Bus
Connection manager dla magistrali komunikacyjnej
"""

import pika
import logging
from typing import Optional, Callable
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BusConfig:
    """Konfiguracja połączenia z RabbitMQ"""
    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    heartbeat: int = 600
    connection_timeout: int = 30


class MessageBus:
    """
    RabbitMQ Message Bus
    Zarządza połączeniem z RabbitMQ i podstawowymi operacjami
    """
    
    def __init__(self, config: Optional[BusConfig] = None):
        """
        Initialize message bus
        
        Args:
            config: Bus configuration (używa domyślnej jeśli None)
        """
        self.config = config or BusConfig()
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self._is_connected = False
        
        logger.info(f"MessageBus zainicjalizowany (host: {self.config.host})")
    
    def connect(self) -> bool:
        """
        Nawiąż połączenie z RabbitMQ
        
        Returns:
            True jeśli sukces, False jeśli błąd
        """
        try:
            # Credentials
            credentials = pika.PlainCredentials(
                self.config.username,
                self.config.password
            )
            
            # Connection parameters
            parameters = pika.ConnectionParameters(
                host=self.config.host,
                port=self.config.port,
                virtual_host=self.config.virtual_host,
                credentials=credentials,
                heartbeat=self.config.heartbeat,
                connection_attempts=3,
                retry_delay=2
            )
            
            # Połącz
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self._is_connected = True
            
            logger.info(f"✅ Połączono z RabbitMQ: {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Błąd połączenia z RabbitMQ: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self):
        """Zamknij połączenie z RabbitMQ"""
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                logger.info("Rozłączono z RabbitMQ")
            self._is_connected = False
        except Exception as e:
            logger.error(f"Błąd podczas rozłączania: {e}")
    
    def is_connected(self) -> bool:
        """Sprawdź czy połączenie jest aktywne"""
        if not self._is_connected:
            return False
        
        try:
            if self.connection and self.connection.is_open:
                return True
        except:
            pass
        
        self._is_connected = False
        return False
    
    def reconnect(self) -> bool:
        """Ponowne połączenie"""
        logger.info("Próba ponownego połączenia...")
        self.disconnect()
        return self.connect()
    
    def declare_exchange(
        self,
        exchange_name: str,
        exchange_type: str = "topic",
        durable: bool = True
    ) -> bool:
        """
        Zadeklaruj exchange
        
        Args:
            exchange_name: Nazwa exchange
            exchange_type: Typ (direct, fanout, topic, headers)
            durable: Czy przetrwa restart brokera
        
        Returns:
            True jeśli sukces
        """
        if not self.is_connected():
            logger.error("Brak połączenia z RabbitMQ")
            return False
        
        try:
            self.channel.exchange_declare(
                exchange=exchange_name,
                exchange_type=exchange_type,
                durable=durable
            )
            logger.info(f"Exchange zadeklarowany: {exchange_name} ({exchange_type})")
            return True
        except Exception as e:
            logger.error(f"Błąd deklaracji exchange: {e}")
            return False
    
    def declare_queue(
        self,
        queue_name: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False
    ) -> bool:
        """
        Zadeklaruj kolejkę
        
        Args:
            queue_name: Nazwa kolejki
            durable: Czy przetrwa restart
            exclusive: Czy tylko dla tego połączenia
            auto_delete: Czy usunąć po odłączeniu konsumenta
        
        Returns:
            True jeśli sukces
        """
        if not self.is_connected():
            logger.error("Brak połączenia z RabbitMQ")
            return False
        
        try:
            self.channel.queue_declare(
                queue=queue_name,
                durable=durable,
                exclusive=exclusive,
                auto_delete=auto_delete
            )
            logger.info(f"Kolejka zadeklarowana: {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Błąd deklaracji kolejki: {e}")
            return False
    
    def bind_queue(
        self,
        queue_name: str,
        exchange_name: str,
        routing_key: str = ""
    ) -> bool:
        """
        Zbinduj kolejkę z exchange
        
        Args:
            queue_name: Nazwa kolejki
            exchange_name: Nazwa exchange
            routing_key: Klucz routingu
        
        Returns:
            True jeśli sukces
        """
        if not self.is_connected():
            logger.error("Brak połączenia z RabbitMQ")
            return False
        
        try:
            self.channel.queue_bind(
                queue=queue_name,
                exchange=exchange_name,
                routing_key=routing_key
            )
            logger.info(f"Kolejka {queue_name} zbindowana z {exchange_name} (key: {routing_key})")
            return True
        except Exception as e:
            logger.error(f"Błąd bindowania kolejki: {e}")
            return False
    
    def __enter__(self):
        """Context manager: connect"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: disconnect"""
        self.disconnect()
