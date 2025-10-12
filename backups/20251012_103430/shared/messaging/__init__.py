"""
Messaging Module
RabbitMQ-based message bus dla komunikacji między agentami
"""

from .message import (
    Message,
    MessageType,
    MessagePriority,
    create_task_request,
    create_broadcast,
    create_code_review_request
)

from .bus import MessageBus, BusConfig
from .publisher import MessagePublisher
from .consumer import MessageConsumer
from .agent_comm import AgentCommunicator

__all__ = [
    # Message
    'Message',
    'MessageType',
    'MessagePriority',
    'create_task_request',
    'create_broadcast',
    'create_code_review_request',
    
    # Bus
    'MessageBus',
    'BusConfig',
    
    # Publisher/Consumer
    'MessagePublisher',
    'MessageConsumer',
    
    # High-level API
    'AgentCommunicator',
]
