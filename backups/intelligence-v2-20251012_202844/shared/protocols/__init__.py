"""
Protocols Module
System protokołów komunikacji i współpracy między agentami
"""

from .base_protocol import (
    BaseProtocol,
    ProtocolMessage,
    ProtocolStatus,
    ProtocolType
)
from .code_review import CodeReviewProtocol, ReviewResult, ReviewSeverity
from .problem_solving import ProblemSolvingProtocol, Solution, SolutionQuality
from .knowledge_sharing import KnowledgeSharingProtocol, KnowledgeItem, KnowledgeCategory
from .escalation import EscalationProtocol, EscalationLevel, EscalationTicket, EscalationReason
from .consensus import ConsensusProtocol, ConsensusMethod, ConsensusResult, VoteChoice

__all__ = [
    # Base
    'BaseProtocol',
    'ProtocolMessage',
    'ProtocolStatus',
    'ProtocolType',
    # Code Review
    'CodeReviewProtocol',
    'ReviewResult',
    'ReviewSeverity',
    # Problem Solving
    'ProblemSolvingProtocol',
    'Solution',
    'SolutionQuality',
    # Knowledge Sharing
    'KnowledgeSharingProtocol',
    'KnowledgeItem',
    'KnowledgeCategory',
    # Escalation
    'EscalationProtocol',
    'EscalationLevel',
    'EscalationTicket',
    'EscalationReason',
    # Consensus
    'ConsensusProtocol',
    'ConsensusMethod',
    'ConsensusResult',
    'VoteChoice'
]

__version__ = '1.0.0'
