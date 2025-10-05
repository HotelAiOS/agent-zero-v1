"""
Consensus Protocol
Protokół osiągania consensusu w decyzjach zespołowych
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from .base_protocol import BaseProtocol, ProtocolMessage, ProtocolStatus, ProtocolType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Metody osiągania consensusu"""
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    WEIGHTED_VOTE = "weighted_vote"
    VETO = "veto"


class VoteChoice(Enum):
    """Opcje głosowania"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    VETO = "veto"


@dataclass
class Vote:
    """Głos w consensusie"""
    voter: str
    choice: VoteChoice
    weight: float = 1.0
    reasoning: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Wynik consensusu"""
    decision_reached: bool
    outcome: str
    votes_approve: int
    votes_reject: int
    votes_abstain: int
    votes_veto: int
    total_votes: int
    approval_rate: float
    vetoed_by: List[str] = field(default_factory=list)
    consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY


class ConsensusProtocol(BaseProtocol):
    """Protokół Consensusu"""
    
    def __init__(self):
        super().__init__(ProtocolType.CONSENSUS)
        self.decision_topic: Optional[str] = None
        self.description: Optional[str] = None
        self.options: List[str] = []
        self.votes: List[Vote] = []
        self.consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY
        self.required_threshold: float = 0.5
        self.voting_deadline: Optional[datetime] = None
        self.expert_agents: List[str] = []
        self.expert_weight: float = 2.0
        self.veto_enabled: bool = False
        self.result: Optional[ConsensusResult] = None
    
    def initiate(self, initiator: str, context: Dict[str, Any]) -> bool:
        """Inicjuj consensus"""
        self.initiated_by = initiator
        self.add_participant(initiator)
        
        self.decision_topic = context.get('topic', '')
        self.description = context.get('description', '')
        self.options = context.get('options', [])
        
        method_str = context.get('method', 'MAJORITY')
        self.consensus_method = ConsensusMethod[method_str]
        
        if self.consensus_method == ConsensusMethod.UNANIMOUS:
            self.required_threshold = 1.0
        elif self.consensus_method == ConsensusMethod.SUPERMAJORITY:
            self.required_threshold = 0.75
        elif self.consensus_method == ConsensusMethod.MAJORITY:
            self.required_threshold = 0.5
        
        for participant in context.get('participants', []):
            self.add_participant(participant)
        
        self.expert_agents = context.get('experts', [])
        self.expert_weight = context.get('expert_weight', 2.0)
        self.veto_enabled = self.consensus_method == ConsensusMethod.VETO
        
        deadline_minutes = context.get('deadline_minutes', 60)
        self.voting_deadline = datetime.now() + timedelta(minutes=deadline_minutes)
        
        self.broadcast_message(
            from_agent=initiator,
            content={
                'action': 'voting_request',
                'topic': self.decision_topic,
                'description': self.description,
                'options': self.options,
                'method': self.consensus_method.value,
                'deadline': self.voting_deadline.isoformat()
            }
        )
        
        self.status = ProtocolStatus.WAITING_RESPONSE
        
        logger.info(
            f"Consensus initiated: {self.decision_topic} "
            f"({self.consensus_method.value}, {len(self.participants)} participants)"
        )
        
        return True
    
    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Przetwórz wiadomość"""
        action = message.content.get('action')
        
        if action == 'cast_vote':
            return self._handle_vote(message)
        elif action == 'change_vote':
            return self._handle_vote_change(message)
        elif action == 'finalize':
            return self._handle_finalization(message)
        
        return None
    
    def _handle_vote(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż oddanie głosu"""
        voter = message.from_agent
        choice_str = message.content.get('choice', 'ABSTAIN')
        reasoning = message.content.get('reasoning')
        
        existing_vote = next((v for v in self.votes if v.voter == voter), None)
        if existing_vote:
            logger.warning(f"{voter} already voted")
            return None
        
        weight = self.expert_weight if voter in self.expert_agents else 1.0
        
        vote = Vote(
            voter=voter,
            choice=VoteChoice[choice_str],
            weight=weight,
            reasoning=reasoning
        )
        
        self.votes.append(vote)
        
        logger.info(f"Vote cast by {voter}: {vote.choice.value} (weight: {weight})")
        
        if len(self.votes) >= len(self.participants):
            self._calculate_consensus()
        
        return None
    
    def _handle_vote_change(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż zmianę głosu"""
        voter = message.from_agent
        new_choice_str = message.content.get('new_choice')
        
        for vote in self.votes:
            if vote.voter == voter:
                old_choice = vote.choice
                vote.choice = VoteChoice[new_choice_str]
                vote.timestamp = datetime.now()
                logger.info(f"Vote changed by {voter}: {old_choice.value} -> {vote.choice.value}")
                break
        
        return None
    
    def _handle_finalization(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż finalizację"""
        self._calculate_consensus()
        return None
    
    def _calculate_consensus(self):
        """Oblicz consensus"""
        if not self.votes:
            logger.warning("No votes to calculate consensus")
            return
        
        approve_votes = sum(v.weight for v in self.votes if v.choice == VoteChoice.APPROVE)
        reject_votes = sum(v.weight for v in self.votes if v.choice == VoteChoice.REJECT)
        abstain_votes = sum(v.weight for v in self.votes if v.choice == VoteChoice.ABSTAIN)
        veto_votes = sum(v.weight for v in self.votes if v.choice == VoteChoice.VETO)
        
        approve_count = sum(1 for v in self.votes if v.choice == VoteChoice.APPROVE)
        reject_count = sum(1 for v in self.votes if v.choice == VoteChoice.REJECT)
        abstain_count = sum(1 for v in self.votes if v.choice == VoteChoice.ABSTAIN)
        veto_count = sum(1 for v in self.votes if v.choice == VoteChoice.VETO)
        
        voting_total = approve_votes + reject_votes
        approval_rate = approve_votes / voting_total if voting_total > 0 else 0.0
        
        vetoed_by = [v.voter for v in self.votes if v.choice == VoteChoice.VETO]
        
        decision_reached = False
        outcome = 'no_consensus'
        
        if veto_count > 0 and self.veto_enabled:
            outcome = 'rejected'
            decision_reached = True
        elif approval_rate >= self.required_threshold:
            outcome = 'approved'
            decision_reached = True
        elif (1.0 - approval_rate) >= self.required_threshold:
            outcome = 'rejected'
            decision_reached = True
        
        self.result = ConsensusResult(
            decision_reached=decision_reached,
            outcome=outcome,
            votes_approve=approve_count,
            votes_reject=reject_count,
            votes_abstain=abstain_count,
            votes_veto=veto_count,
            total_votes=len(self.votes),
            approval_rate=approval_rate,
            vetoed_by=vetoed_by,
            consensus_method=self.consensus_method
        )
        
        self.status = ProtocolStatus.COMPLETED
        self.completed_at = datetime.now()
        
        logger.info(f"Consensus reached: {outcome} (approval rate: {approval_rate:.1%})")
        
        self.broadcast_message(
            from_agent='system',
            content={
                'action': 'consensus_result',
                'outcome': outcome,
                'approval_rate': approval_rate,
                'votes': {
                    'approve': approve_count,
                    'reject': reject_count,
                    'abstain': abstain_count,
                    'veto': veto_count
                }
            }
        )
    
    def cast_vote(self, agent_id: str, choice: VoteChoice, reasoning: Optional[str] = None) -> bool:
        """Oddaj głos"""
        if agent_id not in self.participants:
            logger.warning(f"{agent_id} is not a participant")
            return False
        
        message = ProtocolMessage(
            message_id=f"msg_{len(self.messages)}",
            from_agent=agent_id,
            to_agent=None,
            timestamp=datetime.now(),
            content={
                'action': 'cast_vote',
                'choice': choice.name,
                'reasoning': reasoning
            },
            protocol_type=self.protocol_type
        )
        
        self._handle_vote(message)
        return True
    
    def complete(self) -> Dict[str, Any]:
        """Zakończ consensus"""
        if not self.result:
            self._calculate_consensus()
        
        return {
            'topic': self.decision_topic,
            'method': self.consensus_method.value,
            'decision_reached': self.result.decision_reached if self.result else False,
            'outcome': self.result.outcome if self.result else 'no_consensus',
            'approval_rate': self.result.approval_rate if self.result else 0.0,
            'votes': {
                'approve': self.result.votes_approve if self.result else 0,
                'reject': self.result.votes_reject if self.result else 0,
                'abstain': self.result.votes_abstain if self.result else 0,
                'veto': self.result.votes_veto if self.result else 0
            },
            'total_participants': len(self.participants),
            'total_votes': len(self.votes),
            'participation_rate': len(self.votes) / len(self.participants) if self.participants else 0.0
        }
    
    def get_vote_breakdown(self) -> Dict[str, List[str]]:
        """Breakdown głosów"""
        breakdown = {
            'approve': [],
            'reject': [],
            'abstain': [],
            'veto': []
        }
        
        for vote in self.votes:
            breakdown[vote.choice.value].append(vote.voter)
        
        return breakdown


def create_consensus() -> ConsensusProtocol:
    """Utwórz Consensus Protocol"""
    return ConsensusProtocol()
