"""
Problem Solving Protocol
Protokół rozwiązywania problemów przez zespół agentów
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base_protocol import BaseProtocol, ProtocolMessage, ProtocolStatus, ProtocolType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolutionQuality(Enum):
    """Ocena jakości rozwiązania"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    UNACCEPTABLE = 1


@dataclass
class Solution:
    """Proponowane rozwiązanie problemu"""
    solution_id: str
    proposed_by: str
    description: str
    approach: str
    estimated_effort_hours: float
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    votes: int = 0
    voted_by: List[str] = field(default_factory=list)
    quality_rating: Optional[SolutionQuality] = None
    comments: List[Dict[str, Any]] = field(default_factory=list)
    proposed_at: datetime = field(default_factory=datetime.now)


@dataclass
class Problem:
    """Definicja problemu"""
    problem_id: str
    title: str
    description: str
    context: Dict[str, Any]
    severity: str  # critical, high, medium, low
    reported_by: str
    reported_at: datetime = field(default_factory=datetime.now)


class ProblemSolvingProtocol(BaseProtocol):
    """
    Protokół Problem Solving
    Zespół agentów wspólnie rozwiązuje problem przez:
    - Brainstorming
    - Propozycje rozwiązań
    - Dyskusję
    - Voting
    - Wybór najlepszego rozwiązania
    """
    
    def __init__(self):
        super().__init__(ProtocolType.PROBLEM_SOLVING)
        self.problem: Optional[Problem] = None
        self.solutions: List[Solution] = []
        self.selected_solution: Optional[Solution] = None
        self.brainstorm_duration_seconds: int = 300  # 5 minut domyślnie
        self.voting_enabled: bool = True
        self.expert_agents: List[str] = []  # Agenci eksperci dla tego problemu
    
    def initiate(
        self,
        initiator: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Inicjuj problem solving
        
        Context powinien zawierać:
        - title: str
        - description: str
        - severity: str
        - context: Dict
        - experts: List[str] - lista ekspertów
        - brainstorm_duration: int - czas na brainstorm (sekundy)
        """
        self.initiated_by = initiator
        self.add_participant(initiator)
        
        # Utwórz problem
        self.problem = Problem(
            problem_id=self.protocol_id,
            title=context.get('title', 'Unknown Problem'),
            description=context.get('description', ''),
            context=context.get('context', {}),
            severity=context.get('severity', 'medium'),
            reported_by=initiator
        )
        
        # Dodaj ekspertów
        self.expert_agents = context.get('experts', [])
        for expert in self.expert_agents:
            self.add_participant(expert)
        
        # Konfiguracja
        self.brainstorm_duration_seconds = context.get('brainstorm_duration', 300)
        self.voting_enabled = context.get('voting_enabled', True)
        
        # Broadcast problemu do wszystkich ekspertów
        self.broadcast_message(
            from_agent=initiator,
            content={
                'action': 'problem_announcement',
                'problem': {
                    'title': self.problem.title,
                    'description': self.problem.description,
                    'severity': self.problem.severity,
                    'context': self.problem.context
                },
                'brainstorm_duration': self.brainstorm_duration_seconds,
                'request': 'propose_solutions'
            }
        )
        
        self.status = ProtocolStatus.IN_PROGRESS
        
        logger.info(
            f"Problem Solving initiated: '{self.problem.title}' "
            f"({self.problem.severity}), {len(self.expert_agents)} experts"
        )
        
        return True
    
    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Przetwórz wiadomość w problem solving"""
        action = message.content.get('action')
        
        if action == 'propose_solution':
            return self._handle_solution_proposal(message)
        elif action == 'vote_solution':
            return self._handle_vote(message)
        elif action == 'comment_solution':
            return self._handle_comment(message)
        elif action == 'finalize_selection':
            return self._handle_finalization(message)
        
        return None
    
    def _handle_solution_proposal(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż propozycję rozwiązania"""
        content = message.content
        
        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            proposed_by=message.from_agent,
            description=content.get('description', ''),
            approach=content.get('approach', ''),
            estimated_effort_hours=content.get('estimated_effort', 0.0),
            pros=content.get('pros', []),
            cons=content.get('cons', []),
            risks=content.get('risks', [])
        )
        
        self.solutions.append(solution)
        
        logger.info(
            f"Solution proposed by {message.from_agent}: "
            f"{solution.description[:50]}..."
        )
        
        # Broadcast nowego rozwiązania do wszystkich
        return self.broadcast_message(
            from_agent='system',
            content={
                'action': 'new_solution',
                'solution_id': solution.solution_id,
                'proposed_by': solution.proposed_by,
                'description': solution.description
            }
        )
    
    def _handle_vote(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż głos na rozwiązanie"""
        solution_id = message.content.get('solution_id')
        voter = message.from_agent
        
        # Znajdź rozwiązanie
        for solution in self.solutions:
            if solution.solution_id == solution_id:
                if voter not in solution.voted_by:
                    solution.votes += 1
                    solution.voted_by.append(voter)
                    logger.info(f"{voter} voted for solution {solution_id}")
                else:
                    logger.warning(f"{voter} already voted for {solution_id}")
                break
        
        return None
    
    def _handle_comment(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż komentarz do rozwiązania"""
        solution_id = message.content.get('solution_id')
        comment_text = message.content.get('comment')
        
        for solution in self.solutions:
            if solution.solution_id == solution_id:
                comment = {
                    'agent': message.from_agent,
                    'comment': comment_text,
                    'timestamp': datetime.now().isoformat()
                }
                solution.comments.append(comment)
                logger.info(f"Comment added to {solution_id} by {message.from_agent}")
                break
        
        return None
    
    def _handle_finalization(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż finalizację wyboru rozwiązania"""
        # Wybierz rozwiązanie z największą liczbą głosów
        if self.solutions:
            self.selected_solution = max(self.solutions, key=lambda s: s.votes)
            self.status = ProtocolStatus.COMPLETED
            self.completed_at = datetime.now()
            
            logger.info(
                f"Selected solution: {self.selected_solution.solution_id} "
                f"by {self.selected_solution.proposed_by} "
                f"({self.selected_solution.votes} votes)"
            )
        
        return None
    
    def start_voting(self) -> bool:
        """Rozpocznij fazę głosowania"""
        if not self.voting_enabled:
            logger.warning("Voting is disabled")
            return False
        
        if not self.solutions:
            logger.warning("No solutions to vote on")
            return False
        
        # Broadcast żądania głosowania
        self.broadcast_message(
            from_agent='system',
            content={
                'action': 'start_voting',
                'solutions': [
                    {
                        'id': s.solution_id,
                        'proposed_by': s.proposed_by,
                        'description': s.description
                    }
                    for s in self.solutions
                ]
            }
        )
        
        logger.info(f"Voting started for {len(self.solutions)} solutions")
        return True
    
    def complete(self) -> Dict[str, Any]:
        """Zakończ problem solving i zwróć wynik"""
        if self.status != ProtocolStatus.COMPLETED:
            # Auto-select jeśli nie wybrano
            if self.solutions:
                self.selected_solution = max(self.solutions, key=lambda s: s.votes)
                self.status = ProtocolStatus.COMPLETED
                self.completed_at = datetime.now()
        
        self.result = {
            'problem': {
                'title': self.problem.title,
                'description': self.problem.description,
                'severity': self.problem.severity
            },
            'solutions_proposed': len(self.solutions),
            'participants': len(self.participants),
            'selected_solution': {
                'id': self.selected_solution.solution_id,
                'proposed_by': self.selected_solution.proposed_by,
                'description': self.selected_solution.description,
                'votes': self.selected_solution.votes,
                'estimated_effort': self.selected_solution.estimated_effort_hours
            } if self.selected_solution else None,
            'all_solutions': [
                {
                    'id': s.solution_id,
                    'proposed_by': s.proposed_by,
                    'votes': s.votes,
                    'description': s.description[:100]
                }
                for s in sorted(self.solutions, key=lambda x: x.votes, reverse=True)
            ]
        }
        
        logger.info(f"Problem Solving completed: {self.result}")
        return self.result
    
    def get_solution_ranking(self) -> List[Solution]:
        """Pobierz ranking rozwiązań według liczby głosów"""
        return sorted(self.solutions, key=lambda s: s.votes, reverse=True)
    
    def get_consensus_level(self) -> float:
        """Oblicz poziom consensusu (0.0 - 1.0)"""
        if not self.selected_solution or not self.participants:
            return 0.0
        
        return self.selected_solution.votes / len(self.participants)


def create_problem_solving() -> ProblemSolvingProtocol:
    """Utwórz Problem Solving Protocol"""
    return ProblemSolvingProtocol()
