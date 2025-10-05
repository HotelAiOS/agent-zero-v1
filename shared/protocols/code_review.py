"""
Code Review Protocol
Protokół przeglądu kodu między agentami
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base_protocol import BaseProtocol, ProtocolMessage, ProtocolStatus, ProtocolType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewSeverity(Enum):
    """Poziom ważności uwagi w review"""
    BLOCKER = "blocker"  # Must fix przed merge
    CRITICAL = "critical"  # Should fix
    MAJOR = "major"  # Important
    MINOR = "minor"  # Nice to fix
    INFO = "info"  # Informacyjne


@dataclass
class ReviewComment:
    """Komentarz w code review"""
    comment_id: str
    reviewer: str
    file_path: str
    line_number: Optional[int]
    severity: ReviewSeverity
    message: str
    suggestion: Optional[str] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class ReviewResult:
    """Wynik code review"""
    review_id: str
    reviewer: str
    approved: bool
    comments: List[ReviewComment] = field(default_factory=list)
    blockers_count: int = 0
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    info_count: int = 0
    summary: Optional[str] = None
    reviewed_at: datetime = field(default_factory=datetime.now)


class CodeReviewProtocol(BaseProtocol):
    """
    Protokół Code Review
    Agent prosi innych agentów o review kodu
    """
    
    def __init__(self):
        super().__init__(ProtocolType.CODE_REVIEW)
        self.code_author: Optional[str] = None
        self.reviewers: List[str] = []
        self.required_approvals: int = 1
        self.code_files: List[str] = []
        self.reviews: List[ReviewResult] = []
        self.approved_by: List[str] = []
        self.changes_requested_by: List[str] = []
    
    def initiate(
        self,
        initiator: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Inicjuj code review
        
        Context powinien zawierać:
        - code_files: List[str] - pliki do review
        - reviewers: List[str] - lista reviewerów
        - required_approvals: int - wymagana liczba approval
        - description: str - opis zmian
        """
        self.initiated_by = initiator
        self.code_author = initiator
        self.code_files = context.get('code_files', [])
        self.reviewers = context.get('reviewers', [])
        self.required_approvals = context.get('required_approvals', 1)
        
        # Dodaj uczestników
        self.add_participant(initiator)
        for reviewer in self.reviewers:
            self.add_participant(reviewer)
        
        # Wyślij request o review
        for reviewer in self.reviewers:
            self.send_message(
                from_agent=initiator,
                to_agent=reviewer,
                content={
                    'action': 'review_request',
                    'code_files': self.code_files,
                    'description': context.get('description', ''),
                    'required_approvals': self.required_approvals
                },
                requires_response=True
            )
        
        self.status = ProtocolStatus.WAITING_RESPONSE
        
        logger.info(
            f"Code Review initiated by {initiator}: "
            f"{len(self.code_files)} files, {len(self.reviewers)} reviewers"
        )
        
        return True
    
    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Przetwórz wiadomość w code review"""
        action = message.content.get('action')
        
        if action == 'submit_review':
            return self._handle_review_submission(message)
        elif action == 'resolve_comment':
            return self._handle_comment_resolution(message)
        elif action == 'request_changes':
            return self._handle_changes_request(message)
        
        return None
    
    def _handle_review_submission(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż submission review"""
        reviewer = message.from_agent
        content = message.content
        
        # Utwórz ReviewResult
        review = ReviewResult(
            review_id=message.message_id,
            reviewer=reviewer,
            approved=content.get('approved', False),
            summary=content.get('summary')
        )
        
        # Dodaj komentarze
        for comment_data in content.get('comments', []):
            comment = ReviewComment(
                comment_id=f"cmt_{len(review.comments)}",
                reviewer=reviewer,
                file_path=comment_data['file_path'],
                line_number=comment_data.get('line_number'),
                severity=ReviewSeverity[comment_data['severity']],
                message=comment_data['message'],
                suggestion=comment_data.get('suggestion')
            )
            review.comments.append(comment)
            
            # Zlicz według severity
            if comment.severity == ReviewSeverity.BLOCKER:
                review.blockers_count += 1
            elif comment.severity == ReviewSeverity.CRITICAL:
                review.critical_count += 1
            elif comment.severity == ReviewSeverity.MAJOR:
                review.major_count += 1
            elif comment.severity == ReviewSeverity.MINOR:
                review.minor_count += 1
            else:
                review.info_count += 1
        
        self.reviews.append(review)
        
        # Aktualizuj listy approval/changes requested
        if review.approved and review.blockers_count == 0:
            self.approved_by.append(reviewer)
        else:
            self.changes_requested_by.append(reviewer)
        
        logger.info(
            f"Review submitted by {reviewer}: "
            f"{'APPROVED' if review.approved else 'CHANGES REQUESTED'}, "
            f"{len(review.comments)} comments"
        )
        
        # Sprawdź czy review jest kompletny
        if len(self.reviews) >= len(self.reviewers):
            self._check_review_completion()
        
        # Odpowiedź do autora
        return self.send_message(
            from_agent='system',
            to_agent=self.code_author,
            content={
                'action': 'review_received',
                'reviewer': reviewer,
                'approved': review.approved,
                'comments_count': len(review.comments)
            }
        )
    
    def _handle_comment_resolution(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż rozwiązanie komentarza"""
        comment_id = message.content.get('comment_id')
        
        # Znajdź komentarz
        for review in self.reviews:
            for comment in review.comments:
                if comment.comment_id == comment_id:
                    comment.resolved = True
                    comment.resolved_by = message.from_agent
                    comment.resolved_at = datetime.now()
                    
                    logger.info(f"Comment {comment_id} resolved by {message.from_agent}")
                    return None
        
        return None
    
    def _handle_changes_request(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż request o zmiany"""
        self.changes_requested_by.append(message.from_agent)
        return None
    
    def _check_review_completion(self):
        """Sprawdź czy review jest ukończony"""
        # Sprawdź czy mamy wystarczająco approvals
        if len(self.approved_by) >= self.required_approvals:
            # Sprawdź czy nie ma blockerów
            total_blockers = sum(r.blockers_count for r in self.reviews)
            
            if total_blockers == 0:
                self.status = ProtocolStatus.COMPLETED
                self.completed_at = datetime.now()
                logger.info(f"Code Review APPROVED: {len(self.approved_by)} approvals")
            else:
                logger.warning(f"Code Review has {total_blockers} blockers - cannot approve")
        else:
            needed = self.required_approvals - len(self.approved_by)
            logger.info(f"Waiting for {needed} more approval(s)")
    
    def complete(self) -> Dict[str, Any]:
        """Zakończ code review i zwróć wynik"""
        if self.status != ProtocolStatus.COMPLETED:
            self._check_review_completion()
        
        total_comments = sum(len(r.comments) for r in self.reviews)
        total_blockers = sum(r.blockers_count for r in self.reviews)
        total_critical = sum(r.critical_count for r in self.reviews)
        
        unresolved_blockers = sum(
            1 for r in self.reviews
            for c in r.comments
            if c.severity == ReviewSeverity.BLOCKER and not c.resolved
        )
        
        self.result = {
            'approved': len(self.approved_by) >= self.required_approvals and unresolved_blockers == 0,
            'reviewers_count': len(self.reviewers),
            'reviews_received': len(self.reviews),
            'approved_by': self.approved_by,
            'changes_requested_by': self.changes_requested_by,
            'total_comments': total_comments,
            'blockers': total_blockers,
            'critical': total_critical,
            'unresolved_blockers': unresolved_blockers,
            'can_merge': unresolved_blockers == 0 and len(self.approved_by) >= self.required_approvals
        }
        
        logger.info(f"Code Review completed: {self.result}")
        return self.result
    
    def get_all_comments(self, severity: Optional[ReviewSeverity] = None) -> List[ReviewComment]:
        """Pobierz wszystkie komentarze, opcjonalnie filtrowane po severity"""
        comments = []
        for review in self.reviews:
            for comment in review.comments:
                if severity is None or comment.severity == severity:
                    comments.append(comment)
        return comments
    
    def get_unresolved_comments(self) -> List[ReviewComment]:
        """Pobierz nierozwiązane komentarze"""
        comments = []
        for review in self.reviews:
            for comment in review.comments:
                if not comment.resolved:
                    comments.append(comment)
        return comments


def create_code_review() -> CodeReviewProtocol:
    """Utwórz Code Review Protocol"""
    return CodeReviewProtocol()
