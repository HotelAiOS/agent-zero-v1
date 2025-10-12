"""
Knowledge Sharing Protocol
Protokół dzielenia się wiedzą między agentami
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base_protocol import BaseProtocol, ProtocolMessage, ProtocolStatus, ProtocolType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeCategory(Enum):
    """Kategorie wiedzy"""
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    SOLUTION_PATTERN = "solution_pattern"
    ANTI_PATTERN = "anti_pattern"
    TIP = "tip"
    WARNING = "warning"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"


@dataclass
class KnowledgeItem:
    """Element wiedzy"""
    knowledge_id: str
    category: KnowledgeCategory
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    related_technologies: List[str] = field(default_factory=list)
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    views: int = 0
    upvotes: int = 0
    downvotes: int = 0
    comments: List[Dict[str, Any]] = field(default_factory=list)
    useful_for: List[str] = field(default_factory=list)  # Agent types
    source_project: Optional[str] = None
    verified: bool = False
    verified_by: Optional[str] = None


class KnowledgeSharingProtocol(BaseProtocol):
    """
    Protokół Knowledge Sharing
    Agenci dzielą się wiedzą, best practices, lessons learned
    """
    
    def __init__(self):
        super().__init__(ProtocolType.KNOWLEDGE_SHARING)
        self.knowledge_base: List[KnowledgeItem] = []
        self.broadcast_enabled: bool = True
        self.auto_store: bool = True  # Automatycznie zapisuj do knowledge graph
    
    def initiate(
        self,
        initiator: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Inicjuj knowledge sharing
        
        Context powinien zawierać:
        - broadcast: bool - czy broadcastować
        - auto_store: bool - czy auto-zapisywać
        """
        self.initiated_by = initiator
        self.add_participant(initiator)
        
        self.broadcast_enabled = context.get('broadcast', True)
        self.auto_store = context.get('auto_store', True)
        
        self.status = ProtocolStatus.IN_PROGRESS
        
        logger.info(f"Knowledge Sharing initiated by {initiator}")
        return True
    
    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Przetwórz wiadomość w knowledge sharing"""
        action = message.content.get('action')
        
        if action == 'share_knowledge':
            return self._handle_knowledge_share(message)
        elif action == 'upvote':
            return self._handle_upvote(message)
        elif action == 'downvote':
            return self._handle_downvote(message)
        elif action == 'comment':
            return self._handle_comment(message)
        elif action == 'verify':
            return self._handle_verification(message)
        elif action == 'query_knowledge':
            return self._handle_query(message)
        
        return None
    
    def _handle_knowledge_share(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż sharing wiedzy"""
        content = message.content
        
        knowledge = KnowledgeItem(
            knowledge_id=f"know_{len(self.knowledge_base) + 1}",
            category=KnowledgeCategory[content.get('category', 'TIP')],
            title=content.get('title', ''),
            content=content.get('content', ''),
            tags=content.get('tags', []),
            related_technologies=content.get('technologies', []),
            author=message.from_agent,
            useful_for=content.get('useful_for', []),
            source_project=content.get('source_project')
        )
        
        self.knowledge_base.append(knowledge)
        
        logger.info(
            f"Knowledge shared by {message.from_agent}: "
            f"{knowledge.title} ({knowledge.category.value})"
        )
        
        # Broadcast jeśli włączony
        if self.broadcast_enabled:
            return self.broadcast_message(
                from_agent=message.from_agent,
                content={
                    'action': 'new_knowledge',
                    'knowledge_id': knowledge.knowledge_id,
                    'title': knowledge.title,
                    'category': knowledge.category.value,
                    'tags': knowledge.tags
                }
            )
        
        return None
    
    def _handle_upvote(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż upvote"""
        knowledge_id = message.content.get('knowledge_id')
        
        for item in self.knowledge_base:
            if item.knowledge_id == knowledge_id:
                item.upvotes += 1
                logger.info(f"Upvote for {knowledge_id} by {message.from_agent}")
                break
        
        return None
    
    def _handle_downvote(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż downvote"""
        knowledge_id = message.content.get('knowledge_id')
        
        for item in self.knowledge_base:
            if item.knowledge_id == knowledge_id:
                item.downvotes += 1
                logger.info(f"Downvote for {knowledge_id} by {message.from_agent}")
                break
        
        return None
    
    def _handle_comment(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż komentarz"""
        knowledge_id = message.content.get('knowledge_id')
        comment_text = message.content.get('comment')
        
        for item in self.knowledge_base:
            if item.knowledge_id == knowledge_id:
                comment = {
                    'agent': message.from_agent,
                    'comment': comment_text,
                    'timestamp': datetime.now().isoformat()
                }
                item.comments.append(comment)
                logger.info(f"Comment added to {knowledge_id}")
                break
        
        return None
    
    def _handle_verification(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż weryfikację wiedzy (przez eksperta)"""
        knowledge_id = message.content.get('knowledge_id')
        
        for item in self.knowledge_base:
            if item.knowledge_id == knowledge_id:
                item.verified = True
                item.verified_by = message.from_agent
                logger.info(f"Knowledge {knowledge_id} verified by {message.from_agent}")
                break
        
        return None
    
    def _handle_query(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Obsłuż query o wiedzę"""
        query = message.content.get('query', '')
        tags = message.content.get('tags', [])
        category = message.content.get('category')
        
        # Proste wyszukiwanie
        results = []
        for item in self.knowledge_base:
            match = False
            
            # Sprawdź query w tytule/content
            if query and (query.lower() in item.title.lower() or 
                         query.lower() in item.content.lower()):
                match = True
            
            # Sprawdź tagi
            if tags and any(tag in item.tags for tag in tags):
                match = True
            
            # Sprawdź kategorię
            if category and item.category.value == category:
                match = True
            
            if match:
                item.views += 1
                results.append(item)
        
        logger.info(
            f"Knowledge query by {message.from_agent}: "
            f"found {len(results)} items"
        )
        
        # Odpowiedź z wynikami
        return self.send_message(
            from_agent='system',
            to_agent=message.from_agent,
            content={
                'action': 'query_results',
                'count': len(results),
                'results': [
                    {
                        'id': item.knowledge_id,
                        'title': item.title,
                        'category': item.category.value,
                        'author': item.author,
                        'upvotes': item.upvotes
                    }
                    for item in sorted(results, key=lambda x: x.upvotes, reverse=True)[:10]
                ]
            }
        )
    
    def share_knowledge(
        self,
        agent_id: str,
        category: KnowledgeCategory,
        title: str,
        content: str,
        tags: List[str] = None,
        technologies: List[str] = None
    ) -> KnowledgeItem:
        """Wygodna metoda do sharingu wiedzy"""
        message = ProtocolMessage(
            message_id=f"msg_{len(self.messages)}",
            from_agent=agent_id,
            to_agent=None,
            timestamp=datetime.now(),
            content={
                'action': 'share_knowledge',
                'category': category.name,
                'title': title,
                'content': content,
                'tags': tags or [],
                'technologies': technologies or []
            },
            protocol_type=self.protocol_type
        )
        
        self._handle_knowledge_share(message)
        return self.knowledge_base[-1]
    
    def complete(self) -> Dict[str, Any]:
        """Zakończ knowledge sharing i zwróć wynik"""
        self.status = ProtocolStatus.COMPLETED
        self.completed_at = datetime.now()
        
        # Statystyki
        by_category = {}
        for item in self.knowledge_base:
            cat = item.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        total_upvotes = sum(item.upvotes for item in self.knowledge_base)
        verified_count = sum(1 for item in self.knowledge_base if item.verified)
        
        self.result = {
            'total_items': len(self.knowledge_base),
            'by_category': by_category,
            'total_upvotes': total_upvotes,
            'verified_items': verified_count,
            'top_contributors': self._get_top_contributors(),
            'most_viewed': [
                {'id': item.knowledge_id, 'title': item.title, 'views': item.views}
                for item in sorted(self.knowledge_base, key=lambda x: x.views, reverse=True)[:5]
            ]
        }
        
        logger.info(f"Knowledge Sharing completed: {self.result}")
        return self.result
    
    def _get_top_contributors(self) -> List[Dict[str, Any]]:
        """Pobierz top contributors"""
        contributors = {}
        for item in self.knowledge_base:
            if item.author:
                contributors[item.author] = contributors.get(item.author, 0) + 1
        
        return [
            {'agent': agent, 'contributions': count}
            for agent, count in sorted(contributors.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def get_trending_knowledge(self, limit: int = 10) -> List[KnowledgeItem]:
        """Pobierz trending knowledge (most upvotes, recent)"""
        return sorted(
            self.knowledge_base,
            key=lambda x: (x.upvotes - x.downvotes, x.created_at),
            reverse=True
        )[:limit]


def create_knowledge_sharing() -> KnowledgeSharingProtocol:
    """Utwórz Knowledge Sharing Protocol"""
    return KnowledgeSharingProtocol()
