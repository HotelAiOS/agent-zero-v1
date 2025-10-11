import logging
import tiktoken
from typing import List
from ..models.schemas import Message, MessageRole

logger = logging.getLogger(__name__)

class ContextManager:
    """Zarządzanie kontekstem rozmowy"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Policz tokeny w tekście"""
        return len(self.encoding.encode(text))
    
    def build_context(self, messages: List[Message], system_prompt: str = "") -> str:
        """Zbuduj kontekst z historii wiadomości"""
        context_parts = []
        total_tokens = 0
        
        # System prompt
        if system_prompt:
            system_tokens = self.count_tokens(system_prompt)
            if system_tokens < self.max_tokens:
                context_parts.append(f"System: {system_prompt}")
                total_tokens += system_tokens
        
        # Dodawaj wiadomości od najnowszych
        for message in reversed(messages):
            msg_text = f"{message.role.value.capitalize()}: {message.content}"
            msg_tokens = self.count_tokens(msg_text)
            
            if total_tokens + msg_tokens > self.max_tokens:
                break
            
            context_parts.insert(0 if system_prompt else 0, msg_text)
            total_tokens += msg_tokens
        
        return "\n\n".join(context_parts)
    
    def summarize_if_needed(self, messages: List[Message]) -> tuple[bool, str]:
        """Sprawdź czy kontekst wymaga podsumowania"""
        total_tokens = sum(self.count_tokens(msg.content) for msg in messages)
        
        if total_tokens > self.max_tokens * 0.8:  # 80% limitu
            logger.info(f"Context size ({total_tokens}) exceeds threshold, summarization recommended")
            return True, f"Long conversation ({len(messages)} messages, {total_tokens} tokens)"
        
        return False, ""
