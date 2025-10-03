import redis
import json
import logging
from typing import Optional, List
from ..models.schemas import Message, ChatSession

logger = logging.getLogger(__name__)

class RedisStore:
    """Redis storage dla sesji czatu"""
    
    def __init__(self, redis_url: str = "redis://redis-cluster:6379/0", password: str = ""):
        self.client = redis.from_url(redis_url, password=password, decode_responses=True)
        self.ttl = 86400 * 7  # 7 dni
    
    def save_session(self, session: ChatSession) -> bool:
        """Zapisz sesję czatu"""
        try:
            key = f"session:{session.id}"
            self.client.setex(
                key,
                self.ttl,
                json.dumps(session.dict(), default=str)
            )
            # Dodaj do listy sesji użytkownika
            user_sessions_key = f"user:{session.user_id}:sessions"
            self.client.sadd(user_sessions_key, session.id)
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Pobierz sesję"""
        try:
            key = f"session:{session_id}"
            data = self.client.get(key)
            if data:
                return ChatSession(**json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def save_message(self, message: Message) -> bool:
        """Zapisz wiadomość"""
        try:
            key = f"messages:{message.session_id}"
            self.client.rpush(key, json.dumps(message.dict(), default=str))
            self.client.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def get_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Pobierz wiadomości z sesji"""
        try:
            key = f"messages:{session_id}"
            messages_data = self.client.lrange(key, -limit, -1)
            return [Message(**json.loads(msg)) for msg in messages_data]
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Pobierz listę sesji użytkownika"""
        try:
            key = f"user:{user_id}:sessions"
            return list(self.client.smembers(key))
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Usuń sesję"""
        try:
            self.client.delete(f"session:{session_id}")
            self.client.delete(f"messages:{session_id}")
            self.client.srem(f"user:{user_id}:sessions", session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
