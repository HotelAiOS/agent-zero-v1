"""
Cache Manager
In-memory cache dla często używanych danych
"""

from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache Manager
    Prosty in-memory cache z TTL
    (W produkcji: Redis)
    """
    
    def __init__(self, default_ttl_seconds: int = 300):
        """
        Args:
            default_ttl_seconds: Domyślny TTL (5 minut)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
        logger.info(f"CacheManager zainicjalizowany (TTL: {default_ttl_seconds}s)")
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Ustaw wartość w cache
        
        Args:
            key: Klucz
            value: Wartość (serializable)
            ttl_seconds: TTL w sekundach (None = default)
        
        Returns:
            True jeśli sukces
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Pobierz wartość z cache
        
        Args:
            key: Klucz
        
        Returns:
            Wartość lub None jeśli nie istnieje/expired
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Sprawdź czy expired
        if datetime.utcnow() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def delete(self, key: str) -> bool:
        """Usuń klucz z cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Wyczyść cały cache, zwróć liczbę usuniętych kluczy"""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared: {count} keys")
        return count
    
    def cleanup_expired(self) -> int:
        """Usuń expired entries, zwróć liczbę usuniętych"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired keys")
        
        return len(expired_keys)
    
    def exists(self, key: str) -> bool:
        """Sprawdź czy klucz istnieje (i nie expired)"""
        return self.get(key) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Pobierz statystyki cache"""
        return {
            'total_keys': len(self.cache),
            'expired_keys': sum(
                1 for entry in self.cache.values()
                if datetime.utcnow() > entry['expires_at']
            )
        }
    
    # Cache patterns
    def get_project(self, project_id: str) -> Optional[Any]:
        """Pobierz projekt z cache"""
        return self.get(f"project:{project_id}")
    
    def set_project(self, project_id: str, project_data: Any, ttl: int = 300):
        """Zapisz projekt w cache"""
        return self.set(f"project:{project_id}", project_data, ttl)
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Pobierz agenta z cache"""
        return self.get(f"agent:{agent_id}")
    
    def set_agent(self, agent_id: str, agent_data: Any, ttl: int = 600):
        """Zapisz agenta w cache"""
        return self.set(f"agent:{agent_id}", agent_data, ttl)
    
    def get_system_status(self) -> Optional[Dict]:
        """Pobierz system status z cache"""
        return self.get("system:status")
    
    def set_system_status(self, status: Dict, ttl: int = 30):
        """Zapisz system status w cache (krótki TTL)"""
        return self.set("system:status", status, ttl)


# Global instance
_cache_manager: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Pobierz globalny CacheManager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
