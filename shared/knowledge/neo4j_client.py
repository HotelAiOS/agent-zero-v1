"""
Stub Neo4j Client for testing purposes.
"""

from typing import Any, Optional, Dict

class Neo4jClient:
    """
    Stub Neo4j client for testing.
    """
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        
    async def close(self) -> None:
        pass
        
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        return []

