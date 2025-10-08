
# GENEROWANIE PLIKU 1: Neo4j Client Fix
neo4j_client_code = '''"""
Neo4j Knowledge Graph Client - FIXED VERSION
Agent Zero V1 - Critical Fix A0-5

Fixes:
- Connection retry logic with exponential backoff
- Proper connection pooling
- Health check mechanism
- Comprehensive error handling
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Enhanced Neo4j client with robust connection handling"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 2.0
    ):
        """
        Initialize Neo4j client with retry logic
        
        Args:
            uri: Neo4j connection URI (default: from env)
            username: Database username (default: from env)
            password: Database password (default: from env)
            max_retries: Maximum connection retry attempts
            retry_delay: Base delay between retries (seconds)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "agent-pass")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = None
        
        self._connect_with_retry()
    
    def _connect_with_retry(self) -> None:
        """Establish connection with exponential backoff retry"""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Neo4j connection attempt {attempt}/{self.max_retries}")
                
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_pool_size=50,
                    connection_timeout=30.0,
                    max_transaction_retry_time=15.0
                )
                
                # Verify connection
                self.driver.verify_connectivity()
                logger.info(f"✅ Neo4j connected successfully to {self.uri}")
                return
                
            except ServiceUnavailable as e:
                logger.warning(f"Neo4j service unavailable (attempt {attempt}): {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("❌ Neo4j connection failed after all retries")
                    raise ConnectionError(
                        f"Failed to connect to Neo4j at {self.uri} after {self.max_retries} attempts"
                    )
            
            except AuthError as e:
                logger.error(f"❌ Neo4j authentication failed: {e}")
                raise ValueError(f"Invalid Neo4j credentials for {self.username}")
            
            except Exception as e:
                logger.error(f"❌ Unexpected Neo4j connection error: {e}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check Neo4j database health"""
        try:
            with self.driver.session() as session:
                result = session.run("CALL dbms.components() YIELD versions RETURN versions")
                versions = result.single()
                
                return {
                    "status": "healthy",
                    "uri": self.uri,
                    "versions": versions[0] if versions else [],
                    "connected": True
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "uri": self.uri,
                "error": str(e),
                "connected": False
            }
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query with error handling
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        
        except ServiceUnavailable:
            logger.error("Neo4j service unavailable, attempting reconnection...")
            self._connect_with_retry()
            # Retry query after reconnection
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def store_agent_knowledge(
        self,
        agent_id: str,
        knowledge_type: str,
        content: Dict[str, Any]
    ) -> bool:
        """Store agent knowledge in graph"""
        query = """
        MERGE (a:Agent {id: $agent_id})
        CREATE (k:Knowledge {
            type: $knowledge_type,
            content: $content,
            timestamp: timestamp()
        })
        CREATE (a)-[:KNOWS]->(k)
        RETURN k.timestamp as created_at
        """
        
        try:
            result = self.execute_query(
                query,
                {
                    "agent_id": agent_id,
                    "knowledge_type": knowledge_type,
                    "content": str(content)
                }
            )
            logger.info(f"✅ Stored knowledge for agent {agent_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            return False
    
    def get_agent_knowledge(
        self,
        agent_id: str,
        knowledge_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve agent knowledge from graph"""
        if knowledge_type:
            query = """
            MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(k:Knowledge {type: $knowledge_type})
            RETURN k.content as content, k.timestamp as timestamp
            ORDER BY k.timestamp DESC
            """
            parameters = {"agent_id": agent_id, "knowledge_type": knowledge_type}
        else:
            query = """
            MATCH (a:Agent {id: $agent_id})-[:KNOWS]->(k:Knowledge)
            RETURN k.type as type, k.content as content, k.timestamp as timestamp
            ORDER BY k.timestamp DESC
            """
            parameters = {"agent_id": agent_id}
        
        try:
            return self.execute_query(query, parameters)
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
    def close(self) -> None:
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# Singleton instance
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create Neo4j client singleton"""
    global _neo4j_client
    
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    
    return _neo4j_client


if __name__ == "__main__":
    # Test connection
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Neo4j connection...")
    client = get_neo4j_client()
    
    health = client.health_check()
    print(f"\\nHealth check: {health}")
    
    if health["connected"]:
        print("✅ Neo4j client working correctly!")
    else:
        print("❌ Neo4j client has issues")
'''

with open('neo4j_client_fixed.py', 'w', encoding='utf-8') as f:
    f.write(neo4j_client_code)

print("✅ Generated: neo4j_client_fixed.py")
