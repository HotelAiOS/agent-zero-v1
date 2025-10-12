"""
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
        """Initialize Neo4j client with retry logic"""
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
                    delay = self.retry_delay * (2 ** (attempt - 1))
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
        """Execute Cypher query with error handling"""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]

        except ServiceUnavailable:
            logger.error("Neo4j service unavailable, attempting reconnection...")
            self._connect_with_retry()
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise

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
