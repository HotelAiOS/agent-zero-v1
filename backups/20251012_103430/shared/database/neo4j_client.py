"""
Agent Zero V1 - Neo4j Knowledge Base Client
Production-ready Neo4j connection client with connection pooling,
error handling, retry logic, and monitoring capabilities.

Author: Agent Zero V1 Development Team
Version: 1.0.0
Date: 2025-10-07
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass
from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import ServiceUnavailable, TransientError, ClientError
import threading
from functools import wraps
import json


@dataclass
class Neo4jConfig:
    """Neo4j configuration settings."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "agent_zero_2024!"
    database: str = "neo4j"
    max_connections: int = 50
    connection_timeout: int = 30
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    encrypted: bool = False
    trust: str = "TRUST_ALL_CERTIFICATES"


class Neo4jConnectionError(Exception):
    """Custom exception for Neo4j connection issues."""
    pass


class Neo4jQueryError(Exception):
    """Custom exception for Neo4j query execution issues."""
    pass


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying operations on transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ServiceUnavailable, TransientError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
                except Exception as e:
                    logging.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            raise last_exception
        return wrapper
    return decorator


class Neo4jClient:
    """
    Production-ready Neo4j client with connection pooling, retry logic,
    and comprehensive error handling for Agent Zero V1.
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize Neo4j client with configuration."""
        self.config = config or self._load_config_from_env()
        self.driver: Optional[Driver] = None
        self._connection_lock = threading.Lock()
        self._is_connected = False
        self.logger = self._setup_logging()
        
    def _load_config_from_env(self) -> Neo4jConfig:
        """Load configuration from environment variables."""
        return Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "agent_zero_2024!"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connections=int(os.getenv("NEO4J_MAX_CONNECTIONS", "50")),
            connection_timeout=int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30")),
            max_retry_attempts=int(os.getenv("NEO4J_MAX_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("NEO4J_RETRY_DELAY", "1.0")),
            encrypted=os.getenv("NEO4J_ENCRYPTED", "false").lower() == "true",
            trust=os.getenv("NEO4J_TRUST", "TRUST_ALL_CERTIFICATES")
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for Neo4j operations."""
        logger = logging.getLogger("agent_zero.neo4j")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    def connect(self) -> None:
        """Establish connection to Neo4j database with retry logic."""
        with self._connection_lock:
            if self._is_connected and self.driver:
                return
                
            try:
                self.logger.info(f"Connecting to Neo4j at {self.config.uri}")
                
                self.driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.username, self.config.password),
                    encrypted=self.config.encrypted,
                    trust=self.config.trust,
                    max_connection_pool_size=self.config.max_connections,
                    connection_timeout=self.config.connection_timeout,
                    max_retry_time=30,
                    resolver=None
                )
                
                # Verify connectivity
                self.driver.verify_connectivity()
                self._is_connected = True
                self.logger.info("Successfully connected to Neo4j")
                
            except Exception as e:
                self._is_connected = False
                self.logger.error(f"Failed to connect to Neo4j: {e}")
                raise Neo4jConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def disconnect(self) -> None:
        """Close connection to Neo4j database."""
        with self._connection_lock:
            if self.driver:
                try:
                    self.driver.close()
                    self.logger.info("Disconnected from Neo4j")
                except Exception as e:
                    self.logger.error(f"Error during disconnect: {e}")
                finally:
                    self.driver = None
                    self._is_connected = False
    
    def is_connected(self) -> bool:
        """Check if client is connected to Neo4j."""
        return self._is_connected and self.driver is not None
    
    @contextmanager
    def session(self, database: Optional[str] = None):
        """Context manager for Neo4j sessions."""
        if not self.is_connected():
            self.connect()
            
        session = None
        try:
            session = self.driver.session(
                database=database or self.config.database,
                default_access_mode="WRITE"
            )
            yield session
        except Exception as e:
            self.logger.error(f"Session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Target database name
            
        Returns:
            List of result records as dictionaries
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        parameters = parameters or {}
        
        try:
            with self.session(database) as session:
                self.logger.debug(f"Executing query: {query[:100]}...")
                result = session.run(query, parameters)
                records = [record.data() for record in result]
                self.logger.debug(f"Query returned {len(records)} records")
                return records
                
        except ClientError as e:
            self.logger.error(f"Client error in query execution: {e}")
            raise Neo4jQueryError(f"Query execution failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in query execution: {e}")
            raise Neo4jQueryError(f"Unexpected query error: {e}")
    
    def execute_write_transaction(
        self,
        transaction_function,
        database: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute a write transaction."""
        with self.session(database) as session:
            return session.write_transaction(transaction_function, **kwargs)
    
    def execute_read_transaction(
        self,
        transaction_function,
        database: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute a read transaction."""
        with self.session(database) as session:
            return session.read_transaction(transaction_function, **kwargs)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of Neo4j connection.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            "connected": False,
            "database_accessible": False,
            "node_count": 0,
            "response_time_ms": 0,
            "error": None,
            "timestamp": time.time()
        }
        
        try:
            start_time = time.time()
            
            # Check connection
            if not self.is_connected():
                self.connect()
            
            health_status["connected"] = True
            
            # Test database access with simple query
            result = self.execute_query("MATCH (n) RETURN count(n) as count")
            if result:
                health_status["database_accessible"] = True
                health_status["node_count"] = result[0].get("count", 0)
            
            health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            self.logger.info(f"Health check passed - Response time: {health_status['response_time_ms']}ms")
            
        except Exception as e:
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def create_knowledge_node(
        self,
        node_type: str,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a knowledge node in the graph.
        
        Args:
            node_type: Primary node type/label
            properties: Node properties
            labels: Additional labels for the node
            
        Returns:
            Created node data
        """
        labels = labels or []
        all_labels = [node_type] + labels
        labels_str = ":".join(all_labels)
        
        query = f"""
        CREATE (n:{labels_str} $properties)
        RETURN n, id(n) as node_id
        """
        
        result = self.execute_query(query, {"properties": properties})
        if result:
            node_data = result[0]["n"]
            node_data["node_id"] = result[0]["node_id"]
            self.logger.info(f"Created {node_type} node with ID {node_data['node_id']}")
            return node_data
        
        raise Neo4jQueryError("Failed to create knowledge node")
    
    def find_nodes(
        self,
        node_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by type and optional filters.
        
        Args:
            node_type: Node type/label to search for
            filters: Property filters
            limit: Maximum number of results
            
        Returns:
            List of matching nodes
        """
        filters = filters or {}
        
        where_clauses = []
        parameters = {"limit": limit}
        
        for key, value in filters.items():
            param_name = f"filter_{key}"
            where_clauses.append(f"n.{key} = ${param_name}")
            parameters[param_name] = value
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (n:{node_type})
        {where_clause}
        RETURN n, id(n) as node_id
        LIMIT $limit
        """
        
        results = self.execute_query(query, parameters)
        nodes = []
        for result in results:
            node_data = result["n"]
            node_data["node_id"] = result["node_id"]
            nodes.append(node_data)
        
        return nodes
    
    def create_relationship(
        self,
        from_node_id: int,
        to_node_id: int,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Created relationship data
        """
        properties = properties or {}
        
        query = f"""
        MATCH (a) WHERE id(a) = $from_id
        MATCH (b) WHERE id(b) = $to_id
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r, id(r) as rel_id
        """
        
        parameters = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "properties": properties
        }
        
        result = self.execute_query(query, parameters)
        if result:
            rel_data = result[0]["r"]
            rel_data["rel_id"] = result[0]["rel_id"]
            self.logger.info(f"Created {relationship_type} relationship with ID {rel_data['rel_id']}")
            return rel_data
        
        raise Neo4jQueryError("Failed to create relationship")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            # Get basic stats
            stats_query = """
            CALL apoc.meta.stats() YIELD labels, relTypes, stats
            RETURN labels, relTypes, stats
            """
            
            # Fallback if APOC is not available
            fallback_query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->()
            RETURN 
                count(DISTINCT n) as node_count,
                count(DISTINCT r) as relationship_count,
                collect(DISTINCT labels(n)) as all_labels
            """
            
            try:
                result = self.execute_query(stats_query)
                if result:
                    return result[0]
            except:
                # Use fallback query
                result = self.execute_query(fallback_query)
                if result:
                    return result[0]
            
            return {"error": "Could not retrieve database info"}
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Singleton instance for global use
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get global Neo4j client instance."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client


# Convenience functions for common operations
def execute_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a Cypher query using the global client."""
    client = get_neo4j_client()
    return client.execute_query(query, parameters)


def health_check() -> Dict[str, Any]:
    """Perform health check using the global client."""
    client = get_neo4j_client()
    return client.health_check()


if __name__ == "__main__":
    # Example usage and testing
    print("Agent Zero V1 - Neo4j Client Test")
    print("=" * 40)
    
    # Initialize client
    client = Neo4jClient()
    
    try:
        # Connect and perform health check
        client.connect()
        health = client.health_check()
        print(f"Health check: {json.dumps(health, indent=2)}")
        
        # Test basic operations
        result = client.execute_query("RETURN 'Hello Agent Zero!' as message")
        if result:
            print(f"Test query result: {result[0]['message']}")
        
        # Get database info
        db_info = client.get_database_info()
        print(f"Database info: {json.dumps(db_info, indent=2)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        client.disconnect()
        print("Neo4j client test completed")