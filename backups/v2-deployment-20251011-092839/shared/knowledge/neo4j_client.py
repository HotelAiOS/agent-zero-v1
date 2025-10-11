"""
Neo4j Knowledge Graph Client
Long-term memory storage for Agent Zero
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j client for Agent Zero Knowledge Graph
    Stores agent experiences, code patterns, and task relationships
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "agent-zero-password"
    ):
        """Initialize Neo4j client"""
        self.uri = uri
        self.user = user
        self.driver = None
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info(f"✅ Connected to Neo4j at {uri}")
            self._init_schema()
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j not available at {uri}: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def _init_schema(self):
        """Initialize constraints and indexes"""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE")
            session.run("CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE")
            session.run("CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE")
            session.run("CREATE INDEX agent_type_idx IF NOT EXISTS FOR (a:Agent) ON (a.agent_type)")
            session.run("CREATE INDEX task_status_idx IF NOT EXISTS FOR (t:Task) ON (t.status)")
            logger.info("✅ Schema initialized")
    
    def create_agent_node(self, agent_id: str, agent_type: str, capabilities: List[str]) -> bool:
        """Create agent node in knowledge graph"""
        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (a:Agent {agent_id: $agent_id})
                ON CREATE SET 
                    a.agent_type = $agent_type,
                    a.capabilities = $capabilities,
                    a.created_at = datetime()
                RETURN a
                """,
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities
            )
            node = result.single()
            if node:
                logger.info(f"✅ Agent node created: {agent_id}")
                return True
            return False
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent node data"""
        with self.driver.session() as session:
            result = session.run("MATCH (a:Agent {agent_id: $agent_id}) RETURN a", agent_id=agent_id)
            record = result.single()
            if record:
                return dict(record['a'])
            return None
    
    def create_task_node(self, task_id: str, description: str, agent_id: str, project_id: Optional[str] = None, complexity: int = 5) -> bool:
        """Create task node and link to agent"""
        with self.driver.session() as session:
            query = """
            MATCH (a:Agent {agent_id: $agent_id})
            MERGE (t:Task {task_id: $task_id})
            ON CREATE SET 
                t.description = $description,
                t.status = 'in_progress',
                t.complexity = $complexity,
                t.created_at = datetime()
            MERGE (a)-[:EXECUTED]->(t)
            """
            if project_id:
                query += """
                MERGE (p:Project {project_id: $project_id})
                MERGE (t)-[:BELONGS_TO]->(p)
                """
            query += "RETURN t"
            
            result = session.run(query, task_id=task_id, description=description, agent_id=agent_id, project_id=project_id, complexity=complexity)
            if result.single():
                logger.info(f"✅ Task created: {task_id}")
                return True
            return False
    
    def complete_task(self, task_id: str, success: bool, outcome: str) -> bool:
        """Mark task as completed"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Task {task_id: $task_id})
                SET t.status = $status,
                    t.completed_at = datetime(),
                    t.success = $success,
                    t.outcome = $outcome
                RETURN t
                """,
                task_id=task_id,
                status="completed" if success else "failed",
                success=success,
                outcome=outcome
            )
            return result.single() is not None
    
    def store_experience(self, agent_id: str, context: str, outcome: str, success: bool, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store agent experience for learning (metadata stored as JSON string)"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agent {agent_id: $agent_id})
                CREATE (e:Experience {
                    experience_id: randomUUID(),
                    context: $context,
                    outcome: $outcome,
                    success: $success,
                    metadata: $metadata_json,
                    timestamp: datetime()
                })
                MERGE (a)-[:LEARNED]->(e)
                RETURN e.experience_id AS exp_id
                """,
                agent_id=agent_id,
                context=context,
                outcome=outcome,
                success=success,
                metadata_json=json.dumps(metadata or {})
            )
            record = result.single()
            if record:
                exp_id = record['exp_id']
                logger.info(f"✅ Experience stored: {exp_id}")
                return exp_id
            return None
    
    def get_similar_experiences(self, agent_id: str, context_keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar past experiences"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agent {agent_id: $agent_id})-[:LEARNED]->(e:Experience)
                WHERE ANY(keyword IN $keywords WHERE e.context CONTAINS keyword)
                RETURN e
                ORDER BY e.timestamp DESC
                LIMIT $limit
                """,
                agent_id=agent_id,
                keywords=context_keywords,
                limit=limit
            )
            experiences = []
            for record in result:
                exp = dict(record['e'])
                # Parse metadata JSON back to dict
                if 'metadata' in exp and exp['metadata']:
                    try:
                        exp['metadata'] = json.loads(exp['metadata'])
                    except:
                        pass
                experiences.append(exp)
            return experiences
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get agent statistics"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agent {agent_id: $agent_id})
                OPTIONAL MATCH (a)-[:EXECUTED]->(t:Task)
                OPTIONAL MATCH (a)-[:LEARNED]->(e:Experience)
                RETURN 
                    a.agent_type AS agent_type,
                    count(DISTINCT t) AS total_tasks,
                    count(DISTINCT CASE WHEN t.success = true THEN t END) AS successful_tasks,
                    count(DISTINCT e) AS experiences
                """,
                agent_id=agent_id
            )
            record = result.single()
            if record:
                return dict(record)
            return {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
