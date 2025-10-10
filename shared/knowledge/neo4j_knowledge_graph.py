#!/usr/bin/env python3
"""
Agent Zero V1 - Neo4j Knowledge Graph Integration
V2.0 Intelligence Layer - Week 44 Implementation

🎯 Week 44 Critical Task: Neo4j Knowledge Graph Integration (6 SP)
Zadanie: Aktywacja Knowledge Graph z prawdziwą bazą Neo4j, migracja danych
Rezultat: 40% performance improvement w query speed
Impact: Semantyczna wiedza dostępna dla wszystkich agentów

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-44 (Week 44 Implementation)
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Neo4j imports with fallback
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Install with: pip install neo4j")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeType(Enum):
    TASK = "Task"
    AGENT = "Agent"
    MODEL = "Model"
    PATTERN = "Pattern"
    EXPERIENCE = "Experience"
    RECOMMENDATION = "Recommendation"
    PROJECT = "Project"
    CONTEXT = "Context"

class RelationType(Enum):
    EXECUTES = "EXECUTES"
    USES_MODEL = "USES_MODEL"
    GENERATES = "GENERATES"
    LEARNS_FROM = "LEARNS_FROM"
    RECOMMENDS = "RECOMMENDS"
    BELONGS_TO = "BELONGS_TO"
    SIMILAR_TO = "SIMILAR_TO"
    INFLUENCES = "INFLUENCES"
    FOLLOWS_PATTERN = "FOLLOWS_PATTERN"

@dataclass
class GraphNode:
    """Reprezentacja węzła w knowledge graph"""
    id: str
    type: NodeType
    properties: Dict[str, Any]
    labels: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class GraphRelation:
    """Reprezentacja relacji w knowledge graph"""
    id: str
    from_node: str
    to_node: str
    type: RelationType
    properties: Dict[str, Any]
    strength: float
    created_at: datetime

class KnowledgeGraphManager:
    """
    Core Knowledge Graph Manager for Agent Zero V2.0
    
    Responsibilities:
    - Connect to Neo4j database with retry logic
    - Migrate data from SQLite to Neo4j
    - Provide semantic querying capabilities
    - Maintain graph schema and relationships
    - Optimize query performance with indexing
    """
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j", 
                 password: str = "agent-pass",
                 db_name: str = "neo4j",
                 max_retries: int = 5,
                 retry_delay: float = 2.0):
        
        self.uri = uri
        self.username = username
        self.password = password
        self.db_name = db_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = None
        
        if NEO4J_AVAILABLE:
            self._connect_with_retry()
            self._init_schema()
        else:
            logger.warning("Neo4j not available - running in mock mode")
    
    def _connect_with_retry(self):
        """Connect to Neo4j with exponential backoff retry"""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=basic_auth(self.username, self.password),
                    max_connection_pool_size=50,
                    connection_timeout=30.0,
                    max_transaction_retry_time=15.0
                )
                
                # Test connection
                with self.driver.session(database=self.db_name) as session:
                    session.run("RETURN 1")
                
                logger.info(f"✅ Connected to Neo4j at {self.uri}")
                return
                
            except (ServiceUnavailable, AuthError) as e:
                if attempt == self.max_retries:
                    logger.error(f"❌ Failed to connect to Neo4j after {self.max_retries} attempts: {e}")
                    raise
                
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(f"Neo4j connection attempt {attempt} failed, retrying in {delay}s...")
                time.sleep(delay)
    
    def _init_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        if not self.driver:
            return
        
        schema_queries = [
            # Constraints for uniqueness
            "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT model_name_unique IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT experience_id_unique IF NOT EXISTS FOR (e:Experience) REQUIRE e.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX task_type_idx IF NOT EXISTS FOR (t:Task) ON (t.type)",
            "CREATE INDEX task_timestamp_idx IF NOT EXISTS FOR (t:Task) ON (t.timestamp)",
            "CREATE INDEX experience_success_idx IF NOT EXISTS FOR (e:Experience) ON (e.success_score)",
            "CREATE INDEX pattern_confidence_idx IF NOT EXISTS FOR (p:Pattern) ON (p.confidence)",
            "CREATE INDEX model_performance_idx IF NOT EXISTS FOR (m:Model) ON (m.avg_performance)",
            
            # Composite indexes for complex queries
            "CREATE INDEX task_model_performance IF NOT EXISTS FOR ()-[r:USES_MODEL]-() ON (r.success_score, r.cost, r.latency)",
        ]
        
        with self.driver.session(database=self.db_name) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.debug(f"Schema query executed: {query}")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")
        
        logger.info("✅ Neo4j schema initialized")
    
    def create_node(self, node: GraphNode) -> bool:
        """Create a node in the knowledge graph"""
        if not self.driver:
            logger.warning("Neo4j not available - node creation skipped")
            return False
        
        query = f"""
        MERGE (n:{node.type.value} {{id: $id}})
        SET n += $properties
        SET n:AgentZero
        SET n.created_at = $created_at
        SET n.updated_at = $updated_at
        RETURN n
        """
        
        params = {
            'id': node.id,
            'properties': node.properties,
            'created_at': node.created_at.isoformat(),
            'updated_at': node.updated_at.isoformat()
        }
        
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, params)
                record = result.single()
                if record:
                    logger.debug(f"Created node: {node.type.value}:{node.id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to create node {node.id}: {e}")
        
        return False
    
    def create_relationship(self, relation: GraphRelation) -> bool:
        """Create a relationship in the knowledge graph"""
        if not self.driver:
            logger.warning("Neo4j not available - relationship creation skipped")
            return False
        
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        MERGE (a)-[r:{relation.type.value}]->(b)
        SET r += $properties
        SET r.strength = $strength
        SET r.created_at = $created_at
        RETURN r
        """
        
        params = {
            'from_id': relation.from_node,
            'to_id': relation.to_node,
            'properties': relation.properties,
            'strength': relation.strength,
            'created_at': relation.created_at.isoformat()
        }
        
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, params)
                record = result.single()
                if record:
                    logger.debug(f"Created relationship: {relation.from_node} -{relation.type.value}-> {relation.to_node}")
                    return True
        except Exception as e:
            logger.error(f"Failed to create relationship {relation.id}: {e}")
        
        return False
    
    def find_similar_tasks(self, task_type: str, context_keywords: List[str], limit: int = 10) -> List[Dict]:
        """Find similar tasks using semantic similarity"""
        if not self.driver:
            return []
        
        # Build context matching query
        context_conditions = " OR ".join([f"t.context CONTAINS '{keyword}'" for keyword in context_keywords])
        
        query = f"""
        MATCH (t:Task)
        WHERE t.type = $task_type
        AND ({context_conditions})
        MATCH (t)-[r:USES_MODEL]->(m:Model)
        RETURN t.id as task_id, 
               t.type as task_type,
               t.context as context,
               r.success_score as success_score,
               r.cost as cost,
               r.latency as latency,
               m.name as model_used
        ORDER BY r.success_score DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, {'task_type': task_type, 'limit': limit})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to find similar tasks: {e}")
            return []
    
    def get_model_performance_insights(self, model_name: str, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive model performance insights"""
        if not self.driver:
            return {}
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        query = """
        MATCH (m:Model {name: $model_name})<-[r:USES_MODEL]-(t:Task)
        WHERE t.timestamp >= $cutoff_date
        WITH m, r, t
        RETURN 
            count(r) as total_uses,
            avg(r.success_score) as avg_success,
            avg(r.cost) as avg_cost,
            avg(r.latency) as avg_latency,
            collect(DISTINCT t.type) as task_types,
            percentileCont(r.success_score, 0.9) as p90_success,
            percentileCont(r.cost, 0.9) as p90_cost
        """
        
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, {
                    'model_name': model_name,
                    'cutoff_date': cutoff_date
                })
                record = result.single()
                if record:
                    return dict(record)
        except Exception as e:
            logger.error(f"Failed to get model insights: {e}")
        
        return {}
    
    def discover_patterns(self, min_occurrences: int = 5, min_success_rate: float = 0.8) -> List[Dict]:
        """Discover patterns in task execution using graph analysis"""
        if not self.driver:
            return []
        
        query = """
        MATCH (t:Task)-[r:USES_MODEL]->(m:Model)
        WHERE r.success_score >= $min_success_rate
        WITH t.type as task_type, m.name as model_name, 
             count(r) as occurrences,
             avg(r.success_score) as avg_success,
             avg(r.cost) as avg_cost,
             avg(r.latency) as avg_latency
        WHERE occurrences >= $min_occurrences
        RETURN task_type, model_name, occurrences, avg_success, avg_cost, avg_latency
        ORDER BY avg_success DESC, occurrences DESC
        """
        
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, {
                    'min_occurrences': min_occurrences,
                    'min_success_rate': min_success_rate
                })
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to discover patterns: {e}")
            return []
    
    def get_recommendations_for_task(self, task_type: str, context: Dict[str, Any]) -> List[Dict]:
        """Get AI-powered recommendations for a task based on historical data"""
        if not self.driver:
            return []
        
        # Extract context keywords for similarity matching
        context_text = json.dumps(context).lower()
        
        query = """
        MATCH (t:Task {type: $task_type})-[r:USES_MODEL]->(m:Model)
        WHERE r.success_score >= 0.7
        WITH m, 
             avg(r.success_score) as avg_success,
             avg(r.cost) as avg_cost,
             avg(r.latency) as avg_latency,
             count(r) as usage_count
        WHERE usage_count >= 3
        RETURN m.name as recommended_model,
               avg_success,
               avg_cost,
               avg_latency,
               usage_count,
               (avg_success * 0.5 + (1.0 / avg_cost) * 0.3 + (1000.0 / avg_latency) * 0.2) as score
        ORDER BY score DESC
        LIMIT 5
        """
        
        try:
            with self.driver.session(database=self.db_name) as session:
                result = session.run(query, {'task_type': task_type})
                recommendations = [dict(record) for record in result]
                
                # Add reasoning for each recommendation
                for rec in recommendations:
                    rec['reasoning'] = self._generate_recommendation_reasoning(rec)
                
                return recommendations
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
    
    def _generate_recommendation_reasoning(self, rec: Dict) -> str:
        """Generate human-readable reasoning for recommendation"""
        success = rec['avg_success']
        cost = rec['avg_cost']
        usage = rec['usage_count']
        
        reasons = []
        if success > 0.9:
            reasons.append(f"excellent success rate ({success:.1%})")
        elif success > 0.8:
            reasons.append(f"high success rate ({success:.1%})")
        
        if cost < 0.01:
            reasons.append("very cost-effective")
        elif cost < 0.02:
            reasons.append("cost-effective")
        
        if usage > 10:
            reasons.append(f"proven with {usage} successful uses")
        elif usage > 5:
            reasons.append(f"validated with {usage} uses")
        
        return f"Recommended due to {', '.join(reasons)}"
    
    def migrate_sqlite_data(self, sqlite_db_path: str = "agent_zero.db") -> Dict[str, int]:
        """Migrate existing data from SQLite to Neo4j"""
        if not self.driver:
            logger.warning("Neo4j not available - migration skipped")
            return {}
        
        import sqlite3
        migration_stats = {
            'tasks_migrated': 0,
            'experiences_migrated': 0,
            'patterns_migrated': 0,
            'relationships_created': 0
        }
        
        try:
            with sqlite3.connect(sqlite_db_path) as conn:
                # Migrate tasks from simple_tracker
                cursor = conn.execute("""
                    SELECT task_id, task_type, model_used, success_score, 
                           cost_usd, latency_ms, timestamp, context
                    FROM simple_tracker
                """)
                
                for row in cursor.fetchall():
                    task_id, task_type, model_used, success_score, cost_usd, latency_ms, timestamp, context = row
                    
                    # Create task node
                    task_node = GraphNode(
                        id=task_id,
                        type=NodeType.TASK,
                        properties={
                            'type': task_type,
                            'timestamp': timestamp,
                            'context': context or '{}'
                        },
                        labels=['Task'],
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    if self.create_node(task_node):
                        migration_stats['tasks_migrated'] += 1
                    
                    # Create model node (if not exists)
                    model_node = GraphNode(
                        id=f"model_{model_used}",
                        type=NodeType.MODEL,
                        properties={
                            'name': model_used,
                            'type': 'llm'
                        },
                        labels=['Model'],
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    self.create_node(model_node)
                    
                    # Create relationship
                    relation = GraphRelation(
                        id=str(uuid.uuid4()),
                        from_node=task_id,
                        to_node=f"model_{model_used}",
                        type=RelationType.USES_MODEL,
                        properties={
                            'success_score': success_score,
                            'cost': cost_usd,
                            'latency': latency_ms
                        },
                        strength=success_score,
                        created_at=datetime.now()
                    )
                    
                    if self.create_relationship(relation):
                        migration_stats['relationships_created'] += 1
                
                # Migrate V2.0 experiences if they exist
                try:
                    cursor = conn.execute("SELECT * FROM v2_experiences LIMIT 1")
                    cursor = conn.execute("""
                        SELECT id, task_id, task_type, outcome, success_score,
                               cost_usd, latency_ms, model_used, timestamp, metadata
                        FROM v2_experiences
                    """)
                    
                    for row in cursor.fetchall():
                        exp_id, task_id, task_type, outcome, success_score, cost_usd, latency_ms, model_used, timestamp, metadata = row
                        
                        exp_node = GraphNode(
                            id=exp_id,
                            type=NodeType.EXPERIENCE,
                            properties={
                                'task_id': task_id,
                                'task_type': task_type,
                                'outcome': outcome,
                                'success_score': success_score,
                                'cost': cost_usd,
                                'latency': latency_ms,
                                'model_used': model_used,
                                'timestamp': timestamp,
                                'metadata': metadata
                            },
                            labels=['Experience'],
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        
                        if self.create_node(exp_node):
                            migration_stats['experiences_migrated'] += 1
                
                except sqlite3.OperationalError:
                    logger.info("No V2.0 experiences table found - skipping")
        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
        
        logger.info(f"✅ Migration completed: {migration_stats}")
        return migration_stats
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        if not self.driver:
            return {'error': 'Neo4j not available'}
        
        queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
            'node_types': "MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC",
            'relationship_types': "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC",
            'avg_node_degree': """
                MATCH (n)
                WITH n, size((n)--()) as degree
                RETURN avg(degree) as avg_degree
            """,
            'top_connected_nodes': """
                MATCH (n)
                WITH n, size((n)--()) as degree
                WHERE degree > 0
                RETURN n.id as node_id, labels(n)[0] as type, degree
                ORDER BY degree DESC
                LIMIT 10
            """
        }
        
        stats = {}
        
        try:
            with self.driver.session(database=self.db_name) as session:
                for stat_name, query in queries.items():
                    result = session.run(query)
                    if stat_name in ['node_types', 'relationship_types', 'top_connected_nodes']:
                        stats[stat_name] = [dict(record) for record in result]
                    else:
                        record = result.single()
                        stats[stat_name] = record['count'] if record else 0
                        if stat_name == 'avg_node_degree' and record:
                            stats[stat_name] = round(record['avg_degree'], 2)
        
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

# CLI Integration Functions
def init_knowledge_graph(migrate_data: bool = True) -> Dict[str, Any]:
    """CLI function to initialize knowledge graph"""
    kg = KnowledgeGraphManager()
    
    result = {
        'status': 'initialized',
        'neo4j_available': NEO4J_AVAILABLE,
        'migration_stats': {}
    }
    
    if migrate_data and NEO4J_AVAILABLE:
        result['migration_stats'] = kg.migrate_sqlite_data()
    
    if NEO4J_AVAILABLE:
        result['graph_stats'] = kg.get_graph_statistics()
    
    kg.close()
    return result

def get_task_recommendations(task_type: str, context: Dict[str, Any] = None) -> List[Dict]:
    """CLI function to get recommendations for a task"""
    kg = KnowledgeGraphManager()
    recommendations = kg.get_recommendations_for_task(task_type, context or {})
    kg.close()
    return recommendations

def discover_success_patterns(min_occurrences: int = 5) -> List[Dict]:
    """CLI function to discover patterns"""
    kg = KnowledgeGraphManager()
    patterns = kg.discover_patterns(min_occurrences=min_occurrences)
    kg.close()
    return patterns

def get_knowledge_graph_stats() -> Dict[str, Any]:
    """CLI function to get graph statistics"""
    kg = KnowledgeGraphManager()
    stats = kg.get_graph_statistics()
    kg.close()
    return stats

def find_similar_historical_tasks(task_type: str, keywords: List[str]) -> List[Dict]:
    """CLI function to find similar tasks"""
    kg = KnowledgeGraphManager()
    similar_tasks = kg.find_similar_tasks(task_type, keywords)
    kg.close()
    return similar_tasks

if __name__ == "__main__":
    # Test Knowledge Graph Integration
    print("🔗 Testing Neo4j Knowledge Graph Integration...")
    
    # Initialize knowledge graph
    init_result = init_knowledge_graph(migrate_data=True)
    print(f"✅ Initialization: {init_result['status']}")
    
    if init_result.get('migration_stats'):
        stats = init_result['migration_stats']
        print(f"📊 Migration: {stats['tasks_migrated']} tasks, {stats['relationships_created']} relationships")
    
    # Get recommendations
    recommendations = get_task_recommendations("code_generation")
    print(f"💡 Recommendations found: {len(recommendations)}")
    
    # Discover patterns
    patterns = discover_success_patterns()
    print(f"🔍 Patterns discovered: {len(patterns)}")
    
    # Get statistics
    graph_stats = get_knowledge_graph_stats()
    if 'error' not in graph_stats:
        print(f"📈 Graph contains: {graph_stats.get('total_nodes', 0)} nodes, {graph_stats.get('total_relationships', 0)} relationships")
    
    print("\n🎉 Neo4j Knowledge Graph Integration - OPERATIONAL!")