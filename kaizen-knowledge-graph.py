#!/usr/bin/env python3
"""
Agent Zero V1 - Neo4j Kaizen Knowledge Graph
V2.0 Intelligence Layer Component - Week 43 Implementation

Pattern recognition między projektami z Neo4j:
- Similar Task Finder - znajdowanie analogicznych zadań
- Model Performance Analysis - success rate by model + task type  
- Improvement Opportunities Detection - automatyczna identyfikacja optymalizacji
- Cross-project Learning - współdzielenie wiedzy między projektami

Integruje z istniejącym neo4j_client.py i SimpleTracker.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from enum import Enum
from collections import defaultdict
import logging

# Neo4j imports
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    neo4j_available = True
except ImportError:
    neo4j_available = False
    print("Warning: Neo4j driver not available. Using mock implementation.")

# Import existing components
import sys
sys.path.append('.')
from simple_tracker import SimpleTracker

try:
    # Try to import existing neo4j client
    exec(open('neo4j_client.py').read(), globals())
    has_existing_client = True
except FileNotFoundError:
    has_existing_client = False

class NodeType(Enum):
    """Typy węzłów w knowledge graph"""
    TASK = "Task"
    MODEL = "Model"
    PROJECT = "Project"
    USER = "User"
    PATTERN = "Pattern"
    OUTCOME = "Outcome"

class RelationType(Enum):
    """Typy relacji w knowledge graph"""
    EXECUTED_BY = "EXECUTED_BY"         # Task -> Model
    BELONGS_TO = "BELONGS_TO"           # Task -> Project
    RATED_BY = "RATED_BY"              # Task -> User
    SIMILAR_TO = "SIMILAR_TO"          # Task -> Task
    RESULTED_IN = "RESULTED_IN"        # Task -> Outcome
    LEARNED_FROM = "LEARNED_FROM"      # Pattern -> Task
    RECOMMENDS = "RECOMMENDS"          # Pattern -> Model

@dataclass
class TaskNode:
    """Węzeł Task w knowledge graph"""
    task_id: str
    task_type: str
    description: Optional[str]
    complexity_score: float  # 0.0-1.0
    context_vector: List[float]  # Embedding dla similarity
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ModelNode:
    """Węzeł Model w knowledge graph"""
    model_name: str
    provider: str
    cost_per_token: float
    avg_latency: int
    capabilities: List[str]
    specializations: List[str]
    reliability_score: float

@dataclass
class OutcomeNode:
    """Węzeł Outcome w knowledge graph"""
    outcome_id: str
    success_score: float  # 0.0-1.0
    quality_rating: float
    cost: float
    latency: int
    user_satisfaction: float
    feedback_comment: Optional[str]

@dataclass
class PatternNode:
    """Węzeł Pattern w knowledge graph"""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    success_rate: float
    confidence: float
    sample_count: int
    discovery_date: datetime

class KaizenKnowledgeGraph:
    """
    Główna klasa dla Neo4j Knowledge Graph systemu Kaizen
    """
    
    def __init__(
        self, 
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j", 
        neo4j_password: str = "password",
        tracker: Optional[SimpleTracker] = None
    ):
        
        self.tracker = tracker or SimpleTracker()
        self.logger = self._setup_logging()
        
        # Neo4j connection
        if neo4j_available:
            try:
                self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                self.neo4j_connected = True
                self.logger.info("Connected to Neo4j successfully")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Neo4j: {e}. Using mock mode.")
                self.neo4j_connected = False
        else:
            self.neo4j_connected = False
        
        # Initialize schema
        if self.neo4j_connected:
            self._initialize_schema()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('kaizen_graph')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_schema(self):
        """Inicjalizuje schemat Neo4j dla Kaizen"""
        
        schema_queries = [
            # Constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Outcome) REQUIRE o.outcome_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pat:Pattern) REQUIRE pat.pattern_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.task_type)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (m:Model) ON (m.provider)",
            "CREATE INDEX IF NOT EXISTS FOR (o:Outcome) ON (o.success_score)",
            "CREATE INDEX IF NOT EXISTS FOR (pat:Pattern) ON (pat.success_rate)"
        ]
        
        if self.neo4j_connected:
            with self.driver.session() as session:
                for query in schema_queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        self.logger.warning(f"Schema query failed: {query} - {e}")
        
        self.logger.info("Neo4j schema initialized")
    
    def ingest_task_from_tracker(self, task_id: str) -> bool:
        """
        Ingests task from SimpleTracker do Neo4j Knowledge Graph
        
        Args:
            task_id: ID zadania z trackera
        
        Returns:
            Success boolean
        """
        
        try:
            # Pobierz task data z trackera
            cursor = self.tracker.conn.execute('''
                SELECT 
                    t.id, t.task_type, t.model_used, t.model_recommended,
                    t.cost_usd, t.latency_ms, t.timestamp, t.context,
                    f.rating, f.comment
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.id = ?
            ''', (task_id,))
            
            result = cursor.fetchone()
            if not result:
                self.logger.warning(f"Task {task_id} not found in tracker")
                return False
            
            (task_id, task_type, model_used, model_recommended, 
             cost, latency, timestamp, context_json, rating, comment) = result
            
            # Parse context
            context = json.loads(context_json) if context_json else {}
            
            # Create task node
            task_node = TaskNode(
                task_id=task_id,
                task_type=task_type,
                description=context.get('description'),
                complexity_score=self._calculate_complexity_score(task_type, cost, latency),
                context_vector=self._generate_context_vector(task_type, context),
                timestamp=datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.now(),
                metadata=context
            )
            
            # Create outcome node if feedback exists
            outcome_node = None
            if rating:
                outcome_node = OutcomeNode(
                    outcome_id=f"{task_id}_outcome",
                    success_score=rating / 5.0,
                    quality_rating=rating,
                    cost=cost or 0.0,
                    latency=latency or 0,
                    user_satisfaction=rating / 5.0,
                    feedback_comment=comment
                )
            
            # Write to Neo4j
            if self.neo4j_connected:
                return self._write_task_to_neo4j(task_node, model_used, outcome_node)
            else:
                # Mock mode - just log
                self.logger.info(f"Mock: Would ingest task {task_id} to Neo4j")
                return True
        
        except Exception as e:
            self.logger.error(f"Error ingesting task {task_id}: {e}")
            return False
    
    def _calculate_complexity_score(self, task_type: str, cost: float, latency: int) -> float:
        """Oblicza complexity score dla zadania"""
        
        # Base complexity by task type
        base_complexity = {
            'chat': 0.3,
            'code_generation': 0.7,
            'analysis': 0.6,
            'pipeline': 0.8,
            'business_parsing': 0.5,
            'orchestration': 0.9
        }
        
        base = base_complexity.get(task_type, 0.5)
        
        # Adjust based on cost and latency
        cost_factor = min((cost or 0) * 100, 0.3)  # Cap at 0.3
        latency_factor = min((latency or 1000) / 10000, 0.2)  # Cap at 0.2
        
        complexity = base + cost_factor + latency_factor
        return min(complexity, 1.0)
    
    def _generate_context_vector(self, task_type: str, context: Dict) -> List[float]:
        """Generuje context vector dla similarity matching"""
        
        # Simple embedding - can be replaced with proper embeddings
        vector = [0.0] * 10  # 10-dimensional vector
        
        # Task type encoding
        task_type_encoding = {
            'chat': [1.0, 0.0, 0.0, 0.0, 0.0],
            'code_generation': [0.0, 1.0, 0.0, 0.0, 0.0],
            'analysis': [0.0, 0.0, 1.0, 0.0, 0.0],
            'pipeline': [0.0, 0.0, 0.0, 1.0, 0.0],
            'business_parsing': [0.0, 0.0, 0.0, 0.0, 1.0],
            'orchestration': [0.5, 0.5, 0.0, 0.0, 0.0]
        }
        
        encoding = task_type_encoding.get(task_type, [0.2, 0.2, 0.2, 0.2, 0.2])
        vector[:5] = encoding
        
        # Context features
        if context:
            vector[5] = 1.0 if 'urgent' in str(context).lower() else 0.0
            vector[6] = 1.0 if 'complex' in str(context).lower() else 0.0
            vector[7] = len(str(context)) / 1000.0  # Context length
            vector[8] = 1.0 if 'api' in str(context).lower() else 0.0
            vector[9] = 1.0 if 'database' in str(context).lower() else 0.0
        
        return vector
    
    def _write_task_to_neo4j(
        self, 
        task_node: TaskNode, 
        model_used: str, 
        outcome_node: Optional[OutcomeNode]
    ) -> bool:
        """Zapisuje task node do Neo4j"""
        
        if not self.neo4j_connected:
            return False
        
        try:
            with self.driver.session() as session:
                # Create task node
                task_query = """
                MERGE (t:Task {task_id: $task_id})
                SET t.task_type = $task_type,
                    t.description = $description,
                    t.complexity_score = $complexity_score,
                    t.context_vector = $context_vector,
                    t.timestamp = datetime($timestamp),
                    t.metadata = $metadata
                """
                
                session.run(task_query, 
                    task_id=task_node.task_id,
                    task_type=task_node.task_type,
                    description=task_node.description,
                    complexity_score=task_node.complexity_score,
                    context_vector=task_node.context_vector,
                    timestamp=task_node.timestamp.isoformat(),
                    metadata=task_node.metadata
                )
                
                # Create/update model node
                model_query = """
                MERGE (m:Model {name: $model_name})
                ON CREATE SET m.provider = 'unknown',
                              m.cost_per_token = 0.0,
                              m.avg_latency = 1000,
                              m.capabilities = [],
                              m.specializations = [],
                              m.reliability_score = 0.8
                """
                
                session.run(model_query, model_name=model_used)
                
                # Create relationship Task -> Model
                rel_query = """
                MATCH (t:Task {task_id: $task_id})
                MATCH (m:Model {name: $model_name})
                MERGE (t)-[:EXECUTED_BY]->(m)
                """
                
                session.run(rel_query, task_id=task_node.task_id, model_name=model_used)
                
                # Create outcome node if exists
                if outcome_node:
                    outcome_query = """
                    MERGE (o:Outcome {outcome_id: $outcome_id})
                    SET o.success_score = $success_score,
                        o.quality_rating = $quality_rating,
                        o.cost = $cost,
                        o.latency = $latency,
                        o.user_satisfaction = $user_satisfaction,
                        o.feedback_comment = $feedback_comment
                    
                    WITH o
                    MATCH (t:Task {task_id: $task_id})
                    MERGE (t)-[:RESULTED_IN]->(o)
                    """
                    
                    session.run(outcome_query,
                        outcome_id=outcome_node.outcome_id,
                        success_score=outcome_node.success_score,
                        quality_rating=outcome_node.quality_rating,
                        cost=outcome_node.cost,
                        latency=outcome_node.latency,
                        user_satisfaction=outcome_node.user_satisfaction,
                        feedback_comment=outcome_node.feedback_comment,
                        task_id=task_node.task_id
                    )
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error writing to Neo4j: {e}")
            return False
    
    def find_similar_tasks(
        self, 
        task_id: str, 
        similarity_threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Znajduje podobne zadania w knowledge graph
        
        Args:
            task_id: ID zadania referencyjnego
            similarity_threshold: Próg podobieństwa (0.0-1.0)
            limit: Max liczba wyników
        
        Returns:
            Lista podobnych zadań z metrykami
        """
        
        if not self.neo4j_connected:
            return self._mock_similar_tasks(task_id, limit)
        
        try:
            with self.driver.session() as session:
                # Pobierz context vector dla reference task
                ref_query = """
                MATCH (t:Task {task_id: $task_id})
                RETURN t.context_vector as context_vector, t.task_type as task_type
                """
                
                ref_result = session.run(ref_query, task_id=task_id).single()
                if not ref_result:
                    return []
                
                ref_vector = ref_result['context_vector']
                ref_task_type = ref_result['task_type']
                
                # Znajdź podobne zadania
                similarity_query = """
                MATCH (t:Task)-[:EXECUTED_BY]->(m:Model)
                OPTIONAL MATCH (t)-[:RESULTED_IN]->(o:Outcome)
                WHERE t.task_id <> $task_id
                AND t.task_type = $task_type
                RETURN 
                    t.task_id as task_id,
                    t.task_type as task_type,
                    t.description as description,
                    t.complexity_score as complexity,
                    t.context_vector as context_vector,
                    m.name as model_used,
                    o.success_score as success_score,
                    o.quality_rating as quality_rating,
                    o.cost as cost,
                    o.latency as latency
                ORDER BY t.timestamp DESC
                LIMIT $limit
                """
                
                results = session.run(similarity_query, 
                    task_id=task_id, 
                    task_type=ref_task_type,
                    limit=limit
                ).data()
                
                # Calculate similarity scores
                similar_tasks = []
                for result in results:
                    if result['context_vector']:
                        similarity = self._calculate_vector_similarity(
                            ref_vector, result['context_vector']
                        )
                        
                        if similarity >= similarity_threshold:
                            similar_tasks.append({
                                'task_id': result['task_id'],
                                'similarity_score': similarity,
                                'task_type': result['task_type'],
                                'description': result['description'],
                                'complexity': result['complexity'],
                                'model_used': result['model_used'],
                                'success_score': result['success_score'],
                                'quality_rating': result['quality_rating'],
                                'cost': result['cost'],
                                'latency': result['latency']
                            })
                
                # Sort by similarity
                similar_tasks.sort(key=lambda x: x['similarity_score'], reverse=True)
                return similar_tasks[:limit]
        
        except Exception as e:
            self.logger.error(f"Error finding similar tasks: {e}")
            return []
    
    def _calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Oblicza cosine similarity między wektorami"""
        
        if len(vec1) != len(vec2):
            return 0.0
        
        try:
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        except Exception:
            return 0.0
    
    def analyze_model_performance_by_context(
        self, 
        days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analizuje wydajność modeli w różnych kontekstach
        
        Args:
            days: Okres analizy w dniach
        
        Returns:
            Dict[model_name -> performance_metrics]
        """
        
        if not self.neo4j_connected:
            return self._mock_model_performance()
        
        try:
            with self.driver.session() as session:
                performance_query = """
                MATCH (t:Task)-[:EXECUTED_BY]->(m:Model)
                OPTIONAL MATCH (t)-[:RESULTED_IN]->(o:Outcome)
                WHERE t.timestamp >= datetime() - duration({days: $days})
                
                WITH m.name as model_name, t.task_type as task_type,
                     count(t) as task_count,
                     avg(o.success_score) as avg_success,
                     avg(o.quality_rating) as avg_quality,
                     avg(o.cost) as avg_cost,
                     avg(o.latency) as avg_latency,
                     avg(t.complexity_score) as avg_complexity
                
                WHERE task_count >= 3
                
                RETURN 
                    model_name,
                    task_type,
                    task_count,
                    avg_success,
                    avg_quality,
                    avg_cost,
                    avg_latency,
                    avg_complexity
                ORDER BY model_name, task_type
                """
                
                results = session.run(performance_query, days=days).data()
                
                # Organize by model
                model_performance = defaultdict(lambda: {
                    'contexts': {},
                    'overall_metrics': {
                        'total_tasks': 0,
                        'avg_success': 0.0,
                        'avg_quality': 0.0,
                        'avg_cost': 0.0,
                        'avg_latency': 0.0
                    }
                })
                
                for result in results:
                    model = result['model_name']
                    task_type = result['task_type']
                    
                    # Context-specific metrics
                    model_performance[model]['contexts'][task_type] = {
                        'task_count': result['task_count'],
                        'avg_success': result['avg_success'] or 0.0,
                        'avg_quality': result['avg_quality'] or 0.0,
                        'avg_cost': result['avg_cost'] or 0.0,
                        'avg_latency': result['avg_latency'] or 0,
                        'avg_complexity': result['avg_complexity'] or 0.0
                    }
                    
                    # Update overall metrics
                    model_performance[model]['overall_metrics']['total_tasks'] += result['task_count']
                
                # Calculate overall averages
                for model, data in model_performance.items():
                    contexts = data['contexts']
                    if contexts:
                        data['overall_metrics']['avg_success'] = sum(
                            ctx['avg_success'] for ctx in contexts.values()
                        ) / len(contexts)
                        
                        data['overall_metrics']['avg_quality'] = sum(
                            ctx['avg_quality'] for ctx in contexts.values()
                        ) / len(contexts)
                        
                        data['overall_metrics']['avg_cost'] = sum(
                            ctx['avg_cost'] for ctx in contexts.values()
                        ) / len(contexts)
                        
                        data['overall_metrics']['avg_latency'] = sum(
                            ctx['avg_latency'] for ctx in contexts.values()
                        ) / len(contexts)
                
                return dict(model_performance)
        
        except Exception as e:
            self.logger.error(f"Error analyzing model performance: {e}")
            return {}
    
    def discover_improvement_opportunities(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Wykrywa możliwości poprawy na podstawie graph patterns
        
        Args:
            days: Okres analizy
        
        Returns:
            Lista opportunities z actionable insights
        """
        
        opportunities = []
        
        try:
            if self.neo4j_connected:
                with self.driver.session() as session:
                    # Opportunity 1: Models with declining performance
                    declining_query = """
                    MATCH (t:Task)-[:EXECUTED_BY]->(m:Model)
                    OPTIONAL MATCH (t)-[:RESULTED_IN]->(o:Outcome)
                    WHERE t.timestamp >= datetime() - duration({days: $days})
                    
                    WITH m.name as model,
                         collect({timestamp: t.timestamp, success: o.success_score}) as performances
                    
                    WHERE size(performances) >= 5
                    
                    WITH model, performances,
                         [p in performances WHERE p.timestamp >= datetime() - duration({days: $recent_days}) | p.success] as recent,
                         [p in performances WHERE p.timestamp < datetime() - duration({days: $recent_days}) | p.success] as older
                    
                    WHERE size(recent) >= 2 AND size(older) >= 2
                    
                    WITH model,
                         reduce(sum = 0.0, x IN recent | sum + x) / size(recent) as recent_avg,
                         reduce(sum = 0.0, x IN older | sum + x) / size(older) as older_avg
                    
                    WHERE recent_avg < older_avg - 0.1
                    
                    RETURN model, recent_avg, older_avg, (older_avg - recent_avg) as decline
                    ORDER BY decline DESC
                    """
                    
                    results = session.run(declining_query, 
                        days=days, 
                        recent_days=days//3  # Recent = last 1/3 of period
                    ).data()
                    
                    for result in results:
                        opportunities.append({
                            'type': 'MODEL_PERFORMANCE_DECLINE',
                            'model': result['model'],
                            'description': f"Model {result['model']} performance declined by {result['decline']:.2f}",
                            'severity': 'HIGH' if result['decline'] > 0.3 else 'MEDIUM',
                            'current_performance': result['recent_avg'],
                            'previous_performance': result['older_avg'],
                            'recommendation': f"Investigate {result['model']} configuration or consider alternative models"
                        })
            
            # Opportunity 2: High-cost, low-success combinations
            model_performance = self.analyze_model_performance_by_context(days)
            
            for model, data in model_performance.items():
                overall = data['overall_metrics']
                
                if (overall['avg_cost'] > 0.01 and 
                    overall['avg_success'] < 0.6 and 
                    overall['total_tasks'] >= 5):
                    
                    opportunities.append({
                        'type': 'HIGH_COST_LOW_SUCCESS',
                        'model': model,
                        'description': f"Model {model} has high cost (${overall['avg_cost']:.4f}) with low success ({overall['avg_success']:.2f})",
                        'severity': 'HIGH',
                        'avg_cost': overall['avg_cost'],
                        'avg_success': overall['avg_success'],
                        'total_tasks': overall['total_tasks'],
                        'recommendation': f"Consider replacing {model} with more cost-effective alternatives"
                    })
            
            # Opportunity 3: Identify best performers for underutilized contexts
            context_usage = defaultdict(list)
            for model, data in model_performance.items():
                for context, metrics in data['contexts'].items():
                    context_usage[context].append((model, metrics))
            
            for context, models in context_usage.items():
                if len(models) >= 2:
                    # Sort by success rate
                    models.sort(key=lambda x: x[1]['avg_success'], reverse=True)
                    best_model, best_metrics = models[0]
                    
                    # Check if best model is underutilized
                    total_context_tasks = sum(m[1]['task_count'] for m in models)
                    best_model_share = best_metrics['task_count'] / total_context_tasks
                    
                    if (best_model_share < 0.5 and 
                        best_metrics['avg_success'] > 0.8 and
                        best_metrics['task_count'] >= 3):
                        
                        opportunities.append({
                            'type': 'UNDERUTILIZED_HIGH_PERFORMER',
                            'model': best_model,
                            'context': context,
                            'description': f"Model {best_model} performs excellently in {context} but is underutilized",
                            'severity': 'MEDIUM',
                            'success_rate': best_metrics['avg_success'],
                            'current_share': best_model_share,
                            'recommendation': f"Increase usage of {best_model} for {context} tasks"
                        })
        
        except Exception as e:
            self.logger.error(f"Error discovering improvement opportunities: {e}")
        
        return opportunities
    
    def create_knowledge_pattern(
        self, 
        pattern_type: str,
        conditions: Dict[str, Any],
        outcomes: Dict[str, Any],
        confidence: float
    ) -> str:
        """
        Tworzy pattern node w knowledge graph
        
        Args:
            pattern_type: Typ wzorca
            conditions: Warunki aplikacji wzorca
            outcomes: Oczekiwane rezultaty
            confidence: Confidence score (0.0-1.0)
        
        Returns:
            Pattern ID
        """
        
        pattern_id = f"pattern_{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.neo4j_connected:
            try:
                with self.driver.session() as session:
                    pattern_query = """
                    CREATE (p:Pattern {
                        pattern_id: $pattern_id,
                        pattern_type: $pattern_type,
                        conditions: $conditions,
                        outcomes: $outcomes,
                        confidence: $confidence,
                        sample_count: 1,
                        discovery_date: datetime(),
                        success_rate: $success_rate
                    })
                    RETURN p.pattern_id as pattern_id
                    """
                    
                    success_rate = outcomes.get('success_rate', 0.5)
                    
                    result = session.run(pattern_query,
                        pattern_id=pattern_id,
                        pattern_type=pattern_type,
                        conditions=conditions,
                        outcomes=outcomes,
                        confidence=confidence,
                        success_rate=success_rate
                    ).single()
                    
                    self.logger.info(f"Created knowledge pattern: {pattern_id}")
                    return result['pattern_id']
            
            except Exception as e:
                self.logger.error(f"Error creating knowledge pattern: {e}")
        
        return pattern_id
    
    def get_cross_project_insights(self, project_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Zwraca insights z cross-project learning
        
        Args:
            project_context: Kontekst projektu (opcjonalny)
        
        Returns:
            Dict z insights i recommendations
        """
        
        insights = {
            'similar_tasks_found': 0,
            'best_practices': [],
            'model_recommendations': [],
            'performance_benchmarks': {},
            'learning_opportunities': []
        }
        
        try:
            if self.neo4j_connected:
                with self.driver.session() as session:
                    # Get cross-project model success rates
                    cross_project_query = """
                    MATCH (t:Task)-[:EXECUTED_BY]->(m:Model)
                    OPTIONAL MATCH (t)-[:RESULTED_IN]->(o:Outcome)
                    WHERE o.success_score IS NOT NULL
                    
                    WITH m.name as model, t.task_type as task_type,
                         avg(o.success_score) as avg_success,
                         count(t) as task_count,
                         collect(DISTINCT coalesce(t.metadata.project_id, 'default')) as projects
                    
                    WHERE task_count >= 3 AND size(projects) >= 2
                    
                    RETURN model, task_type, avg_success, task_count, projects
                    ORDER BY avg_success DESC
                    """
                    
                    results = session.run(cross_project_query).data()
                    
                    for result in results:
                        if result['avg_success'] > 0.8:
                            insights['best_practices'].append({
                                'model': result['model'],
                                'task_type': result['task_type'],
                                'success_rate': result['avg_success'],
                                'projects_validated': len(result['projects']),
                                'sample_size': result['task_count']
                            })
                    
                    insights['similar_tasks_found'] = len(results)
        
        except Exception as e:
            self.logger.error(f"Error getting cross-project insights: {e}")
        
        # Add some mock insights if Neo4j not available
        if not self.neo4j_connected:
            insights.update(self._mock_cross_project_insights())
        
        return insights
    
    # === Mock implementations for testing without Neo4j ===
    
    def _mock_similar_tasks(self, task_id: str, limit: int) -> List[Dict[str, Any]]:
        """Mock implementation of similar tasks finder"""
        return [
            {
                'task_id': f'similar_{i}',
                'similarity_score': 0.8 - (i * 0.1),
                'task_type': 'chat',
                'description': f'Similar task {i}',
                'model_used': 'llama3.2-3b',
                'success_score': 0.8,
                'quality_rating': 4.0
            } for i in range(min(3, limit))
        ]
    
    def _mock_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Mock implementation of model performance analysis"""
        return {
            'llama3.2-3b': {
                'contexts': {
                    'chat': {'task_count': 10, 'avg_success': 0.8, 'avg_quality': 4.0, 'avg_cost': 0.0, 'avg_latency': 800},
                    'analysis': {'task_count': 5, 'avg_success': 0.7, 'avg_quality': 3.5, 'avg_cost': 0.0, 'avg_latency': 1200}
                },
                'overall_metrics': {'total_tasks': 15, 'avg_success': 0.75, 'avg_quality': 3.75, 'avg_cost': 0.0, 'avg_latency': 1000}
            },
            'gpt-4': {
                'contexts': {
                    'code_generation': {'task_count': 8, 'avg_success': 0.9, 'avg_quality': 4.5, 'avg_cost': 0.03, 'avg_latency': 2000}
                },
                'overall_metrics': {'total_tasks': 8, 'avg_success': 0.9, 'avg_quality': 4.5, 'avg_cost': 0.03, 'avg_latency': 2000}
            }
        }
    
    def _mock_cross_project_insights(self) -> Dict[str, Any]:
        """Mock cross-project insights"""
        return {
            'similar_tasks_found': 15,
            'best_practices': [
                {
                    'model': 'llama3.2-3b',
                    'task_type': 'chat',
                    'success_rate': 0.85,
                    'projects_validated': 3,
                    'sample_size': 25
                }
            ],
            'performance_benchmarks': {
                'chat_avg_latency': 900,
                'code_generation_avg_success': 0.82
            }
        }
    
    def close(self):
        """Zamyka połączenie z Neo4j"""
        if self.neo4j_connected and hasattr(self, 'driver'):
            self.driver.close()

# === CLI Integration Functions ===

def sync_tracker_to_graph_cli(days: int = 7) -> Dict:
    """
    CLI wrapper for syncing SimpleTracker data to Neo4j
    
    Usage:
        result = sync_tracker_to_graph_cli(7)
        console.print(f"Synced {result['synced_tasks']} tasks")
    """
    
    graph = KaizenKnowledgeGraph()
    
    # Get recent tasks from tracker
    try:
        cursor = graph.tracker.conn.execute('''
            SELECT id FROM tasks 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days))
        
        task_ids = [row[0] for row in cursor.fetchall()]
        
        synced_count = 0
        for task_id in task_ids:
            if graph.ingest_task_from_tracker(task_id):
                synced_count += 1
        
        return {
            'total_tasks': len(task_ids),
            'synced_tasks': synced_count,
            'success_rate': synced_count / len(task_ids) if task_ids else 0.0
        }
    
    except Exception as e:
        return {'error': str(e), 'synced_tasks': 0}

def find_similar_tasks_cli(task_id: str, limit: int = 5) -> Dict:
    """
    CLI wrapper for finding similar tasks
    
    Usage:
        similar = find_similar_tasks_cli("task-123", 5)
        console.print(f"Found {len(similar['tasks'])} similar tasks")
    """
    
    graph = KaizenKnowledgeGraph()
    similar_tasks = graph.find_similar_tasks(task_id, limit=limit)
    
    return {
        'reference_task': task_id,
        'similar_tasks_count': len(similar_tasks),
        'tasks': [
            {
                'task_id': task['task_id'],
                'similarity': task['similarity_score'],
                'model': task['model_used'],
                'success': task['success_score']
            } for task in similar_tasks
        ]
    }

def get_model_insights_cli(days: int = 30) -> Dict:
    """
    CLI wrapper for model performance insights
    
    Usage:
        insights = get_model_insights_cli(30)
        console.print(f"Analyzed {len(insights['models'])} models")
    """
    
    graph = KaizenKnowledgeGraph()
    performance = graph.analyze_model_performance_by_context(days)
    opportunities = graph.discover_improvement_opportunities(days)
    
    return {
        'analysis_period_days': days,
        'models_analyzed': len(performance),
        'improvement_opportunities': len(opportunities),
        'models': {
            model: data['overall_metrics'] 
            for model, data in performance.items()
        },
        'top_opportunities': opportunities[:5]
    }

# === Testing Functions ===

def test_kaizen_knowledge_graph():
    """Testy funkcjonalne dla Kaizen Knowledge Graph"""
    
    print("=== TESTING KAIZEN KNOWLEDGE GRAPH ===\n")
    
    graph = KaizenKnowledgeGraph()
    
    # Test 1: Ingest task from tracker
    print("Test 1: Task ingestion")
    
    # First create a test task in tracker
    test_task_id = "test_graph_1"
    graph.tracker.track_task(
        task_id=test_task_id,
        task_type="chat",
        model_used="llama3.2-3b",
        model_recommended="llama3.2-3b",
        cost=0.0,
        latency=800,
        context={'description': 'Test chat task for graph'}
    )
    
    # Add feedback
    graph.tracker.record_feedback(test_task_id, 4, "Good response")
    
    # Ingest to graph
    success = graph.ingest_task_from_tracker(test_task_id)
    print(f"Task ingestion success: {success}")
    print()
    
    # Test 2: Find similar tasks
    print("Test 2: Similar tasks discovery")
    
    similar = graph.find_similar_tasks(test_task_id, limit=5)
    print(f"Found {len(similar)} similar tasks")
    for task in similar[:3]:
        print(f"  - {task['task_id']}: similarity {task['similarity_score']:.2f}")
    print()
    
    # Test 3: Model performance analysis
    print("Test 3: Model performance analysis")
    
    performance = graph.analyze_model_performance_by_context(days=30)
    print(f"Analyzed {len(performance)} models:")
    for model, data in performance.items():
        metrics = data['overall_metrics']
        print(f"  - {model}: {metrics['total_tasks']} tasks, {metrics['avg_success']:.2f} success")
    print()
    
    # Test 4: Improvement opportunities
    print("Test 4: Improvement opportunities discovery")
    
    opportunities = graph.discover_improvement_opportunities(days=30)
    print(f"Found {len(opportunities)} improvement opportunities:")
    for opp in opportunities[:3]:
        print(f"  - {opp['type']}: {opp['description']}")
    print()
    
    # Test 5: Cross-project insights
    print("Test 5: Cross-project insights")
    
    insights = graph.get_cross_project_insights()
    print(f"Cross-project insights:")
    print(f"  - Similar tasks found: {insights['similar_tasks_found']}")
    print(f"  - Best practices: {len(insights['best_practices'])}")
    print(f"  - Learning opportunities: {len(insights['learning_opportunities'])}")
    print()
    
    # Test 6: CLI Integration
    print("Test 6: CLI Integration")
    
    sync_result = sync_tracker_to_graph_cli(7)
    print(f"Sync result: {sync_result}")
    
    similar_cli = find_similar_tasks_cli(test_task_id, 3)
    print(f"Similar tasks CLI: {similar_cli['similar_tasks_count']} found")
    
    insights_cli = get_model_insights_cli(30)
    print(f"Model insights CLI: {insights_cli['models_analyzed']} models analyzed")
    
    print("\n=== ALL TESTS COMPLETED ===")
    
    # Cleanup
    graph.close()

if __name__ == "__main__":
    test_kaizen_knowledge_graph()