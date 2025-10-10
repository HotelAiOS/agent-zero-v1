# Neo4j Knowledge Graph - Strategic Module for Agent Zero V1  
# Task: A0-24 Neo4j Knowledge Graph Pattern Recognition (Week 43)
# Focus: Pattern recognition i knowledge reuse między projektami
# Core: KaizenKnowledgeGraph schema implementation

"""
Neo4j Knowledge Graph for Agent Zero V1
Advanced pattern recognition and knowledge reuse system

This module provides:
- Project pattern recognition across executions
- Knowledge graph construction and querying
- Reusable solution templates and best practices
- Kaizen-driven continuous improvement insights
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import hashlib
from collections import defaultdict, Counter

# Neo4j imports
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, CypherError
except ImportError:
    logging.warning("Neo4j driver not available - using mock implementation")
    # Mock classes for testing
    class GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            return MockDriver()
    
    class MockDriver:
        def session(self):
            return MockSession()
        def close(self):
            pass
    
    class MockSession:
        def run(self, query, **kwargs):
            return MockResult()
        def close(self):
            pass
    
    class MockResult:
        def data(self):
            return []

# Import existing components
try:
    import sys
    sys.path.insert(0, '.')
    from simple_tracker import SimpleTracker
    from business_requirements_parser import BusinessRequirementsParser, IntentType, ComplexityLevel
    from project_orchestrator import Project, Task, ProjectState, TaskStatus
    from hierarchical_task_planner import HierarchicalTask, TaskType, TaskPriority
except ImportError as e:
    logging.warning(f"Could not import existing components: {e}")
    # Minimal fallback classes
    class SimpleTracker:
        def track_event(self, event): pass

class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    PROJECT = "Project"
    TASK = "Task" 
    INTENT = "Intent"
    SOLUTION = "Solution"
    PATTERN = "Pattern"
    AGENT = "Agent"
    TOOL = "Tool"
    TECHNOLOGY = "Technology"
    DOMAIN = "Domain"
    USER = "User"
    OUTCOME = "Outcome"
    METRIC = "Metric"

class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    CONTAINS = "CONTAINS"              # Project contains Tasks
    DEPENDS_ON = "DEPENDS_ON"          # Task depends on Task
    IMPLEMENTS = "IMPLEMENTS"          # Task implements Intent
    USES = "USES"                      # Task uses Tool/Agent
    SIMILAR_TO = "SIMILAR_TO"          # Pattern similarity
    EVOLVED_FROM = "EVOLVED_FROM"      # Solution evolution
    BELONGS_TO = "BELONGS_TO"          # Domain classification
    ACHIEVED = "ACHIEVED"              # Task achieved Outcome
    MEASURED_BY = "MEASURED_BY"        # Outcome measured by Metric
    CREATED_BY = "CREATED_BY"          # Created by User/Agent
    SUCCEEDED_BY = "SUCCEEDED_BY"      # Temporal succession
    CONFLICTS_WITH = "CONFLICTS_WITH"  # Conflicting patterns
    ENHANCES = "ENHANCES"              # Enhancement relationships

@dataclass
class GraphNode:
    """Generic graph node"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class GraphRelationship:
    """Generic graph relationship"""
    from_node: str
    to_node: str
    relationship_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    created_at: Optional[datetime] = None

@dataclass
class Pattern:
    """Identified pattern in project execution"""
    pattern_id: str
    name: str
    description: str
    pattern_type: str  # e.g., "task_sequence", "resource_allocation", "failure_mode"
    frequency: int = 1
    success_rate: float = 0.0
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    contexts: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)  # Project/Task IDs
    confidence: float = 0.0
    last_seen: Optional[datetime] = None

@dataclass
class SolutionTemplate:
    """Reusable solution template"""
    template_id: str
    name: str
    description: str
    intent_types: List[IntentType] = field(default_factory=list)
    complexity_range: Tuple[ComplexityLevel, ComplexityLevel] = (ComplexityLevel.SIMPLE, ComplexityLevel.COMPLEX)
    task_sequence: List[Dict[str, Any]] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    estimated_metrics: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    usage_count: int = 0
    feedback_scores: List[int] = field(default_factory=list)

class PatternRecognitionEngine:
    """Engine for identifying patterns in project data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._known_patterns = {}
        
        # Pattern detection thresholds
        self.min_pattern_frequency = 3
        self.min_pattern_confidence = 0.7
        self.similarity_threshold = 0.8
    
    def analyze_project_patterns(self, projects: List[Project]) -> List[Pattern]:
        """Analyze projects to identify recurring patterns"""
        
        patterns = []
        
        # Task sequence patterns
        sequence_patterns = self._find_task_sequence_patterns(projects)
        patterns.extend(sequence_patterns)
        
        # Resource allocation patterns
        resource_patterns = self._find_resource_allocation_patterns(projects)
        patterns.extend(resource_patterns)
        
        # Failure mode patterns
        failure_patterns = self._find_failure_mode_patterns(projects)
        patterns.extend(failure_patterns)
        
        # Success factor patterns
        success_patterns = self._find_success_factor_patterns(projects)
        patterns.extend(success_patterns)
        
        # Filter patterns by confidence and frequency
        filtered_patterns = [
            p for p in patterns 
            if p.frequency >= self.min_pattern_frequency and p.confidence >= self.min_pattern_confidence
        ]
        
        self.logger.info(f"Identified {len(filtered_patterns)} patterns from {len(projects)} projects")
        return filtered_patterns
    
    def _find_task_sequence_patterns(self, projects: List[Project]) -> List[Pattern]:
        """Find common task sequencing patterns"""
        
        sequence_groups = defaultdict(list)
        
        # Group projects by task sequences
        for project in projects:
            if not project.tasks:
                continue
            
            # Extract task sequence signature
            task_sequence = []
            sorted_tasks = sorted(
                project.tasks.values(),
                key=lambda t: t.started_at or t.created_at or datetime.now()
            )
            
            for task in sorted_tasks[:10]:  # Limit to first 10 tasks
                # Create signature based on business request keywords
                keywords = self._extract_keywords(task.business_request)
                task_signature = "_".join(sorted(keywords[:3]))  # Top 3 keywords
                task_sequence.append(task_signature)
            
            if len(task_sequence) >= 2:
                sequence_key = "|".join(task_sequence)
                sequence_groups[sequence_key].append(project.project_id)
        
        # Create patterns for frequent sequences
        patterns = []
        for sequence, project_ids in sequence_groups.items():
            if len(project_ids) >= self.min_pattern_frequency:
                # Calculate metrics for this pattern
                pattern_projects = [p for p in projects if p.project_id in project_ids]
                success_rate = sum(1 for p in pattern_projects if p.state == ProjectState.COMPLETED) / len(pattern_projects)
                avg_cost = sum(p.metrics.actual_cost for p in pattern_projects if p.metrics) / len(pattern_projects)
                avg_duration = sum(p.metrics.actual_duration for p in pattern_projects if p.metrics) / len(pattern_projects)
                
                pattern = Pattern(
                    pattern_id=f"seq_{hashlib.md5(sequence.encode()).hexdigest()[:8]}",
                    name=f"Task Sequence: {sequence.replace('|', ' → ')}",
                    description=f"Common task sequence appearing in {len(project_ids)} projects",
                    pattern_type="task_sequence",
                    frequency=len(project_ids),
                    success_rate=success_rate,
                    avg_cost=avg_cost,
                    avg_duration=avg_duration,
                    examples=project_ids[:5],
                    confidence=min(1.0, len(project_ids) / 10),  # Confidence based on frequency
                    last_seen=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_resource_allocation_patterns(self, projects: List[Project]) -> List[Pattern]:
        """Find patterns in resource allocation"""
        
        allocation_groups = defaultdict(list)
        
        for project in projects:
            if not project.tasks:
                continue
            
            # Calculate resource allocation signature
            agent_usage = Counter()
            total_tasks = len(project.tasks)
            
            for task in project.tasks.values():
                if hasattr(task, 'assigned_agents') and task.assigned_agents:
                    for agent in task.assigned_agents:
                        agent_usage[agent] += 1
            
            # Create allocation signature (top 3 most used agents with percentages)
            if agent_usage:
                top_agents = agent_usage.most_common(3)
                allocation_signature = []
                for agent, count in top_agents:
                    percentage = round((count / total_tasks) * 100)
                    allocation_signature.append(f"{agent}:{percentage}%")
                
                allocation_key = "|".join(allocation_signature)
                allocation_groups[allocation_key].append(project.project_id)
        
        # Create patterns for frequent allocations
        patterns = []
        for allocation, project_ids in allocation_groups.items():
            if len(project_ids) >= self.min_pattern_frequency:
                pattern_projects = [p for p in projects if p.project_id in project_ids]
                success_rate = sum(1 for p in pattern_projects if p.state == ProjectState.COMPLETED) / len(pattern_projects)
                
                pattern = Pattern(
                    pattern_id=f"res_{hashlib.md5(allocation.encode()).hexdigest()[:8]}",
                    name=f"Resource Allocation: {allocation.replace('|', ', ')}",
                    description=f"Common resource allocation pattern in {len(project_ids)} projects",
                    pattern_type="resource_allocation",
                    frequency=len(project_ids),
                    success_rate=success_rate,
                    examples=project_ids[:5],
                    confidence=min(1.0, len(project_ids) / 8),
                    last_seen=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_failure_mode_patterns(self, projects: List[Project]) -> List[Pattern]:
        """Find common failure patterns"""
        
        failure_patterns = defaultdict(list)
        
        failed_projects = [p for p in projects if p.state == ProjectState.FAILED]
        
        for project in failed_projects:
            failure_indicators = []
            
            # Check for common failure indicators
            if project.metrics:
                if project.metrics.actual_cost > project.metrics.estimated_cost * 2:
                    failure_indicators.append("cost_overrun")
                if project.metrics.actual_duration > project.metrics.estimated_duration * 2:
                    failure_indicators.append("time_overrun")
                if project.metrics.failed_tasks > project.metrics.total_tasks * 0.5:
                    failure_indicators.append("high_failure_rate")
            
            # Check task failure patterns  
            failed_tasks = [t for t in project.tasks.values() if t.status == TaskStatus.FAILED]
            if failed_tasks:
                common_errors = Counter()
                for task in failed_tasks:
                    if hasattr(task, 'error_message') and task.error_message:
                        # Extract error type keywords
                        error_keywords = self._extract_keywords(task.error_message)
                        for keyword in error_keywords[:2]:
                            common_errors[keyword] += 1
                
                if common_errors:
                    top_error = common_errors.most_common(1)[0][0]
                    failure_indicators.append(f"error_{top_error}")
            
            # Group by failure signature
            if failure_indicators:
                failure_key = "|".join(sorted(failure_indicators))
                failure_patterns[failure_key].append(project.project_id)
        
        # Create failure patterns
        patterns = []
        for failure_sig, project_ids in failure_patterns.items():
            if len(project_ids) >= 2:  # Lower threshold for failure patterns
                pattern = Pattern(
                    pattern_id=f"fail_{hashlib.md5(failure_sig.encode()).hexdigest()[:8]}",
                    name=f"Failure Pattern: {failure_sig.replace('|', ', ')}",
                    description=f"Common failure mode in {len(project_ids)} projects",
                    pattern_type="failure_mode",
                    frequency=len(project_ids),
                    success_rate=0.0,  # Failure patterns have 0% success rate
                    examples=project_ids,
                    confidence=min(1.0, len(project_ids) / 3),
                    last_seen=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_success_factor_patterns(self, projects: List[Project]) -> List[Pattern]:
        """Find patterns that correlate with success"""
        
        successful_projects = [
            p for p in projects 
            if p.state == ProjectState.COMPLETED and p.metrics and p.metrics.success_rate > 0.8
        ]
        
        if len(successful_projects) < self.min_pattern_frequency:
            return []
        
        success_factors = Counter()
        
        for project in successful_projects:
            factors = []
            
            # Time/cost efficiency factors
            if project.metrics:
                if project.metrics.actual_cost <= project.metrics.estimated_cost * 1.1:
                    factors.append("cost_efficient")
                if project.metrics.actual_duration <= project.metrics.estimated_duration * 1.1:
                    factors.append("time_efficient")
                if project.metrics.completion_rate >= 0.95:
                    factors.append("high_completion")
            
            # Team composition factors
            if project.tasks:
                agent_diversity = set()
                for task in project.tasks.values():
                    if hasattr(task, 'assigned_agents') and task.assigned_agents:
                        agent_diversity.update(task.assigned_agents)
                
                if len(agent_diversity) > 1:
                    factors.append("diverse_team")
                if len(agent_diversity) <= 2:
                    factors.append("focused_team")
            
            for factor in factors:
                success_factors[factor] += 1
        
        # Create success factor patterns
        patterns = []
        total_successful = len(successful_projects)
        
        for factor, count in success_factors.items():
            if count >= self.min_pattern_frequency:
                correlation = count / total_successful
                
                pattern = Pattern(
                    pattern_id=f"succ_{hashlib.md5(factor.encode()).hexdigest()[:8]}",
                    name=f"Success Factor: {factor.replace('_', ' ').title()}",
                    description=f"Factor present in {count}/{total_successful} successful projects",
                    pattern_type="success_factor",
                    frequency=count,
                    success_rate=1.0,  # Success factors have 100% success rate by definition
                    confidence=correlation,
                    last_seen=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for pattern analysis"""
        
        # Simple keyword extraction (in production, could use NLP)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        words = text.lower().split()
        keywords = [
            word.strip('.,!?()[]{}";:') 
            for word in words 
            if len(word) > 3 and word not in stop_words
        ]
        
        # Count frequency and return most common
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

class KnowledgeGraphBuilder:
    """Builds and manages the Neo4j knowledge graph"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "password123"):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
            self._test_connection()
            self.connected = True
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = MockDriver()
            self.connected = False
        
        # Initialize graph schema
        self._initialize_schema()
    
    def _test_connection(self):
        """Test Neo4j connection"""
        with self.driver.session() as session:
            result = session.run("RETURN 'Connected' as status")
            data = result.data()
            if data and data[0]['status'] == 'Connected':
                self.logger.info("Successfully connected to Neo4j")
            else:
                raise Exception("Connection test failed")
    
    def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        
        schema_queries = [
            # Unique constraints
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE",
            "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE",
            "CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS FOR (pt:Pattern) REQUIRE pt.pattern_id IS UNIQUE",
            "CREATE CONSTRAINT solution_id_unique IF NOT EXISTS FOR (s:Solution) REQUIRE s.template_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX project_state_idx IF NOT EXISTS FOR (p:Project) ON (p.state)",
            "CREATE INDEX task_status_idx IF NOT EXISTS FOR (t:Task) ON (t.status)",
            "CREATE INDEX pattern_type_idx IF NOT EXISTS FOR (pt:Pattern) ON (pt.pattern_type)",
            "CREATE INDEX intent_type_idx IF NOT EXISTS FOR (i:Intent) ON (i.intent_type)",
            
            # Full-text search indexes
            "CALL db.index.fulltext.createNodeIndex('projectSearch', ['Project'], ['name', 'description']) IF NOT EXISTS",
            "CALL db.index.fulltext.createNodeIndex('taskSearch', ['Task'], ['business_request', 'description']) IF NOT EXISTS"
        ]
        
        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                except Exception as e:
                    # Some constraints might already exist
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Schema query failed: {query} - {e}")
    
    def add_project_to_graph(self, project: Project):
        """Add project and its tasks to the knowledge graph"""
        
        with self.driver.session() as session:
            # Create project node
            project_query = """
            MERGE (p:Project {project_id: $project_id})
            SET p.name = $name,
                p.description = $description,
                p.state = $state,
                p.created_at = datetime($created_at),
                p.started_at = datetime($started_at),
                p.completed_at = datetime($completed_at),
                p.total_tasks = $total_tasks,
                p.completed_tasks = $completed_tasks,
                p.success_rate = $success_rate,
                p.estimated_cost = $estimated_cost,
                p.actual_cost = $actual_cost,
                p.estimated_duration = $estimated_duration,
                p.actual_duration = $actual_duration,
                p.updated_at = datetime()
            """
            
            metrics = project.metrics or type('', (), {
                'total_tasks': 0, 'completed_tasks': 0, 'success_rate': 0.0,
                'estimated_cost': 0.0, 'actual_cost': 0.0,
                'estimated_duration': 0, 'actual_duration': 0
            })()
            
            session.run(project_query, {
                'project_id': project.project_id,
                'name': project.name,
                'description': project.description,
                'state': project.state.value,
                'created_at': project.created_at.isoformat() if project.created_at else None,
                'started_at': project.started_at.isoformat() if project.started_at else None,
                'completed_at': project.completed_at.isoformat() if project.completed_at else None,
                'total_tasks': metrics.total_tasks,
                'completed_tasks': metrics.completed_tasks,
                'success_rate': metrics.success_rate,
                'estimated_cost': metrics.estimated_cost,
                'actual_cost': metrics.actual_cost,
                'estimated_duration': metrics.estimated_duration,
                'actual_duration': metrics.actual_duration
            })
            
            # Add tasks and relationships
            for task_id, task in project.tasks.items():
                self._add_task_to_graph(session, project.project_id, task)
    
    def _add_task_to_graph(self, session, project_id: str, task: Task):
        """Add task to graph and create relationships"""
        
        # Create task node
        task_query = """
        MERGE (t:Task {task_id: $task_id})
        SET t.business_request = $business_request,
            t.status = $status,
            t.created_at = datetime($created_at),
            t.started_at = datetime($started_at),
            t.completed_at = datetime($completed_at),
            t.estimated_duration = $estimated_duration,
            t.actual_duration = $actual_duration,
            t.estimated_cost = $estimated_cost,
            t.actual_cost = $actual_cost,
            t.progress_percentage = $progress_percentage,
            t.updated_at = datetime()
        """
        
        session.run(task_query, {
            'task_id': task.task_id,
            'business_request': task.business_request,
            'status': task.status.value,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'estimated_duration': getattr(task, 'estimated_duration', 0),
            'actual_duration': getattr(task, 'actual_duration', 0),
            'estimated_cost': getattr(task, 'estimated_cost', 0.0),
            'actual_cost': getattr(task, 'actual_cost', 0.0),
            'progress_percentage': getattr(task, 'progress_percentage', 0)
        })
        
        # Create project-task relationship
        rel_query = """
        MATCH (p:Project {project_id: $project_id})
        MATCH (t:Task {task_id: $task_id})
        MERGE (p)-[:CONTAINS]->(t)
        """
        
        session.run(rel_query, {
            'project_id': project_id,
            'task_id': task.task_id
        })
        
        # Create intent node and relationship
        self._add_intent_to_graph(session, task)
        
        # Add agent and tool relationships if available
        if hasattr(task, 'assigned_agents') and task.assigned_agents:
            for agent in task.assigned_agents:
                self._add_agent_relationship(session, task.task_id, agent)
    
    def _add_intent_to_graph(self, session, task: Task):
        """Extract and add intent information to graph"""
        
        # Simple intent extraction (would use BusinessRequirementsParser in production)
        intent_keywords = self._extract_intent_keywords(task.business_request)
        
        if intent_keywords:
            intent_id = f"intent_{hashlib.md5('_'.join(intent_keywords).encode()).hexdigest()[:8]}"
            
            # Create intent node
            intent_query = """
            MERGE (i:Intent {intent_id: $intent_id})
            SET i.keywords = $keywords,
                i.business_request_sample = $business_request,
                i.updated_at = datetime()
            """
            
            session.run(intent_query, {
                'intent_id': intent_id,
                'keywords': intent_keywords,
                'business_request': task.business_request
            })
            
            # Create task-intent relationship
            rel_query = """
            MATCH (t:Task {task_id: $task_id})
            MATCH (i:Intent {intent_id: $intent_id})
            MERGE (t)-[:IMPLEMENTS]->(i)
            """
            
            session.run(rel_query, {
                'task_id': task.task_id,
                'intent_id': intent_id
            })
    
    def _add_agent_relationship(self, session, task_id: str, agent_name: str):
        """Add agent node and relationship to task"""
        
        # Create agent node
        agent_query = """
        MERGE (a:Agent {name: $agent_name, agent_id: $agent_id})
        SET a.updated_at = datetime()
        """
        
        agent_id = f"agent_{agent_name.lower().replace(' ', '_')}"
        session.run(agent_query, {
            'agent_name': agent_name,
            'agent_id': agent_id
        })
        
        # Create task-agent relationship
        rel_query = """
        MATCH (t:Task {task_id: $task_id})
        MATCH (a:Agent {agent_id: $agent_id})
        MERGE (t)-[:USES]->(a)
        """
        
        session.run(rel_query, {
            'task_id': task_id,
            'agent_id': agent_id
        })
    
    def add_patterns_to_graph(self, patterns: List[Pattern]):
        """Add identified patterns to the knowledge graph"""
        
        with self.driver.session() as session:
            for pattern in patterns:
                # Create pattern node
                pattern_query = """
                MERGE (pt:Pattern {pattern_id: $pattern_id})
                SET pt.name = $name,
                    pt.description = $description,
                    pt.pattern_type = $pattern_type,
                    pt.frequency = $frequency,
                    pt.success_rate = $success_rate,
                    pt.avg_cost = $avg_cost,
                    pt.avg_duration = $avg_duration,
                    pt.confidence = $confidence,
                    pt.last_seen = datetime($last_seen),
                    pt.examples = $examples,
                    pt.updated_at = datetime()
                """
                
                session.run(pattern_query, {
                    'pattern_id': pattern.pattern_id,
                    'name': pattern.name,
                    'description': pattern.description,
                    'pattern_type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'success_rate': pattern.success_rate,
                    'avg_cost': pattern.avg_cost,
                    'avg_duration': pattern.avg_duration,
                    'confidence': pattern.confidence,
                    'last_seen': pattern.last_seen.isoformat() if pattern.last_seen else None,
                    'examples': pattern.examples
                })
                
                # Link pattern to related projects
                for project_id in pattern.examples:
                    rel_query = """
                    MATCH (p:Project {project_id: $project_id})
                    MATCH (pt:Pattern {pattern_id: $pattern_id})
                    MERGE (p)-[:EXHIBITS {strength: $confidence}]->(pt)
                    """
                    
                    session.run(rel_query, {
                        'project_id': project_id,
                        'pattern_id': pattern.pattern_id,
                        'confidence': pattern.confidence
                    })
    
    def find_similar_projects(self, target_project: Project, limit: int = 5) -> List[Dict[str, Any]]:
        """Find projects similar to the target project"""
        
        query = """
        MATCH (target:Project {project_id: $target_id})
        MATCH (similar:Project)
        WHERE similar.project_id <> $target_id
        
        // Calculate similarity based on shared patterns
        OPTIONAL MATCH (target)-[:EXHIBITS]->(pattern:Pattern)<-[:EXHIBITS]-(similar)
        WITH target, similar, COUNT(pattern) as shared_patterns,
             COLLECT(pattern.pattern_id) as pattern_ids
        
        // Calculate similarity score
        WITH target, similar, shared_patterns, pattern_ids,
             CASE WHEN shared_patterns > 0 THEN shared_patterns * 1.0 ELSE 0.0 END as similarity_score
        
        WHERE similarity_score > 0
        RETURN similar.project_id as project_id,
               similar.name as name,
               similar.description as description,
               similar.state as state,
               similar.success_rate as success_rate,
               similarity_score,
               pattern_ids
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'target_id': target_project.project_id,
                'limit': limit
            })
            
            return result.data()
    
    def recommend_solutions(self, intent_keywords: List[str], 
                          complexity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recommend solution approaches based on intent and patterns"""
        
        query = """
        // Find intents matching keywords
        MATCH (i:Intent)
        WHERE ANY(keyword IN $keywords WHERE keyword IN i.keywords)
        
        // Find successful tasks implementing these intents
        MATCH (t:Task)-[:IMPLEMENTS]->(i)
        WHERE t.status = 'completed'
        
        // Find projects containing these tasks
        MATCH (p:Project)-[:CONTAINS]->(t)
        WHERE p.success_rate > 0.7
        
        // Find patterns exhibited by successful projects
        OPTIONAL MATCH (p)-[:EXHIBITS]->(pt:Pattern)
        WHERE pt.success_rate > 0.5
        
        // Aggregate recommendations
        WITH i, COUNT(DISTINCT t) as task_count, 
             COUNT(DISTINCT p) as project_count,
             AVG(p.success_rate) as avg_success_rate,
             COLLECT(DISTINCT pt.name) as relevant_patterns,
             COLLECT(DISTINCT p.project_id) as example_projects
        
        WHERE task_count >= 2  // At least 2 successful implementations
        
        RETURN i.keywords as intent_keywords,
               i.business_request_sample as example_request,
               task_count,
               project_count,
               avg_success_rate,
               relevant_patterns,
               example_projects[0..3] as example_project_ids
        ORDER BY avg_success_rate DESC, task_count DESC
        LIMIT 10
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'keywords': intent_keywords
            })
            
            return result.data()
    
    def get_project_insights(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a project"""
        
        query = """
        MATCH (p:Project {project_id: $project_id})
        
        // Get basic project info
        OPTIONAL MATCH (p)-[:CONTAINS]->(t:Task)
        OPTIONAL MATCH (t)-[:IMPLEMENTS]->(i:Intent)
        OPTIONAL MATCH (t)-[:USES]->(a:Agent)
        OPTIONAL MATCH (p)-[:EXHIBITS]->(pt:Pattern)
        
        RETURN p.name as project_name,
               p.description as description,
               p.state as state,
               p.success_rate as success_rate,
               COUNT(DISTINCT t) as total_tasks,
               COUNT(DISTINCT i) as unique_intents,
               COUNT(DISTINCT a) as agents_used,
               COUNT(DISTINCT pt) as patterns_exhibited,
               COLLECT(DISTINCT i.keywords) as intent_keywords,
               COLLECT(DISTINCT a.name) as agent_names,
               COLLECT(DISTINCT pt.name) as pattern_names
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'project_id': project_id})
            data = result.data()
            
            if data:
                return data[0]
            else:
                return {}
    
    def _extract_intent_keywords(self, business_request: str) -> List[str]:
        """Extract intent keywords from business request"""
        
        intent_indicators = {
            'create': ['create', 'build', 'make', 'develop', 'generate', 'add'],
            'analyze': ['analyze', 'examine', 'study', 'investigate', 'research'],
            'optimize': ['optimize', 'improve', 'enhance', 'speed up', 'reduce'],
            'integrate': ['integrate', 'connect', 'link', 'combine', 'merge'],
            'test': ['test', 'verify', 'validate', 'check', 'ensure'],
            'deploy': ['deploy', 'launch', 'release', 'publish', 'deliver']
        }
        
        request_lower = business_request.lower()
        detected_intents = []
        
        for intent, indicators in intent_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                detected_intents.append(intent)
        
        # Add domain-specific keywords
        domains = ['api', 'database', 'ui', 'auth', 'data', 'machine learning', 'web', 'mobile']
        for domain in domains:
            if domain in request_lower:
                detected_intents.append(domain.replace(' ', '_'))
        
        return detected_intents[:5]  # Limit to top 5 keywords
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self.driver, 'close'):
            self.driver.close()

class KaizenKnowledgeGraph:
    """Main Knowledge Graph system integrating pattern recognition and Neo4j"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password123"):
        
        self.pattern_engine = PatternRecognitionEngine()
        self.graph_builder = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
        self.logger = logging.getLogger(__name__)
        
        # Integration components
        try:
            self.tracker = SimpleTracker()
        except:
            self.tracker = None
            self.logger.warning("Could not initialize SimpleTracker")
        
        # Knowledge cache
        self._pattern_cache = {}
        self._solution_cache = {}
        self._last_analysis = None
    
    async def analyze_and_store_projects(self, projects: List[Project]) -> Dict[str, Any]:
        """Analyze projects for patterns and store in knowledge graph"""
        
        start_time = time.time()
        
        # Store projects in graph
        for project in projects:
            self.graph_builder.add_project_to_graph(project)
        
        # Identify patterns
        patterns = self.pattern_engine.analyze_project_patterns(projects)
        
        # Store patterns in graph
        self.graph_builder.add_patterns_to_graph(patterns)
        
        # Update cache
        self._pattern_cache = {p.pattern_id: p for p in patterns}
        self._last_analysis = datetime.now()
        
        analysis_time = time.time() - start_time
        
        # Track analysis
        if self.tracker:
            self.tracker.track_event({
                'type': 'knowledge_graph_analysis',
                'projects_analyzed': len(projects),
                'patterns_identified': len(patterns),
                'analysis_time_seconds': analysis_time,
                'neo4j_connected': self.graph_builder.connected
            })
        
        result = {
            'projects_processed': len(projects),
            'patterns_identified': len(patterns),
            'analysis_time_seconds': round(analysis_time, 2),
            'patterns_by_type': self._group_patterns_by_type(patterns),
            'top_patterns': sorted(patterns, key=lambda p: p.confidence, reverse=True)[:5]
        }
        
        self.logger.info(f"Analyzed {len(projects)} projects, identified {len(patterns)} patterns in {analysis_time:.2f}s")
        
        return result
    
    def _group_patterns_by_type(self, patterns: List[Pattern]) -> Dict[str, int]:
        """Group patterns by type for summary"""
        
        type_counts = defaultdict(int)
        for pattern in patterns:
            type_counts[pattern.pattern_type] += 1
        
        return dict(type_counts)
    
    async def get_recommendations_for_project(self, business_requests: List[str],
                                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get recommendations for a new project based on knowledge graph"""
        
        recommendations = {
            'similar_projects': [],
            'recommended_patterns': [],
            'suggested_solutions': [],
            'risk_warnings': [],
            'success_factors': []
        }
        
        # Extract intent keywords from business requests
        all_keywords = []
        for request in business_requests:
            keywords = self.graph_builder._extract_intent_keywords(request)
            all_keywords.extend(keywords)
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(all_keywords))
        
        # Get solution recommendations
        solution_recommendations = self.graph_builder.recommend_solutions(unique_keywords)
        recommendations['suggested_solutions'] = solution_recommendations
        
        # Get relevant patterns from cache
        relevant_patterns = []
        for pattern in self._pattern_cache.values():
            if any(keyword in pattern.name.lower() for keyword in unique_keywords):
                relevant_patterns.append({
                    'pattern_id': pattern.pattern_id,
                    'name': pattern.name,
                    'description': pattern.description,
                    'success_rate': pattern.success_rate,
                    'frequency': pattern.frequency,
                    'confidence': pattern.confidence
                })
        
        # Sort by success rate and confidence
        relevant_patterns.sort(key=lambda p: (p['success_rate'], p['confidence']), reverse=True)
        recommendations['recommended_patterns'] = relevant_patterns[:10]
        
        # Identify risk patterns
        risk_patterns = [
            p for p in relevant_patterns 
            if p['success_rate'] < 0.5 or 'failure' in p['name'].lower()
        ]
        recommendations['risk_warnings'] = risk_patterns[:5]
        
        # Identify success factors
        success_patterns = [
            p for p in relevant_patterns 
            if p['success_rate'] > 0.8 and 'success' in p['name'].lower()
        ]
        recommendations['success_factors'] = success_patterns[:5]
        
        # Track recommendation request
        if self.tracker:
            self.tracker.track_event({
                'type': 'project_recommendations_requested',
                'business_requests_count': len(business_requests),
                'keywords_extracted': len(unique_keywords),
                'solutions_found': len(solution_recommendations),
                'patterns_found': len(relevant_patterns)
            })
        
        return recommendations
    
    def get_pattern_details(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific pattern"""
        
        pattern = self._pattern_cache.get(pattern_id)
        if not pattern:
            return None
        
        # Get additional details from graph
        query = """
        MATCH (pt:Pattern {pattern_id: $pattern_id})
        OPTIONAL MATCH (p:Project)-[:EXHIBITS]->(pt)
        RETURN pt.name as name,
               pt.description as description,
               pt.pattern_type as pattern_type,
               pt.frequency as frequency,
               pt.success_rate as success_rate,
               pt.confidence as confidence,
               COLLECT(p.project_id) as example_projects,
               COUNT(p) as total_occurrences
        """
        
        with self.graph_builder.driver.session() as session:
            result = session.run(query, {'pattern_id': pattern_id})
            data = result.data()
            
            if data:
                return data[0]
            else:
                return asdict(pattern)
    
    def search_knowledge(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge graph using full-text search"""
        
        search_query = """
        CALL db.index.fulltext.queryNodes('projectSearch', $query_text)
        YIELD node, score
        RETURN node.project_id as id,
               node.name as name,
               node.description as description,
               'Project' as type,
               score
        
        UNION
        
        CALL db.index.fulltext.queryNodes('taskSearch', $query_text)
        YIELD node, score
        RETURN node.task_id as id,
               node.business_request as name,
               '' as description,
               'Task' as type,
               score
        
        ORDER BY score DESC
        LIMIT $limit
        """
        
        with self.graph_builder.driver.session() as session:
            try:
                result = session.run(search_query, {
                    'query_text': query_text,
                    'limit': limit
                })
                return result.data()
            except Exception as e:
                self.logger.warning(f"Full-text search failed: {e}")
                return []
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats_query = """
        MATCH (p:Project) 
        OPTIONAL MATCH (p)-[:CONTAINS]->(t:Task)
        OPTIONAL MATCH (p)-[:EXHIBITS]->(pt:Pattern)
        OPTIONAL MATCH (t)-[:IMPLEMENTS]->(i:Intent)
        OPTIONAL MATCH (t)-[:USES]->(a:Agent)
        
        RETURN COUNT(DISTINCT p) as total_projects,
               COUNT(DISTINCT t) as total_tasks,
               COUNT(DISTINCT pt) as total_patterns,
               COUNT(DISTINCT i) as total_intents,
               COUNT(DISTINCT a) as total_agents,
               AVG(p.success_rate) as avg_project_success_rate,
               AVG(p.actual_cost) as avg_project_cost,
               AVG(p.actual_duration) as avg_project_duration
        """
        
        with self.graph_builder.driver.session() as session:
            result = session.run(stats_query)
            graph_stats = result.data()[0] if result.data() else {}
        
        # Add pattern statistics
        pattern_stats = {
            'patterns_in_cache': len(self._pattern_cache),
            'last_analysis': self._last_analysis.isoformat() if self._last_analysis else None,
            'pattern_types': list(self._group_patterns_by_type(list(self._pattern_cache.values())).keys())
        }
        
        return {
            'knowledge_graph': graph_stats,
            'pattern_engine': pattern_stats,
            'neo4j_connected': self.graph_builder.connected,
            'system_ready': len(self._pattern_cache) > 0
        }
    
    def close(self):
        """Cleanup resources"""
        self.graph_builder.close()

# CLI interface for testing
async def main():
    """CLI interface for testing Knowledge Graph system"""
    
    kg = KaizenKnowledgeGraph()
    
    print("🧠 Agent Zero V1 - Neo4j Knowledge Graph")
    print("=" * 60)
    
    # Create sample projects for testing
    from project_orchestrator import Project, Task, ProjectState, TaskStatus, ProjectMetrics
    
    sample_projects = []
    
    # Project 1: API Development
    project1 = Project(
        project_id="proj_api_001",
        name="User Authentication API",
        description="RESTful API for user authentication with JWT",
        state=ProjectState.COMPLETED,
        created_at=datetime.now() - timedelta(days=30),
        completed_at=datetime.now() - timedelta(days=25),
        metrics=ProjectMetrics(
            total_tasks=5, completed_tasks=5, success_rate=1.0,
            estimated_cost=0.15, actual_cost=0.12,
            estimated_duration=180, actual_duration=165
        )
    )
    
    project1.tasks = {
        "task1": Task(
            task_id="task1", 
            business_request="Design API endpoints for user registration and login",
            status=TaskStatus.COMPLETED
        ),
        "task2": Task(
            task_id="task2",
            business_request="Implement JWT token generation and validation",
            status=TaskStatus.COMPLETED
        ),
        "task3": Task(
            task_id="task3",
            business_request="Add password hashing and security measures",
            status=TaskStatus.COMPLETED
        )
    }
    
    # Project 2: Data Analysis
    project2 = Project(
        project_id="proj_data_001",
        name="Sales Analytics Dashboard",
        description="Interactive dashboard for sales data analysis",
        state=ProjectState.COMPLETED,
        created_at=datetime.now() - timedelta(days=20),
        completed_at=datetime.now() - timedelta(days=15),
        metrics=ProjectMetrics(
            total_tasks=4, completed_tasks=4, success_rate=1.0,
            estimated_cost=0.25, actual_cost=0.22,
            estimated_duration=240, actual_duration=220
        )
    )
    
    project2.tasks = {
        "task4": Task(
            task_id="task4",
            business_request="Analyze quarterly sales data for trends and patterns",
            status=TaskStatus.COMPLETED
        ),
        "task5": Task(
            task_id="task5",
            business_request="Create interactive charts and visualizations",
            status=TaskStatus.COMPLETED
        )
    }
    
    # Project 3: Failed Project
    project3 = Project(
        project_id="proj_fail_001",
        name="Complex ML Pipeline",
        description="Machine learning pipeline that failed due to complexity",
        state=ProjectState.FAILED,
        created_at=datetime.now() - timedelta(days=10),
        metrics=ProjectMetrics(
            total_tasks=8, completed_tasks=3, failed_tasks=5, success_rate=0.375,
            estimated_cost=0.50, actual_cost=0.75,
            estimated_duration=480, actual_duration=360
        )
    )
    
    project3.tasks = {
        "task6": Task(
            task_id="task6",
            business_request="Build machine learning model for prediction",
            status=TaskStatus.FAILED
        ),
        "task7": Task(
            task_id="task7",
            business_request="Create data preprocessing pipeline",
            status=TaskStatus.COMPLETED
        )
    }
    
    sample_projects = [project1, project2, project3]
    
    print(f"\n📊 Analyzing {len(sample_projects)} sample projects...")
    
    # Analyze projects
    analysis_results = await kg.analyze_and_store_projects(sample_projects)
    
    print(f"✅ Analysis completed:")
    print(f"   Projects processed: {analysis_results['projects_processed']}")
    print(f"   Patterns identified: {analysis_results['patterns_identified']}")
    print(f"   Analysis time: {analysis_results['analysis_time_seconds']}s")
    print(f"   Pattern types: {analysis_results['patterns_by_type']}")
    
    # Show top patterns
    if analysis_results['top_patterns']:
        print(f"\n🔍 Top Patterns:")
        for i, pattern in enumerate(analysis_results['top_patterns'][:3]):
            print(f"   {i+1}. {pattern.name}")
            print(f"      Type: {pattern.pattern_type}, Confidence: {pattern.confidence:.2f}")
            print(f"      Success Rate: {pattern.success_rate:.1%}, Frequency: {pattern.frequency}")
    
    # Test recommendations
    print(f"\n💡 Testing recommendations for new project...")
    
    test_requests = [
        "Create user authentication system with secure login",
        "Build API endpoints for data access",
        "Add authorization and permission controls"
    ]
    
    recommendations = await kg.get_recommendations_for_project(test_requests)
    
    print(f"   Solutions found: {len(recommendations['suggested_solutions'])}")
    print(f"   Relevant patterns: {len(recommendations['recommended_patterns'])}")
    print(f"   Risk warnings: {len(recommendations['risk_warnings'])}")
    print(f"   Success factors: {len(recommendations['success_factors'])}")
    
    # Show some recommendations
    if recommendations['suggested_solutions']:
        print(f"\n📋 Top Solution Recommendations:")
        for i, solution in enumerate(recommendations['suggested_solutions'][:2]):
            print(f"   {i+1}. Intent: {', '.join(solution['intent_keywords'])}")
            print(f"      Success Rate: {solution['avg_success_rate']:.1%}")
            print(f"      Used in {solution['task_count']} tasks across {solution['project_count']} projects")
    
    # Test search
    print(f"\n🔍 Testing knowledge search...")
    search_results = kg.search_knowledge("authentication API", limit=5)
    
    print(f"   Found {len(search_results)} results for 'authentication API'")
    for result in search_results[:3]:
        print(f"   • {result['type']}: {result['name'][:50]}... (Score: {result['score']:.2f})")
    
    # System statistics
    stats = kg.get_system_statistics()
    print(f"\n📈 System Statistics:")
    if stats['knowledge_graph']:
        kg_stats = stats['knowledge_graph']
        print(f"   Projects: {kg_stats.get('total_projects', 0)}")
        print(f"   Tasks: {kg_stats.get('total_tasks', 0)}")
        print(f"   Patterns: {kg_stats.get('total_patterns', 0)}")
        print(f"   Intents: {kg_stats.get('total_intents', 0)}")
        print(f"   Agents: {kg_stats.get('total_agents', 0)}")
        if kg_stats.get('avg_project_success_rate'):
            print(f"   Avg Success Rate: {kg_stats['avg_project_success_rate']:.1%}")
    
    print(f"   Neo4j Connected: {stats['neo4j_connected']}")
    print(f"   System Ready: {stats['system_ready']}")
    
    # Cleanup
    kg.close()
    
    print(f"\n✅ Knowledge Graph system test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())