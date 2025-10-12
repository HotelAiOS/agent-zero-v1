#!/usr/bin/env python3
"""
Enhanced Experience Tracker - Priority 1 Implementation
Agent Zero V2.0 Intelligence Layer - Experience Management System

Integrates with existing shared/experience_manager.py
Extends functionality with ML capabilities and Neo4j integration
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing Agent Zero components
try:
    from shared.experience_manager import ExperienceManager
    from shared.knowledge.neo4j_client import Neo4jClient
    from shared.utils.simple_tracker import SimpleTracker
except ImportError as e:
    print(f"Warning: Could not import existing components: {e}")
    print("This is expected if running outside Agent Zero environment")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperienceType(Enum):
    TASK_EXECUTION = "task_execution"
    MODEL_SELECTION = "model_selection"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_TUNING = "performance_tuning"
    ERROR_RESOLUTION = "error_resolution"

class InsightLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MLInsight:
    insight_id: str
    insight_type: str
    confidence: float
    description: str
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: str

@dataclass
class EnhancedExperience:
    """Enhanced experience data structure"""
    experience_id: str
    experience_type: ExperienceType
    task_id: str
    task_type: str
    model_used: str
    success_score: float
    cost_usd: float
    latency_ms: float
    timestamp: str
    user_feedback: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    ml_insights: Optional[List[MLInsight]] = None
    graph_node_id: Optional[str] = None

class MLInsightEngine:
    """Machine Learning engine for generating insights from experiences"""
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
    
    def _load_insight_rules(self) -> List[Dict[str, Any]]:
        """Load ML insight generation rules"""
        return [
            {
                'name': 'cost_efficiency_insight',
                'condition': lambda exp: exp.success_score > 0.8 and exp.cost_usd < 0.01,
                'insight_type': 'cost_optimization',
                'confidence_base': 0.85,
                'description': 'High success rate with low cost - excellent model choice'
            },
            {
                'name': 'performance_concern',
                'condition': lambda exp: exp.latency_ms > 5000 and exp.success_score > 0.7,
                'insight_type': 'performance_tuning',
                'confidence_base': 0.75,
                'description': 'Good results but high latency - consider model optimization'
            },
            {
                'name': 'model_excellence',
                'condition': lambda exp: exp.success_score > 0.9 and exp.latency_ms < 2000,
                'insight_type': 'model_selection',
                'confidence_base': 0.9,
                'description': 'Exceptional performance - optimal model for this task type'
            },
            {
                'name': 'failure_analysis',
                'condition': lambda exp: exp.success_score < 0.5,
                'insight_type': 'error_resolution',
                'confidence_base': 0.8,
                'description': 'Low success rate - requires investigation and optimization'
            }
        ]
    
    async def analyze_experience(self, experience: EnhancedExperience) -> List[MLInsight]:
        """Generate ML insights for an experience"""
        insights = []
        
        for rule in self.insight_rules:
            if rule['condition'](experience):
                insight = MLInsight(
                    insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                    insight_type=rule['insight_type'],
                    confidence=rule['confidence_base'],
                    description=rule['description'],
                    recommendations=self._generate_recommendations(rule, experience),
                    supporting_data={
                        'success_score': experience.success_score,
                        'cost_usd': experience.cost_usd,
                        'latency_ms': experience.latency_ms,
                        'model_used': experience.model_used
                    },
                    generated_at=datetime.utcnow().isoformat()
                )
                insights.append(insight)
        
        return insights
    
    def _generate_recommendations(self, rule: Dict[str, Any], experience: EnhancedExperience) -> List[str]:
        """Generate specific recommendations based on rule and experience"""
        recommendations = []
        
        if rule['insight_type'] == 'cost_optimization':
            recommendations.extend([
                f"Continue using {experience.model_used} for {experience.task_type} tasks",
                f"Expected cost savings: ${experience.cost_usd:.4f} per task",
                "Consider scaling this model choice across similar tasks"
            ])
        
        elif rule['insight_type'] == 'performance_tuning':
            recommendations.extend([
                "Consider using a faster model variant",
                "Implement caching for repeated similar requests",
                f"Target latency reduction from {experience.latency_ms}ms to <2000ms"
            ])
        
        elif rule['insight_type'] == 'model_selection':
            recommendations.extend([
                f"Set {experience.model_used} as default for {experience.task_type}",
                "Document this as a best practice pattern",
                f"Expected performance: {experience.success_score:.1%} success rate"
            ])
        
        elif rule['insight_type'] == 'error_resolution':
            recommendations.extend([
                "Review task parameters and model configuration",
                "Consider switching to a more capable model",
                "Investigate root cause of low success rate",
                "Add additional validation or retry logic"
            ])
        
        return recommendations

class V2ExperienceTracker:
    """
    Enhanced Experience Tracker for Agent Zero V2.0
    Extends existing ExperienceManager with ML capabilities and Neo4j integration
    """
    
    def __init__(self, neo4j_client=None, simple_tracker=None):
        # Initialize existing components if available
        try:
            self.experience_manager = ExperienceManager()
            logger.info("Initialized with existing ExperienceManager")
        except:
            self.experience_manager = None
            logger.info("Running in standalone mode - no existing ExperienceManager")
        
        # Initialize Neo4j client
        self.neo4j_client = neo4j_client or self._get_neo4j_client()
        
        # Initialize SimpleTracker for backward compatibility
        self.simple_tracker = simple_tracker or self._get_simple_tracker()
        
        # Initialize ML engine
        self.ml_engine = MLInsightEngine()
        
        logger.info("V2ExperienceTracker initialized successfully")
    
    def _get_neo4j_client(self):
        """Get Neo4j client instance"""
        try:
            return Neo4jClient()
        except:
            logger.warning("Neo4j client not available - graph features disabled")
            return None
    
    def _get_simple_tracker(self):
        """Get SimpleTracker instance"""
        try:
            return SimpleTracker()
        except:
            logger.warning("SimpleTracker not available - tracking to memory only")
            return None
    
    async def track_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track experience with V2.0 intelligence
        
        Args:
            experience_data: Dictionary containing experience information
            
        Returns:
            Dictionary with tracking results and insights
        """
        try:
            # Create enhanced experience object
            enhanced_exp = EnhancedExperience(
                experience_id=f"exp_{uuid.uuid4().hex[:12]}",
                experience_type=ExperienceType(experience_data.get('experience_type', 'task_execution')),
                task_id=experience_data.get('task_id', f"task_{uuid.uuid4().hex[:8]}"),
                task_type=experience_data.get('task_type', 'unknown'),
                model_used=experience_data.get('model_used', 'unknown'),
                success_score=float(experience_data.get('success_score', 0.0)),
                cost_usd=float(experience_data.get('cost_usd', 0.0)),
                latency_ms=float(experience_data.get('latency_ms', 0.0)),
                timestamp=experience_data.get('timestamp', datetime.utcnow().isoformat()),
                user_feedback=experience_data.get('user_feedback'),
                context=experience_data.get('context', {})
            )
            
            # Generate ML insights
            ml_insights = await self.ml_engine.analyze_experience(enhanced_exp)
            enhanced_exp.ml_insights = ml_insights
            
            # Store in existing SimpleTracker for backward compatibility
            simple_tracker_id = None
            if self.simple_tracker:
                simple_tracker_id = await self._store_in_simple_tracker(enhanced_exp)
            
            # Store in Neo4j graph database if available
            graph_node_id = None
            if self.neo4j_client:
                graph_node_id = await self._store_in_graph_database(enhanced_exp)
                enhanced_exp.graph_node_id = graph_node_id
            
            # Store in existing ExperienceManager if available
            legacy_id = None
            if self.experience_manager:
                legacy_id = await self._store_in_experience_manager(enhanced_exp)
            
            logger.info(f"Tracked experience {enhanced_exp.experience_id} with {len(ml_insights)} insights")
            
            return {
                'experience_id': enhanced_exp.experience_id,
                'task_id': enhanced_exp.task_id,
                'insights': [asdict(insight) for insight in ml_insights],
                'insights_count': len(ml_insights),
                'graph_node_id': graph_node_id,
                'simple_tracker_id': simple_tracker_id,
                'legacy_id': legacy_id,
                'status': 'success',
                'tracked_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking experience: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'tracked_at': datetime.utcnow().isoformat()
            }
    
    async def _store_in_simple_tracker(self, experience: EnhancedExperience) -> str:
        """Store in existing SimpleTracker"""
        try:
            task_id = self.simple_tracker.track_event(
                task_id=experience.task_id,
                task_type=experience.task_type,
                model_used=experience.model_used,
                success_score=experience.success_score,
                cost_usd=experience.cost_usd,
                latency_ms=experience.latency_ms,
                user_feedback=experience.user_feedback
            )
            return task_id
        except Exception as e:
            logger.error(f"Error storing in SimpleTracker: {e}")
            return None
    
    async def _store_in_graph_database(self, experience: EnhancedExperience) -> str:
        """Store experience in Neo4j graph database"""
        try:
            # Create experience node
            query = """
            CREATE (e:Experience {
                id: $experience_id,
                type: $experience_type,
                task_id: $task_id,
                task_type: $task_type,
                model_used: $model_used,
                success_score: $success_score,
                cost_usd: $cost_usd,
                latency_ms: $latency_ms,
                timestamp: datetime($timestamp),
                user_feedback: $user_feedback,
                context: $context
            })
            RETURN e.id as node_id
            """
            
            result = await self.neo4j_client.execute_query(query, {
                'experience_id': experience.experience_id,
                'experience_type': experience.experience_type.value,
                'task_id': experience.task_id,
                'task_type': experience.task_type,
                'model_used': experience.model_used,
                'success_score': experience.success_score,
                'cost_usd': experience.cost_usd,
                'latency_ms': experience.latency_ms,
                'timestamp': experience.timestamp,
                'user_feedback': experience.user_feedback,
                'context': json.dumps(experience.context) if experience.context else None
            })
            
            # Store insights as separate nodes and relationships
            if experience.ml_insights:
                await self._store_insights_in_graph(experience.experience_id, experience.ml_insights)
            
            return result[0]['node_id'] if result else experience.experience_id
            
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
            return None
    
    async def _store_insights_in_graph(self, experience_id: str, insights: List[MLInsight]):
        """Store ML insights in graph database"""
        try:
            for insight in insights:
                query = """
                MATCH (e:Experience {id: $experience_id})
                CREATE (i:Insight {
                    id: $insight_id,
                    type: $insight_type,
                    confidence: $confidence,
                    description: $description,
                    recommendations: $recommendations,
                    supporting_data: $supporting_data,
                    generated_at: datetime($generated_at)
                })
                CREATE (e)-[:HAS_INSIGHT]->(i)
                """
                
                await self.neo4j_client.execute_query(query, {
                    'experience_id': experience_id,
                    'insight_id': insight.insight_id,
                    'insight_type': insight.insight_type,
                    'confidence': insight.confidence,
                    'description': insight.description,
                    'recommendations': insight.recommendations,
                    'supporting_data': json.dumps(insight.supporting_data),
                    'generated_at': insight.generated_at
                })
                
        except Exception as e:
            logger.error(f"Error storing insights in graph: {e}")
    
    async def _store_in_experience_manager(self, experience: EnhancedExperience) -> str:
        """Store in existing ExperienceManager for backward compatibility"""
        try:
            # Convert to format expected by existing ExperienceManager
            legacy_data = {
                'task_id': experience.task_id,
                'task_type': experience.task_type,
                'success_score': experience.success_score,
                'timestamp': experience.timestamp,
                'metadata': {
                    'model_used': experience.model_used,
                    'cost_usd': experience.cost_usd,
                    'latency_ms': experience.latency_ms,
                    'ml_insights_count': len(experience.ml_insights) if experience.ml_insights else 0
                }
            }
            
            result = await self.experience_manager.capture_experience(legacy_data)
            return result.get('experience_id')
            
        except Exception as e:
            logger.error(f"Error storing in ExperienceManager: {e}")
            return None
    
    async def get_experiences_with_insights(self, 
                                          task_type: Optional[str] = None,
                                          min_confidence: float = 0.7,
                                          limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiences with their ML insights"""
        try:
            if not self.neo4j_client:
                return []
            
            where_clause = ""
            if task_type:
                where_clause = "WHERE e.task_type = $task_type"
            
            query = f"""
            MATCH (e:Experience)-[:HAS_INSIGHT]->(i:Insight)
            {where_clause}
            AND i.confidence >= $min_confidence
            RETURN e, collect(i) as insights
            ORDER BY e.timestamp DESC
            LIMIT $limit
            """
            
            params = {
                'min_confidence': min_confidence,
                'limit': limit
            }
            if task_type:
                params['task_type'] = task_type
            
            results = await self.neo4j_client.execute_query(query, params)
            
            experiences = []
            for record in results:
                exp_data = dict(record['e'])
                insights_data = [dict(insight) for insight in record['insights']]
                
                experiences.append({
                    'experience': exp_data,
                    'insights': insights_data,
                    'insights_count': len(insights_data)
                })
            
            return experiences
            
        except Exception as e:
            logger.error(f"Error getting experiences with insights: {e}")
            return []
    
    async def get_aggregated_insights(self, 
                                    task_type: Optional[str] = None,
                                    insight_type: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated insights for analytics"""
        try:
            if not self.neo4j_client:
                return {'error': 'Neo4j not available'}
            
            where_conditions = []
            if task_type:
                where_conditions.append("e.task_type = $task_type")
            if insight_type:
                where_conditions.append("i.type = $insight_type")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
            MATCH (e:Experience)-[:HAS_INSIGHT]->(i:Insight)
            {where_clause}
            RETURN 
                count(DISTINCT e) as total_experiences,
                count(i) as total_insights,
                avg(i.confidence) as avg_confidence,
                collect(DISTINCT i.type) as insight_types,
                avg(e.success_score) as avg_success_score,
                sum(e.cost_usd) as total_cost
            """
            
            params = {}
            if task_type:
                params['task_type'] = task_type
            if insight_type:
                params['insight_type'] = insight_type
            
            result = await self.neo4j_client.execute_query(query, params)
            
            if result:
                return dict(result[0])
            else:
                return {'total_experiences': 0, 'total_insights': 0}
                
        except Exception as e:
            logger.error(f"Error getting aggregated insights: {e}")
            return {'error': str(e)}

# Standalone testing and demo
async def demo_enhanced_tracker():
    """Demo the Enhanced Experience Tracker"""
    print("🚀 Agent Zero V2.0 - Enhanced Experience Tracker Demo")
    print("=" * 60)
    
    # Initialize tracker
    tracker = V2ExperienceTracker()
    
    # Test experiences
    test_experiences = [
        {
            'task_id': 'demo_001',
            'task_type': 'text_analysis',
            'model_used': 'deepseek-coder-33b',
            'success_score': 0.92,
            'cost_usd': 0.0045,
            'latency_ms': 1800,
            'user_feedback': 'Excellent results'
        },
        {
            'task_id': 'demo_002', 
            'task_type': 'code_generation',
            'model_used': 'qwen2.5-14b',
            'success_score': 0.87,
            'cost_usd': 0.0032,
            'latency_ms': 5500,
            'user_feedback': 'Good but slow'
        },
        {
            'task_id': 'demo_003',
            'task_type': 'text_analysis',
            'model_used': 'llama3.1-8b', 
            'success_score': 0.45,
            'cost_usd': 0.0012,
            'latency_ms': 900,
            'user_feedback': 'Poor results, needs improvement'
        }
    ]
    
    # Track experiences
    results = []
    for i, exp in enumerate(test_experiences, 1):
        print(f"\n{i}. Tracking experience: {exp['task_id']}")
        result = await tracker.track_experience(exp)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"   ✅ Tracked with {result['insights_count']} insights")
            for insight in result['insights']:
                print(f"   💡 {insight['insight_type']}: {insight['description']}")
        else:
            print(f"   ❌ Error: {result.get('error')}")
    
    # Get aggregated insights
    print(f"\n📊 Aggregated Insights")
    print("-" * 30)
    insights = await tracker.get_aggregated_insights()
    if 'error' not in insights:
        print(f"Total Experiences: {insights.get('total_experiences', 0)}")
        print(f"Total Insights: {insights.get('total_insights', 0)}")
        print(f"Average Confidence: {insights.get('avg_confidence', 0):.2f}")
        print(f"Average Success Score: {insights.get('avg_success_score', 0):.2f}")
    
    print(f"\n✅ Demo completed successfully!")
    return results

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_enhanced_tracker())