#!/usr/bin/env python3
"""
Pattern Mining Engine - Agent Zero V2.0
Advanced pattern detection with ML capabilities
"""

import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from shared.knowledge.neo4j_client import Neo4jClient
except ImportError:
    logging.warning("Neo4j client not available")
    Neo4jClient = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    SUCCESS_PATTERN = "success"
    FAILURE_PATTERN = "failure" 
    COST_OPTIMIZATION = "cost_opt"
    PERFORMANCE_PATTERN = "performance"
    USAGE_PATTERN = "usage"

@dataclass
class DetectedPattern:
    pattern_id: str
    pattern_type: PatternType
    confidence: float
    frequency: int
    conditions: Dict[str, Any]
    outcomes: Dict[str, float]
    recommendations: List[str]
    created_at: str

class PatternMiningEngine:
    """Advanced pattern mining with ML capabilities"""
    
    def __init__(self, neo4j_client=None):
        self.neo4j_client = neo4j_client or (Neo4jClient() if Neo4jClient else None)
        self.min_sample_size = 3
        self.confidence_threshold = 0.7
    
    async def discover_patterns(self, time_window_days=30) -> List[DetectedPattern]:
        """Discover patterns in recent experiences"""
        if not self.neo4j_client:
            logger.warning("Neo4j not available - returning empty patterns")
            return []
        
        try:
            # Get recent experiences
            experiences = await self._get_recent_experiences(time_window_days)
            
            patterns = []
            
            # 1. Success patterns
            success_patterns = await self._detect_success_patterns(experiences)
            patterns.extend(success_patterns)
            
            # 2. Cost optimization patterns  
            cost_patterns = await self._detect_cost_patterns(experiences)
            patterns.extend(cost_patterns)
            
            # 3. Performance patterns
            perf_patterns = await self._detect_performance_patterns(experiences) 
            patterns.extend(perf_patterns)
            
            # Store patterns in Neo4j
            await self._store_patterns(patterns)
            
            logger.info(f"Discovered {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
            return []
    
    async def _get_recent_experiences(self, days: int) -> List[Dict[str, Any]]:
        """Get experiences from specified time window"""
        query = """
        MATCH (e:Experience)
        WHERE e.timestamp > datetime() - duration({days: $days})
        RETURN e.id as id,
               e.task_type as task_type,
               e.model_used as model,
               e.success_score as success_score,
               e.cost_usd as cost,
               e.latency_ms as latency
        ORDER BY e.timestamp DESC
        """
        
        results = await self.neo4j_client.execute_query(query, {'days': days})
        return [dict(record) for record in results]
    
    async def _detect_success_patterns(self, experiences: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect patterns that lead to high success rates"""
        patterns = []
        
        # Group by task_type and model
        groups = {}
        for exp in experiences:
            key = (exp['task_type'], exp['model'])
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)
        
        for (task_type, model), group in groups.items():
            if len(group) < self.min_sample_size:
                continue
            
            success_scores = [e['success_score'] for e in group]
            avg_success = sum(success_scores) / len(success_scores)
            
            if avg_success > 0.8:  # High success rate
                pattern = DetectedPattern(
                    pattern_id=f"success_{task_type}_{model}_{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.SUCCESS_PATTERN,
                    confidence=avg_success,
                    frequency=len(group),
                    conditions={
                        'task_type': task_type,
                        'model': model
                    },
                    outcomes={
                        'success_rate': avg_success,
                        'avg_cost': sum(e['cost'] for e in group) / len(group),
                        'avg_latency': sum(e['latency'] for e in group) / len(group)
                    },
                    recommendations=[
                        f"Use {model} for {task_type} tasks for {avg_success:.1%} success rate",
                        f"This combination shows consistent high performance"
                    ],
                    created_at=datetime.utcnow().isoformat()
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_cost_patterns(self, experiences: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect cost optimization patterns"""
        patterns = []
        
        # Find cost-efficient combinations
        groups = {}
        for exp in experiences:
            key = exp['task_type']
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)
        
        for task_type, group in groups.items():
            if len(group) < self.min_sample_size:
                continue
            
            # Calculate efficiency ratios
            for exp in group:
                if exp['cost'] > 0:
                    exp['efficiency'] = exp['success_score'] / exp['cost']
                else:
                    exp['efficiency'] = exp['success_score'] * 1000  # Free is very efficient
            
            # Find the most efficient model for this task type
            by_model = {}
            for exp in group:
                model = exp['model']
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(exp)
            
            best_efficiency = 0
            best_model = None
            
            for model, model_exps in by_model.items():
                if len(model_exps) >= self.min_sample_size:
                    avg_efficiency = sum(e['efficiency'] for e in model_exps) / len(model_exps)
                    if avg_efficiency > best_efficiency:
                        best_efficiency = avg_efficiency
                        best_model = model
            
            if best_model and best_efficiency > 50:  # Good efficiency threshold
                best_model_exps = by_model[best_model]
                avg_cost = sum(e['cost'] for e in best_model_exps) / len(best_model_exps)
                avg_success = sum(e['success_score'] for e in best_model_exps) / len(best_model_exps)
                
                pattern = DetectedPattern(
                    pattern_id=f"cost_{task_type}_{best_model}_{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.COST_OPTIMIZATION,
                    confidence=min(avg_success, 0.95),  # Cap confidence
                    frequency=len(best_model_exps),
                    conditions={
                        'task_type': task_type,
                        'model': best_model
                    },
                    outcomes={
                        'efficiency_ratio': best_efficiency,
                        'avg_cost': avg_cost,
                        'avg_success': avg_success
                    },
                    recommendations=[
                        f"Use {best_model} for cost-efficient {task_type} tasks",
                        f"Efficiency ratio: {best_efficiency:.1f} (success/cost)",
                        f"Average cost: ${avg_cost:.4f}"
                    ],
                    created_at=datetime.utcnow().isoformat()
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_performance_patterns(self, experiences: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect performance-related patterns"""
        patterns = []
        
        # Find fast-performing combinations
        groups = {}
        for exp in experiences:
            if exp['success_score'] > 0.7:  # Only consider successful tasks
                key = (exp['task_type'], exp['model'])
                if key not in groups:
                    groups[key] = []
                groups[key].append(exp)
        
        for (task_type, model), group in groups.items():
            if len(group) < self.min_sample_size:
                continue
            
            avg_latency = sum(e['latency'] for e in group) / len(group)
            avg_success = sum(e['success_score'] for e in group) / len(group)
            
            if avg_latency < 2000 and avg_success > 0.8:  # Fast and successful
                pattern = DetectedPattern(
                    pattern_id=f"perf_{task_type}_{model}_{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.PERFORMANCE_PATTERN,
                    confidence=avg_success,
                    frequency=len(group),
                    conditions={
                        'task_type': task_type,
                        'model': model
                    },
                    outcomes={
                        'avg_latency_ms': avg_latency,
                        'avg_success': avg_success,
                        'speed_score': 2000 / avg_latency  # Higher is better
                    },
                    recommendations=[
                        f"Use {model} for fast {task_type} tasks",
                        f"Average response time: {avg_latency:.0f}ms",
                        f"Success rate: {avg_success:.1%}"
                    ],
                    created_at=datetime.utcnow().isoformat()
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _store_patterns(self, patterns: List[DetectedPattern]):
        """Store discovered patterns in Neo4j"""
        if not patterns:
            return
        
        for pattern in patterns:
            query = """
            CREATE (p:Pattern {
                id: $pattern_id,
                type: $pattern_type,
                confidence: $confidence,
                frequency: $frequency,
                conditions: $conditions,
                outcomes: $outcomes,
                recommendations: $recommendations,
                created_at: datetime($created_at)
            })
            RETURN p.id
            """
            
            await self.neo4j_client.execute_query(query, {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type.value,
                'confidence': pattern.confidence,
                'frequency': pattern.frequency,
                'conditions': json.dumps(pattern.conditions),
                'outcomes': json.dumps(pattern.outcomes),
                'recommendations': pattern.recommendations,
                'created_at': pattern.created_at
            })

# Demo function
async def demo_pattern_mining():
    """Demo pattern mining engine"""
    print("🔍 Agent Zero V2.0 - Pattern Mining Engine Demo")
    print("=" * 50)
    
    engine = PatternMiningEngine()
    patterns = await engine.discover_patterns(time_window_days=30)
    
    print(f"📊 Discovered {len(patterns)} patterns:")
    for pattern in patterns[:5]:  # Show first 5
        print(f"   🎯 {pattern.pattern_type.value}: {pattern.confidence:.1%} confidence")
        print(f"      📝 {pattern.recommendations[0] if pattern.recommendations else 'No recommendations'}")
    
    print("✅ Pattern mining demo completed")

if __name__ == "__main__":
    asyncio.run(demo_pattern_mining())
