#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 2 - Priority 2: Experience Management System
Saturday, October 11, 2025 @ 10:05 CEST

Experience Management System - AI-powered experience matching and knowledge reuse
Integration z existing AI Intelligence Layer for enhanced decision making
"""

import os
import sys
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EXPERIENCE DATA MODELS
# =============================================================================

class TaskTypeEnum(Enum):
    """Task type classifications"""
    DEVELOPMENT = "development"
    ANALYSIS = "analysis" 
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    PLANNING = "planning"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    RESEARCH = "research"

class ComplexityLevelEnum(Enum):
    """Complexity level classifications"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class SuccessLevelEnum(Enum):
    """Success level classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class Experience:
    """Individual experience record"""
    id: str
    task_type: TaskTypeEnum
    complexity: ComplexityLevelEnum
    context: Dict[str, Any]
    approach_used: str
    model_used: str
    success_level: SuccessLevelEnum
    cost_usd: float
    duration_minutes: int
    quality_score: float  # 0.0-1.0
    user_satisfaction: float  # 0.0-1.0
    lessons_learned: List[str]
    reusable_components: List[str]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class ExperiencePattern:
    """Identified pattern from experiences"""
    id: str
    pattern_type: str
    description: str
    confidence_score: float
    frequency: int
    success_rate: float
    avg_cost: float
    avg_duration: float
    conditions: Dict[str, Any]
    recommendations: List[str]
    supporting_experiences: List[str]
    created_at: datetime

@dataclass
class ExperienceMatch:
    """Match between current request and past experience"""
    experience_id: str
    similarity_score: float
    matching_factors: List[str]
    success_probability: float
    estimated_cost: float
    estimated_duration: int
    recommended_approach: str
    potential_issues: List[str]
    confidence: float

# =============================================================================
# EXPERIENCE MANAGEMENT SYSTEM
# =============================================================================

class ExperienceManager:
    """
    Experience Management System for Agent Zero V2.0
    
    Capabilities:
    - Experience aggregation and storage
    - Semantic similarity matching
    - Pattern discovery and analysis
    - Recommendation generation
    - Success prediction
    """
    
    def __init__(self, db_path: str = "data/experience_db.sqlite"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_db_directory()
        self._init_database()
        
        # Similarity weights for matching
        self.similarity_weights = {
            'task_type': 0.3,
            'complexity': 0.25,
            'context_keywords': 0.2,
            'approach': 0.15,
            'model_used': 0.1
        }
        
        logger.info("🧠 Experience Management System initialized")
    
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with experience schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    approach_used TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    success_level TEXT NOT NULL,
                    cost_usd REAL NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    user_satisfaction REAL NOT NULL,
                    lessons_learned_json TEXT NOT NULL,
                    reusable_components_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            ''')
            
            # Experience patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experience_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    frequency INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    avg_duration REAL NOT NULL,
                    conditions_json TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    supporting_experiences_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_type ON experiences(task_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_complexity ON experiences(complexity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_success_level ON experiences(success_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON experiences(created_at)')
            
            conn.commit()
    
    async def record_experience(self, experience: Experience) -> str:
        """Record a new experience"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiences (
                        id, task_type, complexity, context_json, approach_used,
                        model_used, success_level, cost_usd, duration_minutes,
                        quality_score, user_satisfaction, lessons_learned_json,
                        reusable_components_json, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experience.id,
                    experience.task_type.value,
                    experience.complexity.value,
                    json.dumps(experience.context),
                    experience.approach_used,
                    experience.model_used,
                    experience.success_level.value,
                    experience.cost_usd,
                    experience.duration_minutes,
                    experience.quality_score,
                    experience.user_satisfaction,
                    json.dumps(experience.lessons_learned),
                    json.dumps(experience.reusable_components),
                    experience.created_at.isoformat(),
                    json.dumps(experience.metadata)
                ))
                conn.commit()
                
            logger.info(f"✅ Recorded experience {experience.id} for {experience.task_type.value}")
            
            # Update patterns after recording new experience
            await self._update_patterns()
            
            return experience.id
            
        except Exception as e:
            logger.error(f"❌ Failed to record experience: {e}")
            raise
    
    async def find_similar_experiences(self, 
                                     request_context: Dict[str, Any], 
                                     limit: int = 5) -> List[ExperienceMatch]:
        """Find experiences similar to current request"""
        try:
            # Extract key parameters from request context
            task_type = request_context.get('task_type', 'development')
            complexity = request_context.get('complexity', 'moderate') 
            keywords = request_context.get('keywords', [])
            
            matches = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM experiences 
                    WHERE task_type = ? OR complexity = ?
                    ORDER BY created_at DESC
                    LIMIT 50
                ''', (task_type, complexity))
                
                experiences = cursor.fetchall()
                
            # Calculate similarity scores
            for exp_row in experiences:
                experience = self._row_to_experience(exp_row)
                similarity_score = self._calculate_similarity(request_context, experience)
                
                if similarity_score > 0.3:  # Minimum similarity threshold
                    match = ExperienceMatch(
                        experience_id=experience.id,
                        similarity_score=similarity_score,
                        matching_factors=self._identify_matching_factors(request_context, experience),
                        success_probability=self._predict_success(experience, request_context),
                        estimated_cost=experience.cost_usd * (1 + (0.5 - similarity_score)),
                        estimated_duration=int(experience.duration_minutes * (1 + (0.5 - similarity_score))),
                        recommended_approach=experience.approach_used,
                        potential_issues=self._identify_potential_issues(experience, request_context),
                        confidence=similarity_score * experience.quality_score
                    )
                    matches.append(match)
            
            # Sort by similarity score and return top matches
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"🔍 Found {len(matches[:limit])} similar experiences")
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to find similar experiences: {e}")
            return []
    
    def _calculate_similarity(self, request_context: Dict, experience: Experience) -> float:
        """Calculate similarity score between request and experience"""
        score = 0.0
        
        # Task type similarity
        if request_context.get('task_type') == experience.task_type.value:
            score += self.similarity_weights['task_type']
        
        # Complexity similarity
        complexity_scores = {
            ('simple', 'simple'): 1.0,
            ('simple', 'moderate'): 0.7,
            ('moderate', 'moderate'): 1.0,
            ('moderate', 'complex'): 0.7,
            ('complex', 'complex'): 1.0,
            ('complex', 'very_complex'): 0.7,
            ('very_complex', 'very_complex'): 1.0
        }
        
        req_complexity = request_context.get('complexity', 'moderate')
        complexity_pair = (req_complexity, experience.complexity.value)
        complexity_sim = complexity_scores.get(complexity_pair, 0.3)
        score += self.similarity_weights['complexity'] * complexity_sim
        
        # Context keywords similarity
        req_keywords = set(request_context.get('keywords', []))
        exp_keywords = set(experience.context.get('keywords', []))
        if req_keywords or exp_keywords:
            keyword_intersection = len(req_keywords & exp_keywords)
            keyword_union = len(req_keywords | exp_keywords)
            if keyword_union > 0:
                keyword_sim = keyword_intersection / keyword_union
                score += self.similarity_weights['context_keywords'] * keyword_sim
        
        # Approach similarity (semantic)
        req_approach = request_context.get('approach', '')
        if req_approach and experience.approach_used:
            approach_sim = self._calculate_text_similarity(req_approach, experience.approach_used)
            score += self.similarity_weights['approach'] * approach_sim
        
        # Model preference
        if request_context.get('preferred_model') == experience.model_used:
            score += self.similarity_weights['model_used']
        
        return min(score, 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_matching_factors(self, request_context: Dict, experience: Experience) -> List[str]:
        """Identify what factors make this experience similar"""
        factors = []
        
        if request_context.get('task_type') == experience.task_type.value:
            factors.append(f"Same task type: {experience.task_type.value}")
        
        if request_context.get('complexity') == experience.complexity.value:
            factors.append(f"Same complexity: {experience.complexity.value}")
        
        req_keywords = set(request_context.get('keywords', []))
        exp_keywords = set(experience.context.get('keywords', []))
        common_keywords = req_keywords & exp_keywords
        if common_keywords:
            factors.append(f"Common keywords: {', '.join(common_keywords)}")
        
        if request_context.get('preferred_model') == experience.model_used:
            factors.append(f"Same model preference: {experience.model_used}")
        
        return factors
    
    def _predict_success(self, experience: Experience, request_context: Dict) -> float:
        """Predict success probability based on experience"""
        base_success = {
            SuccessLevelEnum.EXCELLENT: 0.9,
            SuccessLevelEnum.GOOD: 0.75,
            SuccessLevelEnum.MODERATE: 0.6,
            SuccessLevelEnum.POOR: 0.3,
            SuccessLevelEnum.FAILED: 0.1
        }
        
        success_prob = base_success.get(experience.success_level, 0.5)
        
        # Adjust based on quality and satisfaction scores
        success_prob = (success_prob + experience.quality_score + experience.user_satisfaction) / 3
        
        return min(success_prob, 0.95)  # Cap at 95%
    
    def _identify_potential_issues(self, experience: Experience, request_context: Dict) -> List[str]:
        """Identify potential issues based on past experience"""
        issues = []
        
        # If experience had poor outcomes, warn about lessons learned
        if experience.success_level in [SuccessLevelEnum.POOR, SuccessLevelEnum.FAILED]:
            issues.extend(experience.lessons_learned)
        
        # High cost warning
        if experience.cost_usd > 0.05:
            issues.append(f"High cost approach (${experience.cost_usd:.4f})")
        
        # Long duration warning  
        if experience.duration_minutes > 60:
            issues.append(f"Time-intensive approach ({experience.duration_minutes} minutes)")
        
        # Low satisfaction warning
        if experience.user_satisfaction < 0.6:
            issues.append("Previous users had low satisfaction with this approach")
        
        return issues
    
    async def _update_patterns(self):
        """Update experience patterns after new experience is recorded"""
        try:
            patterns = await self._discover_patterns()
            
            # Store patterns in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear old patterns
                cursor.execute('DELETE FROM experience_patterns')
                
                # Insert new patterns
                for pattern in patterns:
                    cursor.execute('''
                        INSERT INTO experience_patterns (
                            id, pattern_type, description, confidence_score,
                            frequency, success_rate, avg_cost, avg_duration,
                            conditions_json, recommendations_json,
                            supporting_experiences_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern.id,
                        pattern.pattern_type,
                        pattern.description,
                        pattern.confidence_score,
                        pattern.frequency,
                        pattern.success_rate,
                        pattern.avg_cost,
                        pattern.avg_duration,
                        json.dumps(pattern.conditions),
                        json.dumps(pattern.recommendations),
                        json.dumps(pattern.supporting_experiences),
                        pattern.created_at.isoformat()
                    ))
                
                conn.commit()
                
            logger.info(f"📊 Updated {len(patterns)} experience patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to update patterns: {e}")
    
    async def _discover_patterns(self) -> List[ExperiencePattern]:
        """Discover patterns from experiences"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Pattern 1: Task type success rates
            cursor.execute('''
                SELECT task_type, COUNT(*) as frequency,
                       AVG(CASE WHEN success_level IN ('excellent', 'good') THEN 1.0 ELSE 0.0 END) as success_rate,
                       AVG(cost_usd) as avg_cost,
                       AVG(duration_minutes) as avg_duration,
                       GROUP_CONCAT(id) as experience_ids
                FROM experiences
                GROUP BY task_type
                HAVING COUNT(*) >= 3
            ''')
            
            for row in cursor.fetchall():
                pattern = ExperiencePattern(
                    id=f"task_type_{row[0]}",
                    pattern_type="task_type_performance",
                    description=f"{row[0].title()} tasks have {row[2]:.1%} success rate",
                    confidence_score=min(row[1] / 10, 0.9),  # Confidence based on frequency
                    frequency=row[1],
                    success_rate=row[2],
                    avg_cost=row[3],
                    avg_duration=row[4],
                    conditions={'task_type': row[0]},
                    recommendations=self._generate_recommendations(row[0], row[2], row[3]),
                    supporting_experiences=row[5].split(',') if row[5] else [],
                    created_at=datetime.now()
                )
                patterns.append(pattern)
            
            # Pattern 2: Model effectiveness
            cursor.execute('''
                SELECT model_used, task_type, COUNT(*) as frequency,
                       AVG(quality_score) as avg_quality,
                       AVG(cost_usd) as avg_cost,
                       GROUP_CONCAT(id) as experience_ids
                FROM experiences
                GROUP BY model_used, task_type
                HAVING COUNT(*) >= 2
            ''')
            
            for row in cursor.fetchall():
                pattern = ExperiencePattern(
                    id=f"model_{row[0]}_{row[1]}",
                    pattern_type="model_task_effectiveness",
                    description=f"{row[0]} achieves {row[3]:.2f} quality for {row[1]} tasks",
                    confidence_score=min(row[2] / 5, 0.8),
                    frequency=row[2],
                    success_rate=row[3],
                    avg_cost=row[4],
                    avg_duration=0,  # Not tracked in this query
                    conditions={'model_used': row[0], 'task_type': row[1]},
                    recommendations=self._generate_model_recommendations(row[0], row[1], row[3], row[4]),
                    supporting_experiences=row[5].split(',') if row[5] else [],
                    created_at=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _generate_recommendations(self, task_type: str, success_rate: float, avg_cost: float) -> List[str]:
        """Generate recommendations based on pattern analysis"""
        recommendations = []
        
        if success_rate > 0.8:
            recommendations.append(f"Continue using current approaches for {task_type} tasks")
        elif success_rate < 0.6:
            recommendations.append(f"Review and improve approaches for {task_type} tasks")
            recommendations.append("Consider different models or methodologies")
        
        if avg_cost > 0.03:
            recommendations.append(f"Optimize costs for {task_type} tasks - currently averaging ${avg_cost:.4f}")
        
        return recommendations
    
    def _generate_model_recommendations(self, model: str, task_type: str, quality: float, cost: float) -> List[str]:
        """Generate model-specific recommendations"""
        recommendations = []
        
        if quality > 0.8:
            recommendations.append(f"{model} is highly effective for {task_type} tasks")
        elif quality < 0.6:
            recommendations.append(f"Consider alternatives to {model} for {task_type} tasks")
        
        if cost > 0.02:
            recommendations.append(f"{model} has high costs (${cost:.4f}) for {task_type}")
        
        return recommendations
    
    def _row_to_experience(self, row) -> Experience:
        """Convert database row to Experience object"""
        return Experience(
            id=row[0],
            task_type=TaskTypeEnum(row[1]),
            complexity=ComplexityLevelEnum(row[2]),
            context=json.loads(row[3]),
            approach_used=row[4],
            model_used=row[5],
            success_level=SuccessLevelEnum(row[6]),
            cost_usd=row[7],
            duration_minutes=row[8],
            quality_score=row[9],
            user_satisfaction=row[10],
            lessons_learned=json.loads(row[11]),
            reusable_components=json.loads(row[12]),
            created_at=datetime.fromisoformat(row[13]),
            metadata=json.loads(row[14])
        )
    
    async def get_recommendations(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive recommendations for current request"""
        try:
            # Find similar experiences
            similar_experiences = await self.find_similar_experiences(request_context)
            
            # Get relevant patterns
            patterns = await self._get_relevant_patterns(request_context)
            
            # Generate recommendations
            recommendations = {
                "status": "success",
                "request_context": request_context,
                "similar_experiences": [asdict(match) for match in similar_experiences],
                "relevant_patterns": [asdict(pattern) for pattern in patterns],
                "recommendations": self._generate_comprehensive_recommendations(
                    similar_experiences, patterns, request_context
                ),
                "confidence_score": self._calculate_recommendation_confidence(similar_experiences, patterns),
                "estimated_metrics": self._estimate_request_metrics(similar_experiences),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"📋 Generated recommendations with {len(similar_experiences)} similar experiences")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_relevant_patterns(self, request_context: Dict) -> List[ExperiencePattern]:
        """Get patterns relevant to current request"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM experience_patterns
                WHERE conditions_json LIKE ? OR conditions_json LIKE ?
            ''', (
                f'%{request_context.get("task_type", "")}%',
                f'%{request_context.get("preferred_model", "")}%'
            ))
            
            for row in cursor.fetchall():
                pattern = ExperiencePattern(
                    id=row[0],
                    pattern_type=row[1],
                    description=row[2],
                    confidence_score=row[3],
                    frequency=row[4],
                    success_rate=row[5],
                    avg_cost=row[6],
                    avg_duration=row[7],
                    conditions=json.loads(row[8]),
                    recommendations=json.loads(row[9]),
                    supporting_experiences=json.loads(row[10]),
                    created_at=datetime.fromisoformat(row[11])
                )
                patterns.append(pattern)
        
        return patterns
    
    def _generate_comprehensive_recommendations(self, 
                                             similar_experiences: List[ExperienceMatch],
                                             patterns: List[ExperiencePattern],
                                             request_context: Dict) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        if similar_experiences:
            best_match = similar_experiences[0]
            recommendations.append(
                f"Based on similar experience ({best_match.similarity_score:.2f} similarity), "
                f"recommend: {best_match.recommended_approach}"
            )
            
            if best_match.success_probability > 0.8:
                recommendations.append("High success probability - proceed with confidence")
            elif best_match.success_probability < 0.6:
                recommendations.append("Lower success probability - consider risk mitigation")
            
            if best_match.potential_issues:
                recommendations.append(f"Potential issues to watch: {'; '.join(best_match.potential_issues)}")
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.confidence_score > 0.7:
                recommendations.extend(pattern.recommendations)
        
        # General recommendations
        task_type = request_context.get('task_type', 'development')
        if not similar_experiences:
            recommendations.append(f"No similar experiences found for {task_type} - proceed with standard approach")
            recommendations.append("Document this experience for future reference")
        
        return recommendations
    
    def _calculate_recommendation_confidence(self, 
                                          similar_experiences: List[ExperienceMatch],
                                          patterns: List[ExperiencePattern]) -> float:
        """Calculate overall confidence in recommendations"""
        if not similar_experiences and not patterns:
            return 0.3  # Low confidence with no historical data
        
        confidence = 0.5  # Base confidence
        
        if similar_experiences:
            # Weight by best match similarity and success probability
            best_match = similar_experiences[0]
            experience_confidence = (best_match.similarity_score + best_match.success_probability) / 2
            confidence += 0.3 * experience_confidence
        
        if patterns:
            # Weight by pattern confidence
            avg_pattern_confidence = sum(p.confidence_score for p in patterns) / len(patterns)
            confidence += 0.2 * avg_pattern_confidence
        
        return min(confidence, 0.95)
    
    def _estimate_request_metrics(self, similar_experiences: List[ExperienceMatch]) -> Dict[str, Any]:
        """Estimate metrics for current request"""
        if not similar_experiences:
            return {
                "estimated_cost": 0.01,  # Default estimate
                "estimated_duration": 30,  # 30 minutes
                "success_probability": 0.7,
                "confidence": 0.3
            }
        
        costs = [exp.estimated_cost for exp in similar_experiences[:3]]
        durations = [exp.estimated_duration for exp in similar_experiences[:3]]
        success_probs = [exp.success_probability for exp in similar_experiences[:3]]
        
        return {
            "estimated_cost": sum(costs) / len(costs),
            "estimated_duration": int(sum(durations) / len(durations)),
            "success_probability": sum(success_probs) / len(success_probs),
            "confidence": similar_experiences[0].confidence
        }
    
    async def get_experience_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded experiences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic stats
                cursor.execute('SELECT COUNT(*) FROM experiences')
                total_experiences = cursor.fetchone()[0]
                
                # Success rate
                cursor.execute('''
                    SELECT AVG(CASE WHEN success_level IN ('excellent', 'good') THEN 1.0 ELSE 0.0 END)
                    FROM experiences
                ''')
                success_rate = cursor.fetchone()[0] or 0
                
                # Cost statistics
                cursor.execute('SELECT AVG(cost_usd), MIN(cost_usd), MAX(cost_usd) FROM experiences')
                cost_stats = cursor.fetchone()
                
                # Most common task types
                cursor.execute('''
                    SELECT task_type, COUNT(*) as count
                    FROM experiences
                    GROUP BY task_type
                    ORDER BY count DESC
                    LIMIT 5
                ''')
                popular_task_types = cursor.fetchall()
                
                # Pattern count
                cursor.execute('SELECT COUNT(*) FROM experience_patterns')
                pattern_count = cursor.fetchone()[0]
                
                summary = {
                    "status": "success",
                    "total_experiences": total_experiences,
                    "overall_success_rate": success_rate,
                    "cost_statistics": {
                        "average": cost_stats[0] or 0,
                        "minimum": cost_stats[1] or 0,
                        "maximum": cost_stats[2] or 0
                    },
                    "popular_task_types": [{"type": row[0], "count": row[1]} for row in popular_task_types],
                    "identified_patterns": pattern_count,
                    "system_health": "excellent" if total_experiences > 10 else "good" if total_experiences > 3 else "learning",
                    "timestamp": datetime.now().isoformat()
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"❌ Failed to generate experience summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# =============================================================================
# CLI INTEGRATION FUNCTIONS
# =============================================================================

async def get_experience_recommendations(request_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Get experience-based recommendations for CLI integration"""
    experience_manager = ExperienceManager()
    
    # Parse request context
    request_context = {
        'task_type': 'development',  # Default
        'complexity': 'moderate',
        'keywords': request_text.lower().split(),
        'request_text': request_text,
        **(context or {})
    }
    
    # Basic intent classification
    if any(word in request_text.lower() for word in ['analyze', 'study', 'research']):
        request_context['task_type'] = 'analysis'
    elif any(word in request_text.lower() for word in ['integrate', 'connect', 'link']):
        request_context['task_type'] = 'integration'
    elif any(word in request_text.lower() for word in ['optimize', 'improve', 'enhance']):
        request_context['task_type'] = 'optimization'
    elif any(word in request_text.lower() for word in ['plan', 'design', 'architecture']):
        request_context['task_type'] = 'planning'
    elif any(word in request_text.lower() for word in ['deploy', 'release', 'publish']):
        request_context['task_type'] = 'deployment'
    
    # Complexity assessment
    if len(request_text.split()) > 30 or any(word in request_text.lower() for word in ['complex', 'advanced', 'sophisticated']):
        request_context['complexity'] = 'complex'
    elif len(request_text.split()) < 10 or any(word in request_text.lower() for word in ['simple', 'basic', 'easy']):
        request_context['complexity'] = 'simple'
    
    return await experience_manager.get_recommendations(request_context)

async def record_task_experience(task_id: str, task_type: str, model_used: str, 
                               success_level: str, cost: float, duration: int,
                               quality_score: float, user_feedback: str = "") -> str:
    """Record experience for CLI integration"""
    experience_manager = ExperienceManager()
    
    experience = Experience(
        id=task_id,
        task_type=TaskTypeEnum(task_type.lower()),
        complexity=ComplexityLevelEnum.MODERATE,  # Default
        context={'keywords': [], 'source': 'cli'},
        approach_used=f"CLI execution with {model_used}",
        model_used=model_used,
        success_level=SuccessLevelEnum(success_level.lower()),
        cost_usd=cost,
        duration_minutes=duration,
        quality_score=quality_score,
        user_satisfaction=0.8 if success_level in ['excellent', 'good'] else 0.5,
        lessons_learned=[user_feedback] if user_feedback else [],
        reusable_components=[],
        created_at=datetime.now(),
        metadata={'source': 'agent_zero_cli'}
    )
    
    return await experience_manager.record_experience(experience)

# =============================================================================
# TESTING AND DEMO
# =============================================================================

async def demo_experience_system():
    """Demonstrate Experience Management System"""
    print("🧠 Agent Zero V2.0 - Experience Management System Demo")
    print("=" * 60)
    
    experience_manager = ExperienceManager()
    
    # Demo 1: Record some sample experiences
    print("\n1. Recording sample experiences...")
    
    experiences = [
        Experience(
            id="exp_001",
            task_type=TaskTypeEnum.DEVELOPMENT,
            complexity=ComplexityLevelEnum.MODERATE,
            context={'keywords': ['api', 'authentication', 'jwt'], 'domain': 'web_development'},
            approach_used="FastAPI with JWT authentication",
            model_used="claude-3-sonnet",
            success_level=SuccessLevelEnum.EXCELLENT,
            cost_usd=0.023,
            duration_minutes=45,
            quality_score=0.92,
            user_satisfaction=0.88,
            lessons_learned=["JWT secret management is crucial", "Rate limiting prevents abuse"],
            reusable_components=["JWT middleware", "User validation schema"],
            created_at=datetime.now() - timedelta(days=5),
            metadata={'complexity_factors': ['authentication', 'security']}
        ),
        Experience(
            id="exp_002", 
            task_type=TaskTypeEnum.ANALYSIS,
            complexity=ComplexityLevelEnum.COMPLEX,
            context={'keywords': ['database', 'performance', 'optimization'], 'domain': 'data'},
            approach_used="Query analysis with EXPLAIN and indexing",
            model_used="claude-3-sonnet",
            success_level=SuccessLevelEnum.GOOD,
            cost_usd=0.045,
            duration_minutes=78,
            quality_score=0.84,
            user_satisfaction=0.75,
            lessons_learned=["Index selection is critical", "Query rewriting can help"],
            reusable_components=["Performance analyzer script", "Index recommendations"],
            created_at=datetime.now() - timedelta(days=3),
            metadata={'complexity_factors': ['database', 'performance']}
        )
    ]
    
    for exp in experiences:
        exp_id = await experience_manager.record_experience(exp)
        print(f"  ✅ Recorded experience: {exp_id}")
    
    # Demo 2: Find similar experiences
    print("\n2. Finding similar experiences...")
    
    similar_request = {
        'task_type': 'development',
        'complexity': 'moderate',
        'keywords': ['api', 'security', 'user'],
        'request_text': 'I need to create a secure API with user authentication'
    }
    
    matches = await experience_manager.find_similar_experiences(similar_request)
    print(f"  🔍 Found {len(matches)} similar experiences")
    for match in matches:
        print(f"    - Experience {match.experience_id}: {match.similarity_score:.2f} similarity")
        print(f"      Approach: {match.recommended_approach}")
        print(f"      Success probability: {match.success_probability:.2f}")
    
    # Demo 3: Get recommendations
    print("\n3. Getting recommendations...")
    
    recommendations = await experience_manager.get_recommendations(similar_request)
    print(f"  📋 Generated {len(recommendations.get('recommendations', []))} recommendations")
    for i, rec in enumerate(recommendations.get('recommendations', [])[:3]):
        print(f"    {i+1}. {rec}")
    
    # Demo 4: Experience summary
    print("\n4. Experience system summary...")
    
    summary = await experience_manager.get_experience_summary()
    print(f"  📊 Total experiences: {summary.get('total_experiences', 0)}")
    print(f"  📊 Success rate: {summary.get('overall_success_rate', 0):.1%}")
    print(f"  📊 System health: {summary.get('system_health', 'unknown')}")
    
    print(f"\n✅ Experience Management System operational!")

if __name__ == "__main__":
    asyncio.run(demo_experience_system())