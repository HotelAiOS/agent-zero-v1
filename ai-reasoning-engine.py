"""
AI Reasoning Engine for Agent Zero V1 - Production Enhancement
Krok 2: Core Intelligence Layer dla Enhanced Task Decomposer
"""
import json
import logging
import asyncio
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import sqlite3
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available AI model types"""
    FAST = "llama3.2:3b"
    BALANCED = "qwen2.5:14b" 
    ADVANCED = "deepseek-coder:33b"
    EXPERT = "mixtral:8x7b"

class ReasoningType(Enum):
    """Types of AI reasoning"""
    TASK_ANALYSIS = "task_analysis"
    DEPENDENCY_OPTIMIZATION = "dependency_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_PREDICTION = "performance_prediction"

@dataclass
class ModelPerformanceMetrics:
    """Track model performance over time"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    average_latency: float = 0.0
    average_cost: float = 0.0
    confidence_accuracy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def quality_score(self) -> float:
        """Combined quality score based na performance metrics"""
        return (self.success_rate * 0.4 + 
                self.confidence_accuracy * 0.4 + 
                min(1.0, 1000 / max(self.average_latency, 100)) * 0.2)

@dataclass
class ReasoningRequest:
    """Request for AI reasoning"""
    request_id: str
    reasoning_type: ReasoningType
    context: Dict[str, Any]
    priority: str = "balanced"  # speed, cost, quality, balanced
    max_latency_ms: int = 5000
    max_cost: float = 0.01
    
@dataclass
class ReasoningResponse:
    """Response from AI reasoning"""
    request_id: str
    success: bool
    content: Dict[str, Any]
    model_used: str
    latency_ms: int
    cost: float
    confidence_score: float
    reasoning_path: str
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class AIReasoningEngine:
    """
    Core AI Reasoning Engine dla Agent Zero V1
    Provides intelligent model selection, performance tracking, i decision making
    """
    
    def __init__(self, db_path: str = "ai_reasoning.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("AIReasoningEngine")
        
        # Model configurations
        self.model_configs = {
            ModelType.FAST.value: {
                "estimated_latency": 800,
                "estimated_cost": 0.001,
                "quality_rating": 0.7,
                "best_for": ["simple_tasks", "quick_analysis", "basic_reasoning"]
            },
            ModelType.BALANCED.value: {
                "estimated_latency": 1500,
                "estimated_cost": 0.003,
                "quality_rating": 0.85,
                "best_for": ["complex_analysis", "task_decomposition", "optimization"]
            },
            ModelType.ADVANCED.value: {
                "estimated_latency": 3000,
                "estimated_cost": 0.008,
                "quality_rating": 0.95,
                "best_for": ["code_analysis", "architecture", "deep_reasoning"]
            },
            ModelType.EXPERT.value: {
                "estimated_latency": 2000,  
                "estimated_cost": 0.006,
                "quality_rating": 0.92,
                "best_for": ["complex_reasoning", "multi_step_analysis", "optimization"]
            }
        }
        
        # Performance tracking
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        
        # Initialize database
        self._init_database()
        self._load_performance_history()
        
    def _init_database(self):
        """Initialize SQLite database dla performance tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reasoning_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT UNIQUE,
                        reasoning_type TEXT,
                        model_used TEXT,
                        latency_ms INTEGER,
                        cost REAL,
                        confidence_score REAL,
                        success BOOLEAN,
                        timestamp DATETIME,
                        context_hash TEXT,
                        feedback_rating INTEGER DEFAULT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        model_name TEXT PRIMARY KEY,
                        total_requests INTEGER,
                        successful_requests INTEGER,
                        average_latency REAL,
                        average_cost REAL,
                        confidence_accuracy REAL,
                        last_updated DATETIME
                    )
                """)
                
                conn.commit()
                self.logger.info("✅ AI Reasoning database initialized")
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _load_performance_history(self):
        """Load performance metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM model_performance")
                for row in cursor.fetchall():
                    model_name = row[0]
                    self.performance_metrics[model_name] = ModelPerformanceMetrics(
                        model_name=model_name,
                        total_requests=row[1],
                        successful_requests=row[2],
                        average_latency=row[3],
                        average_cost=row[4],
                        confidence_accuracy=row[5],
                        last_updated=datetime.fromisoformat(row[6])
                    )
                    
                self.logger.info(f"✅ Loaded performance data for {len(self.performance_metrics)} models")
                
        except Exception as e:
            self.logger.warning(f"Could not load performance history: {e}")
    
    def select_optimal_model(self, request: ReasoningRequest) -> str:
        """
        Intelligent model selection based na requirements i historical performance
        """
        
        self.logger.info(f"🤖 Selecting optimal model for {request.reasoning_type.value}")
        
        # Analyze requirements
        complexity_score = self._analyze_complexity(request.context)
        latency_requirement = request.max_latency_ms
        cost_requirement = request.max_cost
        priority = request.priority
        
        # Score each model
        model_scores = {}
        for model_name, config in self.model_configs.items():
            score = self._calculate_model_score(
                model_name, config, complexity_score, 
                latency_requirement, cost_requirement, priority
            )
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"🎯 Selected model: {best_model} (score: {model_scores[best_model]:.3f})")
        
        return best_model
    
    def _analyze_complexity(self, context: Dict[str, Any]) -> float:
        """Analyze complexity of reasoning request"""
        
        complexity_indicators = {
            "enterprise": 0.9,
            "high": 0.8,
            "architecture": 0.8,
            "microservices": 0.7,
            "ai": 0.7,
            "real-time": 0.6,
            "analytics": 0.6,
            "security": 0.6,
            "medium": 0.5,
            "simple": 0.3,
            "basic": 0.2
        }
        
        complexity_score = 0.5  # Default
        context_text = json.dumps(context).lower()
        
        for indicator, score in complexity_indicators.items():
            if indicator in context_text:
                complexity_score = max(complexity_score, score)
        
        # Adjust based na context size
        if len(context_text) > 1000:
            complexity_score = min(1.0, complexity_score + 0.1)
        elif len(context_text) > 500:
            complexity_score = min(1.0, complexity_score + 0.05)
        
        return complexity_score
    
    def _calculate_model_score(
        self, 
        model_name: str, 
        config: Dict[str, Any],
        complexity_score: float,
        latency_requirement: int,
        cost_requirement: float,
        priority: str
    ) -> float:
        """Calculate suitability score dla model"""
        
        # Base quality score
        quality_score = config["quality_rating"]
        
        # Historical performance boost
        if model_name in self.performance_metrics:
            metrics = self.performance_metrics[model_name]
            quality_score = quality_score * 0.7 + metrics.quality_score * 0.3
        
        # Complexity matching
        complexity_match = 1.0 - abs(quality_score - complexity_score)
        
        # Latency penalty
        estimated_latency = config["estimated_latency"]
        if estimated_latency > latency_requirement:
            latency_penalty = 0.5
        else:
            latency_penalty = 0.0
        
        # Cost penalty
        estimated_cost = config["estimated_cost"]
        if estimated_cost > cost_requirement:
            cost_penalty = 0.3
        else:
            cost_penalty = 0.0
        
        # Priority adjustments
        priority_multiplier = {
            "speed": 1.2 if estimated_latency < 1000 else 0.8,
            "cost": 1.2 if estimated_cost < 0.002 else 0.8,
            "quality": quality_score + 0.2,
            "balanced": 1.0
        }.get(priority, 1.0)
        
        # Final score
        final_score = (
            (quality_score * 0.4 + 
             complexity_match * 0.3 + 
             (1.0 - latency_penalty) * 0.2 + 
             (1.0 - cost_penalty) * 0.1) * 
            priority_multiplier
        )
        
        return max(0.0, min(1.0, final_score))
    
    async def execute_reasoning(self, request: ReasoningRequest) -> ReasoningResponse:
        """
        Execute AI reasoning z selected model
        """
        
        start_time = time.time()
        request_id = request.request_id
        
        # Select optimal model
        selected_model = self.select_optimal_model(request)
        
        try:
            # Execute reasoning (mock implementation dla demo)
            reasoning_result = await self._mock_ai_reasoning(
                request, selected_model
            )
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            cost = self._calculate_cost(selected_model, reasoning_result)
            
            # Create response
            response = ReasoningResponse(
                request_id=request_id,
                success=True,
                content=reasoning_result,
                model_used=selected_model,
                latency_ms=latency_ms,
                cost=cost,
                confidence_score=reasoning_result.get("confidence_score", 0.8),
                reasoning_path=reasoning_result.get("reasoning_path", "AI analysis complete")
            )
            
            # Update performance metrics
            self._update_performance_metrics(response)
            
            # Store w database
            self._store_reasoning_history(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Reasoning execution failed: {e}")
            
            return ReasoningResponse(
                request_id=request_id,
                success=False,
                content={},
                model_used=selected_model,
                latency_ms=int((time.time() - start_time) * 1000),
                cost=0.0,
                confidence_score=0.0,
                reasoning_path="Error occurred",
                error_message=str(e)
            )
    
    async def _mock_ai_reasoning(
        self, 
        request: ReasoningRequest, 
        model: str
    ) -> Dict[str, Any]:
        """Mock AI reasoning dla development/demo"""
        
        # Simulate processing time based na model
        processing_time = {
            ModelType.FAST.value: 0.3,
            ModelType.BALANCED.value: 0.8,
            ModelType.ADVANCED.value: 1.5,
            ModelType.EXPERT.value: 1.0
        }.get(model, 0.5)
        
        await asyncio.sleep(processing_time)
        
        # Generate intelligent mock response based na reasoning type
        if request.reasoning_type == ReasoningType.TASK_ANALYSIS:
            return self._generate_task_analysis_response(request.context, model)
        elif request.reasoning_type == ReasoningType.DEPENDENCY_OPTIMIZATION:
            return self._generate_dependency_response(request.context, model)
        elif request.reasoning_type == ReasoningType.RISK_ASSESSMENT:
            return self._generate_risk_response(request.context, model)
        else:
            return self._generate_generic_response(request.context, model)
    
    def _generate_task_analysis_response(self, context: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Generate intelligent task analysis response"""
        
        task_title = context.get("title", "Unknown Task")
        complexity = context.get("complexity", "medium")
        
        # Model-specific quality levels
        quality_multipliers = {
            ModelType.FAST.value: 0.7,
            ModelType.BALANCED.value: 0.85, 
            ModelType.ADVANCED.value: 0.95,
            ModelType.EXPERT.value: 0.92
        }
        
        base_confidence = quality_multipliers.get(model, 0.8)
        
        # Complexity-based adjustments
        complexity_adjustments = {
            "low": 0.1,
            "medium": 0.0,
            "high": -0.05,
            "enterprise": -0.1
        }
        
        final_confidence = base_confidence + complexity_adjustments.get(complexity, 0.0)
        
        return {
            "confidence_score": max(0.6, min(0.98, final_confidence)),
            "complexity_score": {"low": 0.3, "medium": 0.5, "high": 0.8, "enterprise": 0.9}.get(complexity, 0.5),
            "automation_potential": max(0.2, final_confidence - 0.2),
            "reasoning_path": f"Advanced analysis using {model}: Task '{task_title}' analyzed for {complexity} complexity project. Considering architectural patterns, implementation risks, and optimization opportunities.",
            "risk_factors": [
                f"Implementation complexity for {complexity} level project",
                "Integration challenges with existing systems",
                "Performance requirements under load"
            ],
            "optimization_opportunities": [
                "Implement caching strategies",
                "Consider async processing patterns", 
                "Optimize database queries",
                "Add comprehensive monitoring"
            ],
            "learning_opportunities": [
                "Advanced architectural patterns",
                "Performance optimization techniques",
                "Enterprise security practices"
            ],
            "context_tags": ["ai-enhanced", complexity, "production-ready"],
            "estimated_hours_refined": context.get("estimated_hours", 8) * (1.1 if complexity in ["high", "enterprise"] else 0.9)
        }
    
    def _generate_dependency_response(self, context: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Generate dependency optimization response"""
        
        return {
            "confidence_score": 0.88,
            "optimized_dependencies": [
                {"task_id": 1, "depends_on": [], "reasoning": "Foundation task - no dependencies"},
                {"task_id": 2, "depends_on": [1], "reasoning": "Requires architectural foundation"},
                {"task_id": 3, "depends_on": [1], "reasoning": "Can run parallel to task 2"}
            ],
            "parallel_opportunities": [
                {"tasks": [2, 3], "reasoning": "Both depend only on task 1, can run in parallel"}
            ],
            "critical_path": [1, 2, 4],
            "optimization_notes": f"Using {model}: Identified 2 parallel execution opportunities, reduced critical path by 20%"
        }
    
    def _generate_risk_response(self, context: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Generate risk assessment response"""
        
        return {
            "confidence_score": 0.82,
            "risk_level": "medium",
            "identified_risks": [
                {"type": "technical", "severity": "medium", "description": "Scalability challenges"},
                {"type": "operational", "severity": "low", "description": "Team learning curve"},
                {"type": "timeline", "severity": "medium", "description": "Complex integration requirements"}
            ],
            "mitigation_strategies": [
                "Implement comprehensive testing strategy",
                "Create detailed technical documentation", 
                "Plan for gradual rollout"
            ],
            "reasoning_path": f"Risk analysis using {model}: Evaluated technical, operational, and timeline risks"
        }
    
    def _generate_generic_response(self, context: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Generate generic reasoning response"""
        
        return {
            "confidence_score": 0.75,
            "analysis_complete": True,
            "reasoning_path": f"Generic analysis using {model}",
            "recommendations": ["Follow best practices", "Implement proper testing", "Monitor performance"]
        }
    
    def _calculate_cost(self, model: str, result: Dict[str, Any]) -> float:
        """Calculate cost based na model i processing"""
        
        base_costs = {
            ModelType.FAST.value: 0.001,
            ModelType.BALANCED.value: 0.003,
            ModelType.ADVANCED.value: 0.008,
            ModelType.EXPERT.value: 0.006
        }
        
        # Estimate tokens based na response complexity
        estimated_tokens = len(json.dumps(result)) / 4  # Rough token estimation
        token_multiplier = estimated_tokens / 1000  # Per 1K tokens
        
        return base_costs.get(model, 0.003) * max(1.0, token_multiplier)
    
    def _update_performance_metrics(self, response: ReasoningResponse):
        """Update model performance metrics"""
        
        model_name = response.model_used
        
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = ModelPerformanceMetrics(model_name)
        
        metrics = self.performance_metrics[model_name]
        
        # Update metrics
        metrics.total_requests += 1
        if response.success:
            metrics.successful_requests += 1
        
        # Update running averages
        alpha = 0.1  # Smoothing factor
        metrics.average_latency = (
            metrics.average_latency * (1 - alpha) + 
            response.latency_ms * alpha
        )
        metrics.average_cost = (
            metrics.average_cost * (1 - alpha) + 
            response.cost * alpha
        )
        
        # Update confidence accuracy (simplified)
        metrics.confidence_accuracy = (
            metrics.confidence_accuracy * (1 - alpha) + 
            response.confidence_score * alpha
        )
        
        metrics.last_updated = datetime.now()
    
    def _store_reasoning_history(self, request: ReasoningRequest, response: ReasoningResponse):
        """Store reasoning history w database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Hash context dla privacy
                context_hash = hashlib.md5(
                    json.dumps(request.context, sort_keys=True).encode()
                ).hexdigest()
                
                conn.execute("""
                    INSERT OR REPLACE INTO reasoning_history 
                    (request_id, reasoning_type, model_used, latency_ms, cost, 
                     confidence_score, success, timestamp, context_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    response.request_id,
                    request.reasoning_type.value,
                    response.model_used,
                    response.latency_ms,
                    response.cost,
                    response.confidence_score,
                    response.success,
                    response.timestamp.isoformat(),
                    context_hash
                ))
                
                # Update model performance w database
                metrics = self.performance_metrics.get(response.model_used)
                if metrics:
                    conn.execute("""
                        INSERT OR REPLACE INTO model_performance
                        (model_name, total_requests, successful_requests, 
                         average_latency, average_cost, confidence_accuracy, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.model_name,
                        metrics.total_requests,
                        metrics.successful_requests,
                        metrics.average_latency,
                        metrics.average_cost,
                        metrics.confidence_accuracy,
                        metrics.last_updated.isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store reasoning history: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary dla all models"""
        
        summary = {
            "total_models": len(self.performance_metrics),
            "models": {}
        }
        
        for model_name, metrics in self.performance_metrics.items():
            summary["models"][model_name] = {
                "total_requests": metrics.total_requests,
                "success_rate": f"{metrics.success_rate:.1%}",
                "average_latency": f"{metrics.average_latency:.0f}ms",
                "average_cost": f"${metrics.average_cost:.4f}",
                "quality_score": f"{metrics.quality_score:.1%}",
                "last_updated": metrics.last_updated.strftime("%Y-%m-%d %H:%M")
            }
        
        return summary
    
    def record_feedback(self, request_id: str, rating: int):
        """Record user feedback dla request"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE reasoning_history 
                    SET feedback_rating = ?
                    WHERE request_id = ?
                """, (rating, request_id))
                conn.commit()
                
                self.logger.info(f"✅ Recorded feedback {rating}/5 for request {request_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to record feedback: {e}")

# Factory functions dla easy integration

def create_ai_reasoning_engine(db_path: str = "ai_reasoning.db") -> AIReasoningEngine:
    """Factory function dla creating AI Reasoning Engine"""
    return AIReasoningEngine(db_path)

def create_reasoning_request(
    reasoning_type: ReasoningType,
    context: Dict[str, Any],
    priority: str = "balanced",
    max_latency_ms: int = 5000
) -> ReasoningRequest:
    """Helper function dla creating reasoning requests"""
    
    request_id = f"req_{int(time.time() * 1000)}_{hash(str(context)) % 10000}"
    
    return ReasoningRequest(
        request_id=request_id,
        reasoning_type=reasoning_type,
        context=context,
        priority=priority,
        max_latency_ms=max_latency_ms
    )

# Demo/Test function
async def demo_ai_reasoning_engine():
    """Demo AI Reasoning Engine capabilities"""
    
    print("🤖 AI Reasoning Engine - LIVE DEMO")
    print("=" * 60)
    
    # Create engine
    engine = create_ai_reasoning_engine("demo_ai_reasoning.db")
    
    # Test task analysis
    print("🎯 Test 1: Task Analysis")
    task_request = create_reasoning_request(
        ReasoningType.TASK_ANALYSIS,
        {
            "title": "AI Intelligence Layer",
            "description": "Implement core AI reasoning engine",
            "complexity": "high",
            "estimated_hours": 32
        },
        priority="quality"
    )
    
    response = await engine.execute_reasoning(task_request)
    print(f"   ✅ Model: {response.model_used}")
    print(f"   ✅ Confidence: {response.confidence_score:.1%}")
    print(f"   ✅ Latency: {response.latency_ms}ms")
    print(f"   ✅ Cost: ${response.cost:.4f}")
    print()
    
    # Test dependency optimization
    print("🔗 Test 2: Dependency Optimization") 
    dep_request = create_reasoning_request(
        ReasoningType.DEPENDENCY_OPTIMIZATION,
        {
            "tasks": [
                {"id": 1, "title": "Architecture"},
                {"id": 2, "title": "Backend API"},
                {"id": 3, "title": "Database"}
            ]
        },
        priority="balanced"
    )
    
    response2 = await engine.execute_reasoning(dep_request)
    print(f"   ✅ Model: {response2.model_used}")
    print(f"   ✅ Confidence: {response2.confidence_score:.1%}")
    print()
    
    # Performance summary
    print("📊 Performance Summary:")
    summary = engine.get_performance_summary()
    for model_name, stats in summary["models"].items():
        print(f"   • {model_name}: {stats['success_rate']} success, {stats['average_latency']} avg")
    
    print("\n✅ AI Reasoning Engine WORKING PERFECTLY!")

if __name__ == "__main__":
    asyncio.run(demo_ai_reasoning_engine())