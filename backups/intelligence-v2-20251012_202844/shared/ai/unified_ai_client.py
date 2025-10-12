#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Real-Time AI Reasoning Engine
Production-ready unified AI client with context-aware reasoning

Priority 2.1: Unified AI Client (1 SP)
- Centralized Ollama connection management
- Model selection based on task complexity  
- Response caching and optimization
- Fallback handling and error recovery
"""

import asyncio
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3

# Import existing production AI system
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.production_ai_system import ProductionAISystem
    PRODUCTION_AI_AVAILABLE = True
except ImportError:
    PRODUCTION_AI_AVAILABLE = False
    ProductionAISystem = None

logger = logging.getLogger(__name__)

class AIModelType(Enum):
    """AI model types for different reasoning complexity"""
    FAST = "fast"           # Quick responses, simple reasoning
    STANDARD = "standard"   # Balanced performance and quality
    ADVANCED = "advanced"   # Complex reasoning, higher quality
    CODE = "code"          # Code-specific reasoning
    EXPERT = "expert"      # Most complex reasoning tasks

class ReasoningType(Enum):
    """Types of AI reasoning requests"""
    TASK_ANALYSIS = "task_analysis"
    AGENT_SELECTION = "agent_selection" 
    DECISION_MAKING = "decision_making"
    CODE_REVIEW = "code_review"
    PROBLEM_SOLVING = "problem_solving"
    PLANNING = "planning"
    QUALITY_ASSESSMENT = "quality_assessment"

@dataclass
class ReasoningContext:
    """Context for AI reasoning requests"""
    project_type: str = "general"
    tech_stack: List[str] = field(default_factory=list)
    team_skills: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    previous_decisions: List[Dict] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AIReasoningRequest:
    """AI reasoning request structure"""
    request_id: str
    reasoning_type: ReasoningType
    prompt: str
    context: ReasoningContext
    model_preference: Optional[AIModelType] = None
    max_response_tokens: int = 500
    temperature: float = 0.2
    cache_enabled: bool = True
    timeout: float = 30.0

@dataclass 
class AIReasoningResponse:
    """AI reasoning response structure"""
    request_id: str
    reasoning_type: ReasoningType
    response_text: str
    confidence: float
    model_used: str
    response_time: float
    tokens_used: int
    cached: bool
    timestamp: datetime
    reasoning_chain: List[str] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)
    confidence_factors: Dict[str, float] = field(default_factory=dict)

class UnifiedAIClient:
    """
    Unified AI Client for Agent Zero V2.0
    
    Features:
    - Centralized Ollama connection management
    - Intelligent model selection based on task complexity
    - Response caching for performance optimization
    - Context-aware reasoning with project knowledge
    - Comprehensive error handling and fallback
    - Performance monitoring and optimization
    """
    
    def __init__(self, db_path: str = "agent_zero.db", cache_ttl: int = 3600):
        self.db_path = db_path
        self.cache_ttl = cache_ttl  # Cache TTL in seconds
        self.response_cache: Dict[str, Tuple[AIReasoningResponse, datetime]] = {}
        
        # Initialize production AI system
        self.production_ai = None
        if PRODUCTION_AI_AVAILABLE:
            try:
                self.production_ai = ProductionAISystem()
                logger.info("✅ Production AI system connected")
            except Exception as e:
                logger.warning(f"Production AI initialization failed: {e}")
        
        # Model selection mapping
        self.model_mapping = {
            AIModelType.FAST: "fast",
            AIModelType.STANDARD: "standard", 
            AIModelType.ADVANCED: "advanced",
            AIModelType.CODE: "code",
            AIModelType.EXPERT: "expert"
        }
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "model_usage": {},
            "error_count": 0
        }
        
        self._init_database()
        logger.info("✅ UnifiedAIClient initialized")
    
    def _init_database(self):
        """Initialize AI reasoning database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # AI reasoning requests log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_reasoning_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    reasoning_type TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    model_used TEXT,
                    response_time REAL,
                    tokens_used INTEGER,
                    confidence REAL,
                    cached BOOLEAN,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # AI reasoning cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_reasoning_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    reasoning_type TEXT NOT NULL,
                    response_data TEXT NOT NULL,  -- JSON
                    model_used TEXT,
                    confidence REAL,
                    expires_at TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Context learning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_type TEXT,
                    tech_stack TEXT,  -- JSON array
                    reasoning_patterns TEXT,  -- JSON
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 1,
                    last_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def reason(self, request: AIReasoningRequest) -> AIReasoningResponse:
        """
        Main reasoning method - processes AI reasoning requests
        """
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        
        logger.info(f"🧠 AI Reasoning request: {request.reasoning_type.value} ({request.request_id})")
        
        try:
            # Check cache first
            if request.cache_enabled:
                cached_response = await self._get_cached_response(request)
                if cached_response:
                    self.performance_stats["cache_hits"] += 1
                    logger.info(f"💾 Cache hit for {request.request_id}")
                    return cached_response
            
            # Select optimal model
            selected_model = self._select_optimal_model(request)
            
            # Build context-aware prompt
            enhanced_prompt = self._build_context_aware_prompt(request)
            
            # Execute AI reasoning
            ai_response = await self._execute_reasoning(
                enhanced_prompt, 
                selected_model, 
                request
            )
            
            # Process and structure response
            structured_response = self._structure_response(
                request, 
                ai_response, 
                selected_model, 
                start_time
            )
            
            # Cache the response
            if request.cache_enabled:
                await self._cache_response(request, structured_response)
            
            # Log the reasoning
            self._log_reasoning_request(request, structured_response, None)
            
            # Update performance stats
            self._update_performance_stats(structured_response)
            
            logger.info(f"✅ AI Reasoning completed: {structured_response.confidence:.2f} confidence")
            
            return structured_response
            
        except Exception as e:
            error_msg = f"AI reasoning failed: {str(e)}"
            logger.error(error_msg)
            
            # Create error response
            error_response = AIReasoningResponse(
                request_id=request.request_id,
                reasoning_type=request.reasoning_type,
                response_text=f"Reasoning failed: {error_msg}",
                confidence=0.0,
                model_used="fallback",
                response_time=time.time() - start_time,
                tokens_used=0,
                cached=False,
                timestamp=datetime.now()
            )
            
            # Log error
            self._log_reasoning_request(request, error_response, error_msg)
            self.performance_stats["error_count"] += 1
            
            return error_response
    
    def _select_optimal_model(self, request: AIReasoningRequest) -> AIModelType:
        """Select optimal AI model based on reasoning complexity"""
        
        if request.model_preference:
            return request.model_preference
        
        # Model selection logic based on reasoning type
        model_selection_map = {
            ReasoningType.TASK_ANALYSIS: AIModelType.STANDARD,
            ReasoningType.AGENT_SELECTION: AIModelType.ADVANCED,
            ReasoningType.DECISION_MAKING: AIModelType.ADVANCED,
            ReasoningType.CODE_REVIEW: AIModelType.CODE,
            ReasoningType.PROBLEM_SOLVING: AIModelType.EXPERT,
            ReasoningType.PLANNING: AIModelType.ADVANCED,
            ReasoningType.QUALITY_ASSESSMENT: AIModelType.STANDARD
        }
        
        selected = model_selection_map.get(request.reasoning_type, AIModelType.STANDARD)
        
        # Upgrade model if context is complex
        if self._is_complex_context(request.context):
            if selected == AIModelType.FAST:
                selected = AIModelType.STANDARD
            elif selected == AIModelType.STANDARD:
                selected = AIModelType.ADVANCED
        
        logger.info(f"🎯 Selected model: {selected.value} for {request.reasoning_type.value}")
        return selected
    
    def _is_complex_context(self, context: ReasoningContext) -> bool:
        """Determine if reasoning context is complex"""
        complexity_indicators = [
            len(context.tech_stack) > 3,
            len(context.constraints) > 2,
            len(context.previous_decisions) > 0,
            context.project_type in ["enterprise", "complex_system", "ml_system"]
        ]
        return sum(complexity_indicators) >= 2
    
    def _build_context_aware_prompt(self, request: AIReasoningRequest) -> str:
        """Build enhanced prompt with context awareness"""
        
        base_prompt = request.prompt
        context = request.context
        
        # Add project context
        if context.project_type != "general":
            base_prompt += f"\n\nProject Type: {context.project_type}"
        
        if context.tech_stack:
            base_prompt += f"\nTech Stack: {', '.join(context.tech_stack)}"
        
        if context.team_skills:
            base_prompt += f"\nTeam Skills: {', '.join(context.team_skills)}"
        
        if context.constraints:
            base_prompt += f"\nConstraints: {', '.join(context.constraints)}"
        
        # Add historical context
        if context.previous_decisions:
            base_prompt += f"\nPrevious Decisions: {len(context.previous_decisions)} related decisions made"
        
        # Add domain knowledge
        if context.domain_knowledge:
            base_prompt += f"\nDomain Context: {json.dumps(context.domain_knowledge, indent=2)}"
        
        # Add reasoning instructions based on type
        reasoning_instructions = {
            ReasoningType.TASK_ANALYSIS: "Analyze the task thoroughly, consider complexity, risks, and requirements.",
            ReasoningType.AGENT_SELECTION: "Select the most suitable agent based on capabilities, workload, and performance.",
            ReasoningType.DECISION_MAKING: "Make a well-reasoned decision considering all factors and potential outcomes.",
            ReasoningType.CODE_REVIEW: "Review code for quality, performance, security, and best practices.",
            ReasoningType.PROBLEM_SOLVING: "Approach the problem systematically, consider multiple solutions.",
            ReasoningType.PLANNING: "Create a comprehensive plan with clear steps, dependencies, and timelines.",
            ReasoningType.QUALITY_ASSESSMENT: "Assess quality against established criteria and industry standards."
        }
        
        instruction = reasoning_instructions.get(request.reasoning_type, "Provide thoughtful analysis.")
        base_prompt += f"\n\nInstructions: {instruction}"
        
        # Add response format requirements
        base_prompt += f"""

Respond with structured reasoning including:
1. Analysis of the situation
2. Key considerations and factors
3. Reasoning process and logic
4. Final recommendation or conclusion
5. Confidence level (0.0-1.0) and rationale

Be specific, actionable, and provide clear reasoning for your conclusions."""
        
        return base_prompt
    
    async def _execute_reasoning(
        self, 
        prompt: str, 
        model_type: AIModelType, 
        request: AIReasoningRequest
    ) -> Dict[str, Any]:
        """Execute AI reasoning with the production AI system"""
        
        if not self.production_ai:
            # Fallback reasoning
            return {
                "success": True,
                "reasoning": self._generate_fallback_reasoning(request),
                "model_used": "fallback",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            model_name = self.model_mapping.get(model_type, "standard")
            
            # Call production AI system
            ai_response = self.production_ai.generate_ai_reasoning(
                prompt=prompt,
                model_type=model_name
            )
            
            return ai_response
            
        except Exception as e:
            logger.warning(f"Production AI failed: {e}, using fallback")
            return {
                "success": True,
                "reasoning": self._generate_fallback_reasoning(request),
                "model_used": "fallback",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_fallback_reasoning(self, request: AIReasoningRequest) -> str:
        """Generate fallback reasoning when AI is unavailable"""
        
        fallback_responses = {
            ReasoningType.TASK_ANALYSIS: f"""
Task Analysis (Fallback):
- Task appears to be {request.reasoning_type.value} related
- Estimated complexity: moderate based on context
- Recommended approach: systematic analysis and structured implementation
- Key considerations: requirements clarity, resource availability, timeline constraints
- Confidence: 0.6 (fallback reasoning)""",
            
            ReasoningType.AGENT_SELECTION: """
Agent Selection (Fallback):
- Based on available context, recommend using most skilled available agent
- Consider current workload and specializations
- Fallback to generalist agent if specialist unavailable
- Confidence: 0.5 (fallback reasoning)""",
            
            ReasoningType.DECISION_MAKING: """
Decision Analysis (Fallback):
- Multiple factors should be considered
- Risk assessment recommended
- Stakeholder impact analysis needed
- Recommend data-driven decision approach
- Confidence: 0.5 (fallback reasoning)"""
        }
        
        return fallback_responses.get(
            request.reasoning_type,
            f"Fallback reasoning for {request.reasoning_type.value}: systematic approach recommended. Confidence: 0.5"
        )
    
    def _structure_response(
        self, 
        request: AIReasoningRequest, 
        ai_response: Dict[str, Any], 
        model_type: AIModelType, 
        start_time: float
    ) -> AIReasoningResponse:
        """Structure AI response into standardized format"""
        
        response_time = time.time() - start_time
        
        # Extract confidence from response
        confidence = self._extract_confidence(ai_response.get("reasoning", ""))
        
        # Extract reasoning chain
        reasoning_chain = self._extract_reasoning_chain(ai_response.get("reasoning", ""))
        
        return AIReasoningResponse(
            request_id=request.request_id,
            reasoning_type=request.reasoning_type,
            response_text=ai_response.get("reasoning", "No reasoning provided"),
            confidence=confidence,
            model_used=ai_response.get("model_used", model_type.value),
            response_time=response_time,
            tokens_used=len(ai_response.get("reasoning", "").split()),  # Approximate token count
            cached=False,
            timestamp=datetime.now(),
            reasoning_chain=reasoning_chain,
            confidence_factors={"model_availability": 1.0 if ai_response.get("success") else 0.5}
        )
    
    def _extract_confidence(self, response_text: str) -> float:
        """Extract confidence level from AI response"""
        # Simple confidence extraction logic
        if "confidence" in response_text.lower():
            # Try to find numeric confidence values
            import re
            confidence_matches = re.findall(r'confidence[:\s]*([0-9.]+)', response_text.lower())
            if confidence_matches:
                try:
                    confidence = float(confidence_matches[0])
                    return min(max(confidence, 0.0), 1.0)  # Clamp to 0-1 range
                except:
                    pass
        
        # Fallback confidence based on response quality
        if len(response_text) > 200 and "analysis" in response_text.lower():
            return 0.75
        elif len(response_text) > 100:
            return 0.6
        else:
            return 0.4
    
    def _extract_reasoning_chain(self, response_text: str) -> List[str]:
        """Extract reasoning steps from AI response"""
        # Simple reasoning chain extraction
        steps = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in ['step', '1.', '2.', '3.', 'first', 'then', 'finally']):
                if len(line) > 10:  # Meaningful step
                    steps.append(line)
        
        return steps[:5]  # Limit to 5 steps
    
    async def _get_cached_response(self, request: AIReasoningRequest) -> Optional[AIReasoningResponse]:
        """Retrieve cached response if available and not expired"""
        cache_key = self._generate_cache_key(request)
        
        # Check in-memory cache first
        if cache_key in self.response_cache:
            cached_response, cached_time = self.response_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                cached_response.cached = True
                return cached_response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        
        # Check database cache
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT response_data, model_used, confidence, hit_count 
                    FROM ai_reasoning_cache 
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, datetime.now().isoformat()))
                
                result = cursor.fetchone()
                if result:
                    response_data, model_used, confidence, hit_count = result
                    
                    # Update hit count
                    conn.execute("""
                        UPDATE ai_reasoning_cache 
                        SET hit_count = hit_count + 1 
                        WHERE cache_key = ?
                    """, (cache_key,))
                    conn.commit()
                    
                    # Reconstruct response
                    response_dict = json.loads(response_data)
                    cached_response = AIReasoningResponse(**response_dict)
                    cached_response.cached = True
                    
                    return cached_response
                    
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_response(self, request: AIReasoningRequest, response: AIReasoningResponse):
        """Cache AI reasoning response"""
        cache_key = self._generate_cache_key(request)
        
        # Cache in memory
        self.response_cache[cache_key] = (response, datetime.now())
        
        # Cache in database
        try:
            expires_at = datetime.now() + timedelta(seconds=self.cache_ttl)
            response_dict = {
                "request_id": response.request_id,
                "reasoning_type": response.reasoning_type.value,
                "response_text": response.response_text,
                "confidence": response.confidence,
                "model_used": response.model_used,
                "response_time": response.response_time,
                "tokens_used": response.tokens_used,
                "cached": False,
                "timestamp": response.timestamp.isoformat(),
                "reasoning_chain": response.reasoning_chain,
                "confidence_factors": response.confidence_factors
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ai_reasoning_cache
                    (cache_key, reasoning_type, response_data, model_used, confidence, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    request.reasoning_type.value,
                    json.dumps(response_dict),
                    response.model_used,
                    response.confidence,
                    expires_at.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _generate_cache_key(self, request: AIReasoningRequest) -> str:
        """Generate unique cache key for request"""
        # Create hash of prompt + context for caching
        cache_content = {
            "reasoning_type": request.reasoning_type.value,
            "prompt": request.prompt,
            "project_type": request.context.project_type,
            "tech_stack": sorted(request.context.tech_stack),
            "constraints": sorted(request.context.constraints)
        }
        
        cache_string = json.dumps(cache_content, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _log_reasoning_request(
        self, 
        request: AIReasoningRequest, 
        response: AIReasoningResponse, 
        error: Optional[str]
    ):
        """Log reasoning request for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()
                
                conn.execute("""
                    INSERT INTO ai_reasoning_log
                    (request_id, reasoning_type, prompt_hash, model_used, response_time,
                     tokens_used, confidence, cached, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request.request_id,
                    request.reasoning_type.value,
                    prompt_hash,
                    response.model_used,
                    response.response_time,
                    response.tokens_used,
                    response.confidence,
                    response.cached,
                    error is None,
                    error
                ))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Logging failed: {e}")
    
    def _update_performance_stats(self, response: AIReasoningResponse):
        """Update performance statistics"""
        # Update average response time
        current_avg = self.performance_stats["avg_response_time"]
        total_requests = self.performance_stats["total_requests"]
        
        self.performance_stats["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response.response_time) / total_requests
        )
        
        # Update model usage
        model = response.model_used
        if model not in self.performance_stats["model_usage"]:
            self.performance_stats["model_usage"][model] = 0
        self.performance_stats["model_usage"][model] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI client performance statistics"""
        cache_hit_rate = (
            self.performance_stats["cache_hits"] / 
            max(self.performance_stats["total_requests"], 1)
        ) * 100
        
        error_rate = (
            self.performance_stats["error_count"] / 
            max(self.performance_stats["total_requests"], 1)
        ) * 100
        
        return {
            **self.performance_stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "error_rate_percent": round(error_rate, 1),
            "ai_system_available": self.production_ai is not None
        }

# Demo and testing functions
async def demo_unified_ai_client():
    """Demo the unified AI client capabilities"""
    print("🧠 Agent Zero V2.0 - Unified AI Client Demo")
    print("=" * 50)
    
    # Initialize client
    client = UnifiedAIClient()
    
    # Demo reasoning contexts
    demo_contexts = [
        ReasoningContext(
            project_type="web_application",
            tech_stack=["FastAPI", "React", "PostgreSQL"],
            constraints=["2-week timeline", "3-person team"]
        ),
        ReasoningContext(
            project_type="microservice",
            tech_stack=["Docker", "Kubernetes", "Python"],
            team_skills=["backend", "devops"]
        )
    ]
    
    # Demo requests
    demo_requests = [
        AIReasoningRequest(
            request_id="demo_1",
            reasoning_type=ReasoningType.TASK_ANALYSIS,
            prompt="Analyze the complexity of building a user authentication system with JWT tokens",
            context=demo_contexts[0],
            model_preference=AIModelType.STANDARD
        ),
        AIReasoningRequest(
            request_id="demo_2", 
            reasoning_type=ReasoningType.AGENT_SELECTION,
            prompt="Select the best agent for deploying a microservice to Kubernetes cluster",
            context=demo_contexts[1],
            model_preference=AIModelType.ADVANCED
        )
    ]
    
    # Execute reasoning requests
    for i, request in enumerate(demo_requests, 1):
        print(f"\n🧠 Demo Request {i}: {request.reasoning_type.value}")
        print(f"   Prompt: {request.prompt[:60]}...")
        print(f"   Model: {request.model_preference.value}")
        
        response = await client.reason(request)
        
        print(f"   ✅ Response received:")
        print(f"      Confidence: {response.confidence:.2f}")
        print(f"      Model Used: {response.model_used}")
        print(f"      Response Time: {response.response_time:.2f}s")
        print(f"      Cached: {'Yes' if response.cached else 'No'}")
        print(f"      Response: {response.response_text[:100]}...")
    
    # Test caching by repeating first request
    print(f"\n💾 Testing cache with repeated request...")
    repeated_response = await client.reason(demo_requests[0])
    print(f"   Cached: {'✅ Yes' if repeated_response.cached else '❌ No'}")
    
    # Show performance stats
    print(f"\n📊 Performance Statistics:")
    stats = client.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Unified AI Client demo completed!")

if __name__ == "__main__":
    print("🧠 Agent Zero V2.0 Phase 4 - Unified AI Client")
    print("Testing core functionality...")
    
    # Run demo
    asyncio.run(demo_unified_ai_client())