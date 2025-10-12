#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Context-Aware Reasoning Engine
Multi-step reasoning chains with historical decision learning

Priority 2.2: Context-Aware Reasoning Engine (2 SP)
- Chain-of-thought reasoning with multi-step analysis
- Historical decision learning from previous projects
- Cross-project pattern recognition for better decisions
- Enhanced context injection with domain expertise
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

# Import unified AI client
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from unified_ai_client import UnifiedAIClient, AIReasoningRequest, ReasoningContext, ReasoningType, AIModelType
    UNIFIED_AI_AVAILABLE = True
except ImportError:
    UNIFIED_AI_AVAILABLE = False
    print("⚠️ UnifiedAIClient not available")

logger = logging.getLogger(__name__)

class ReasoningChainStep(Enum):
    """Steps in multi-step reasoning chain"""
    PROBLEM_ANALYSIS = "problem_analysis"
    CONTEXT_GATHERING = "context_gathering" 
    OPTION_GENERATION = "option_generation"
    PROS_CONS_ANALYSIS = "pros_cons_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION_SYNTHESIS = "decision_synthesis"
    CONFIDENCE_EVALUATION = "confidence_evaluation"

@dataclass
class HistoricalDecision:
    """Historical decision record for learning"""
    decision_id: str
    project_type: str
    tech_stack: List[str]
    context_hash: str
    reasoning_steps: List[str]
    final_decision: str
    outcome_success: bool
    confidence_level: float
    lessons_learned: List[str]
    timestamp: datetime
    usage_count: int = 0

@dataclass
class ReasoningChain:
    """Multi-step reasoning chain result"""
    chain_id: str
    reasoning_type: ReasoningType
    steps: List[Tuple[ReasoningChainStep, str]]  # (step_type, reasoning_text)
    final_reasoning: str
    confidence: float
    historical_matches: List[HistoricalDecision]
    pattern_insights: List[str]
    execution_time: float
    total_tokens: int

@dataclass
class DomainPattern:
    """Identified pattern in domain expertise"""
    pattern_id: str
    domain: str
    tech_stack_match: List[str]
    common_challenges: List[str]
    best_practices: List[str]
    success_indicators: List[str]
    confidence: float
    usage_frequency: int

class ContextAwareReasoningEngine:
    """
    Context-Aware Reasoning Engine with Historical Learning
    
    Features:
    - Chain-of-thought multi-step reasoning
    - Historical decision pattern matching
    - Cross-project learning and insights
    - Domain expertise pattern recognition
    - Enhanced context injection with experience
    - Confidence calibration based on historical success
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.unified_ai = None
        
        # Initialize UnifiedAIClient
        if UNIFIED_AI_AVAILABLE:
            try:
                self.unified_ai = UnifiedAIClient(db_path=db_path)
                logger.info("✅ UnifiedAIClient connected")
            except Exception as e:
                logger.warning(f"UnifiedAIClient initialization failed: {e}")
        
        # Historical learning storage
        self.historical_decisions: Dict[str, HistoricalDecision] = {}
        self.domain_patterns: Dict[str, DomainPattern] = {}
        
        # Performance tracking
        self.reasoning_stats = {
            "total_chains": 0,
            "avg_chain_length": 0.0,
            "avg_confidence": 0.0,
            "historical_matches": 0,
            "pattern_utilization": 0
        }
        
        self._init_database()
        self._load_historical_data()
        logger.info("✅ ContextAwareReasoningEngine initialized")
    
    def _init_database(self):
        """Initialize context-aware reasoning database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Historical decisions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT UNIQUE NOT NULL,
                    project_type TEXT NOT NULL,
                    tech_stack TEXT NOT NULL,  -- JSON array
                    context_hash TEXT NOT NULL,
                    reasoning_steps TEXT NOT NULL,  -- JSON array
                    final_decision TEXT NOT NULL,
                    outcome_success BOOLEAN,
                    confidence_level REAL,
                    lessons_learned TEXT,  -- JSON array
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_used TEXT
                )
            """)
            
            # Domain patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domain_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    domain TEXT NOT NULL,
                    tech_stack_match TEXT NOT NULL,  -- JSON array
                    common_challenges TEXT NOT NULL,  -- JSON array
                    best_practices TEXT NOT NULL,  -- JSON array
                    success_indicators TEXT NOT NULL,  -- JSON array
                    confidence REAL NOT NULL,
                    usage_frequency INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Reasoning chains log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chain_id TEXT UNIQUE NOT NULL,
                    reasoning_type TEXT NOT NULL,
                    steps_data TEXT NOT NULL,  -- JSON
                    final_reasoning TEXT NOT NULL,
                    confidence REAL,
                    historical_matches TEXT,  -- JSON array
                    pattern_insights TEXT,  -- JSON array
                    execution_time REAL,
                    total_tokens INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def reason_with_context(
        self, 
        problem_statement: str, 
        context: ReasoningContext,
        reasoning_type: ReasoningType = ReasoningType.DECISION_MAKING
    ) -> ReasoningChain:
        """
        Execute context-aware multi-step reasoning with historical learning
        """
        start_time = time.time()
        chain_id = f"chain_{int(time.time())}_{hashlib.md5(problem_statement.encode()).hexdigest()[:8]}"
        
        logger.info(f"🧠 Starting context-aware reasoning: {reasoning_type.value}")
        
        # Step 1: Gather historical context and patterns
        historical_matches = await self._find_historical_matches(context, problem_statement)
        domain_patterns = self._identify_domain_patterns(context)
        
        # Step 2: Build enhanced context with historical insights
        enhanced_context = self._enhance_context_with_history(context, historical_matches, domain_patterns)
        
        # Step 3: Execute multi-step reasoning chain
        reasoning_steps = await self._execute_reasoning_chain(
            problem_statement, enhanced_context, reasoning_type, historical_matches
        )
        
        # Step 4: Synthesize final reasoning with confidence calibration
        final_reasoning, confidence = await self._synthesize_reasoning(
            reasoning_steps, historical_matches, domain_patterns
        )
        
        # Step 5: Extract pattern insights for future learning
        pattern_insights = self._extract_pattern_insights(reasoning_steps, historical_matches)
        
        # Calculate execution metrics
        execution_time = time.time() - start_time
        total_tokens = sum(len(step[1].split()) for step in reasoning_steps)
        
        # Create reasoning chain result
        chain = ReasoningChain(
            chain_id=chain_id,
            reasoning_type=reasoning_type,
            steps=reasoning_steps,
            final_reasoning=final_reasoning,
            confidence=confidence,
            historical_matches=historical_matches,
            pattern_insights=pattern_insights,
            execution_time=execution_time,
            total_tokens=total_tokens
        )
        
        # Log reasoning chain for future learning
        self._log_reasoning_chain(chain)
        
        # Update statistics
        self._update_reasoning_stats(chain)
        
        logger.info(f"✅ Context-aware reasoning completed: {confidence:.2f} confidence, {len(reasoning_steps)} steps")
        
        return chain
    
    async def _find_historical_matches(
        self, 
        context: ReasoningContext, 
        problem_statement: str
    ) -> List[HistoricalDecision]:
        """Find historically similar decisions for learning"""
        
        matches = []
        context_hash = self._generate_context_hash(context)
        
        # Search for similar contexts
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find decisions with similar tech stack and project type
                cursor = conn.execute("""
                    SELECT * FROM historical_decisions 
                    WHERE project_type = ? 
                    AND outcome_success = 1
                    ORDER BY confidence_level DESC, usage_count DESC
                    LIMIT 5
                """, (context.project_type,))
                
                results = cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                
                for result in results:
                    row_dict = dict(zip(column_names, result))
                    
                    # Calculate similarity based on tech stack overlap
                    stored_tech_stack = json.loads(row_dict['tech_stack'])
                    tech_overlap = len(set(context.tech_stack) & set(stored_tech_stack))
                    
                    if tech_overlap > 0:  # At least one tech in common
                        historical_decision = HistoricalDecision(
                            decision_id=row_dict['decision_id'],
                            project_type=row_dict['project_type'],
                            tech_stack=stored_tech_stack,
                            context_hash=row_dict['context_hash'],
                            reasoning_steps=json.loads(row_dict['reasoning_steps']),
                            final_decision=row_dict['final_decision'],
                            outcome_success=row_dict['outcome_success'],
                            confidence_level=row_dict['confidence_level'],
                            lessons_learned=json.loads(row_dict['lessons_learned'] or '[]'),
                            timestamp=datetime.fromisoformat(row_dict['created_at']),
                            usage_count=row_dict['usage_count']
                        )
                        matches.append(historical_decision)
        
        except Exception as e:
            logger.warning(f"Historical search failed: {e}")
        
        logger.info(f"📚 Found {len(matches)} historical matches")
        return matches
    
    def _identify_domain_patterns(self, context: ReasoningContext) -> List[DomainPattern]:
        """Identify relevant domain patterns"""
        
        patterns = []
        
        # Load domain patterns from database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM domain_patterns
                    WHERE domain = ? OR domain = 'general'
                    ORDER BY confidence DESC, usage_frequency DESC
                    LIMIT 3
                """, (context.project_type,))
                
                results = cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                
                for result in results:
                    row_dict = dict(zip(column_names, result))
                    
                    # Check tech stack relevance
                    pattern_tech_stack = json.loads(row_dict['tech_stack_match'])
                    tech_relevance = len(set(context.tech_stack) & set(pattern_tech_stack))
                    
                    if tech_relevance > 0 or row_dict['domain'] == 'general':
                        pattern = DomainPattern(
                            pattern_id=row_dict['pattern_id'],
                            domain=row_dict['domain'],
                            tech_stack_match=pattern_tech_stack,
                            common_challenges=json.loads(row_dict['common_challenges']),
                            best_practices=json.loads(row_dict['best_practices']),
                            success_indicators=json.loads(row_dict['success_indicators']),
                            confidence=row_dict['confidence'],
                            usage_frequency=row_dict['usage_frequency']
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Domain pattern search failed: {e}")
        
        # Add default patterns if none found
        if not patterns:
            patterns = self._get_default_domain_patterns(context)
        
        logger.info(f"🎨 Identified {len(patterns)} domain patterns")
        return patterns
    
    def _get_default_domain_patterns(self, context: ReasoningContext) -> List[DomainPattern]:
        """Get default domain patterns when none are found in database"""
        
        default_patterns = {
            "web_application": DomainPattern(
                pattern_id="web_app_default",
                domain="web_application",
                tech_stack_match=["FastAPI", "React", "PostgreSQL", "Docker"],
                common_challenges=["authentication", "scalability", "security", "performance"],
                best_practices=["RESTful APIs", "component architecture", "database normalization", "containerization"],
                success_indicators=["response time < 200ms", "test coverage > 80%", "security audit passed"],
                confidence=0.7,
                usage_frequency=0
            ),
            "microservice": DomainPattern(
                pattern_id="microservice_default",
                domain="microservice", 
                tech_stack_match=["Docker", "Kubernetes", "API Gateway"],
                common_challenges=["service communication", "data consistency", "monitoring", "deployment"],
                best_practices=["service isolation", "circuit breakers", "health checks", "logging"],
                success_indicators=["service uptime > 99%", "deployment automation", "monitoring coverage"],
                confidence=0.7,
                usage_frequency=0
            )
        }
        
        return [default_patterns.get(context.project_type, default_patterns["web_application"])]
    
    def _enhance_context_with_history(
        self, 
        context: ReasoningContext, 
        historical_matches: List[HistoricalDecision],
        domain_patterns: List[DomainPattern]
    ) -> ReasoningContext:
        """Enhance reasoning context with historical insights"""
        
        enhanced_context = ReasoningContext(
            project_type=context.project_type,
            tech_stack=context.tech_stack.copy(),
            team_skills=context.team_skills.copy(),
            constraints=context.constraints.copy(),
            previous_decisions=context.previous_decisions.copy(),
            domain_knowledge=context.domain_knowledge.copy(),
            user_preferences=context.user_preferences.copy()
        )
        
        # Add historical insights
        if historical_matches:
            historical_lessons = []
            successful_patterns = []
            
            for match in historical_matches:
                historical_lessons.extend(match.lessons_learned)
                if match.outcome_success:
                    successful_patterns.append(match.final_decision)
            
            enhanced_context.domain_knowledge["historical_lessons"] = list(set(historical_lessons))
            enhanced_context.domain_knowledge["successful_patterns"] = successful_patterns
        
        # Add domain expertise
        if domain_patterns:
            all_best_practices = []
            all_challenges = []
            
            for pattern in domain_patterns:
                all_best_practices.extend(pattern.best_practices)
                all_challenges.extend(pattern.common_challenges)
            
            enhanced_context.domain_knowledge["best_practices"] = list(set(all_best_practices))
            enhanced_context.domain_knowledge["common_challenges"] = list(set(all_challenges))
        
        return enhanced_context
    
    async def _execute_reasoning_chain(
        self, 
        problem_statement: str, 
        context: ReasoningContext,
        reasoning_type: ReasoningType,
        historical_matches: List[HistoricalDecision]
    ) -> List[Tuple[ReasoningChainStep, str]]:
        """Execute multi-step reasoning chain"""
        
        reasoning_steps = []
        
        # Define reasoning chain based on type
        chain_steps = self._get_reasoning_chain_steps(reasoning_type)
        
        # Execute each step
        for step in chain_steps:
            step_reasoning = await self._execute_reasoning_step(
                step, problem_statement, context, historical_matches, reasoning_steps
            )
            reasoning_steps.append((step, step_reasoning))
            
            # Small delay between steps for better processing
            await asyncio.sleep(0.1)
        
        return reasoning_steps
    
    def _get_reasoning_chain_steps(self, reasoning_type: ReasoningType) -> List[ReasoningChainStep]:
        """Get reasoning chain steps based on reasoning type"""
        
        chain_templates = {
            ReasoningType.DECISION_MAKING: [
                ReasoningChainStep.PROBLEM_ANALYSIS,
                ReasoningChainStep.CONTEXT_GATHERING,
                ReasoningChainStep.OPTION_GENERATION,
                ReasoningChainStep.PROS_CONS_ANALYSIS,
                ReasoningChainStep.RISK_ASSESSMENT,
                ReasoningChainStep.DECISION_SYNTHESIS,
                ReasoningChainStep.CONFIDENCE_EVALUATION
            ],
            ReasoningType.TASK_ANALYSIS: [
                ReasoningChainStep.PROBLEM_ANALYSIS,
                ReasoningChainStep.CONTEXT_GATHERING,
                ReasoningChainStep.RISK_ASSESSMENT,
                ReasoningChainStep.DECISION_SYNTHESIS,
                ReasoningChainStep.CONFIDENCE_EVALUATION
            ],
            ReasoningType.AGENT_SELECTION: [
                ReasoningChainStep.PROBLEM_ANALYSIS,
                ReasoningChainStep.OPTION_GENERATION,
                ReasoningChainStep.PROS_CONS_ANALYSIS,
                ReasoningChainStep.DECISION_SYNTHESIS,
                ReasoningChainStep.CONFIDENCE_EVALUATION
            ]
        }
        
        return chain_templates.get(reasoning_type, chain_templates[ReasoningType.DECISION_MAKING])
    
    async def _execute_reasoning_step(
        self, 
        step: ReasoningChainStep, 
        problem: str, 
        context: ReasoningContext,
        historical_matches: List[HistoricalDecision],
        previous_steps: List[Tuple[ReasoningChainStep, str]]
    ) -> str:
        """Execute individual reasoning step"""
        
        # Build step-specific prompt
        step_prompt = self._build_step_prompt(step, problem, context, historical_matches, previous_steps)
        
        if not self.unified_ai:
            return self._generate_fallback_step_reasoning(step, problem, context)
        
        try:
            # Create AI reasoning request
            request = AIReasoningRequest(
                request_id=f"step_{step.value}_{int(time.time())}",
                reasoning_type=ReasoningType.DECISION_MAKING,
                prompt=step_prompt,
                context=context,
                model_preference=AIModelType.ADVANCED if step in [
                    ReasoningChainStep.PROS_CONS_ANALYSIS,
                    ReasoningChainStep.DECISION_SYNTHESIS
                ] else AIModelType.STANDARD,
                cache_enabled=True
            )
            
            # Execute reasoning
            response = await self.unified_ai.reason(request)
            return response.response_text
            
        except Exception as e:
            logger.warning(f"Step {step.value} failed: {e}, using fallback")
            return self._generate_fallback_step_reasoning(step, problem, context)
    
    def _build_step_prompt(
        self, 
        step: ReasoningChainStep, 
        problem: str, 
        context: ReasoningContext,
        historical_matches: List[HistoricalDecision],
        previous_steps: List[Tuple[ReasoningChainStep, str]]
    ) -> str:
        """Build step-specific reasoning prompt"""
        
        base_context = f"""
Problem: {problem}
Project Type: {context.project_type}
Tech Stack: {', '.join(context.tech_stack)}
"""
        
        # Add historical context if available
        if historical_matches:
            base_context += f"\nHistorical Insights: {len(historical_matches)} similar decisions found with lessons learned"
        
        # Add previous step context
        if previous_steps:
            base_context += f"\nPrevious Analysis:\n"
            for prev_step, prev_reasoning in previous_steps[-2:]:  # Last 2 steps
                base_context += f"- {prev_step.value}: {prev_reasoning[:100]}...\n"
        
        step_prompts = {
            ReasoningChainStep.PROBLEM_ANALYSIS: f"""
{base_context}

Step 1: Problem Analysis
Analyze the core problem thoroughly:
- What is the fundamental challenge?
- What are the key constraints and requirements?
- What are the success criteria?
- What assumptions should be validated?

Provide clear, structured analysis.""",

            ReasoningChainStep.CONTEXT_GATHERING: f"""
{base_context}

Step 2: Context Gathering
Gather and analyze relevant context:
- Technical environment and dependencies
- Team capabilities and constraints
- Business requirements and priorities
- Available resources and timeline
- External factors and dependencies

Synthesize the contextual landscape.""",

            ReasoningChainStep.OPTION_GENERATION: f"""
{base_context}

Step 3: Option Generation
Generate viable solution options:
- Identify 3-5 distinct approaches
- Consider both conventional and innovative solutions
- Account for technical and resource constraints
- Include quick wins and long-term solutions

Present options clearly with brief descriptions.""",

            ReasoningChainStep.PROS_CONS_ANALYSIS: f"""
{base_context}

Step 4: Pros & Cons Analysis
Analyze advantages and disadvantages of each option:
- Technical feasibility and complexity
- Resource requirements and timeline
- Risk factors and mitigation strategies
- Long-term maintainability and scalability
- Impact on team and organization

Provide balanced analysis for each option.""",

            ReasoningChainStep.RISK_ASSESSMENT: f"""
{base_context}

Step 5: Risk Assessment
Identify and assess key risks:
- Technical risks and complexity factors
- Timeline and resource risks
- Integration and dependency risks
- Performance and scalability risks
- Mitigation strategies for each risk

Categorize risks by severity and probability.""",

            ReasoningChainStep.DECISION_SYNTHESIS: f"""
{base_context}

Step 6: Decision Synthesis
Synthesize analysis into clear recommendation:
- Recommended solution with rationale
- Implementation approach and key milestones
- Critical success factors and monitoring points
- Contingency plans for major risks
- Next steps and immediate actions

Provide actionable, clear recommendation.""",

            ReasoningChainStep.CONFIDENCE_EVALUATION: f"""
{base_context}

Step 7: Confidence Evaluation
Evaluate confidence in the recommendation:
- Strength of evidence and analysis
- Quality of available information
- Alignment with historical patterns
- Potential for unforeseen complications
- Overall confidence level (0.0-1.0) with rationale

Provide honest confidence assessment."""
        }
        
        return step_prompts.get(step, f"{base_context}\nAnalyze this step: {step.value}")
    
    def _generate_fallback_step_reasoning(
        self, 
        step: ReasoningChainStep, 
        problem: str, 
        context: ReasoningContext
    ) -> str:
        """Generate fallback reasoning for step when AI unavailable"""
        
        fallback_reasoning = {
            ReasoningChainStep.PROBLEM_ANALYSIS: f"""
Problem Analysis (Fallback):
- Core challenge: {problem[:100]}...
- Project context: {context.project_type} with {len(context.tech_stack)} technologies
- Key constraints: {', '.join(context.constraints[:3])}
- Analysis confidence: Moderate (fallback mode)""",

            ReasoningChainStep.OPTION_GENERATION: f"""
Option Generation (Fallback):
- Option 1: Standard approach using proven technologies
- Option 2: Incremental development with MVP focus
- Option 3: Comprehensive solution with full feature set
- Recommendation: Start with incremental approach for risk mitigation""",

            ReasoningChainStep.DECISION_SYNTHESIS: f"""
Decision Synthesis (Fallback):
- Recommended approach: Systematic implementation using established patterns
- Key factors: Technology alignment, team skills, timeline constraints
- Implementation: Phased approach with iterative validation
- Confidence: Moderate (fallback reasoning applied)"""
        }
        
        return fallback_reasoning.get(step, f"Fallback analysis for {step.value}: systematic approach recommended")
    
    async def _synthesize_reasoning(
        self, 
        reasoning_steps: List[Tuple[ReasoningChainStep, str]], 
        historical_matches: List[HistoricalDecision],
        domain_patterns: List[DomainPattern]
    ) -> Tuple[str, float]:
        """Synthesize final reasoning with confidence calibration"""
        
        # Extract key insights from each step
        synthesis_points = []
        confidence_factors = []
        
        for step, reasoning in reasoning_steps:
            if step == ReasoningChainStep.DECISION_SYNTHESIS:
                synthesis_points.append(f"Decision: {reasoning[:200]}...")
            elif step == ReasoningChainStep.CONFIDENCE_EVALUATION:
                confidence_factors.append(reasoning)
            elif step == ReasoningChainStep.RISK_ASSESSMENT:
                synthesis_points.append(f"Risks: {reasoning[:150]}...")
        
        # Build final reasoning
        final_reasoning = "Multi-Step Analysis Summary:\n\n"
        final_reasoning += "\n".join(synthesis_points)
        
        # Add historical context
        if historical_matches:
            success_rate = sum(1 for match in historical_matches if match.outcome_success) / len(historical_matches)
            final_reasoning += f"\n\nHistorical Context: {len(historical_matches)} similar cases with {success_rate:.1%} success rate"
        
        # Calculate confidence
        base_confidence = 0.7  # Base confidence for structured reasoning
        
        # Adjust based on historical matches
        if historical_matches:
            historical_confidence = sum(match.confidence_level for match in historical_matches) / len(historical_matches)
            base_confidence = (base_confidence + historical_confidence) / 2
        
        # Adjust based on domain patterns
        if domain_patterns:
            pattern_confidence = sum(pattern.confidence for pattern in domain_patterns) / len(domain_patterns)
            base_confidence = (base_confidence + pattern_confidence) / 2
        
        # Cap confidence between 0.3 and 0.95
        final_confidence = max(0.3, min(0.95, base_confidence))
        
        return final_reasoning, final_confidence
    
    def _extract_pattern_insights(
        self, 
        reasoning_steps: List[Tuple[ReasoningChainStep, str]], 
        historical_matches: List[HistoricalDecision]
    ) -> List[str]:
        """Extract insights for future pattern learning"""
        
        insights = []
        
        # Extract decision patterns
        for step, reasoning in reasoning_steps:
            if step == ReasoningChainStep.DECISION_SYNTHESIS:
                if "incremental" in reasoning.lower():
                    insights.append("Incremental development approach preferred")
                if "mvp" in reasoning.lower():
                    insights.append("MVP strategy recommended")
                if "risk" in reasoning.lower():
                    insights.append("Risk-first decision making applied")
        
        # Add historical pattern insights
        if historical_matches:
            successful_approaches = []
            for match in historical_matches:
                if match.outcome_success:
                    successful_approaches.extend(match.lessons_learned)
            
            if successful_approaches:
                insights.append(f"Historical success patterns: {', '.join(set(successful_approaches[:3]))}")
        
        return insights
    
    def _log_reasoning_chain(self, chain: ReasoningChain):
        """Log reasoning chain for future learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                steps_data = [{"step": step.value, "reasoning": reasoning} for step, reasoning in chain.steps]
                historical_ids = [match.decision_id for match in chain.historical_matches]
                
                conn.execute("""
                    INSERT INTO reasoning_chains
                    (chain_id, reasoning_type, steps_data, final_reasoning, confidence,
                     historical_matches, pattern_insights, execution_time, total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chain.chain_id,
                    chain.reasoning_type.value,
                    json.dumps(steps_data),
                    chain.final_reasoning,
                    chain.confidence,
                    json.dumps(historical_ids),
                    json.dumps(chain.pattern_insights),
                    chain.execution_time,
                    chain.total_tokens
                ))
                conn.commit()
        
        except Exception as e:
            logger.warning(f"Chain logging failed: {e}")
    
    def _update_reasoning_stats(self, chain: ReasoningChain):
        """Update reasoning statistics"""
        self.reasoning_stats["total_chains"] += 1
        
        # Update average chain length
        current_avg_length = self.reasoning_stats["avg_chain_length"]
        total_chains = self.reasoning_stats["total_chains"]
        chain_length = len(chain.steps)
        
        self.reasoning_stats["avg_chain_length"] = (
            (current_avg_length * (total_chains - 1) + chain_length) / total_chains
        )
        
        # Update average confidence
        current_avg_confidence = self.reasoning_stats["avg_confidence"]
        self.reasoning_stats["avg_confidence"] = (
            (current_avg_confidence * (total_chains - 1) + chain.confidence) / total_chains
        )
        
        # Update historical matches count
        if chain.historical_matches:
            self.reasoning_stats["historical_matches"] += len(chain.historical_matches)
        
        # Update pattern utilization
        if chain.pattern_insights:
            self.reasoning_stats["pattern_utilization"] += len(chain.pattern_insights)
    
    def _generate_context_hash(self, context: ReasoningContext) -> str:
        """Generate hash for context matching"""
        context_data = {
            "project_type": context.project_type,
            "tech_stack": sorted(context.tech_stack),
            "constraints": sorted(context.constraints)
        }
        context_string = json.dumps(context_data, sort_keys=True)
        return hashlib.md5(context_string.encode()).hexdigest()
    
    def _load_historical_data(self):
        """Load historical data from database"""
        # This would load from database in production
        # For now, we'll initialize with empty data
        pass
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            **self.reasoning_stats,
            "unified_ai_available": self.unified_ai is not None,
            "historical_decisions_loaded": len(self.historical_decisions),
            "domain_patterns_loaded": len(self.domain_patterns)
        }

# Demo and testing functions
async def demo_context_aware_reasoning():
    """Demo the context-aware reasoning engine"""
    print("🧠 Agent Zero V2.0 - Context-Aware Reasoning Engine Demo")
    print("=" * 60)
    
    # Initialize reasoning engine
    reasoning_engine = ContextAwareReasoningEngine()
    
    # Demo reasoning context
    context = ReasoningContext(
        project_type="web_application",
        tech_stack=["FastAPI", "React", "PostgreSQL", "Docker"],
        team_skills=["Python", "JavaScript", "SQL"],
        constraints=["2-week timeline", "3-person team", "high security requirements"],
        domain_knowledge={"industry": "fintech", "compliance": ["GDPR", "SOX"]}
    )
    
    # Demo problem
    problem = """
    Design an authentication system for a financial application that requires:
    - Multi-factor authentication
    - Session management
    - Audit logging
    - Integration with existing user database
    - High security standards for financial data
    """
    
    print(f"📋 Problem: Authentication system design for fintech application")
    print(f"🔧 Context: {context.project_type} with {len(context.tech_stack)} technologies")
    print(f"⏰ Constraints: {len(context.constraints)} constraints identified")
    
    # Execute context-aware reasoning
    print(f"\n🧠 Executing multi-step reasoning chain...")
    
    reasoning_chain = await reasoning_engine.reason_with_context(
        problem_statement=problem,
        context=context,
        reasoning_type=ReasoningType.DECISION_MAKING
    )
    
    # Display results
    print(f"\n✅ Reasoning completed!")
    print(f"   Chain ID: {reasoning_chain.chain_id}")
    print(f"   Steps executed: {len(reasoning_chain.steps)}")
    print(f"   Confidence: {reasoning_chain.confidence:.2f}")
    print(f"   Execution time: {reasoning_chain.execution_time:.2f}s")
    print(f"   Historical matches: {len(reasoning_chain.historical_matches)}")
    print(f"   Pattern insights: {len(reasoning_chain.pattern_insights)}")
    
    print(f"\n📋 Reasoning Steps:")
    for i, (step, reasoning) in enumerate(reasoning_chain.steps, 1):
        print(f"   {i}. {step.value.replace('_', ' ').title()}")
        print(f"      {reasoning[:80]}...")
    
    print(f"\n🎯 Final Reasoning:")
    print(f"   {reasoning_chain.final_reasoning[:200]}...")
    
    if reasoning_chain.pattern_insights:
        print(f"\n💡 Pattern Insights:")
        for insight in reasoning_chain.pattern_insights:
            print(f"   - {insight}")
    
    # Show reasoning statistics
    print(f"\n📊 Reasoning Engine Statistics:")
    stats = reasoning_engine.get_reasoning_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Context-aware reasoning demo completed!")

if __name__ == "__main__":
    print("🧠 Agent Zero V2.0 Phase 4 - Context-Aware Reasoning Engine")
    print("Testing advanced reasoning capabilities...")
    
    # Run demo
    asyncio.run(demo_context_aware_reasoning())