#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Intelligent Agent Selection System
Production-ready AI-powered agent selection with Ollama integration

Priority 1.1: Core AI Agent Selector (2 SP)
- Real Ollama-powered agent selection reasoning  
- Task complexity → Agent capability matching
- Historical performance learning
- Dynamic load balancing
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3

# Import existing production components
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.production_ai_system import ProductionAISystem
except ImportError:
    # Fallback for testing
    class ProductionAISystem:
        def __init__(self):
            self.available_models = {"standard": "llama3.2:3b"}
        
        def generate_ai_reasoning(self, prompt, model_type="standard"):
            return {"success": True, "reasoning": "Mock reasoning for testing"}

logger = logging.getLogger(__name__)

class AgentCapability(Enum):
    """Agent capability categories"""
    CODE_DEVELOPMENT = "code_development"
    DATA_ANALYSIS = "data_analysis" 
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    ARCHITECTURE = "architecture"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    SECURITY = "security"
    DOCUMENTATION = "documentation"

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"        # 1-4 hours
    MODERATE = "moderate"    # 4-12 hours  
    COMPLEX = "complex"      # 12-40 hours
    EXPERT = "expert"        # 40+ hours

@dataclass
class AgentProfile:
    """Enhanced agent profile with capabilities and metrics"""
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    skill_level: float  # 0.0-1.0
    performance_score: float  # 0.0-1.0 based on historical data
    current_workload: int  # number of active tasks
    max_workload: int  # maximum concurrent tasks
    specializations: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.85  # historical success rate
    avg_completion_time: float = 8.0  # hours
    cost_per_hour: float = 100.0  # USD
    availability: bool = True

    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return (self.availability and 
                self.current_workload < self.max_workload and
                (datetime.now() - self.last_active).seconds < 3600)  # Active in last hour
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (performance vs cost)"""
        return (self.performance_score * self.success_rate) / (self.cost_per_hour / 100.0)

@dataclass
class TaskAnalysis:
    """AI-powered task analysis results"""
    task_id: str
    complexity: TaskComplexity
    required_capabilities: List[AgentCapability]
    estimated_hours: float
    priority: str
    risk_factors: List[str]
    confidence: float
    reasoning: str
    
@dataclass
class AgentSelection:
    """Agent selection result"""
    selected_agent: AgentProfile
    confidence: float
    reasoning: str
    alternative_agents: List[AgentProfile]
    estimated_completion_time: float
    estimated_cost: float
    risk_assessment: Dict[str, Any]

class IntelligentAgentSelector:
    """
    Production AI-Powered Agent Selection System
    
    Uses Ollama LLM for intelligent task analysis and agent matching
    Integrates with historical performance data and real-time metrics
    """
    
    def __init__(self, db_path: str = "agent_zero.db", ai_model: str = "standard"):
        self.db_path = db_path
        self.ai_system = ProductionAISystem()
        self.ai_model = ai_model
        self.agents: Dict[str, AgentProfile] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        self._init_database()
        self._load_default_agents()
        logger.info("✅ IntelligentAgentSelector initialized with AI reasoning")
    
    def _init_database(self):
        """Initialize agent selection database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Agent profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    agent_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    capabilities TEXT NOT NULL,  -- JSON array
                    skill_level REAL NOT NULL,
                    performance_score REAL NOT NULL,
                    current_workload INTEGER DEFAULT 0,
                    max_workload INTEGER DEFAULT 3,
                    specializations TEXT,  -- JSON array
                    success_rate REAL DEFAULT 0.85,
                    avg_completion_time REAL DEFAULT 8.0,
                    cost_per_hour REAL DEFAULT 100.0,
                    availability BOOLEAN DEFAULT TRUE,
                    last_active TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Agent performance history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_type TEXT,
                    complexity TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    success BOOLEAN,
                    duration_hours REAL,
                    quality_score REAL,
                    cost REAL,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (agent_id) REFERENCES agent_profiles (agent_id)
                )
            """)
            
            # Selection decisions log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS selection_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    selected_agent_id TEXT NOT NULL,
                    confidence REAL,
                    reasoning TEXT,
                    alternative_agents TEXT,  -- JSON array
                    ai_model_used TEXT,
                    selection_time REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _load_default_agents(self):
        """Load default agent profiles for development"""
        default_agents = [
            AgentProfile(
                agent_id="backend_specialist_01",
                agent_type="backend_developer",
                capabilities=[AgentCapability.BACKEND, AgentCapability.DATABASE, AgentCapability.CODE_DEVELOPMENT],
                skill_level=0.9,
                performance_score=0.88,
                max_workload=2,
                specializations=["FastAPI", "PostgreSQL", "Python", "Redis"],
                success_rate=0.92,
                avg_completion_time=6.5,
                cost_per_hour=120.0
            ),
            AgentProfile(
                agent_id="frontend_expert_01", 
                agent_type="frontend_developer",
                capabilities=[AgentCapability.FRONTEND, AgentCapability.CODE_DEVELOPMENT],
                skill_level=0.85,
                performance_score=0.82,
                max_workload=3,
                specializations=["React", "TypeScript", "CSS", "JavaScript"],
                success_rate=0.88,
                avg_completion_time=7.2,
                cost_per_hour=110.0
            ),
            AgentProfile(
                agent_id="fullstack_generalist_01",
                agent_type="fullstack_developer", 
                capabilities=[AgentCapability.FRONTEND, AgentCapability.BACKEND, AgentCapability.CODE_DEVELOPMENT],
                skill_level=0.75,
                performance_score=0.78,
                max_workload=4,
                specializations=["React", "FastAPI", "SQL", "Docker"],
                success_rate=0.83,
                avg_completion_time=9.1,
                cost_per_hour=95.0
            ),
            AgentProfile(
                agent_id="devops_engineer_01",
                agent_type="devops_engineer",
                capabilities=[AgentCapability.DEPLOYMENT, AgentCapability.ARCHITECTURE, AgentCapability.SECURITY],
                skill_level=0.92,
                performance_score=0.90,
                max_workload=2,
                specializations=["Docker", "Kubernetes", "CI/CD", "AWS", "Neo4j"],
                success_rate=0.94,
                avg_completion_time=5.8,
                cost_per_hour=140.0
            ),
            AgentProfile(
                agent_id="testing_specialist_01",
                agent_type="qa_engineer",
                capabilities=[AgentCapability.TESTING, AgentCapability.CODE_DEVELOPMENT],
                skill_level=0.88,
                performance_score=0.86,
                max_workload=3,
                specializations=["pytest", "integration testing", "automation"],
                success_rate=0.91,
                avg_completion_time=4.5,
                cost_per_hour=105.0
            )
        ]
        
        for agent in default_agents:
            self.agents[agent.agent_id] = agent
            self._save_agent_profile(agent)
    
    async def analyze_task_with_ai(self, task_description: str, context: Dict[str, Any] = None) -> TaskAnalysis:
        """
        Use AI to analyze task requirements and complexity
        """
        context = context or {}
        
        prompt = f"""Analyze this development task and provide detailed assessment:

TASK: {task_description}

CONTEXT:
- Project Type: {context.get('project_type', 'web_application')}
- Tech Stack: {context.get('tech_stack', [])}
- Timeline: {context.get('timeline', 'flexible')}
- Budget: {context.get('budget', 'standard')}

Analyze and respond with JSON:
{{
    "complexity": "SIMPLE|MODERATE|COMPLEX|EXPERT",
    "estimated_hours": 8.0,
    "required_capabilities": ["CODE_DEVELOPMENT", "BACKEND"],
    "priority": "HIGH|MEDIUM|LOW",
    "risk_factors": ["technical_complexity", "integration_challenges"],
    "confidence": 0.85,
    "reasoning": "Detailed explanation of the analysis"
}}

Consider:
1. Technical complexity and skill requirements
2. Integration points and dependencies  
3. Testing and deployment needs
4. Potential risks and challenges
5. Estimated effort and timeline"""

        try:
            ai_response = self.ai_system.generate_ai_reasoning(prompt, self.ai_model)
            
            if ai_response["success"]:
                # Parse AI response
                analysis_data = json.loads(ai_response["reasoning"])
                
                # Map capabilities from strings to enums
                capabilities = []
                for cap_str in analysis_data.get("required_capabilities", ["CODE_DEVELOPMENT"]):
                    try:
                        capabilities.append(AgentCapability(cap_str.lower()))
                    except ValueError:
                        logger.warning(f"Unknown capability: {cap_str}")
                        capabilities.append(AgentCapability.CODE_DEVELOPMENT)
                
                # Map complexity
                complexity_str = analysis_data.get("complexity", "MODERATE")
                try:
                    complexity = TaskComplexity(complexity_str.lower())
                except ValueError:
                    complexity = TaskComplexity.MODERATE
                
                return TaskAnalysis(
                    task_id=context.get("task_id", f"task_{int(time.time())}"),
                    complexity=complexity,
                    required_capabilities=capabilities,
                    estimated_hours=analysis_data.get("estimated_hours", 8.0),
                    priority=analysis_data.get("priority", "MEDIUM"),
                    risk_factors=analysis_data.get("risk_factors", []),
                    confidence=analysis_data.get("confidence", 0.7),
                    reasoning=analysis_data.get("reasoning", "AI analysis completed")
                )
            else:
                # Fallback analysis
                return self._fallback_task_analysis(task_description, context)
                
        except Exception as e:
            logger.warning(f"AI task analysis failed: {e}, using fallback")
            return self._fallback_task_analysis(task_description, context)
    
    def _fallback_task_analysis(self, task_description: str, context: Dict) -> TaskAnalysis:
        """Fallback task analysis without AI"""
        desc_lower = task_description.lower()
        
        # Simple rule-based analysis
        if any(word in desc_lower for word in ["complex", "advanced", "machine learning", "ai"]):
            complexity = TaskComplexity.EXPERT
            hours = 32.0
        elif any(word in desc_lower for word in ["integration", "system", "architecture"]):
            complexity = TaskComplexity.COMPLEX
            hours = 16.0
        elif any(word in desc_lower for word in ["api", "database", "backend"]):
            complexity = TaskComplexity.MODERATE
            hours = 8.0
        else:
            complexity = TaskComplexity.SIMPLE
            hours = 4.0
        
        # Determine capabilities
        capabilities = [AgentCapability.CODE_DEVELOPMENT]
        if "frontend" in desc_lower or "ui" in desc_lower or "react" in desc_lower:
            capabilities.append(AgentCapability.FRONTEND)
        if "backend" in desc_lower or "api" in desc_lower or "server" in desc_lower:
            capabilities.append(AgentCapability.BACKEND)
        if "test" in desc_lower:
            capabilities.append(AgentCapability.TESTING)
        if "deploy" in desc_lower or "docker" in desc_lower:
            capabilities.append(AgentCapability.DEPLOYMENT)
        
        return TaskAnalysis(
            task_id=context.get("task_id", f"task_{int(time.time())}"),
            complexity=complexity,
            required_capabilities=capabilities,
            estimated_hours=hours,
            priority="MEDIUM",
            risk_factors=[],
            confidence=0.6,
            reasoning="Rule-based fallback analysis"
        )
    
    async def select_optimal_agent(self, task_analysis: TaskAnalysis, context: Dict[str, Any] = None) -> AgentSelection:
        """
        Select the optimal agent using AI-powered decision making
        """
        start_time = time.time()
        context = context or {}
        
        # Filter available agents with required capabilities
        candidate_agents = self._filter_candidate_agents(task_analysis)
        
        if not candidate_agents:
            raise ValueError(f"No available agents found with required capabilities: {[c.value for c in task_analysis.required_capabilities]}")
        
        # Use AI to make final selection
        selected_agent, reasoning = await self._ai_agent_selection(task_analysis, candidate_agents, context)
        
        # Calculate estimates
        completion_time = self._estimate_completion_time(selected_agent, task_analysis)
        cost = self._estimate_cost(selected_agent, completion_time)
        risk_assessment = self._assess_selection_risk(selected_agent, task_analysis)
        
        # Get alternatives
        alternatives = [agent for agent in candidate_agents if agent.agent_id != selected_agent.agent_id][:3]
        
        selection = AgentSelection(
            selected_agent=selected_agent,
            confidence=task_analysis.confidence * 0.9,  # Slight confidence reduction for selection
            reasoning=reasoning,
            alternative_agents=alternatives,
            estimated_completion_time=completion_time,
            estimated_cost=cost,
            risk_assessment=risk_assessment
        )
        
        # Log decision
        self._log_selection_decision(selection, task_analysis, time.time() - start_time)
        
        logger.info(f"✅ Selected agent {selected_agent.agent_id} for task {task_analysis.task_id} (confidence: {selection.confidence:.2f})")
        
        return selection
    
    def _filter_candidate_agents(self, task_analysis: TaskAnalysis) -> List[AgentProfile]:
        """Filter agents based on capabilities and availability"""
        candidates = []
        
        for agent in self.agents.values():
            # Check availability
            if not agent.is_available:
                continue
                
            # Check capability match
            agent_caps = set(agent.capabilities)
            required_caps = set(task_analysis.required_capabilities)
            
            if not required_caps.intersection(agent_caps):
                continue  # No capability overlap
            
            # Check complexity vs skill level
            complexity_requirement = {
                TaskComplexity.SIMPLE: 0.3,
                TaskComplexity.MODERATE: 0.5, 
                TaskComplexity.COMPLEX: 0.7,
                TaskComplexity.EXPERT: 0.85
            }
            
            if agent.skill_level < complexity_requirement.get(task_analysis.complexity, 0.5):
                continue  # Insufficient skill level
            
            candidates.append(agent)
        
        # Sort by efficiency score
        candidates.sort(key=lambda a: a.efficiency_score, reverse=True)
        
        return candidates
    
    async def _ai_agent_selection(self, task_analysis: TaskAnalysis, candidates: List[AgentProfile], context: Dict) -> Tuple[AgentProfile, str]:
        """Use AI to select the best agent from candidates"""
        
        # Prepare candidate data for AI
        candidate_data = []
        for agent in candidates:
            candidate_data.append({
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "capabilities": [c.value for c in agent.capabilities],
                "skill_level": agent.skill_level,
                "performance_score": agent.performance_score,
                "success_rate": agent.success_rate,
                "current_workload": agent.current_workload,
                "avg_completion_time": agent.avg_completion_time,
                "cost_per_hour": agent.cost_per_hour,
                "specializations": agent.specializations,
                "efficiency_score": agent.efficiency_score
            })
        
        prompt = f"""Select the optimal agent for this task:

TASK ANALYSIS:
- Task ID: {task_analysis.task_id}
- Complexity: {task_analysis.complexity.value}
- Required Capabilities: {[c.value for c in task_analysis.required_capabilities]}
- Estimated Hours: {task_analysis.estimated_hours}
- Priority: {task_analysis.priority}
- Risk Factors: {task_analysis.risk_factors}

CANDIDATE AGENTS:
{json.dumps(candidate_data, indent=2)}

SELECTION CRITERIA:
- Capability match (highest priority)
- Performance score and success rate
- Current workload and availability
- Cost efficiency
- Specializations alignment
- Risk mitigation

Select the best agent and respond with JSON:
{{
    "selected_agent_id": "backend_specialist_01",
    "confidence": 0.88,
    "reasoning": "Detailed explanation of why this agent is optimal",
    "key_factors": ["high_performance", "capability_match", "availability"]
}}"""

        try:
            ai_response = self.ai_system.generate_ai_reasoning(prompt, "advanced")  # Use advanced model for selection
            
            if ai_response["success"]:
                selection_data = json.loads(ai_response["reasoning"])
                selected_id = selection_data.get("selected_agent_id")
                
                # Find selected agent
                selected_agent = next((a for a in candidates if a.agent_id == selected_id), None)
                if selected_agent:
                    return selected_agent, selection_data.get("reasoning", "AI selection completed")
        
        except Exception as e:
            logger.warning(f"AI agent selection failed: {e}, using fallback")
        
        # Fallback: select highest efficiency score
        return candidates[0], "Fallback selection: highest efficiency score agent"
    
    def _estimate_completion_time(self, agent: AgentProfile, task_analysis: TaskAnalysis) -> float:
        """Estimate task completion time for agent"""
        base_time = task_analysis.estimated_hours
        
        # Adjust for agent performance
        performance_multiplier = 1.0 / max(agent.performance_score, 0.1)
        
        # Adjust for workload
        workload_multiplier = 1.0 + (agent.current_workload * 0.2)
        
        # Adjust for complexity vs skill match
        complexity_scores = {
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MODERATE: 0.5,
            TaskComplexity.COMPLEX: 0.7, 
            TaskComplexity.EXPERT: 0.9
        }
        
        skill_ratio = agent.skill_level / complexity_scores.get(task_analysis.complexity, 0.5)
        skill_multiplier = 1.0 / max(skill_ratio, 0.5)  # Better skills = faster completion
        
        estimated_time = base_time * performance_multiplier * workload_multiplier * skill_multiplier
        
        return round(estimated_time, 1)
    
    def _estimate_cost(self, agent: AgentProfile, completion_time: float) -> float:
        """Estimate task cost"""
        return round(agent.cost_per_hour * completion_time, 2)
    
    def _assess_selection_risk(self, agent: AgentProfile, task_analysis: TaskAnalysis) -> Dict[str, Any]:
        """Assess risks of agent selection"""
        risks = {
            "overall_risk": "LOW",
            "risk_factors": [],
            "risk_score": 0.0,
            "mitigation_suggestions": []
        }
        
        risk_score = 0.0
        
        # Performance risk
        if agent.performance_score < 0.7:
            risks["risk_factors"].append("low_performance_score")
            risk_score += 0.3
        
        # Workload risk
        if agent.current_workload >= agent.max_workload * 0.8:
            risks["risk_factors"].append("high_workload")  
            risk_score += 0.2
        
        # Capability mismatch risk
        required_caps = set(task_analysis.required_capabilities)
        agent_caps = set(agent.capabilities)
        if not required_caps.issubset(agent_caps):
            risks["risk_factors"].append("capability_gap")
            risk_score += 0.4
        
        # Complexity vs skill risk
        complexity_requirements = {
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MODERATE: 0.5,
            TaskComplexity.COMPLEX: 0.7,
            TaskComplexity.EXPERT: 0.85
        }
        
        required_skill = complexity_requirements.get(task_analysis.complexity, 0.5)
        if agent.skill_level < required_skill:
            risks["risk_factors"].append("insufficient_skill_level")
            risk_score += 0.5
        
        # Set overall risk level
        if risk_score < 0.3:
            risks["overall_risk"] = "LOW"
        elif risk_score < 0.6:
            risks["overall_risk"] = "MEDIUM"
        else:
            risks["overall_risk"] = "HIGH"
        
        risks["risk_score"] = round(risk_score, 2)
        
        # Add mitigation suggestions
        if "high_workload" in risks["risk_factors"]:
            risks["mitigation_suggestions"].append("Consider load balancing or timeline adjustment")
        if "capability_gap" in risks["risk_factors"]:
            risks["mitigation_suggestions"].append("Provide additional training or pair with specialist")
        if "insufficient_skill_level" in risks["risk_factors"]:
            risks["mitigation_suggestions"].append("Add senior review checkpoints")
        
        return risks
    
    def _save_agent_profile(self, agent: AgentProfile):
        """Save agent profile to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_profiles
                (agent_id, agent_type, capabilities, skill_level, performance_score,
                 current_workload, max_workload, specializations, success_rate,
                 avg_completion_time, cost_per_hour, availability, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.agent_id,
                agent.agent_type,
                json.dumps([c.value for c in agent.capabilities]),
                agent.skill_level,
                agent.performance_score,
                agent.current_workload,
                agent.max_workload,
                json.dumps(agent.specializations),
                agent.success_rate,
                agent.avg_completion_time,
                agent.cost_per_hour,
                agent.availability,
                agent.last_active.isoformat()
            ))
            conn.commit()
    
    def _log_selection_decision(self, selection: AgentSelection, task_analysis: TaskAnalysis, selection_time: float):
        """Log selection decision for learning"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO selection_decisions
                (task_id, selected_agent_id, confidence, reasoning, alternative_agents,
                 ai_model_used, selection_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_analysis.task_id,
                selection.selected_agent.agent_id,
                selection.confidence,
                selection.reasoning,
                json.dumps([a.agent_id for a in selection.alternative_agents]),
                self.ai_model,
                selection_time
            ))
            conn.commit()
    
    # Public API methods
    
    def register_agent(self, agent_profile: AgentProfile):
        """Register new agent in the system"""
        self.agents[agent_profile.agent_id] = agent_profile
        self._save_agent_profile(agent_profile)
        logger.info(f"✅ Registered agent: {agent_profile.agent_id}")
    
    def update_agent_workload(self, agent_id: str, workload_delta: int):
        """Update agent's current workload"""
        if agent_id in self.agents:
            self.agents[agent_id].current_workload += workload_delta
            self.agents[agent_id].current_workload = max(0, self.agents[agent_id].current_workload)
            self._save_agent_profile(self.agents[agent_id])
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        available_agents = sum(1 for agent in self.agents.values() if agent.is_available)
        total_workload = sum(agent.current_workload for agent in self.agents.values())
        avg_performance = sum(agent.performance_score for agent in self.agents.values()) / len(self.agents)
        
        return {
            "total_agents": len(self.agents),
            "available_agents": available_agents,
            "total_active_tasks": total_workload,
            "avg_performance_score": round(avg_performance, 2),
            "system_capacity": sum(agent.max_workload for agent in self.agents.values())
        }

# Demo and testing functions
async def demo_intelligent_agent_selection():
    """Demo the intelligent agent selection system"""
    print("🤖 Agent Zero V2.0 - Intelligent Agent Selection Demo")
    print("=" * 60)
    
    # Initialize selector
    selector = IntelligentAgentSelector()
    
    # Demo tasks
    demo_tasks = [
        {
            "description": "Create a FastAPI backend with PostgreSQL database for user authentication",
            "context": {"project_type": "web_api", "tech_stack": ["FastAPI", "PostgreSQL"], "timeline": "1 week"}
        },
        {
            "description": "Build a React dashboard with real-time data visualization", 
            "context": {"project_type": "frontend", "tech_stack": ["React", "TypeScript"], "timeline": "2 weeks"}
        },
        {
            "description": "Set up CI/CD pipeline with Docker and Kubernetes deployment",
            "context": {"project_type": "devops", "tech_stack": ["Docker", "Kubernetes"], "timeline": "3 days"}
        }
    ]
    
    for i, task_data in enumerate(demo_tasks, 1):
        print(f"\n📋 Demo Task {i}: {task_data['description'][:50]}...")
        
        # Analyze task
        print("🧠 Analyzing task with AI...")
        task_analysis = await selector.analyze_task_with_ai(task_data["description"], task_data["context"])
        
        print(f"   Complexity: {task_analysis.complexity.value}")
        print(f"   Required Capabilities: {[c.value for c in task_analysis.required_capabilities]}")
        print(f"   Estimated Hours: {task_analysis.estimated_hours}")
        print(f"   Confidence: {task_analysis.confidence:.2f}")
        
        # Select agent
        print("🎯 Selecting optimal agent...")
        selection = await selector.select_optimal_agent(task_analysis, task_data["context"])
        
        print(f"   Selected: {selection.selected_agent.agent_id}")
        print(f"   Agent Type: {selection.selected_agent.agent_type}")
        print(f"   Confidence: {selection.confidence:.2f}")
        print(f"   Estimated Completion: {selection.estimated_completion_time} hours")
        print(f"   Estimated Cost: ${selection.estimated_cost}")
        print(f"   Risk Level: {selection.risk_assessment['overall_risk']}")
        print(f"   Reasoning: {selection.reasoning[:100]}...")
    
    # Show system stats
    print(f"\n📊 System Statistics:")
    stats = selector.get_agent_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    print("🚀 Agent Zero V2.0 Phase 4 - Intelligent Agent Selection")
    print("Testing core functionality...")
    
    # Run demo
    asyncio.run(demo_intelligent_agent_selection())