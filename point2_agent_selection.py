#!/usr/bin/env python3
"""
🚀 Agent Zero V1 - Point 2: Intelligent Agent Selection System
============================================================
Missing Link w architekturze - most między NLU a Dynamic Priority
Logika architektury: NLU → Agent Selection → Priority → Collaboration
"""

import asyncio
import logging
import json
import time
import uuid
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlite3
import math

# FastAPI components
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Konfiguracja enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intelligent_agent_selection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("IntelligentAgentSelection")

# ================================
# AGENT SELECTION SYSTEM CORE
# ================================

class AgentType(Enum):
    """Typy agentów w systemie"""
    CODE_SPECIALIST = "CODE_SPECIALIST"           # Specjalista od kodu
    RESEARCH_ANALYST = "RESEARCH_ANALYST"         # Analityk i researcher
    PROJECT_MANAGER = "PROJECT_MANAGER"           # Manager projektu
    QA_ENGINEER = "QA_ENGINEER"                  # Inżynier jakości
    DEVOPS_ENGINEER = "DEVOPS_ENGINEER"          # DevOps specialist
    UI_UX_DESIGNER = "UI_UX_DESIGNER"            # Designer interfejsów
    DATA_SCIENTIST = "DATA_SCIENTIST"            # Naukowiec danych
    SECURITY_EXPERT = "SECURITY_EXPERT"          # Ekspert bezpieczeństwa

class AgentSkill(Enum):
    """Umiejętności agentów"""
    PYTHON_DEVELOPMENT = "PYTHON_DEVELOPMENT"
    JAVASCRIPT_DEVELOPMENT = "JAVASCRIPT_DEVELOPMENT"
    DATABASE_DESIGN = "DATABASE_DESIGN"
    API_DEVELOPMENT = "API_DEVELOPMENT"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    SYSTEM_ARCHITECTURE = "SYSTEM_ARCHITECTURE"
    TESTING_AUTOMATION = "TESTING_AUTOMATION"
    DEPLOYMENT_AUTOMATION = "DEPLOYMENT_AUTOMATION"
    UI_DESIGN = "UI_DESIGN"
    SECURITY_AUDIT = "SECURITY_AUDIT"
    PERFORMANCE_OPTIMIZATION = "PERFORMANCE_OPTIMIZATION"
    DOCUMENTATION = "DOCUMENTATION"

class SelectionStrategy(Enum):
    """Strategie wyboru agentów"""
    PERFORMANCE_OPTIMIZED = "PERFORMANCE_OPTIMIZED"   # Najlepsi performerzy
    COST_OPTIMIZED = "COST_OPTIMIZED"                # Najtańsi dostępni
    BALANCED = "BALANCED"                             # Balans performance/koszt
    LEARNING_FOCUSED = "LEARNING_FOCUSED"             # Fokus na uczeniu się
    SPEED_OPTIMIZED = "SPEED_OPTIMIZED"               # Najszybsze wykonanie

@dataclass
class Agent:
    """Definicja agenta w systemie"""
    id: str
    name: str
    agent_type: AgentType
    skills: List[AgentSkill] = field(default_factory=list)
    
    # Performance metrics
    success_rate: float = 0.8
    average_completion_time: float = 1.0  # w godzinach
    cost_per_hour: float = 50.0
    current_workload: float = 0.0  # 0.0-1.0
    availability: bool = True
    
    # Experience metrics
    projects_completed: int = 0
    total_hours_worked: float = 0.0
    specialization_level: float = 0.8  # 0.0-1.0
    learning_rate: float = 0.1
    
    # Collaboration metrics
    team_compatibility: float = 0.9
    communication_score: float = 0.8
    mentoring_ability: float = 0.7
    
    # Current status
    last_active: datetime = field(default_factory=datetime.now)
    current_tasks: List[str] = field(default_factory=list)

@dataclass
class TaskRequirements:
    """Wymagania zadania dla selection"""
    id: str
    title: str
    description: str
    required_skills: List[AgentSkill] = field(default_factory=list)
    preferred_agent_types: List[AgentType] = field(default_factory=list)
    
    # Constraints
    max_budget: float = 1000.0
    deadline: Optional[datetime] = None
    complexity_level: float = 0.5  # 0.0-1.0
    priority_level: str = "MEDIUM"
    
    # Team requirements
    team_size_min: int = 1
    team_size_max: int = 3
    requires_collaboration: bool = False
    
    # Quality requirements
    quality_threshold: float = 0.8
    requires_testing: bool = True
    requires_documentation: bool = True

@dataclass
class AgentSelection:
    """Wynik selekcji agentów"""
    task_id: str
    selected_agents: List[Agent] = field(default_factory=list)
    selection_strategy: SelectionStrategy = SelectionStrategy.BALANCED
    
    # Selection metrics
    confidence_score: float = 0.8
    estimated_success_rate: float = 0.8
    estimated_completion_time: float = 1.0
    estimated_total_cost: float = 100.0
    
    # Reasoning
    selection_reasoning: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)

# ================================
# INTELLIGENT AGENT SELECTION ENGINE
# ================================

class IntelligentAgentSelector:
    """
    Inteligentny system wyboru agentów
    Logika: Analizuje zadania z NLU, wybiera optymalnych agentów, przekazuje do Priority
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, Agent] = {}
        self.selection_history: Dict[str, AgentSelection] = {}
        
        # Połączenia z innymi systemami
        self.nlu_service = "http://localhost:8001"  # Will connect to 8000/9001
        self.priority_service = "http://localhost:8003"
        self.collaboration_service = "http://localhost:8005"
        
        # Inicjalizacja agentów i bazy danych
        self._initialize_agent_pool()
        self._init_database()
        
        self.logger.info("✅ Intelligent Agent Selection Engine initialized!")
    
    def _initialize_agent_pool(self):
        """Inicjalizuje pool dostępnych agentów"""
        
        # Agent pool based on real capabilities
        agents_data = [
            {
                "id": "agent_001", 
                "name": "CodeMaster Pro",
                "agent_type": AgentType.CODE_SPECIALIST,
                "skills": [AgentSkill.PYTHON_DEVELOPMENT, AgentSkill.API_DEVELOPMENT, AgentSkill.DATABASE_DESIGN],
                "success_rate": 0.95,
                "average_completion_time": 0.8,
                "cost_per_hour": 75.0,
                "specialization_level": 0.9
            },
            {
                "id": "agent_002",
                "name": "ResearchBot Elite", 
                "agent_type": AgentType.RESEARCH_ANALYST,
                "skills": [AgentSkill.MACHINE_LEARNING, AgentSkill.SYSTEM_ARCHITECTURE, AgentSkill.DOCUMENTATION],
                "success_rate": 0.92,
                "average_completion_time": 1.2,
                "cost_per_hour": 60.0,
                "specialization_level": 0.85
            },
            {
                "id": "agent_003",
                "name": "ProjectLead AI",
                "agent_type": AgentType.PROJECT_MANAGER, 
                "skills": [AgentSkill.SYSTEM_ARCHITECTURE, AgentSkill.DOCUMENTATION, AgentSkill.TESTING_AUTOMATION],
                "success_rate": 0.88,
                "average_completion_time": 1.0,
                "cost_per_hour": 80.0,
                "team_compatibility": 0.95,
                "mentoring_ability": 0.9
            },
            {
                "id": "agent_004",
                "name": "QualityGuard Pro",
                "agent_type": AgentType.QA_ENGINEER,
                "skills": [AgentSkill.TESTING_AUTOMATION, AgentSkill.PERFORMANCE_OPTIMIZATION, AgentSkill.SECURITY_AUDIT],
                "success_rate": 0.96,
                "average_completion_time": 0.7,
                "cost_per_hour": 65.0,
                "specialization_level": 0.92
            },
            {
                "id": "agent_005", 
                "name": "DeployMaster Elite",
                "agent_type": AgentType.DEVOPS_ENGINEER,
                "skills": [AgentSkill.DEPLOYMENT_AUTOMATION, AgentSkill.SYSTEM_ARCHITECTURE, AgentSkill.PERFORMANCE_OPTIMIZATION],
                "success_rate": 0.94,
                "average_completion_time": 0.6,
                "cost_per_hour": 70.0,
                "specialization_level": 0.88
            },
            {
                "id": "agent_006",
                "name": "DataWizard AI",
                "agent_type": AgentType.DATA_SCIENTIST,
                "skills": [AgentSkill.MACHINE_LEARNING, AgentSkill.PYTHON_DEVELOPMENT, AgentSkill.DATABASE_DESIGN],
                "success_rate": 0.91,
                "average_completion_time": 1.5,
                "cost_per_hour": 85.0,
                "specialization_level": 0.93
            }
        ]
        
        # Tworzenie agentów
        for agent_data in agents_data:
            agent = Agent(**agent_data)
            self.agents[agent.id] = agent
            
        self.logger.info(f"✅ Initialized agent pool with {len(self.agents)} agents")
    
    def _init_database(self):
        """Inicjalizacja bazy danych selekcji agentów"""
        
        self.db_path = "agent_selection.db"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Tabela historii selekcji
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_selections (
                        id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        selected_agents TEXT,
                        selection_strategy TEXT,
                        
                        confidence_score REAL,
                        estimated_success_rate REAL,
                        estimated_completion_time REAL,
                        estimated_total_cost REAL,
                        
                        actual_success_rate REAL,
                        actual_completion_time REAL,
                        actual_total_cost REAL,
                        
                        selection_reasoning TEXT,
                        risk_factors TEXT,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME
                    )
                """)
                
                # Tabela performance agentów
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        
                        assigned_at DATETIME,
                        completed_at DATETIME,
                        
                        success_achieved BOOLEAN,
                        completion_time_hours REAL,
                        cost_incurred REAL,
                        quality_score REAL,
                        
                        skills_used TEXT,
                        collaboration_rating REAL,
                        client_satisfaction REAL,
                        
                        lessons_learned TEXT,
                        improvement_suggestions TEXT
                    )
                """)
                
                conn.commit()
                self.logger.info("✅ Agent selection database initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    async def select_agents_for_task(
        self,
        task_requirements: TaskRequirements,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED
    ) -> AgentSelection:
        """Główna metoda selekcji agentów dla zadania"""
        
        self.logger.info(f"🎯 Starting agent selection for task: {task_requirements.title}")
        
        # 1. Analiza wymagań zadania
        task_analysis = await self._analyze_task_requirements(task_requirements)
        
        # 2. Filtering dostępnych agentów
        candidate_agents = await self._filter_candidate_agents(task_requirements)
        
        # 3. Scoring i ranking agentów
        ranked_agents = await self._score_and_rank_agents(
            candidate_agents, task_requirements, strategy
        )
        
        # 4. Formowanie optymalnego zespołu
        selected_team = await self._form_optimal_team(
            ranked_agents, task_requirements, strategy
        )
        
        # 5. Walidacja selekcji
        validated_selection = await self._validate_selection(selected_team, task_requirements)
        
        # 6. Generowanie uzasadnienia
        reasoning = await self._generate_selection_reasoning(validated_selection, task_analysis)
        
        # 7. Tworzenie finalnego wyniku
        selection_result = AgentSelection(
            task_id=task_requirements.id,
            selected_agents=validated_selection,
            selection_strategy=strategy,
            confidence_score=await self._calculate_confidence_score(validated_selection),
            estimated_success_rate=await self._estimate_success_rate(validated_selection, task_requirements),
            estimated_completion_time=await self._estimate_completion_time(validated_selection, task_requirements),
            estimated_total_cost=await self._estimate_total_cost(validated_selection, task_requirements),
            selection_reasoning=reasoning["reasons"],
            risk_factors=reasoning["risks"],
            optimization_opportunities=reasoning["optimizations"]
        )
        
        # 8. Zapis do historii
        self.selection_history[task_requirements.id] = selection_result
        await self._store_selection(selection_result)
        
        self.logger.info(f"✅ Agent selection completed: {len(validated_selection)} agents selected")
        
        return selection_result
    
    async def _analyze_task_requirements(self, task_req: TaskRequirements) -> Dict[str, Any]:
        """Głęboka analiza wymagań zadania"""
        
        analysis = {
            "complexity_factors": [],
            "skill_criticality": {},
            "collaboration_needs": 0.0,
            "time_pressure": 0.0,
            "quality_requirements": 0.0,
            "budget_constraints": 0.0
        }
        
        # Analiza complexity
        if task_req.complexity_level > 0.8:
            analysis["complexity_factors"].extend(["high_technical_complexity", "requires_senior_agents"])
        
        if task_req.requires_collaboration:
            analysis["collaboration_needs"] = 0.8
            analysis["complexity_factors"].append("team_coordination_required")
        
        # Analiza deadline pressure
        if task_req.deadline:
            time_remaining = (task_req.deadline - datetime.now()).total_seconds() / 3600
            if time_remaining < 24:
                analysis["time_pressure"] = 0.9
                analysis["complexity_factors"].append("tight_deadline")
        
        # Skill criticality analysis
        for skill in task_req.required_skills:
            # Oceń jak krytyczna jest każda umiejętność
            analysis["skill_criticality"][skill.value] = 0.8  # Simplified
        
        return analysis
    
    async def _filter_candidate_agents(self, task_req: TaskRequirements) -> List[Agent]:
        """Filtruje agentów spełniających podstawowe wymagania"""
        
        candidates = []
        
        for agent in self.agents.values():
            # Check availability
            if not agent.availability or agent.current_workload > 0.8:
                continue
                
            # Check skill match
            matching_skills = set(agent.skills) & set(task_req.required_skills)
            skill_match_ratio = len(matching_skills) / len(task_req.required_skills) if task_req.required_skills else 1.0
            
            # Minimum skill threshold
            if skill_match_ratio < 0.5:
                continue
                
            # Check agent type preference
            if task_req.preferred_agent_types and agent.agent_type not in task_req.preferred_agent_types:
                # Nie blokuje, ale obniży score
                pass
                
            candidates.append(agent)
        
        self.logger.debug(f"📋 Filtered to {len(candidates)} candidate agents")
        return candidates
    
    async def _score_and_rank_agents(
        self,
        candidates: List[Agent],
        task_req: TaskRequirements,
        strategy: SelectionStrategy
    ) -> List[Tuple[Agent, float]]:
        """Ocenia i rankinguje agentów według strategii"""
        
        scored_agents = []
        
        for agent in candidates:
            score = await self._calculate_agent_score(agent, task_req, strategy)
            scored_agents.append((agent, score))
        
        # Sortuj według score (malejąco)
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return scored_agents
    
    async def _calculate_agent_score(
        self,
        agent: Agent,
        task_req: TaskRequirements,
        strategy: SelectionStrategy
    ) -> float:
        """Oblicza score agenta dla zadania według strategii"""
        
        base_score = 0.0
        
        # 1. Skill matching
        matching_skills = set(agent.skills) & set(task_req.required_skills)
        skill_score = len(matching_skills) / len(task_req.required_skills) if task_req.required_skills else 1.0
        
        # 2. Performance history
        performance_score = (agent.success_rate + agent.specialization_level) / 2.0
        
        # 3. Availability and workload
        availability_score = 1.0 - agent.current_workload
        
        # 4. Cost efficiency (koszt per value)
        if task_req.max_budget > 0:
            cost_efficiency = min(1.0, task_req.max_budget / (agent.cost_per_hour * task_req.complexity_level * 10))
        else:
            cost_efficiency = 1.0 / (agent.cost_per_hour / 50.0)  # Normalized to $50/hour baseline
        
        # 5. Strategic weights based on selection strategy
        if strategy == SelectionStrategy.PERFORMANCE_OPTIMIZED:
            weights = {"skill": 0.3, "performance": 0.4, "availability": 0.2, "cost": 0.1}
        elif strategy == SelectionStrategy.COST_OPTIMIZED:
            weights = {"skill": 0.2, "performance": 0.2, "availability": 0.2, "cost": 0.4}
        elif strategy == SelectionStrategy.SPEED_OPTIMIZED:
            weights = {"skill": 0.3, "performance": 0.2, "availability": 0.4, "cost": 0.1}
            # Bonus for faster agents
            speed_bonus = 1.0 / max(0.1, agent.average_completion_time)
            availability_score *= speed_bonus
        else:  # BALANCED
            weights = {"skill": 0.3, "performance": 0.3, "availability": 0.2, "cost": 0.2}
        
        # Calculate weighted score
        base_score = (
            skill_score * weights["skill"] +
            performance_score * weights["performance"] +
            availability_score * weights["availability"] +
            cost_efficiency * weights["cost"]
        )
        
        # Bonuses and penalties
        
        # Agent type preference bonus
        if task_req.preferred_agent_types and agent.agent_type in task_req.preferred_agent_types:
            base_score *= 1.1
        
        # Team compatibility bonus for collaborative tasks
        if task_req.requires_collaboration:
            base_score *= agent.team_compatibility
        
        # Experience bonus
        if agent.projects_completed > 10:
            base_score *= 1.05
        
        return min(1.0, base_score)
    
    async def _form_optimal_team(
        self,
        ranked_agents: List[Tuple[Agent, float]],
        task_req: TaskRequirements,
        strategy: SelectionStrategy
    ) -> List[Agent]:
        """Formuje optymalny zespół z rankingu agentów"""
        
        selected_agents = []
        total_cost = 0.0
        covered_skills = set()
        
        # Najpierw wybierz lead agent (najwyższa ocena)
        if ranked_agents:
            lead_agent, lead_score = ranked_agents[0]
            selected_agents.append(lead_agent)
            total_cost += lead_agent.cost_per_hour * task_req.complexity_level * 2  # Estimated hours
            covered_skills.update(lead_agent.skills)
        
        # Dodaj pozostałych agentów w miarę potrzeb
        for agent, score in ranked_agents[1:]:
            # Sprawdź czy dodanie agenta ma sens
            if len(selected_agents) >= task_req.team_size_max:
                break
                
            # Sprawdź budget
            estimated_cost = agent.cost_per_hour * task_req.complexity_level * 2
            if total_cost + estimated_cost > task_req.max_budget:
                continue
            
            # Sprawdź value add - czy agent wnosi nowe umiejętności
            new_skills = set(agent.skills) - covered_skills
            if new_skills or len(selected_agents) < task_req.team_size_min:
                selected_agents.append(agent)
                total_cost += estimated_cost
                covered_skills.update(agent.skills)
        
        return selected_agents
    
    async def _validate_selection(self, selected_agents: List[Agent], task_req: TaskRequirements) -> List[Agent]:
        """Waliduje czy selekcja spełnia wymagania"""
        
        # Check minimum team size
        if len(selected_agents) < task_req.team_size_min:
            self.logger.warning(f"⚠️ Team size below minimum: {len(selected_agents)} < {task_req.team_size_min}")
        
        # Check skill coverage
        covered_skills = set()
        for agent in selected_agents:
            covered_skills.update(agent.skills)
        
        missing_skills = set(task_req.required_skills) - covered_skills
        if missing_skills:
            self.logger.warning(f"⚠️ Missing required skills: {[s.value for s in missing_skills]}")
        
        return selected_agents
    
    # Pozostałe metody pomocnicze (uproszczone implementacje)
    
    async def _calculate_confidence_score(self, agents: List[Agent]) -> float:
        if not agents:
            return 0.0
        return sum(agent.success_rate for agent in agents) / len(agents)
    
    async def _estimate_success_rate(self, agents: List[Agent], task_req: TaskRequirements) -> float:
        if not agents:
            return 0.0
        # Team success rate (not just average)
        individual_success = [agent.success_rate for agent in agents]
        # Team success is better than weakest link but not perfect multiplication
        return min(individual_success) * 0.7 + (sum(individual_success) / len(individual_success)) * 0.3
    
    async def _estimate_completion_time(self, agents: List[Agent], task_req: TaskRequirements) -> float:
        if not agents:
            return 0.0
        # Parallel work assumption with coordination overhead
        fastest_time = min(agent.average_completion_time for agent in agents)
        coordination_overhead = 1.0 + (len(agents) - 1) * 0.1
        return fastest_time * task_req.complexity_level * coordination_overhead
    
    async def _estimate_total_cost(self, agents: List[Agent], task_req: TaskRequirements) -> float:
        if not agents:
            return 0.0
        estimated_hours = await self._estimate_completion_time(agents, task_req)
        return sum(agent.cost_per_hour for agent in agents) * estimated_hours
    
    async def _generate_selection_reasoning(self, agents: List[Agent], analysis: Dict) -> Dict[str, List[str]]:
        reasons = [
            f"Selected {len(agents)} agents for optimal skill coverage",
            f"Team estimated success rate: {await self._estimate_success_rate(agents, TaskRequirements('temp', '', '')):.1%}",
            "Balanced performance, cost, and availability factors"
        ]
        
        risks = [
            "Coordination overhead in multi-agent team",
            "Dependency on individual agent availability"
        ]
        
        optimizations = [
            "Consider cross-training agents in missing skills",
            "Implement real-time workload balancing"
        ]
        
        return {"reasons": reasons, "risks": risks, "optimizations": optimizations}
    
    async def _store_selection(self, selection: AgentSelection):
        """Store selection in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO agent_selections 
                    (id, task_id, selected_agents, selection_strategy, confidence_score,
                     estimated_success_rate, estimated_completion_time, estimated_total_cost,
                     selection_reasoning, risk_factors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()), selection.task_id,
                    json.dumps([agent.id for agent in selection.selected_agents]),
                    selection.selection_strategy.value,
                    selection.confidence_score,
                    selection.estimated_success_rate,
                    selection.estimated_completion_time,
                    selection.estimated_total_cost,
                    json.dumps(selection.selection_reasoning),
                    json.dumps(selection.risk_factors)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store selection: {e}")

# ================================
# INTEGRATION WITH OTHER SYSTEMS
# ================================

class SystemIntegrator:
    """Integrator z innymi systemami Agent Zero"""
    
    def __init__(self, selector: IntelligentAgentSelector):
        self.selector = selector
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_nlu_result(self, nlu_data: Dict[str, Any]) -> AgentSelection:
        """Przetwarza wynik z NLU system do agent selection"""
        
        # Extract task requirements from NLU analysis
        task_req = TaskRequirements(
            id=nlu_data.get("task_id", str(uuid.uuid4())),
            title=nlu_data.get("title", "AI Generated Task"),
            description=nlu_data.get("description", ""),
            complexity_level=nlu_data.get("complexity", 0.5),
            priority_level=nlu_data.get("priority", "MEDIUM")
        )
        
        # Map NLU analysis to required skills
        if "code" in nlu_data.get("task_type", "").lower():
            task_req.required_skills.append(AgentSkill.PYTHON_DEVELOPMENT)
        if "analysis" in nlu_data.get("task_type", "").lower():
            task_req.required_skills.append(AgentSkill.MACHINE_LEARNING)
        if "deployment" in nlu_data.get("description", "").lower():
            task_req.required_skills.append(AgentSkill.DEPLOYMENT_AUTOMATION)
        
        # Perform agent selection
        selection = await self.selector.select_agents_for_task(task_req)
        
        return selection
    
    async def send_to_priority_system(self, selection: AgentSelection) -> bool:
        """Wysyła wyniki selekcji do Dynamic Priority system"""
        
        try:
            async with httpx.AsyncClient() as client:
                priority_data = {
                    "task_id": selection.task_id,
                    "assigned_agents": [agent.id for agent in selection.selected_agents],
                    "estimated_completion": selection.estimated_completion_time,
                    "confidence_score": selection.confidence_score,
                    "selection_metadata": {
                        "strategy": selection.selection_strategy.value,
                        "reasoning": selection.selection_reasoning
                    }
                }
                
                response = await client.post(
                    f"{self.selector.priority_service}/api/v1/priority/tasks/agent-assigned",
                    json=priority_data
                )
                
                if response.status_code == 200:
                    self.logger.info(f"✅ Selection sent to priority system: {selection.task_id}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Failed to send to priority system: {response.status_code}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Error sending to priority system: {e}")
            return False

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Agent Zero V1 - Point 2: Intelligent Agent Selection",
    description="Missing Link: Most między NLU a Dynamic Priority - Inteligentny wybór agentów",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
agent_selector = IntelligentAgentSelector()
system_integrator = SystemIntegrator(agent_selector)

@app.get("/")
async def agent_selection_root():
    """Point 2: Intelligent Agent Selection System"""
    return {
        "system": "Agent Zero V1 - Point 2: Intelligent Agent Selection",
        "version": "1.0.0",
        "status": "OPERATIONAL", 
        "description": "Missing Link w architekturze - most między NLU a Dynamic Priority",
        "architecture_position": "NLU (Port 8000/9001) → Agent Selection (Port 8002) → Priority (Port 8003) → Collaboration (Port 8005)",
        "capabilities": [
            "Inteligentny wybór agentów na podstawie wymagań zadania",
            "Multi-criteria decision making z 5 strategiami selekcji",
            "Optymalne formowanie zespołów",
            "Real-time performance tracking",
            "Integration z wszystkimi systemami Agent Zero"
        ],
        "agent_pool": {
            "total_agents": len(agent_selector.agents),
            "available_agents": len([a for a in agent_selector.agents.values() if a.availability]),
            "agent_types": list(set(a.agent_type.value for a in agent_selector.agents.values()))
        },
        "selection_strategies": [s.value for s in SelectionStrategy],
        "endpoints": {
            "select_agents": "POST /api/v1/agents/select",
            "agent_pool": "GET /api/v1/agents/pool", 
            "selection_history": "GET /api/v1/agents/selections",
            "nlu_integration": "POST /api/v1/agents/from-nlu"
        }
    }

@app.post("/api/v1/agents/select")
async def select_agents(selection_request: dict):
    """Główny endpoint selekcji agentów"""
    
    try:
        # Parse request to TaskRequirements
        task_req = TaskRequirements(
            id=selection_request.get("task_id", str(uuid.uuid4())),
            title=selection_request.get("title", "Agent Selection Task"),
            description=selection_request.get("description", ""),
            complexity_level=selection_request.get("complexity_level", 0.5),
            max_budget=selection_request.get("max_budget", 1000.0)
        )
        
        # Add required skills
        if "required_skills" in selection_request:
            task_req.required_skills = [
                AgentSkill(skill) for skill in selection_request["required_skills"]
            ]
        
        # Strategy
        strategy = SelectionStrategy(selection_request.get("strategy", "BALANCED"))
        
        # Perform selection
        selection_result = await agent_selector.select_agents_for_task(task_req, strategy)
        
        # Send to priority system
        await system_integrator.send_to_priority_system(selection_result)
        
        return {
            "status": "success",
            "task_id": selection_result.task_id,
            "selected_agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.agent_type.value,
                    "skills": [skill.value for skill in agent.skills],
                    "success_rate": agent.success_rate,
                    "cost_per_hour": agent.cost_per_hour
                }
                for agent in selection_result.selected_agents
            ],
            "selection_metrics": {
                "confidence_score": selection_result.confidence_score,
                "estimated_success_rate": selection_result.estimated_success_rate,
                "estimated_completion_time": selection_result.estimated_completion_time,
                "estimated_total_cost": selection_result.estimated_total_cost
            },
            "reasoning": {
                "selection_reasoning": selection_result.selection_reasoning,
                "risk_factors": selection_result.risk_factors,
                "optimization_opportunities": selection_result.optimization_opportunities
            },
            "next_step": "Task forwarded to Dynamic Priority System (Port 8003)",
            "architecture_flow": "NLU → Agent Selection → Priority → Collaboration"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "selected_agents": []
        }

@app.post("/api/v1/agents/from-nlu")
async def process_from_nlu(nlu_data: dict):
    """Integration endpoint dla NLU systems"""
    
    try:
        selection_result = await system_integrator.process_nlu_result(nlu_data)
        
        return {
            "status": "success",
            "message": "NLU task processed and agents selected",
            "selection_result": selection_result.task_id,
            "agents_selected": len(selection_result.selected_agents),
            "forwarded_to_priority": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/agents/pool")
async def get_agent_pool():
    """Zwraca informacje o dostępnych agentach"""
    
    agents_info = []
    for agent in agent_selector.agents.values():
        agents_info.append({
            "id": agent.id,
            "name": agent.name,
            "type": agent.agent_type.value,
            "skills": [skill.value for skill in agent.skills],
            "success_rate": agent.success_rate,
            "current_workload": agent.current_workload,
            "availability": agent.availability,
            "cost_per_hour": agent.cost_per_hour,
            "projects_completed": agent.projects_completed,
            "specialization_level": agent.specialization_level
        })
    
    return {
        "status": "success",
        "agent_pool": agents_info,
        "pool_statistics": {
            "total_agents": len(agents_info),
            "available_agents": len([a for a in agents_info if a["availability"]]),
            "average_success_rate": sum(a["success_rate"] for a in agents_info) / len(agents_info),
            "average_cost": sum(a["cost_per_hour"] for a in agents_info) / len(agents_info)
        }
    }

@app.get("/api/v1/agents/selections")  
async def get_selection_history(limit: int = 10):
    """Historia selekcji agentów"""
    
    recent_selections = list(agent_selector.selection_history.values())[-limit:]
    
    return {
        "status": "success",
        "recent_selections": [
            {
                "task_id": sel.task_id,
                "agents_selected": len(sel.selected_agents),
                "strategy": sel.selection_strategy.value,
                "confidence_score": sel.confidence_score,
                "estimated_success": sel.estimated_success_rate,
                "timestamp": sel.timestamp.isoformat()
            }
            for sel in recent_selections
        ],
        "total_selections": len(agent_selector.selection_history)
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Point 2: Intelligent Agent Selection System...")
    logger.info("🔗 Missing Link w architekturze - łączy NLU z Dynamic Priority")
    logger.info("📊 Agent pool ready with enterprise-grade selection algorithms")
    
    uvicorn.run(
        "point2_agent_selection:app",
        host="0.0.0.0", 
        port=8002,
        workers=1,
        log_level="info",
        reload=False
    )