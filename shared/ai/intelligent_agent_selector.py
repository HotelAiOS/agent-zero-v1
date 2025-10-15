# shared/ai/intelligent_agent_selector.py - Production Implementation

"""
Intelligent Agent Selector - AI-Driven Agent Routing
Week 43 Priority 1 Task (6 SP)

Komponent wykorzystuje Ollama przez ProductionAISystem do inteligentnego
wyboru najlepszego agenta dla danego zadania na podstawie:
- Analizy złożoności zadania
- Dopasowania kompetencji agentów
- Historii performance
- Obciążenia systemu
- Kosztów operacyjnych
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from shared.ai.production_ai_system import ProductionAISystem
from shared.knowledge.neo4j_client import Neo4jClient
from shared.experience.experience_tracker import ExperienceTracker

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Reprezentacja możliwości agenta"""
    agent_id: str
    name: str
    capabilities: List[str]
    performance_score: float = 0.0
    current_load: int = 0
    max_load: int = 10
    cost_per_task: float = 0.01
    success_rate: float = 0.85
    avg_latency_ms: float = 500.0
    specializations: List[str] = field(default_factory=list)


@dataclass
class TaskComplexity:
    """Analiza złożoności zadania"""
    task_id: str
    description: str
    complexity_score: float  # 0.0 - 1.0
    required_capabilities: List[str]
    estimated_duration_ms: float
    priority: str  # "low", "medium", "high", "critical"
    context: Dict = field(default_factory=dict)


@dataclass
class AgentSelection:
    """Wynik selekcji agenta"""
    agent_id: str
    agent_name: str
    confidence: float  # 0.0 - 1.0
    reasoning: str
    estimated_cost: float
    estimated_duration_ms: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # (agent_id, confidence)


class IntelligentAgentSelector:
    """
    AI-driven agent selection system wykorzystujący Ollama do analizy
    zadań i inteligentnego routingu do najbardziej odpowiednich agentów.
    """
    
    def __init__(
        self,
        ai_system: ProductionAISystem,
        neo4j_client: Optional[Neo4jClient] = None,
        experience_tracker: Optional[ExperienceTracker] = None,
        enable_learning: bool = True
    ):
        self.ai_system = ai_system
        self.neo4j_client = neo4j_client
        self.experience_tracker = experience_tracker
        self.enable_learning = enable_learning
        
        # Registry agentów
        self.agents: Dict[str, AgentCapability] = {}
        
        # Cache dla optymalizacji
        self._selection_cache: Dict[str, AgentSelection] = {}
        self._cache_ttl_seconds = 300  # 5 minut
        
        logger.info("IntelligentAgentSelector initialized")
    
    def register_agent(self, agent: AgentCapability) -> None:
        """Rejestracja agenta w systemie"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    async def analyze_task_complexity(self, task_description: str, context: Dict = None) -> TaskComplexity:
        """
        Analiza złożoności zadania przy użyciu AI
        
        Args:
            task_description: Opis zadania do wykonania
            context: Dodatkowy kontekst (user_id, project_id, etc.)
        
        Returns:
            TaskComplexity z pełną analizą zadania
        """
        task_id = context.get("task_id", f"task_{datetime.now().timestamp()}") if context else f"task_{datetime.now().timestamp()}"
        
        # Prompt dla Ollama do analizy zadania
        analysis_prompt = f"""Analyze the following task and provide a structured assessment:

Task: {task_description}
Context: {json.dumps(context or {}, indent=2)}

Provide analysis in JSON format:
{{
  "complexity_score": <0.0 to 1.0>,
  "required_capabilities": ["capability1", "capability2", ...],
  "estimated_duration_ms": <number>,
  "priority": "<low|medium|high|critical>",
  "reasoning": "Brief explanation of complexity assessment"
}}

Consider:
1. Technical complexity (algorithms, data processing)
2. Resource requirements (computation, memory, I/O)
3. Dependencies on external systems
4. Time sensitivity
5. Domain expertise required"""

        try:
            # Wywołanie AI system dla analizy
            response = await self.ai_system.generate(
                prompt=analysis_prompt,
                model="llama3.2:3b",  # Szybki model dla analizy
                temperature=0.3,  # Niska temperatura dla spójnych wyników
                max_tokens=500
            )
            
            # Parsowanie odpowiedzi AI
            analysis_data = json.loads(response.strip())
            
            complexity = TaskComplexity(
                task_id=task_id,
                description=task_description,
                complexity_score=analysis_data["complexity_score"],
                required_capabilities=analysis_data["required_capabilities"],
                estimated_duration_ms=analysis_data["estimated_duration_ms"],
                priority=analysis_data["priority"],
                context=context or {}
            )
            
            logger.info(f"Task {task_id} analyzed: complexity={complexity.complexity_score:.2f}")
            return complexity
            
        except Exception as e:
            logger.error(f"Task analysis failed: {e}", exc_info=True)
            # Fallback do prostej heurystyki
            return TaskComplexity(
                task_id=task_id,
                description=task_description,
                complexity_score=0.5,
                required_capabilities=["general"],
                estimated_duration_ms=1000.0,
                priority="medium",
                context=context or {}
            )
    
    async def select_agent(
        self,
        task: TaskComplexity,
        constraints: Optional[Dict] = None
    ) -> AgentSelection:
        """
        Wybór najlepszego agenta dla zadania
        
        Args:
            task: Zadanie do wykonania z analizą złożoności
            constraints: Ograniczenia (max_cost, max_duration, required_agent, etc.)
        
        Returns:
            AgentSelection z wyborem i uzasadnieniem
        """
        constraints = constraints or {}
        
        # Sprawdź cache
        cache_key = f"{task.task_id}_{hash(json.dumps(constraints, sort_keys=True))}"
        if cache_key in self._selection_cache:
            cached = self._selection_cache[cache_key]
            if (datetime.now() - cached.timestamp).total_seconds() < self._cache_ttl_seconds:
                logger.info(f"Using cached selection for task {task.task_id}")
                return cached
        
        # Filtruj agentów według dostępności i ograniczeń
        available_agents = [
            agent for agent in self.agents.values()
            if agent.current_load < agent.max_load
            and (not constraints.get("max_cost") or agent.cost_per_task <= constraints["max_cost"])
        ]
        
        if not available_agents:
            raise ValueError("No available agents matching constraints")
        
        # Pobierz historyczne dane performance dla agentów
        agent_history = await self._get_agent_performance_history(
            [agent.agent_id for agent in available_agents],
            task.required_capabilities
        )
        
        # Prompt dla AI do wyboru agenta
        selection_prompt = f"""Select the best agent for the following task:

Task Analysis:
- Complexity: {task.complexity_score:.2f}
- Required Capabilities: {', '.join(task.required_capabilities)}
- Estimated Duration: {task.estimated_duration_ms}ms
- Priority: {task.priority}

Available Agents:
{self._format_agents_for_prompt(available_agents, agent_history)}

Constraints:
{json.dumps(constraints, indent=2)}

Select the optimal agent and provide reasoning in JSON format:
{{
  "selected_agent_id": "<agent_id>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "Detailed explanation of selection",
  "alternatives": [
    {{"agent_id": "<id>", "confidence": <score>}},
    ...
  ]
}}

Consider:
1. Capability match with task requirements
2. Current load and availability
3. Historical performance (success rate, latency)
4. Cost efficiency
5. Task priority and urgency"""

        try:
            # Wywołanie AI dla selekcji
            response = await self.ai_system.generate(
                prompt=selection_prompt,
                model="llama3.2:3b",
                temperature=0.2,  # Bardzo niska dla deterministycznych wyborów
                max_tokens=800
            )
            
            # Parsowanie decyzji AI
            selection_data = json.loads(response.strip())
            
            selected_agent = self.agents[selection_data["selected_agent_id"]]
            
            selection = AgentSelection(
                agent_id=selected_agent.agent_id,
                agent_name=selected_agent.name,
                confidence=selection_data["confidence"],
                reasoning=selection_data["reasoning"],
                estimated_cost=selected_agent.cost_per_task,
                estimated_duration_ms=selected_agent.avg_latency_ms,
                alternatives=[(alt["agent_id"], alt["confidence"]) for alt in selection_data.get("alternatives", [])]
            )
            
            # Cache wynik
            selection.timestamp = datetime.now()
            self._selection_cache[cache_key] = selection
            
            # Record decision dla learning
            if self.enable_learning and self.experience_tracker:
                await self._record_selection_decision(task, selection)
            
            logger.info(f"Selected agent {selection.agent_name} for task {task.task_id} (confidence: {selection.confidence:.2f})")
            return selection
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}", exc_info=True)
            # Fallback do prostej heurystyki
            return self._fallback_selection(available_agents, task)
    
    async def update_agent_load(self, agent_id: str, delta: int) -> None:
        """Aktualizacja obciążenia agenta"""
        if agent_id in self.agents:
            self.agents[agent_id].current_load += delta
            self.agents[agent_id].current_load = max(0, self.agents[agent_id].current_load)
            logger.debug(f"Agent {agent_id} load: {self.agents[agent_id].current_load}/{self.agents[agent_id].max_load}")
    
    async def record_task_outcome(
        self,
        task_id: str,
        agent_id: str,
        success: bool,
        actual_duration_ms: float,
        actual_cost: float
    ) -> None:
        """
        Rejestracja wyniku wykonania zadania dla learning
        
        Aktualizuje performance metrics agenta na podstawie rzeczywistych wyników
        """
        if agent_id not in self.agents:
            logger.warning(f"Unknown agent {agent_id} in outcome recording")
            return
        
        agent = self.agents[agent_id]
        
        # Exponential moving average dla metryk
        alpha = 0.2  # Współczynnik uczenia
        
        if success:
            agent.success_rate = (1 - alpha) * agent.success_rate + alpha * 1.0
        else:
            agent.success_rate = (1 - alpha) * agent.success_rate + alpha * 0.0
        
        agent.avg_latency_ms = (1 - alpha) * agent.avg_latency_ms + alpha * actual_duration_ms
        agent.cost_per_task = (1 - alpha) * agent.cost_per_task + alpha * actual_cost
        
        # Zapisz w Neo4j dla długoterminowego uczenia
        if self.neo4j_client:
            await self._persist_outcome_to_neo4j(task_id, agent_id, success, actual_duration_ms, actual_cost)
        
        logger.info(f"Recorded outcome for task {task_id} by agent {agent_id}: success={success}")
    
    def _format_agents_for_prompt(self, agents: List[AgentCapability], history: Dict) -> str:
        """Formatowanie agentów dla promptu AI"""
        lines = []
        for agent in agents:
            hist = history.get(agent.agent_id, {})
            lines.append(f"""
Agent: {agent.name} ({agent.agent_id})
- Capabilities: {', '.join(agent.capabilities)}
- Specializations: {', '.join(agent.specializations) if agent.specializations else 'None'}
- Performance Score: {agent.performance_score:.2f}
- Success Rate: {agent.success_rate:.1%}
- Avg Latency: {agent.avg_latency_ms:.0f}ms
- Cost per Task: ${agent.cost_per_task:.4f}
- Current Load: {agent.current_load}/{agent.max_load}
- Historical Tasks: {hist.get('total_tasks', 0)}
- Recent Success Rate: {hist.get('recent_success_rate', 0.0):.1%}
""")
        return "\n".join(lines)
    
    def _fallback_selection(self, agents: List[AgentCapability], task: TaskComplexity) -> AgentSelection:
        """Prosta heurystyka fallback gdy AI zawiedzie"""
        # Sortuj według score (kombinacja success_rate, load, cost)
        def score_agent(agent: AgentCapability) -> float:
            load_factor = 1.0 - (agent.current_load / agent.max_load)
            cost_factor = 1.0 / (1.0 + agent.cost_per_task)
            return agent.success_rate * 0.5 + load_factor * 0.3 + cost_factor * 0.2
        
        sorted_agents = sorted(agents, key=score_agent, reverse=True)
        best = sorted_agents[0]
        
        return AgentSelection(
            agent_id=best.agent_id,
            agent_name=best.name,
            confidence=0.6,  # Niższa confidence dla fallback
            reasoning="Fallback heuristic selection based on success rate, load, and cost",
            estimated_cost=best.cost_per_task,
            estimated_duration_ms=best.avg_latency_ms,
            alternatives=[(agent.agent_id, score_agent(agent)) for agent in sorted_agents[1:3]]
        )
    
    async def _get_agent_performance_history(
        self,
        agent_ids: List[str],
        capabilities: List[str]
    ) -> Dict:
        """Pobierz historyczne dane performance z Neo4j"""
        if not self.neo4j_client:
            return {}
        
        # Neo4j query dla historical performance
        query = """
        MATCH (a:Agent)-[:EXECUTED]->(t:Task)
        WHERE a.agent_id IN $agent_ids
        AND any(cap IN $capabilities WHERE cap IN t.required_capabilities)
        WITH a, t
        RETURN a.agent_id as agent_id,
               count(t) as total_tasks,
               avg(CASE WHEN t.success THEN 1.0 ELSE 0.0 END) as recent_success_rate,
               avg(t.duration_ms) as avg_duration
        """
        
        try:
            results = await self.neo4j_client.execute_query(query, {
                "agent_ids": agent_ids,
                "capabilities": capabilities
            })
            
            return {
                record["agent_id"]: {
                    "total_tasks": record["total_tasks"],
                    "recent_success_rate": record["recent_success_rate"],
                    "avg_duration": record["avg_duration"]
                }
                for record in results
            }
        except Exception as e:
            logger.error(f"Failed to fetch agent history: {e}")
            return {}
    
    async def _persist_outcome_to_neo4j(
        self,
        task_id: str,
        agent_id: str,
        success: bool,
        duration_ms: float,
        cost: float
    ) -> None:
        """Persist task outcome w Neo4j dla długoterminowego uczenia"""
        if not self.neo4j_client:
            return
        
        query = """
        MERGE (a:Agent {agent_id: $agent_id})
        MERGE (t:Task {task_id: $task_id})
        MERGE (a)-[r:EXECUTED]->(t)
        SET r.timestamp = datetime(),
            r.success = $success,
            r.duration_ms = $duration_ms,
            r.cost = $cost
        SET t.success = $success,
            t.duration_ms = $duration_ms,
            t.cost = $cost
        """
        
        try:
            await self.neo4j_client.execute_query(query, {
                "agent_id": agent_id,
                "task_id": task_id,
                "success": success,
                "duration_ms": duration_ms,
                "cost": cost
            })
        except Exception as e:
            logger.error(f"Failed to persist outcome to Neo4j: {e}")
    
    async def _record_selection_decision(self, task: TaskComplexity, selection: AgentSelection) -> None:
        """Zapisz decyzję selekcji w ExperienceTracker"""
        if not self.experience_tracker:
            return
        
        await self.experience_tracker.record_decision(
            task_id=task.task_id,
            decision_type="agent_selection",
            decision_data={
                "selected_agent": selection.agent_id,
                "confidence": selection.confidence,
                "reasoning": selection.reasoning,
                "alternatives": selection.alternatives,
                "task_complexity": task.complexity_score,
                "required_capabilities": task.required_capabilities
            },
            context=task.context
        )


# Factory function dla łatwej inicjalizacji
async def create_intelligent_agent_selector(
    ai_system: ProductionAISystem,
    neo4j_url: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    enable_learning: bool = True
) -> IntelligentAgentSelector:
    """
    Utworzenie w pełni skonfigurowanego IntelligentAgentSelector
    
    Args:
        ai_system: Instancja ProductionAISystem
        neo4j_url: URL Neo4j (opcjonalnie)
        neo4j_user: Username Neo4j
        neo4j_password: Password Neo4j
        enable_learning: Czy włączyć system uczenia
    
    Returns:
        Skonfigurowany IntelligentAgentSelector
    """
    neo4j_client = None
    experience_tracker = None
    
    if neo4j_url and neo4j_user and neo4j_password:
        neo4j_client = Neo4jClient(neo4j_url, neo4j_user, neo4j_password)
        await neo4j_client.connect()
    
    if enable_learning:
        from shared.experience.experience_tracker import ExperienceTracker
        experience_tracker = ExperienceTracker(neo4j_client=neo4j_client)
    
    return IntelligentAgentSelector(
        ai_system=ai_system,
        neo4j_client=neo4j_client,
        experience_tracker=experience_tracker,
        enable_learning=enable_learning
    )
