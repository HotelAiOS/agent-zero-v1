"""
Agent Zero Core Engine
Główny silnik łączący wszystkie moduły systemu
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_factory import AgentFactory
from orchestration import (
    IntelligentPlanner,
    TaskDecomposer,
    TeamFormationEngine,
    QualityGateManager,
    TaskScheduler,
    ScheduleStrategy
)
from protocols import (
    CodeReviewProtocol,
    ProblemSolvingProtocol,
    KnowledgeSharingProtocol,
    EscalationProtocol,
    ConsensusProtocol
)
from learning import (
    PostMortemAnalyzer,
    PatternDetector,
    AntiPatternDetector,
    RecommendationEngine,
    AgentEvolutionEngine
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProjectExecution:
    """Reprezentacja wykonania projektu"""
    project_id: str
    project_name: str
    business_requirements: List[str]
    
    # Components
    team: Optional[Any] = None
    plan: Optional[Any] = None
    schedule: Optional[List] = None
    
    # Status
    status: str = 'initialized'
    progress: float = 0.0
    
    # Results
    completed_tasks: List[str] = field(default_factory=list)
    active_protocols: Dict[str, Any] = field(default_factory=dict)
    quality_gates_status: Dict[str, str] = field(default_factory=dict)
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Learning
    post_mortem_analysis: Optional[Any] = None
    lessons_learned: List[str] = field(default_factory=list)


class AgentZeroCore:
    """
    Agent Zero Core Engine
    
    Główny silnik integrujący:
    - Agent Factory (tworzenie agentów)
    - Orchestration (planowanie, scheduling)
    - Protocols (komunikacja między agentami)
    - Learning (analiza, wzorce, ewolucja)
    """
    
    def __init__(self):
        logger.info("Inicjalizacja Agent Zero Core Engine...")
        
        # Moduł 1: Agent Factory
        self.agent_factory = AgentFactory()
        logger.info(f"✓ Agent Factory: {len(self.agent_factory.templates)} typów agentów")
        
        # Moduł 2: Orchestration
        self.task_decomposer = TaskDecomposer()
        self.team_formation = TeamFormationEngine()
        self.quality_gates = QualityGateManager()
        self.quality_gates.define_standard_gates()
        self.scheduler = TaskScheduler()
        self.planner = IntelligentPlanner()
        logger.info("✓ Orchestration: planner, decomposer, team formation")
        
        # Moduł 3: Protocols
        self.active_protocols: Dict[str, Any] = {}
        logger.info("✓ Protocols: 5 protokołów dostępnych")
        
        # Moduł 4: Learning
        self.post_mortem = PostMortemAnalyzer()
        self.pattern_detector = PatternDetector(neo4j_enabled=False)
        self.antipattern_detector = AntiPatternDetector()
        self.recommendation_engine = RecommendationEngine()
        self.evolution_engine = AgentEvolutionEngine()
        logger.info("✓ Learning: analiza, wzorce, rekomendacje, ewolucja")
        
        # Storage
        self.active_projects: Dict[str, ProjectExecution] = {}
        self.completed_projects: List[ProjectExecution] = []
        
        logger.info("✅ Agent Zero Core Engine ready!")
    
    def create_project(
        self,
        project_name: str,
        project_type: str,
        business_requirements: List[str],
        schedule_strategy: ScheduleStrategy = ScheduleStrategy.LOAD_BALANCED
    ) -> ProjectExecution:
        """
        Utwórz i zaplanuj projekt
        
        Args:
            project_name: Nazwa projektu
            project_type: Typ projektu (fullstack_web_app, api_backend, etc.)
            business_requirements: Wymagania biznesowe
            schedule_strategy: Strategia schedulowania
        
        Returns:
            ProjectExecution
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Tworzenie projektu: {project_name}")
        logger.info(f"{'='*70}")
        
        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        execution = ProjectExecution(
            project_id=project_id,
            project_name=project_name,
            business_requirements=business_requirements,
            started_at=datetime.now()
        )
        
        # FAZA 1: Rekomendacje (Learning)
        logger.info("\n📊 FAZA 1: Analiza i rekomendacje")
        recommendations = self._generate_recommendations(
            project_type,
            business_requirements
        )
        logger.info(f"   ✓ Wygenerowano {len(recommendations)} rekomendacji")
        
        # FAZA 2: Tworzenie zespołu (Agent Factory + Orchestration)
        logger.info("\n👥 FAZA 2: Formowanie zespołu agentów")
        execution.team = self._create_team(project_type, business_requirements)
        logger.info(f"   ✓ Zespół: {len(execution.team)} agentów")
        
        # FAZA 3: Planowanie (Orchestration)
        logger.info("\n📋 FAZA 3: Dekompozycja i planowanie")
        execution.plan = self._create_plan(
            project_id,
            project_name,
            project_type,
            business_requirements,
            execution.team,
            schedule_strategy
        )
        logger.info(f"   ✓ Plan: {len(execution.plan.tasks)} zadań")
        logger.info(f"   ✓ Czas: {execution.plan.estimated_duration_days:.1f} dni")
        logger.info(f"   ✓ Koszt: {execution.plan.total_cost_estimate:,.2f} PLN")
        
        # FAZA 4: Quality Gates (Orchestration)
        logger.info("\n🚦 FAZA 4: Quality Gates setup")
        execution.quality_gates_status = {
            gate_id: gate.status.value
            for gate_id, gate in execution.plan.quality_gates.items()
        }
        logger.info(f"   ✓ Quality Gates: {len(execution.quality_gates_status)}")
        
        execution.status = 'planned'
        self.active_projects[project_id] = execution
        
        logger.info(f"\n✅ Projekt {project_name} zaplanowany!")
        logger.info(f"{'='*70}\n")
        
        return execution
    
    def _generate_recommendations(
        self,
        project_type: str,
        requirements: List[str]
    ) -> List[Any]:
        """Generuj rekomendacje dla projektu"""
        # Symulacja historycznych danych
        historical_data = [
            {
                'project_id': 'hist_1',
                'success': True,
                'tech_stack': ['Python', 'FastAPI', 'PostgreSQL'],
                'team_size': 4,
                'test_coverage': 0.85,
                'actual_duration_days': 35
            }
        ]
        
        project_plan = {
            'tech_stack': ['Python', 'FastAPI', 'PostgreSQL'],
            'team_size': 5,
            'target_test_coverage': 0.8,
            'estimated_duration_days': 30
        }
        
        return self.recommendation_engine.generate_recommendations(
            project_plan,
            historical_data
        )
    
    def _create_team(self, project_type: str, requirements: List[str]) -> Dict[str, Any]:
        """Utwórz zespół agentów"""
        # Określ wymagane role na podstawie typu projektu
        if project_type == 'fullstack_web_app':
            roles = ['architect', 'backend', 'frontend', 'database', 'tester', 'devops']
        elif project_type == 'api_backend':
            roles = ['architect', 'backend', 'database', 'tester', 'security']
        elif project_type == 'microservices':
            roles = ['architect', 'backend', 'database', 'devops', 'tester']
        else:
            roles = ['architect', 'backend', 'tester']
        
        # Utwórz zespół przez Agent Factory
        team = self.agent_factory.create_team(roles)
        
        return team
    
    def _create_plan(
        self,
        project_id: str,
        project_name: str,
        project_type: str,
        requirements: List[str],
        team: Dict[str, Any],
        schedule_strategy: ScheduleStrategy
    ) -> Any:
        """Utwórz kompletny plan projektu"""
        # Zarejestruj agentów w plannerze
        for agent_id, agent in team.items():
            agent_type = agent.agent_type
            if agent_type not in self.planner.team_formation.agent_pool:
                self.planner.team_formation.agent_pool[agent_type] = []
            self.planner.team_formation.agent_pool[agent_type].append(agent_id)
        
        # Utwórz plan
        plan = self.planner.create_project_plan(
            project_name=project_name,
            project_type=project_type,
            business_requirements=requirements,
            schedule_strategy=schedule_strategy
        )
        
        return plan
    
    def start_protocol(
        self,
        project_id: str,
        protocol_type: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        Uruchom protokół komunikacji
        
        Args:
            project_id: ID projektu
            protocol_type: Typ protokołu (code_review, problem_solving, etc.)
            context: Kontekst protokołu
        
        Returns:
            Protocol instance
        """
        if project_id not in self.active_projects:
            logger.error(f"Project {project_id} not found")
            return None
        
        execution = self.active_projects[project_id]
        
        # Utwórz protokół
        if protocol_type == 'code_review':
            protocol = CodeReviewProtocol()
        elif protocol_type == 'problem_solving':
            protocol = ProblemSolvingProtocol()
        elif protocol_type == 'knowledge_sharing':
            protocol = KnowledgeSharingProtocol()
        elif protocol_type == 'escalation':
            protocol = EscalationProtocol()
        elif protocol_type == 'consensus':
            protocol = ConsensusProtocol()
        else:
            logger.error(f"Unknown protocol type: {protocol_type}")
            return None
        
        # Inicjuj
        protocol.initiate(
            initiator=context.get('initiator', 'system'),
            context=context
        )
        
        # Zapisz
        protocol_id = protocol.protocol_id
        execution.active_protocols[protocol_id] = protocol
        self.active_protocols[protocol_id] = protocol
        
        logger.info(f"✓ Started {protocol_type} protocol: {protocol_id}")
        
        return protocol
    
    def complete_project(
        self,
        project_id: str,
        perform_post_mortem: bool = True
    ) -> Optional[Any]:
        """
        Zakończ projekt i przeprowadź analizę
        
        Args:
            project_id: ID projektu
            perform_post_mortem: Czy przeprowadzić post-mortem
        
        Returns:
            Post-mortem analysis lub None
        """
        if project_id not in self.active_projects:
            logger.error(f"Project {project_id} not found")
            return None
        
        execution = self.active_projects[project_id]
        execution.status = 'completed'
        execution.completed_at = datetime.now()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Zakończenie projektu: {execution.project_name}")
        logger.info(f"{'='*70}")
        
        # Post-mortem analysis
        post_mortem = None
        if perform_post_mortem:
            logger.info("\n📊 Post-mortem Analysis...")
            
            project_data = {
                'planned_duration_days': execution.plan.estimated_duration_days,
                'actual_duration_days': (execution.completed_at - execution.started_at).days,
                'planned_cost': execution.plan.total_cost_estimate,
                'actual_cost': execution.plan.total_cost_estimate * 1.05,
                'completion_rate': 1.0,
                'test_coverage': 0.85,
                'code_quality_score': 0.8,
                'team_size': len(execution.team),
                'tasks_completed': len(execution.completed_tasks)
            }
            
            post_mortem = self.post_mortem.analyze_project(
                project_id=project_id,
                project_name=execution.project_name,
                project_data=project_data
            )
            
            execution.post_mortem_analysis = post_mortem
            execution.lessons_learned = post_mortem.lessons_learned
            
            logger.info(f"   ✓ Quality Score: {post_mortem.quality_score:.2f}")
            logger.info(f"   ✓ Insights: {len(post_mortem.insights)}")
            logger.info(f"   ✓ Lessons learned: {len(post_mortem.lessons_learned)}")
        
        # Przenieś do completed
        self.completed_projects.append(execution)
        del self.active_projects[project_id]
        
        logger.info(f"\n✅ Projekt zakończony!")
        logger.info(f"{'='*70}\n")
        
        return post_mortem
    
    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Pobierz status projektu"""
        if project_id in self.active_projects:
            execution = self.active_projects[project_id]
        else:
            execution = next(
                (p for p in self.completed_projects if p.project_id == project_id),
                None
            )
        
        if not execution:
            return None
        
        return {
            'project_id': execution.project_id,
            'project_name': execution.project_name,
            'status': execution.status,
            'progress': execution.progress,
            'team_size': len(execution.team) if execution.team else 0,
            'total_tasks': len(execution.plan.tasks) if execution.plan else 0,
            'completed_tasks': len(execution.completed_tasks),
            'active_protocols': len(execution.active_protocols),
            'quality_gates': execution.quality_gates_status,
            'started_at': execution.started_at.isoformat() if execution.started_at else None,
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Pobierz status całego systemu"""
        return {
            'active_projects': len(self.active_projects),
            'completed_projects': len(self.completed_projects),
            'active_protocols': len(self.active_protocols),
            'agent_types_available': len(self.agent_factory.templates),
            'patterns_detected': len(self.pattern_detector.patterns),
            'antipatterns_known': len(self.antipattern_detector.known_antipatterns)
        }


def create_agent_zero_core() -> AgentZeroCore:
    """Utwórz Agent Zero Core Engine"""
    return AgentZeroCore()
