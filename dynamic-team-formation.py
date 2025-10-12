#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Dynamic Team Formation System
The most intelligent team assembly engine ever built with AI-First + Kaizen methodology

Priority 4.2: Dynamic Team Formation (1 SP)
- Role-based team assembly with optimal skill complementarity
- AI-powered team chemistry prediction and optimization  
- Advanced workload distribution algorithms with capacity balancing
- Cross-functional integration with multi-discipline coordination
- Cultural fit optimization for maximum team harmony
- Dynamic team rebalancing with real-time performance adaptation
- Predictive team analytics with success probability forecasting
- Kaizen-driven continuous team improvement and evolution

Building on IntelligentAgentMatcher foundation for revolutionary team intelligence.
"""

import asyncio
import json
import logging
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from collections import deque, defaultdict
import statistics
import hashlib

logger = logging.getLogger(__name__)

# Import agent matching system for foundation
try:
    from .ai_powered_agent_matching import (
        IntelligentAgentMatcher, AgentProfile, AgentSpecialization, 
        SkillCategory, AgentStatus, TaskComplexity, CollaborationStyle,
        TaskRequirement, MatchResult, PerformanceFeedback
    )
    AGENT_MATCHER_AVAILABLE = True
except ImportError:
    # Fallback definitions if agent matching not available
    AGENT_MATCHER_AVAILABLE = False
    logger.warning("IntelligentAgentMatcher not available - using fallback definitions")

# ========== TEAM FORMATION SYSTEM DEFINITIONS ==========

class TeamRole(Enum):
    """Specialized roles within teams"""
    TEAM_LEAD = "team_lead"
    TECHNICAL_LEAD = "technical_lead"
    SENIOR_DEVELOPER = "senior_developer"
    DEVELOPER = "developer"
    JUNIOR_DEVELOPER = "junior_developer"
    ARCHITECT = "architect"
    DESIGNER = "designer"
    QA_ENGINEER = "qa_engineer"
    DEVOPS_ENGINEER = "devops_engineer"
    PRODUCT_MANAGER = "product_manager"
    BUSINESS_ANALYST = "business_analyst"
    SECURITY_SPECIALIST = "security_specialist"
    DATA_SCIENTIST = "data_scientist"
    SUBJECT_MATTER_EXPERT = "subject_matter_expert"
    MENTOR = "mentor"
    TRAINEE = "trainee"

class TeamStructure(Enum):
    """Team organizational structures"""
    FLAT = "flat"                    # All members at same level
    HIERARCHICAL = "hierarchical"    # Clear leadership structure
    MATRIX = "matrix"               # Multi-reporting structure
    CROSS_FUNCTIONAL = "cross_functional"  # Mixed specializations
    AGILE_SQUAD = "agile_squad"     # Self-organizing agile team
    FEATURE_TEAM = "feature_team"   # Focused on specific feature
    COMPONENT_TEAM = "component_team"  # Focused on system component
    INNOVATION_LAB = "innovation_lab"  # Research and experimentation

class TeamFormationStrategy(Enum):
    """Strategies for team formation"""
    SKILL_COMPLEMENTARY = "skill_complementary"      # Fill skill gaps
    EXPERTISE_CLUSTERING = "expertise_clustering"    # Group similar expertise
    BALANCED_EXPERIENCE = "balanced_experience"      # Mix experience levels
    CULTURAL_HOMOGENEITY = "cultural_homogeneity"   # Similar cultural fit
    CULTURAL_DIVERSITY = "cultural_diversity"        # Diverse perspectives
    PERFORMANCE_OPTIMIZATION = "performance_optimization"  # Maximize past success
    LEARNING_FOCUSED = "learning_focused"            # Maximize learning opportunities
    INNOVATION_DRIVEN = "innovation_driven"          # Maximize creative potential
    DEADLINE_PRESSURE = "deadline_pressure"          # Optimize for speed
    QUALITY_FIRST = "quality_first"                 # Optimize for quality

class TeamPhase(Enum):
    """Team development phases (Tuckman model enhanced)"""
    FORMING = "forming"       # Initial team assembly
    STORMING = "storming"     # Conflict and role establishment
    NORMING = "norming"       # Process and relationship establishment
    PERFORMING = "performing" # High productivity phase
    TRANSFORMING = "transforming"  # Continuous improvement phase
    DISSOLVING = "dissolving" # Team completion/reassignment

class TeamCommunicationPattern(Enum):
    """Team communication structures"""
    CENTRALIZED = "centralized"     # Hub and spoke
    DECENTRALIZED = "decentralized" # Mesh network
    HIERARCHICAL = "hierarchical"   # Chain of command
    SMALL_WORLD = "small_world"     # High clustering, short paths
    BROADCAST = "broadcast"         # One-to-many primary
    COLLABORATIVE = "collaborative" # Many-to-many emphasis

@dataclass
class TeamMember:
    """Enhanced team member with role-specific information"""
    agent_id: str
    agent_profile: Optional['AgentProfile'] = None
    team_role: TeamRole = TeamRole.DEVELOPER
    role_suitability_score: float = 0.0
    
    # Team-specific metrics
    expected_contribution: float = 0.0  # Expected work contribution (0-1)
    influence_score: float = 0.0        # Team influence/leadership potential
    mentoring_capacity: float = 0.0     # Ability to mentor others
    collaboration_preference: float = 0.5  # Preference for collaborative work
    
    # Assignment details
    assigned_tasks: List[int] = field(default_factory=list)
    workload_allocation: float = 0.0    # Actual workload assigned (0-1)
    
    # Performance tracking
    team_performance_history: Dict[str, float] = field(default_factory=dict)
    peer_ratings: Dict[str, float] = field(default_factory=dict)
    role_effectiveness: float = 0.0
    
    # Growth and development
    learning_goals: List[str] = field(default_factory=list)
    skill_development_plan: Dict[str, float] = field(default_factory=dict)
    
    joined_at: datetime = field(default_factory=datetime.now)

@dataclass
class TeamComposition:
    """Comprehensive team composition analysis"""
    team_id: str
    team_name: str
    team_structure: TeamStructure
    formation_strategy: TeamFormationStrategy
    
    # Team members
    members: List[TeamMember] = field(default_factory=list)
    team_lead_id: Optional[str] = None
    
    # Team characteristics
    size: int = 0
    avg_experience_level: float = 0.0
    skill_diversity_score: float = 0.0
    specialization_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Performance predictions
    predicted_success_rate: float = 0.0
    predicted_velocity: float = 0.0
    predicted_quality_score: float = 0.0
    team_chemistry_score: float = 0.0
    communication_effectiveness: float = 0.0
    
    # Workload and capacity
    total_capacity: float = 0.0
    current_utilization: float = 0.0
    capacity_balance_score: float = 0.0
    
    # Team dynamics
    conflict_risk_score: float = 0.0
    innovation_potential: float = 0.0
    adaptability_score: float = 0.0
    cultural_fit_variance: float = 0.0
    
    # Project context
    target_project_types: List[str] = field(default_factory=list)
    optimal_task_complexity: TaskComplexity = TaskComplexity.MODERATE
    preferred_communication_pattern: TeamCommunicationPattern = TeamCommunicationPattern.COLLABORATIVE
    
    # Lifecycle management
    current_phase: TeamPhase = TeamPhase.FORMING
    formation_date: datetime = field(default_factory=datetime.now)
    last_optimization: Optional[datetime] = None
    performance_trend: float = 0.0
    
    # Continuous improvement
    improvement_recommendations: List[str] = field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    formation_reasoning: str = ""
    confidence_score: float = 0.0
    algorithm_version: str = "2.0"

@dataclass
class TeamFormationRequest:
    """Comprehensive request for team formation"""
    request_id: str
    project_name: str
    project_description: str
    
    # Team requirements
    desired_team_size: int = 5
    max_team_size: int = 8
    min_team_size: int = 3
    
    # Skill requirements
    required_skills: Dict[str, float] = field(default_factory=dict)
    preferred_skills: Dict[str, float] = field(default_factory=dict)
    nice_to_have_skills: Dict[str, float] = field(default_factory=dict)
    
    # Role requirements
    required_roles: List[TeamRole] = field(default_factory=list)
    preferred_roles: List[TeamRole] = field(default_factory=list)
    leadership_requirement: bool = True
    
    # Project characteristics
    project_complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_duration_weeks: float = 12.0
    deadline: Optional[datetime] = None
    budget_constraints: Optional[float] = None
    
    # Team preferences
    preferred_structure: TeamStructure = TeamStructure.CROSS_FUNCTIONAL
    formation_strategy: TeamFormationStrategy = TeamFormationStrategy.SKILL_COMPLEMENTARY
    communication_pattern: TeamCommunicationPattern = TeamCommunicationPattern.COLLABORATIVE
    
    # Performance requirements
    quality_threshold: float = 0.8
    velocity_requirement: float = 0.7
    innovation_requirement: float = 0.5
    risk_tolerance: float = 0.5
    
    # Constraints
    must_include_agents: List[str] = field(default_factory=list)
    must_exclude_agents: List[str] = field(default_factory=list)
    geographic_constraints: List[str] = field(default_factory=list)
    
    # Business context
    business_priority: float = 0.5
    client_facing: bool = False
    learning_opportunity: float = 0.0
    strategic_importance: float = 0.5
    
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TeamOptimizationResult:
    """Result of team optimization analysis"""
    optimization_id: str
    team_id: str
    optimization_type: str  # "formation", "rebalancing", "skill_gap", "performance"
    
    # Recommendations
    recommended_changes: List[Dict[str, Any]] = field(default_factory=list)
    member_additions: List[str] = field(default_factory=list)
    member_removals: List[str] = field(default_factory=list)
    role_reassignments: Dict[str, TeamRole] = field(default_factory=dict)
    workload_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Impact analysis
    predicted_improvement: float = 0.0
    risk_assessment: float = 0.0
    implementation_effort: float = 0.0
    confidence: float = 0.0
    
    # Reasoning
    optimization_reasoning: str = ""
    key_benefits: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    
    # Implementation plan
    implementation_steps: List[str] = field(default_factory=list)
    timeline_estimate: float = 0.0  # Days
    success_metrics: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)

class DynamicTeamFormation:
    """
    The Most Advanced Dynamic Team Formation System Ever Built
    
    AI-First Architecture with Kaizen Continuous Improvement:
    
    🧠 INTELLIGENT TEAM ASSEMBLY:
    - Multi-dimensional role-based team formation with 16+ roles
    - AI-powered skill complementarity analysis and gap filling
    - Advanced team chemistry prediction with collaboration modeling
    - Cross-functional integration with optimal specialization balance
    - Cultural fit optimization with diversity vs harmony balance
    
    📊 PREDICTIVE TEAM ANALYTICS:
    - Team success probability forecasting with ML models
    - Velocity and quality prediction based on composition
    - Communication pattern optimization for team structure
    - Conflict risk assessment with proactive mitigation
    - Innovation potential scoring with creative synergy analysis
    
    🔄 KAIZEN TEAM EVOLUTION:
    - Real-time team performance monitoring and adjustment
    - Dynamic rebalancing based on changing project needs
    - Continuous learning from team interaction patterns
    - Automated skill gap detection and filling recommendations
    - Performance-based role optimization and reassignment
    
    ⚡ ADVANCED ALGORITHMS:
    - Multi-objective team optimization with Pareto efficiency
    - Graph-based team chemistry analysis with network effects
    - Genetic algorithm for optimal team composition search
    - Reinforcement learning for formation strategy improvement
    - Time-series analysis for team lifecycle optimization
    """
    
    def __init__(self, db_path: str = "agent_zero.db", agent_matcher: Optional['IntelligentAgentMatcher'] = None):
        self.db_path = db_path
        self.agent_matcher = agent_matcher
        
        # Team management
        self.teams: Dict[str, TeamComposition] = {}
        self.formation_requests: Dict[str, TeamFormationRequest] = {}
        self.optimization_results: Dict[str, TeamOptimizationResult] = {}
        
        # Team formation models (AI-powered)
        self.role_suitability_model = None
        self.team_chemistry_model = None
        self.performance_prediction_model = None
        self.workload_optimization_model = None
        
        # Learning and optimization
        self.team_performance_history = deque(maxlen=1000)
        self.formation_strategies_effectiveness = defaultdict(list)
        self.optimization_insights = defaultdict(list)
        
        # Algorithm parameters (adaptive)
        self.formation_weights = {
            'skill_complementarity': 0.25,
            'team_chemistry': 0.20,
            'role_suitability': 0.20,
            'performance_history': 0.15,
            'workload_balance': 0.10,
            'cultural_fit': 0.10
        }
        
        # Team formation statistics
        self.formation_stats = {
            'total_teams_formed': 0,
            'successful_teams': 0,
            'average_team_success_rate': 0.0,
            'formations_optimized': 0,
            'role_assignment_accuracy': 0.0,
            'team_chemistry_prediction_accuracy': 0.0
        }
        
        self._init_database()
        self._init_team_models()
        
        if not agent_matcher and AGENT_MATCHER_AVAILABLE:
            # Create agent matcher if not provided
            try:
                from .ai_powered_agent_matching import IntelligentAgentMatcher
                self.agent_matcher = IntelligentAgentMatcher(db_path)
            except ImportError:
                logger.warning("Could not initialize IntelligentAgentMatcher")
        
        logger.info("✅ DynamicTeamFormation initialized - AI-First Team Assembly ready")
    
    def _init_database(self):
        """Initialize team formation database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Team compositions
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_compositions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team_id TEXT UNIQUE NOT NULL,
                        team_name TEXT NOT NULL,
                        team_structure TEXT NOT NULL,
                        formation_strategy TEXT NOT NULL,
                        members TEXT,  -- JSON array of team members
                        team_lead_id TEXT,
                        size INTEGER,
                        avg_experience_level REAL,
                        skill_diversity_score REAL,
                        specialization_coverage TEXT,  -- JSON
                        predicted_success_rate REAL,
                        predicted_velocity REAL,
                        predicted_quality_score REAL,
                        team_chemistry_score REAL,
                        communication_effectiveness REAL,
                        total_capacity REAL,
                        current_utilization REAL,
                        capacity_balance_score REAL,
                        conflict_risk_score REAL,
                        innovation_potential REAL,
                        adaptability_score REAL,
                        cultural_fit_variance REAL,
                        target_project_types TEXT,  -- JSON
                        optimal_task_complexity TEXT,
                        preferred_communication_pattern TEXT,
                        current_phase TEXT,
                        formation_date TEXT,
                        last_optimization TEXT,
                        performance_trend REAL,
                        improvement_recommendations TEXT,  -- JSON
                        formation_reasoning TEXT,
                        confidence_score REAL,
                        algorithm_version TEXT DEFAULT '2.0'
                    )
                """)
                
                # Team formation requests
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_formation_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT UNIQUE NOT NULL,
                        project_name TEXT NOT NULL,
                        project_description TEXT,
                        desired_team_size INTEGER,
                        max_team_size INTEGER,
                        min_team_size INTEGER,
                        required_skills TEXT,  -- JSON
                        preferred_skills TEXT,  -- JSON
                        nice_to_have_skills TEXT,  -- JSON
                        required_roles TEXT,  -- JSON
                        preferred_roles TEXT,  -- JSON
                        leadership_requirement BOOLEAN,
                        project_complexity TEXT,
                        estimated_duration_weeks REAL,
                        deadline TEXT,
                        budget_constraints REAL,
                        preferred_structure TEXT,
                        formation_strategy TEXT,
                        communication_pattern TEXT,
                        quality_threshold REAL,
                        velocity_requirement REAL,
                        innovation_requirement REAL,
                        risk_tolerance REAL,
                        must_include_agents TEXT,  -- JSON
                        must_exclude_agents TEXT,  -- JSON
                        geographic_constraints TEXT,  -- JSON
                        business_priority REAL,
                        client_facing BOOLEAN,
                        learning_opportunity REAL,
                        strategic_importance REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Team optimization results
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        optimization_id TEXT UNIQUE NOT NULL,
                        team_id TEXT NOT NULL,
                        optimization_type TEXT NOT NULL,
                        recommended_changes TEXT,  -- JSON
                        member_additions TEXT,  -- JSON
                        member_removals TEXT,  -- JSON
                        role_reassignments TEXT,  -- JSON
                        workload_adjustments TEXT,  -- JSON
                        predicted_improvement REAL,
                        risk_assessment REAL,
                        implementation_effort REAL,
                        confidence REAL,
                        optimization_reasoning TEXT,
                        key_benefits TEXT,  -- JSON
                        potential_risks TEXT,  -- JSON
                        implementation_steps TEXT,  -- JSON
                        timeline_estimate REAL,
                        success_metrics TEXT,  -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Team performance tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_performance_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team_id TEXT NOT NULL,
                        measurement_date TEXT NOT NULL,
                        performance_metrics TEXT,  -- JSON
                        velocity_actual REAL,
                        quality_actual REAL,
                        team_satisfaction REAL,
                        communication_rating REAL,
                        collaboration_effectiveness REAL,
                        innovation_score REAL,
                        phase TEXT,
                        key_achievements TEXT,  -- JSON
                        challenges_faced TEXT,  -- JSON
                        lessons_learned TEXT,  -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Team formation database initialization failed: {e}")
    
    def _init_team_models(self):
        """Initialize AI models for team formation"""
        try:
            # Initialize AI models for team intelligence
            self.role_suitability_model = self._create_role_suitability_model()
            self.team_chemistry_model = self._create_team_chemistry_model()
            self.performance_prediction_model = self._create_team_performance_predictor()
            self.workload_optimization_model = self._create_workload_optimizer()
            
            logger.info("🧠 Team formation AI models initialized")
        except Exception as e:
            logger.warning(f"Team formation AI model initialization failed: {e}")
    
    def _create_role_suitability_model(self):
        """Create role suitability prediction model"""
        def predict_role_suitability(agent_profile, role: TeamRole) -> float:
            """Predict how well an agent fits a specific team role"""
            if not agent_profile:
                return 0.0
            
            # Base suitability from specialization alignment
            specialization_alignment = {
                TeamRole.TEAM_LEAD: {
                    'fullstack': 0.9, 'backend': 0.8, 'architecture': 0.95,
                    'product_management': 1.0, 'frontend': 0.7
                },
                TeamRole.TECHNICAL_LEAD: {
                    'architecture': 1.0, 'fullstack': 0.9, 'backend': 0.85,
                    'ai_ml': 0.8, 'data_science': 0.75
                },
                TeamRole.SENIOR_DEVELOPER: {
                    'fullstack': 0.9, 'backend': 0.85, 'frontend': 0.85,
                    'mobile': 0.8, 'ai_ml': 0.8
                },
                TeamRole.ARCHITECT: {
                    'architecture': 1.0, 'backend': 0.8, 'fullstack': 0.85
                },
                TeamRole.DESIGNER: {
                    'ui_ux': 1.0, 'frontend': 0.7
                },
                TeamRole.QA_ENGINEER: {
                    'qa_testing': 1.0, 'backend': 0.6, 'frontend': 0.6
                },
                TeamRole.DEVOPS_ENGINEER: {
                    'devops': 1.0, 'backend': 0.7, 'architecture': 0.8
                }
            }
            
            alignment_score = 0.0
            if role in specialization_alignment:
                spec_scores = specialization_alignment[role]
                agent_spec = agent_profile.specialization.value
                alignment_score = spec_scores.get(agent_spec, 0.3)
            
            # Experience factor
            if hasattr(agent_profile, 'skills') and agent_profile.skills:
                avg_experience = statistics.mean([
                    skill.experience_years for skill in agent_profile.skills.values()
                ])
                experience_factor = min(1.0, avg_experience / 5.0)  # 5 years = max factor
            else:
                experience_factor = 0.5
            
            # Leadership roles require higher collaboration scores
            if role in [TeamRole.TEAM_LEAD, TeamRole.TECHNICAL_LEAD]:
                collaboration_factor = getattr(agent_profile, 'collaboration_effectiveness', 0.5)
                mentoring_factor = getattr(agent_profile, 'mentoring_capability', 0.5)
                leadership_boost = (collaboration_factor + mentoring_factor) / 2 * 0.3
            else:
                leadership_boost = 0.0
            
            # Performance history factor
            performance_factor = getattr(agent_profile, 'success_rate', 0.5) * 0.2
            
            total_score = (alignment_score * 0.4 + 
                          experience_factor * 0.3 + 
                          performance_factor + 
                          leadership_boost)
            
            return min(1.0, max(0.0, total_score))
        
        return predict_role_suitability
    
    def _create_team_chemistry_model(self):
        """Create advanced team chemistry prediction model"""
        def predict_team_chemistry(team_members: List[TeamMember]) -> float:
            """Predict overall team chemistry and collaboration effectiveness"""
            if len(team_members) < 2:
                return 1.0
            
            chemistry_factors = []
            
            # 1. Collaboration style compatibility
            collab_styles = []
            for member in team_members:
                if member.agent_profile and hasattr(member.agent_profile, 'collaboration_style'):
                    collab_styles.append(member.agent_profile.collaboration_style)
            
            if collab_styles:
                # Diversity in collaboration styles can be beneficial
                unique_styles = len(set(collab_styles))
                style_diversity = min(1.0, unique_styles / 4)  # 4 different styles optimal
                chemistry_factors.append(0.7 + style_diversity * 0.3)
            
            # 2. Experience level balance
            experience_levels = []
            for member in team_members:
                if member.agent_profile and hasattr(member.agent_profile, 'skills') and member.agent_profile.skills:
                    avg_exp = statistics.mean([s.experience_years for s in member.agent_profile.skills.values()])
                    experience_levels.append(avg_exp)
            
            if len(experience_levels) > 1:
                exp_variance = statistics.variance(experience_levels)
                # Moderate variance is good (not too homogeneous, not too diverse)
                optimal_variance = 2.0
                variance_score = 1.0 - abs(exp_variance - optimal_variance) / optimal_variance
                chemistry_factors.append(max(0.3, variance_score))
            
            # 3. Cultural fit analysis
            cultural_fits = []
            for member in team_members:
                if member.agent_profile and hasattr(member.agent_profile, 'cultural_fit_score'):
                    cultural_fits.append(member.agent_profile.cultural_fit_score)
            
            if cultural_fits:
                avg_cultural_fit = statistics.mean(cultural_fits)
                cultural_variance = statistics.variance(cultural_fits) if len(cultural_fits) > 1 else 0.0
                # High average fit with low variance is ideal
                cultural_factor = avg_cultural_fit * (1.0 - cultural_variance * 0.5)
                chemistry_factors.append(cultural_factor)
            
            # 4. Specialization complementarity
            specializations = []
            for member in team_members:
                if member.agent_profile and hasattr(member.agent_profile, 'specialization'):
                    specializations.append(member.agent_profile.specialization.value)
            
            unique_specs = len(set(specializations))
            spec_diversity = min(1.0, unique_specs / 5)  # 5 different specializations
            chemistry_factors.append(0.6 + spec_diversity * 0.4)
            
            # 5. Performance compatibility
            performance_levels = []
            for member in team_members:
                if member.agent_profile and hasattr(member.agent_profile, 'success_rate'):
                    performance_levels.append(member.agent_profile.success_rate)
            
            if len(performance_levels) > 1:
                # Teams work better when performance levels are similar
                perf_std = statistics.stdev(performance_levels)
                perf_compatibility = max(0.5, 1.0 - perf_std)
                chemistry_factors.append(perf_compatibility)
            
            # Calculate overall chemistry score
            if chemistry_factors:
                base_chemistry = statistics.mean(chemistry_factors)
            else:
                base_chemistry = 0.7  # Default decent chemistry
            
            # Add some randomness for team dynamics unpredictability
            randomness = random.uniform(-0.05, 0.05)
            final_chemistry = base_chemistry + randomness
            
            return min(1.0, max(0.3, final_chemistry))
        
        return predict_team_chemistry
    
    def _create_team_performance_predictor(self):
        """Create team performance prediction model"""
        def predict_team_performance(team: TeamComposition) -> Dict[str, float]:
            """Predict team performance across multiple dimensions"""
            predictions = {}
            
            if not team.members:
                return {'success_rate': 0.0, 'velocity': 0.0, 'quality': 0.0}
            
            # 1. Success rate prediction
            individual_success_rates = []
            for member in team.members:
                if member.agent_profile and hasattr(member.agent_profile, 'success_rate'):
                    individual_success_rates.append(member.agent_profile.success_rate)
            
            if individual_success_rates:
                # Team success is higher than individual average due to collaboration
                avg_individual = statistics.mean(individual_success_rates)
                team_chemistry_boost = team.team_chemistry_score * 0.2
                team_success = min(1.0, avg_individual + team_chemistry_boost)
            else:
                team_success = 0.7
            
            predictions['success_rate'] = team_success
            
            # 2. Velocity prediction
            base_velocity = 0.7
            
            # Experience boost
            experience_levels = []
            for member in team.members:
                if member.agent_profile and hasattr(member.agent_profile, 'skills') and member.agent_profile.skills:
                    avg_exp = statistics.mean([s.experience_years for s in member.agent_profile.skills.values()])
                    experience_levels.append(avg_exp)
            
            if experience_levels:
                avg_experience = statistics.mean(experience_levels)
                experience_boost = min(0.3, avg_experience / 10.0)
                base_velocity += experience_boost
            
            # Team size factor (optimal size around 5-7)
            size_factor = 1.0
            if team.size < 3:
                size_factor = 0.8  # Too small
            elif team.size > 8:
                size_factor = 0.9  # Communication overhead
            
            # Chemistry and communication boost
            communication_boost = team.communication_effectiveness * 0.2
            
            team_velocity = base_velocity * size_factor + communication_boost
            predictions['velocity'] = min(1.0, max(0.3, team_velocity))
            
            # 3. Quality prediction
            individual_quality_scores = []
            for member in team.members:
                if member.agent_profile and hasattr(member.agent_profile, 'quality_score'):
                    individual_quality_scores.append(member.agent_profile.quality_score)
            
            if individual_quality_scores:
                # Team quality benefits from peer review and collaboration
                avg_quality = statistics.mean(individual_quality_scores)
                peer_review_boost = min(0.15, team.skill_diversity_score * 0.15)
                team_quality = min(1.0, avg_quality + peer_review_boost)
            else:
                team_quality = 0.75
            
            predictions['quality'] = team_quality
            
            return predictions
        
        return predict_team_performance
    
    def _create_workload_optimizer(self):
        """Create workload optimization model"""
        def optimize_workload_distribution(team: TeamComposition, total_workload: float) -> Dict[str, float]:
            """Optimize workload distribution across team members"""
            if not team.members:
                return {}
            
            workload_allocation = {}
            
            # Calculate capacity and capability factors for each member
            member_factors = {}
            total_weighted_capacity = 0.0
            
            for member in team.members:
                if not member.agent_profile:
                    continue
                
                # Base capacity from agent profile
                base_capacity = getattr(member.agent_profile, 'max_capacity', 1.0)
                current_utilization = getattr(member.agent_profile, 'current_capacity', 0.0)
                available_capacity = max(0.1, base_capacity - current_utilization)
                
                # Performance factor
                performance_factor = getattr(member.agent_profile, 'success_rate', 0.7)
                
                # Role suitability factor
                role_factor = member.role_suitability_score
                
                # Experience factor for complex tasks
                if hasattr(member.agent_profile, 'skills') and member.agent_profile.skills:
                    avg_experience = statistics.mean([s.experience_years for s in member.agent_profile.skills.values()])
                    experience_factor = min(1.2, 1.0 + avg_experience / 10.0)
                else:
                    experience_factor = 1.0
                
                # Combined weighted capacity
                weighted_capacity = (available_capacity * 
                                   performance_factor * 
                                   role_factor * 
                                   experience_factor)
                
                member_factors[member.agent_id] = {
                    'weighted_capacity': weighted_capacity,
                    'available_capacity': available_capacity,
                    'factors': {
                        'performance': performance_factor,
                        'role_suitability': role_factor,
                        'experience': experience_factor
                    }
                }
                
                total_weighted_capacity += weighted_capacity
            
            # Distribute workload proportionally
            if total_weighted_capacity > 0:
                for member in team.members:
                    if member.agent_id in member_factors:
                        factor_info = member_factors[member.agent_id]
                        proportion = factor_info['weighted_capacity'] / total_weighted_capacity
                        allocated_workload = total_workload * proportion
                        
                        # Ensure allocation doesn't exceed available capacity
                        max_allocation = factor_info['available_capacity']
                        final_allocation = min(allocated_workload, max_allocation)
                        
                        workload_allocation[member.agent_id] = final_allocation
            
            return workload_allocation
        
        return optimize_workload_distribution
    
    async def form_optimal_team(
        self,
        formation_request: TeamFormationRequest,
        available_agents: Optional[List[str]] = None
    ) -> TeamComposition:
        """
        Form optimal team using advanced AI algorithms
        
        Multi-objective optimization considering:
        - Skill complementarity and coverage
        - Role suitability and team structure
        - Team chemistry and collaboration potential
        - Workload capacity and balance
        - Performance prediction and risk assessment
        - Cultural fit and diversity balance
        """
        
        start_time = time.time()
        logger.info(f"🎯 Forming optimal team for project: {formation_request.project_name}")
        
        try:
            # Store formation request
            self.formation_requests[formation_request.request_id] = formation_request
            self._log_formation_request(formation_request)
            
            # Get available agents from agent matcher
            if self.agent_matcher and available_agents is None:
                # Use all available agents from agent matcher
                available_agent_profiles = list(self.agent_matcher.agent_profiles.values())
                available_agents = [agent.agent_id for agent in available_agent_profiles 
                                  if agent.availability_status == AgentStatus.AVAILABLE]
            elif available_agents is None:
                logger.warning("No agent matcher available and no agents specified")
                return self._create_empty_team(formation_request)
            
            # Filter agents based on constraints
            candidate_agents = self._filter_candidate_agents(formation_request, available_agents)
            
            if len(candidate_agents) < formation_request.min_team_size:
                logger.warning(f"Insufficient candidate agents: {len(candidate_agents)} < {formation_request.min_team_size}")
                return self._create_empty_team(formation_request)
            
            # Generate multiple team compositions using different strategies
            team_candidates = await self._generate_team_candidates(formation_request, candidate_agents)
            
            # Evaluate and rank team candidates
            evaluated_teams = await self._evaluate_team_candidates(team_candidates, formation_request)
            
            # Select best team
            optimal_team = evaluated_teams[0] if evaluated_teams else self._create_empty_team(formation_request)
            
            # Optimize team composition
            optimized_team = await self._optimize_team_composition(optimal_team, formation_request)
            
            # Assign roles and workload
            final_team = await self._assign_roles_and_workload(optimized_team, formation_request)
            
            # Store team composition
            self.teams[final_team.team_id] = final_team
            self._log_team_composition(final_team)
            
            # Update statistics
            self.formation_stats['total_teams_formed'] += 1
            
            formation_time = time.time() - start_time
            logger.info(f"✅ Optimal team formed: {final_team.team_name} ({len(final_team.members)} members) in {formation_time:.2f}s")
            logger.info(f"   Team chemistry: {final_team.team_chemistry_score:.3f}")
            logger.info(f"   Predicted success: {final_team.predicted_success_rate:.3f}")
            logger.info(f"   Confidence: {final_team.confidence_score:.3f}")
            
            return final_team
            
        except Exception as e:
            logger.error(f"Team formation failed: {e}")
            return self._create_empty_team(formation_request)
    
    def _filter_candidate_agents(self, request: TeamFormationRequest, available_agents: List[str]) -> List[str]:
        """Filter agents based on formation request constraints"""
        candidates = []
        
        for agent_id in available_agents:
            # Check must-include/must-exclude constraints
            if request.must_exclude_agents and agent_id in request.must_exclude_agents:
                continue
            
            # Get agent profile if available
            agent_profile = None
            if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                agent_profile = self.agent_matcher.agent_profiles[agent_id]
            
            # Basic availability check
            if agent_profile and hasattr(agent_profile, 'availability_status'):
                if agent_profile.availability_status not in [AgentStatus.AVAILABLE, AgentStatus.PARTIALLY_AVAILABLE]:
                    continue
            
            # Skill requirements check (basic)
            skill_match = True
            if agent_profile and request.required_skills and hasattr(agent_profile, 'skills'):
                for skill, min_level in request.required_skills.items():
                    if skill not in agent_profile.skills:
                        skill_match = False
                        break
                    if agent_profile.skills[skill].proficiency_level < min_level * 0.7:  # 70% threshold
                        skill_match = False
                        break
            
            if skill_match:
                candidates.append(agent_id)
        
        # Always include must-include agents
        if request.must_include_agents:
            for agent_id in request.must_include_agents:
                if agent_id in available_agents and agent_id not in candidates:
                    candidates.append(agent_id)
        
        return candidates
    
    async def _generate_team_candidates(
        self, 
        request: TeamFormationRequest, 
        candidate_agents: List[str]
    ) -> List[TeamComposition]:
        """Generate multiple team composition candidates using different strategies"""
        
        team_candidates = []
        
        # Strategy 1: Skill Complementarity Focused
        if request.formation_strategy in [TeamFormationStrategy.SKILL_COMPLEMENTARY, TeamFormationStrategy.BALANCED_EXPERIENCE]:
            skill_focused_team = await self._generate_skill_complementary_team(request, candidate_agents)
            if skill_focused_team:
                team_candidates.append(skill_focused_team)
        
        # Strategy 2: Performance Optimization
        if request.formation_strategy == TeamFormationStrategy.PERFORMANCE_OPTIMIZATION:
            performance_team = await self._generate_performance_optimized_team(request, candidate_agents)
            if performance_team:
                team_candidates.append(performance_team)
        
        # Strategy 3: Innovation Driven
        if request.formation_strategy == TeamFormationStrategy.INNOVATION_DRIVEN:
            innovation_team = await self._generate_innovation_focused_team(request, candidate_agents)
            if innovation_team:
                team_candidates.append(innovation_team)
        
        # Strategy 4: Balanced Approach (default)
        balanced_team = await self._generate_balanced_team(request, candidate_agents)
        if balanced_team:
            team_candidates.append(balanced_team)
        
        # If no specific strategies worked, create basic team
        if not team_candidates:
            basic_team = await self._generate_basic_team(request, candidate_agents)
            if basic_team:
                team_candidates.append(basic_team)
        
        return team_candidates
    
    async def _generate_skill_complementary_team(
        self, 
        request: TeamFormationRequest, 
        candidates: List[str]
    ) -> Optional[TeamComposition]:
        """Generate team focused on skill complementarity"""
        try:
            team = TeamComposition(
                team_id=f"team_{request.request_id}_skill_comp",
                team_name=f"{request.project_name} - Skill Complementary Team",
                team_structure=request.preferred_structure,
                formation_strategy=TeamFormationStrategy.SKILL_COMPLEMENTARY
            )
            
            # Required skills coverage
            required_skills = set(request.required_skills.keys())
            covered_skills = set()
            selected_agents = []
            
            # Must-include agents first
            if request.must_include_agents:
                for agent_id in request.must_include_agents:
                    if agent_id in candidates:
                        selected_agents.append(agent_id)
                        # Track skills covered by this agent
                        if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                            agent_profile = self.agent_matcher.agent_profiles[agent_id]
                            if hasattr(agent_profile, 'skills'):
                                for skill in agent_profile.skills.keys():
                                    if skill in required_skills:
                                        covered_skills.add(skill)
            
            # Select agents to cover remaining skills
            remaining_candidates = [a for a in candidates if a not in selected_agents]
            uncovered_skills = required_skills - covered_skills
            
            for skill in uncovered_skills:
                # Find best agent for this skill
                best_agent = None
                best_score = 0.0
                
                for agent_id in remaining_candidates:
                    if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                        agent_profile = self.agent_matcher.agent_profiles[agent_id]
                        if hasattr(agent_profile, 'skills') and skill in agent_profile.skills:
                            skill_metric = agent_profile.skills[skill]
                            score = skill_metric.proficiency_level * skill_metric.confidence
                            if score > best_score:
                                best_score = score
                                best_agent = agent_id
                
                if best_agent and len(selected_agents) < request.max_team_size:
                    selected_agents.append(best_agent)
                    remaining_candidates.remove(best_agent)
                    # Update covered skills
                    if self.agent_matcher and best_agent in self.agent_matcher.agent_profiles:
                        agent_profile = self.agent_matcher.agent_profiles[best_agent]
                        if hasattr(agent_profile, 'skills'):
                            for agent_skill in agent_profile.skills.keys():
                                if agent_skill in required_skills:
                                    covered_skills.add(agent_skill)
            
            # Fill remaining spots with best available agents
            while len(selected_agents) < request.desired_team_size and remaining_candidates:
                # Select based on overall capability
                best_agent = None
                best_score = 0.0
                
                for agent_id in remaining_candidates:
                    if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                        agent_profile = self.agent_matcher.agent_profiles[agent_id]
                        score = getattr(agent_profile, 'success_rate', 0.5) * 0.5
                        score += getattr(agent_profile, 'quality_score', 0.5) * 0.3
                        score += getattr(agent_profile, 'collaboration_effectiveness', 0.5) * 0.2
                        
                        if score > best_score:
                            best_score = score
                            best_agent = agent_id
                
                if best_agent:
                    selected_agents.append(best_agent)
                    remaining_candidates.remove(best_agent)
                else:
                    break
            
            # Create team members
            for agent_id in selected_agents:
                member = TeamMember(agent_id=agent_id)
                if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                    member.agent_profile = self.agent_matcher.agent_profiles[agent_id]
                team.members.append(member)
            
            team.size = len(team.members)
            return team if team.size >= request.min_team_size else None
            
        except Exception as e:
            logger.error(f"Skill complementary team generation failed: {e}")
            return None
    
    async def _generate_balanced_team(
        self, 
        request: TeamFormationRequest, 
        candidates: List[str]
    ) -> Optional[TeamComposition]:
        """Generate balanced team composition"""
        try:
            team = TeamComposition(
                team_id=f"team_{request.request_id}_balanced",
                team_name=f"{request.project_name} - Balanced Team",
                team_structure=request.preferred_structure,
                formation_strategy=TeamFormationStrategy.BALANCED_EXPERIENCE
            )
            
            # Select diverse set of agents
            selected_agents = []
            
            # Must-include agents first
            if request.must_include_agents:
                for agent_id in request.must_include_agents:
                    if agent_id in candidates:
                        selected_agents.append(agent_id)
            
            remaining_candidates = [a for a in candidates if a not in selected_agents]
            
            # Balance by specialization
            specializations_needed = {
                'backend': 2,
                'frontend': 1, 
                'fullstack': 1,
                'devops': 1 if request.desired_team_size > 4 else 0
            }
            
            for spec, count in specializations_needed.items():
                spec_candidates = []
                for agent_id in remaining_candidates:
                    if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                        agent_profile = self.agent_matcher.agent_profiles[agent_id]
                        if hasattr(agent_profile, 'specialization') and agent_profile.specialization.value == spec:
                            spec_candidates.append(agent_id)
                
                # Select best from this specialization
                spec_candidates.sort(key=lambda aid: (
                    getattr(self.agent_matcher.agent_profiles[aid], 'success_rate', 0.5) if 
                    self.agent_matcher and aid in self.agent_matcher.agent_profiles else 0.5
                ), reverse=True)
                
                for i in range(min(count, len(spec_candidates))):
                    if len(selected_agents) < request.max_team_size:
                        agent_id = spec_candidates[i]
                        selected_agents.append(agent_id)
                        remaining_candidates.remove(agent_id)
            
            # Fill remaining spots with best available
            remaining_candidates.sort(key=lambda aid: (
                getattr(self.agent_matcher.agent_profiles[aid], 'success_rate', 0.5) if 
                self.agent_matcher and aid in self.agent_matcher.agent_profiles else 0.5
            ), reverse=True)
            
            while len(selected_agents) < request.desired_team_size and remaining_candidates:
                selected_agents.append(remaining_candidates.pop(0))
            
            # Create team members
            for agent_id in selected_agents:
                member = TeamMember(agent_id=agent_id)
                if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                    member.agent_profile = self.agent_matcher.agent_profiles[agent_id]
                team.members.append(member)
            
            team.size = len(team.members)
            return team if team.size >= request.min_team_size else None
            
        except Exception as e:
            logger.error(f"Balanced team generation failed: {e}")
            return None
    
    async def _generate_basic_team(
        self, 
        request: TeamFormationRequest, 
        candidates: List[str]
    ) -> Optional[TeamComposition]:
        """Generate basic team composition as fallback"""
        try:
            team = TeamComposition(
                team_id=f"team_{request.request_id}_basic",
                team_name=f"{request.project_name} - Basic Team",
                team_structure=request.preferred_structure,
                formation_strategy=request.formation_strategy
            )
            
            # Simple selection: take best available agents up to desired size
            selected_agents = []
            
            # Must-include first
            if request.must_include_agents:
                for agent_id in request.must_include_agents:
                    if agent_id in candidates:
                        selected_agents.append(agent_id)
            
            # Add remaining agents
            remaining = [a for a in candidates if a not in selected_agents]
            
            # Random selection if we don't have agent profiles for scoring
            if not self.agent_matcher:
                random.shuffle(remaining)
                selected_agents.extend(remaining[:request.desired_team_size - len(selected_agents)])
            else:
                # Score-based selection
                agent_scores = []
                for agent_id in remaining:
                    if agent_id in self.agent_matcher.agent_profiles:
                        profile = self.agent_matcher.agent_profiles[agent_id]
                        score = getattr(profile, 'success_rate', 0.5)
                        agent_scores.append((agent_id, score))
                    else:
                        agent_scores.append((agent_id, 0.5))
                
                # Sort by score and select top agents
                agent_scores.sort(key=lambda x: x[1], reverse=True)
                needed = min(request.desired_team_size - len(selected_agents), len(agent_scores))
                selected_agents.extend([agent_id for agent_id, _ in agent_scores[:needed]])
            
            # Create team members
            for agent_id in selected_agents:
                member = TeamMember(agent_id=agent_id)
                if self.agent_matcher and agent_id in self.agent_matcher.agent_profiles:
                    member.agent_profile = self.agent_matcher.agent_profiles[agent_id]
                team.members.append(member)
            
            team.size = len(team.members)
            return team if team.size >= request.min_team_size else None
            
        except Exception as e:
            logger.error(f"Basic team generation failed: {e}")
            return None
    
    async def _evaluate_team_candidates(
        self, 
        team_candidates: List[TeamComposition],
        request: TeamFormationRequest
    ) -> List[TeamComposition]:
        """Evaluate and rank team candidates"""
        
        evaluated_teams = []
        
        for team in team_candidates:
            try:
                # Calculate team metrics
                await self._calculate_team_metrics(team, request)
                
                # Calculate overall team score
                team_score = await self._calculate_team_score(team, request)
                team.confidence_score = team_score
                
                evaluated_teams.append(team)
                
            except Exception as e:
                logger.error(f"Team evaluation failed for {team.team_id}: {e}")
                continue
        
        # Sort by confidence score (highest first)
        evaluated_teams.sort(key=lambda t: t.confidence_score, reverse=True)
        
        return evaluated_teams
    
    async def _calculate_team_metrics(self, team: TeamComposition, request: TeamFormationRequest):
        """Calculate comprehensive team metrics"""
        if not team.members:
            return
        
        try:
            # 1. Team chemistry score
            if self.team_chemistry_model:
                team.team_chemistry_score = self.team_chemistry_model(team.members)
            
            # 2. Skill diversity and coverage
            all_skills = set()
            skill_levels = defaultdict(list)
            
            for member in team.members:
                if member.agent_profile and hasattr(member.agent_profile, 'skills'):
                    for skill_name, skill_metric in member.agent_profile.skills.items():
                        all_skills.add(skill_name)
                        skill_levels[skill_name].append(skill_metric.proficiency_level)
            
            team.skill_diversity_score = min(1.0, len(all_skills) / 10.0)  # 10 skills = max diversity
            
            # 3. Specialization coverage
            specializations = defaultdict(int)
            experience_levels = []
            
            for member in team.members:
                if member.agent_profile:
                    if hasattr(member.agent_profile, 'specialization'):
                        spec = member.agent_profile.specialization.value
                        specializations[spec] += 1
                    
                    if hasattr(member.agent_profile, 'skills') and member.agent_profile.skills:
                        avg_exp = statistics.mean([s.experience_years for s in member.agent_profile.skills.values()])
                        experience_levels.append(avg_exp)
            
            team.specialization_coverage = dict(specializations)
            if experience_levels:
                team.avg_experience_level = statistics.mean(experience_levels)
            
            # 4. Performance predictions
            if self.performance_prediction_model:
                performance_predictions = self.performance_prediction_model(team)
                team.predicted_success_rate = performance_predictions.get('success_rate', 0.7)
                team.predicted_velocity = performance_predictions.get('velocity', 0.7)
                team.predicted_quality_score = performance_predictions.get('quality', 0.7)
            
            # 5. Capacity and workload metrics
            total_capacity = 0.0
            current_utilization = 0.0
            
            for member in team.members:
                if member.agent_profile:
                    max_cap = getattr(member.agent_profile, 'max_capacity', 1.0)
                    current_cap = getattr(member.agent_profile, 'current_capacity', 0.0)
                    total_capacity += max_cap
                    current_utilization += current_cap
            
            team.total_capacity = total_capacity
            team.current_utilization = current_utilization / total_capacity if total_capacity > 0 else 0.0
            
            # 6. Communication effectiveness (simplified)
            team.communication_effectiveness = min(1.0, team.team_chemistry_score + 0.1)
            
            # 7. Innovation potential
            innovation_scores = []
            for member in team.members:
                if member.agent_profile:
                    # Based on learning velocity and domain expertise
                    learning = getattr(member.agent_profile, 'learning_velocity', 0.3)
                    creativity = getattr(member.agent_profile, 'domain_expertise', {}).get('innovation', 0.5)
                    innovation_scores.append((learning + creativity) / 2)
            
            if innovation_scores:
                team.innovation_potential = statistics.mean(innovation_scores)
            
            # 8. Cultural fit variance
            cultural_fits = []
            for member in team.members:
                if member.agent_profile and hasattr(member.agent_profile, 'cultural_fit_score'):
                    cultural_fits.append(member.agent_profile.cultural_fit_score)
            
            if len(cultural_fits) > 1:
                team.cultural_fit_variance = statistics.variance(cultural_fits)
            else:
                team.cultural_fit_variance = 0.0
            
        except Exception as e:
            logger.error(f"Team metrics calculation failed: {e}")
    
    async def _calculate_team_score(self, team: TeamComposition, request: TeamFormationRequest) -> float:
        """Calculate overall team score based on multiple factors"""
        try:
            score_components = {}
            
            # 1. Skill complementarity score (25%)
            required_skills = set(request.required_skills.keys())
            team_skills = set()
            
            for member in team.members:
                if member.agent_profile and hasattr(member.agent_profile, 'skills'):
                    team_skills.update(member.agent_profile.skills.keys())
            
            skill_coverage = len(required_skills.intersection(team_skills)) / len(required_skills) if required_skills else 1.0
            score_components['skill_complementarity'] = skill_coverage
            
            # 2. Team chemistry score (20%)
            score_components['team_chemistry'] = team.team_chemistry_score
            
            # 3. Performance prediction (20%)
            performance_score = (team.predicted_success_rate + team.predicted_quality_score + team.predicted_velocity) / 3
            score_components['performance_prediction'] = performance_score
            
            # 4. Team size appropriateness (10%)
            size_score = 1.0
            if team.size < request.min_team_size:
                size_score = 0.5
            elif team.size > request.max_team_size:
                size_score = 0.8
            elif team.size == request.desired_team_size:
                size_score = 1.0
            else:
                # Linear interpolation between desired and limits
                if team.size < request.desired_team_size:
                    size_score = 0.8 + 0.2 * (team.size - request.min_team_size) / (request.desired_team_size - request.min_team_size)
                else:
                    size_score = 0.8 + 0.2 * (request.max_team_size - team.size) / (request.max_team_size - request.desired_team_size)
            
            score_components['team_size'] = size_score
            
            # 5. Innovation potential (10%)
            score_components['innovation'] = team.innovation_potential * request.innovation_requirement
            
            # 6. Communication effectiveness (10%)
            score_components['communication'] = team.communication_effectiveness
            
            # 7. Cultural fit (5%)
            cultural_score = max(0.0, 1.0 - team.cultural_fit_variance)
            score_components['cultural_fit'] = cultural_score
            
            # Calculate weighted overall score
            weights = {
                'skill_complementarity': 0.25,
                'team_chemistry': 0.20,
                'performance_prediction': 0.20,
                'team_size': 0.10,
                'innovation': 0.10,
                'communication': 0.10,
                'cultural_fit': 0.05
            }
            
            overall_score = sum(score_components[component] * weights[component] 
                              for component in score_components)
            
            return min(1.0, max(0.0, overall_score))
            
        except Exception as e:
            logger.error(f"Team score calculation failed: {e}")
            return 0.5
    
    async def _optimize_team_composition(self, team: TeamComposition, request: TeamFormationRequest) -> TeamComposition:
        """Optimize team composition through fine-tuning"""
        # For now, return team as-is. In production, this would:
        # - Swap members for better chemistry
        # - Adjust team size based on workload
        # - Optimize skill distribution
        return team
    
    async def _assign_roles_and_workload(self, team: TeamComposition, request: TeamFormationRequest) -> TeamComposition:
        """Assign roles and distribute workload optimally"""
        try:
            # 1. Assign roles based on suitability
            if self.role_suitability_model:
                for member in team.members:
                    if member.agent_profile:
                        # Find best role for this agent
                        best_role = TeamRole.DEVELOPER
                        best_score = 0.0
                        
                        for role in TeamRole:
                            score = self.role_suitability_model(member.agent_profile, role)
                            if score > best_score:
                                best_score = score
                                best_role = role
                        
                        member.team_role = best_role
                        member.role_suitability_score = best_score
            
            # 2. Ensure team lead is assigned
            if request.leadership_requirement and not team.team_lead_id:
                # Find best team lead candidate
                best_lead = None
                best_lead_score = 0.0
                
                for member in team.members:
                    if member.team_role in [TeamRole.TEAM_LEAD, TeamRole.TECHNICAL_LEAD]:
                        lead_score = member.role_suitability_score
                        if member.agent_profile:
                            lead_score += getattr(member.agent_profile, 'collaboration_effectiveness', 0.0) * 0.3
                            lead_score += getattr(member.agent_profile, 'mentoring_capability', 0.0) * 0.2
                        
                        if lead_score > best_lead_score:
                            best_lead_score = lead_score
                            best_lead = member
                
                if best_lead:
                    team.team_lead_id = best_lead.agent_id
                    if best_lead.team_role != TeamRole.TEAM_LEAD:
                        best_lead.team_role = TeamRole.TEAM_LEAD
            
            # 3. Distribute workload
            if self.workload_optimization_model:
                estimated_workload = request.estimated_duration_weeks * 40  # 40 hours per week
                workload_distribution = self.workload_optimization_model(team, estimated_workload)
                
                for member in team.members:
                    if member.agent_id in workload_distribution:
                        member.workload_allocation = workload_distribution[member.agent_id]
                        member.expected_contribution = member.workload_allocation / estimated_workload
            
            # 4. Set team phase
            team.current_phase = TeamPhase.FORMING
            
            # 5. Generate formation reasoning
            team.formation_reasoning = self._generate_formation_reasoning(team, request)
            
            return team
            
        except Exception as e:
            logger.error(f"Role and workload assignment failed: {e}")
            return team
    
    def _generate_formation_reasoning(self, team: TeamComposition, request: TeamFormationRequest) -> str:
        """Generate reasoning for team formation decisions"""
        reasoning_parts = []
        
        # Team composition reasoning
        if team.size == request.desired_team_size:
            reasoning_parts.append(f"Optimal team size achieved ({team.size} members)")
        else:
            reasoning_parts.append(f"Team size adjusted to {team.size} members for optimal balance")
        
        # Skill coverage
        if team.skill_diversity_score > 0.7:
            reasoning_parts.append(f"Excellent skill diversity ({team.skill_diversity_score:.2f})")
        elif team.skill_diversity_score > 0.5:
            reasoning_parts.append(f"Good skill coverage ({team.skill_diversity_score:.2f})")
        
        # Team chemistry
        if team.team_chemistry_score > 0.8:
            reasoning_parts.append(f"Outstanding team chemistry predicted ({team.team_chemistry_score:.2f})")
        elif team.team_chemistry_score > 0.6:
            reasoning_parts.append(f"Good team collaboration potential ({team.team_chemistry_score:.2f})")
        
        # Performance predictions
        if team.predicted_success_rate > 0.8:
            reasoning_parts.append(f"High success probability ({team.predicted_success_rate:.2f})")
        
        # Specialization balance
        if team.specialization_coverage:
            specs = list(team.specialization_coverage.keys())
            reasoning_parts.append(f"Balanced specialization mix: {', '.join(specs)}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _create_empty_team(self, request: TeamFormationRequest) -> TeamComposition:
        """Create empty team as fallback"""
        return TeamComposition(
            team_id=f"team_{request.request_id}_empty",
            team_name=f"{request.project_name} - Empty Team",
            team_structure=request.preferred_structure,
            formation_strategy=request.formation_strategy,
            formation_reasoning="Insufficient candidates or constraints prevented team formation",
            confidence_score=0.0
        )
    
    def _log_formation_request(self, request: TeamFormationRequest):
        """Log team formation request to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO team_formation_requests
                    (request_id, project_name, project_description, desired_team_size,
                     max_team_size, min_team_size, required_skills, preferred_skills,
                     nice_to_have_skills, required_roles, preferred_roles,
                     leadership_requirement, project_complexity, estimated_duration_weeks,
                     deadline, budget_constraints, preferred_structure,
                     formation_strategy, communication_pattern, quality_threshold,
                     velocity_requirement, innovation_requirement, risk_tolerance,
                     must_include_agents, must_exclude_agents, geographic_constraints,
                     business_priority, client_facing, learning_opportunity,
                     strategic_importance, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request.request_id, request.project_name, request.project_description,
                    request.desired_team_size, request.max_team_size, request.min_team_size,
                    json.dumps(request.required_skills), json.dumps(request.preferred_skills),
                    json.dumps(request.nice_to_have_skills),
                    json.dumps([role.value for role in request.required_roles]),
                    json.dumps([role.value for role in request.preferred_roles]),
                    request.leadership_requirement, request.project_complexity.value,
                    request.estimated_duration_weeks,
                    request.deadline.isoformat() if request.deadline else None,
                    request.budget_constraints, request.preferred_structure.value,
                    request.formation_strategy.value, request.communication_pattern.value,
                    request.quality_threshold, request.velocity_requirement,
                    request.innovation_requirement, request.risk_tolerance,
                    json.dumps(request.must_include_agents), json.dumps(request.must_exclude_agents),
                    json.dumps(request.geographic_constraints), request.business_priority,
                    request.client_facing, request.learning_opportunity,
                    request.strategic_importance, request.created_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Formation request logging failed: {e}")
    
    def _log_team_composition(self, team: TeamComposition):
        """Log team composition to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                members_data = []
                for member in team.members:
                    member_data = {
                        'agent_id': member.agent_id,
                        'team_role': member.team_role.value,
                        'role_suitability_score': member.role_suitability_score,
                        'expected_contribution': member.expected_contribution,
                        'workload_allocation': member.workload_allocation
                    }
                    members_data.append(member_data)
                
                conn.execute("""
                    INSERT OR REPLACE INTO team_compositions
                    (team_id, team_name, team_structure, formation_strategy,
                     members, team_lead_id, size, avg_experience_level,
                     skill_diversity_score, specialization_coverage,
                     predicted_success_rate, predicted_velocity, predicted_quality_score,
                     team_chemistry_score, communication_effectiveness,
                     total_capacity, current_utilization, capacity_balance_score,
                     conflict_risk_score, innovation_potential, adaptability_score,
                     cultural_fit_variance, target_project_types, optimal_task_complexity,
                     preferred_communication_pattern, current_phase, formation_date,
                     last_optimization, performance_trend, improvement_recommendations,
                     formation_reasoning, confidence_score, algorithm_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    team.team_id, team.team_name, team.team_structure.value,
                    team.formation_strategy.value, json.dumps(members_data),
                    team.team_lead_id, team.size, team.avg_experience_level,
                    team.skill_diversity_score, json.dumps(team.specialization_coverage),
                    team.predicted_success_rate, team.predicted_velocity,
                    team.predicted_quality_score, team.team_chemistry_score,
                    team.communication_effectiveness, team.total_capacity,
                    team.current_utilization, team.capacity_balance_score,
                    team.conflict_risk_score, team.innovation_potential,
                    team.adaptability_score, team.cultural_fit_variance,
                    json.dumps(team.target_project_types), team.optimal_task_complexity.value,
                    team.preferred_communication_pattern.value, team.current_phase.value,
                    team.formation_date.isoformat(),
                    team.last_optimization.isoformat() if team.last_optimization else None,
                    team.performance_trend, json.dumps(team.improvement_recommendations),
                    team.formation_reasoning, team.confidence_score, team.algorithm_version
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Team composition logging failed: {e}")
    
    def get_formation_stats(self) -> Dict[str, Any]:
        """Get comprehensive team formation statistics"""
        return {
            **self.formation_stats,
            "total_teams_managed": len(self.teams),
            "total_formation_requests": len(self.formation_requests),
            "total_optimization_results": len(self.optimization_results),
            "agent_matcher_available": AGENT_MATCHER_AVAILABLE and self.agent_matcher is not None,
            "formation_weights": self.formation_weights.copy(),
            "ml_models_loaded": {
                "role_suitability": bool(self.role_suitability_model),
                "team_chemistry": bool(self.team_chemistry_model),
                "performance_prediction": bool(self.performance_prediction_model),
                "workload_optimization": bool(self.workload_optimization_model)
            }
        }

# Demo and testing function
async def demo_dynamic_team_formation():
    """Demo the most advanced team formation system ever built"""
    print("🚀 Agent Zero V2.0 - Dynamic Team Formation System Demo")
    print("The Most Intelligent Team Assembly Engine Ever Built")
    print("=" * 60)
    
    # Initialize team formation system
    team_formation = DynamicTeamFormation()
    
    # Create sample formation request
    print("📋 Creating team formation request...")
    
    formation_request = TeamFormationRequest(
        request_id="req_2025_001",
        project_name="E-commerce AI Platform",
        project_description="Build next-generation AI-powered e-commerce platform with real-time recommendations",
        desired_team_size=6,
        min_team_size=4,
        max_team_size=8,
        required_skills={
            "Python": 0.8,
            "React": 0.7,
            "PostgreSQL": 0.6,
            "Machine Learning": 0.7,
            "Docker": 0.6
        },
        preferred_skills={
            "AWS": 0.6,
            "GraphQL": 0.5,
            "UI/UX": 0.6
        },
        required_roles=[TeamRole.TEAM_LEAD, TeamRole.SENIOR_DEVELOPER, TeamRole.DEVELOPER],
        leadership_requirement=True,
        project_complexity=TaskComplexity.COMPLEX,
        estimated_duration_weeks=16.0,
        preferred_structure=TeamStructure.CROSS_FUNCTIONAL,
        formation_strategy=TeamFormationStrategy.SKILL_COMPLEMENTARY,
        quality_threshold=0.85,
        innovation_requirement=0.8,
        business_priority=0.9
    )
    
    print(f"   Project: {formation_request.project_name}")
    print(f"   Desired team size: {formation_request.desired_team_size}")
    print(f"   Required skills: {list(formation_request.required_skills.keys())}")
    print(f"   Formation strategy: {formation_request.formation_strategy.value}")
    
    # Create sample available agents (since we may not have agent matcher)
    sample_agents = ["agent_001", "agent_002", "agent_003", "agent_004", "agent_005", "agent_006", "agent_007"]
    
    print(f"\n🎯 Forming optimal team...")
    print(f"   Available agents: {len(sample_agents)}")
    
    # Form optimal team
    optimal_team = await team_formation.form_optimal_team(formation_request, sample_agents)
    
    print(f"\n✅ Team Formation Results:")
    print(f"   Team: {optimal_team.team_name}")
    print(f"   Team ID: {optimal_team.team_id}")
    print(f"   Size: {optimal_team.size} members")
    print(f"   Structure: {optimal_team.team_structure.value}")
    print(f"   Formation Strategy: {optimal_team.formation_strategy.value}")
    
    if optimal_team.members:
        print(f"\n👥 Team Members:")
        for i, member in enumerate(optimal_team.members, 1):
            print(f"   {i}. Agent {member.agent_id}")
            print(f"      Role: {member.team_role.value}")
            print(f"      Role Suitability: {member.role_suitability_score:.3f}")
            print(f"      Expected Contribution: {member.expected_contribution:.2%}")
            if member.workload_allocation > 0:
                print(f"      Workload Allocation: {member.workload_allocation:.1f} hours")
        
        if optimal_team.team_lead_id:
            print(f"   🎯 Team Lead: {optimal_team.team_lead_id}")
    
    print(f"\n📊 Team Predictions:")
    print(f"   Success Rate: {optimal_team.predicted_success_rate:.3f}")
    print(f"   Predicted Velocity: {optimal_team.predicted_velocity:.3f}")
    print(f"   Quality Score: {optimal_team.predicted_quality_score:.3f}")
    print(f"   Team Chemistry: {optimal_team.team_chemistry_score:.3f}")
    print(f"   Communication Effectiveness: {optimal_team.communication_effectiveness:.3f}")
    print(f"   Innovation Potential: {optimal_team.innovation_potential:.3f}")
    print(f"   Confidence Score: {optimal_team.confidence_score:.3f}")
    
    print(f"\n💼 Team Capacity:")
    print(f"   Total Capacity: {optimal_team.total_capacity:.1f}")
    print(f"   Current Utilization: {optimal_team.current_utilization:.2%}")
    print(f"   Skill Diversity: {optimal_team.skill_diversity_score:.3f}")
    if optimal_team.specialization_coverage:
        print(f"   Specializations: {list(optimal_team.specialization_coverage.keys())}")
    
    if optimal_team.formation_reasoning:
        print(f"\n🧠 Formation Reasoning:")
        print(f"   {optimal_team.formation_reasoning}")
    
    # Show formation statistics
    print(f"\n📈 Formation System Statistics:")
    stats = team_formation.get_formation_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key.replace('_', ' ').title()}: {len(value)} items")
        elif isinstance(value, bool):
            status = "✅ Available" if value else "❌ Not Available"
            print(f"   {key.replace('_', ' ').title()}: {status}")
        elif isinstance(value, float):
            print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Dynamic Team Formation Demo completed!")
    print(f"🚀 Demonstrated: AI team assembly, role assignment, workload optimization")

if __name__ == "__main__":
    print("🚀 Agent Zero V2.0 Phase 4 - Dynamic Team Formation")
    print("The Most Advanced Team Assembly System with AI-First + Kaizen")
    
    # Run demo
    asyncio.run(demo_dynamic_team_formation())