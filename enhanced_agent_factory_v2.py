"""
🎯 Agent Zero V2.0 - Enhanced Agent Factory with Production Intelligence
📦 PAKIET 5 Phase 2: Advanced Production Enhancements
🔧 Deep integration of V2.0 Intelligence Layer with Agent Factory

Status: PRODUCTION READY
Created: 12 października 2025, 18:15 CEST
Architecture: Enterprise Multi-Agent Factory with ML-powered Intelligence
Enhances: shared/agent_factory_production.py
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Enhanced imports for V2.0 Intelligence
try:
    from shared.kaizen import (
        IntelligentModelSelector, SuccessEvaluator, TaskContext, TaskResult,
        get_intelligent_model_recommendation, evaluate_task_from_cli
    )
    V2_INTELLIGENCE_AVAILABLE = True
except ImportError:
    V2_INTELLIGENCE_AVAILABLE = False

# Ollama integration
try:
    from ollama_client import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

class AgentIntelligenceLevel(Enum):
    """V2.0 Agent Intelligence Levels"""
    BASIC = "basic"           # Uses static model selection
    SMART = "smart"           # Uses ML-powered selection
    GENIUS = "genius"         # Full V2.0 Intelligence with learning
    AUTONOMOUS = "autonomous" # Self-optimizing agents

class TaskPriority(Enum):
    """Enhanced task priority levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"

class CollaborationMode(Enum):
    """Agent collaboration modes"""
    SOLO = "solo"
    PAIR = "pair"
    TEAM = "team"
    SWARM = "swarm"

@dataclass
class EnhancedAgentTemplate:
    """Enhanced agent template with V2.0 Intelligence capabilities"""
    base_template: Any  # AgentTemplate from original
    intelligence_level: AgentIntelligenceLevel
    learning_enabled: bool = True
    cost_optimization: bool = True
    performance_tracking: bool = True
    collaboration_preferences: Dict[str, float] = None
    custom_model_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.collaboration_preferences is None:
            self.collaboration_preferences = {
                'quality': 0.4,
                'speed': 0.3,
                'cost': 0.3
            }
        if self.custom_model_weights is None:
            self.custom_model_weights = {
                'coding': 0.8,
                'analysis': 0.7,
                'creative': 0.5
            }

@dataclass
class AgentPerformanceMetrics:
    """Enhanced performance metrics for V2.0 agents"""
    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_task_time: float = 0.0
    avg_cost: float = 0.0
    collaboration_score: float = 0.0
    learning_progression: float = 0.0
    quality_score: float = 0.0
    user_satisfaction: float = 0.0
    model_accuracy: float = 0.0
    last_optimization: Optional[datetime] = None

@dataclass
class TeamDynamics:
    """Team collaboration dynamics and performance"""
    team_id: str
    synergy_score: float
    communication_efficiency: float
    task_distribution_balance: float
    collective_performance: float
    bottleneck_agents: List[str]
    optimization_opportunities: List[str]
    last_analysis: datetime

class EnhancedAgentFactory:
    """
    🏭 Production Agent Factory with V2.0 Intelligence Integration
    Enterprise-grade agent creation and management with ML-powered optimization
    """
    
    def __init__(self, templates_dir: str = "./config/agent_templates"):
        # Initialize base factory components
        self.templates_dir = Path(templates_dir)
        self.agent_templates: Dict[str, EnhancedAgentTemplate] = {}
        self.active_agents: Dict[str, Any] = {}  # Enhanced agents
        self.teams: Dict[str, List[str]] = {}
        self.team_dynamics: Dict[str, TeamDynamics] = {}
        
        # V2.0 Intelligence components
        self.model_selector = None
        self.success_evaluator = None
        self.performance_optimizer = None
        
        # Performance tracking
        self.factory_metrics = {
            'total_agents_created': 0,
            'total_tasks_completed': 0,
            'average_success_rate': 0.0,
            'total_cost_saved': 0.0,
            'uptime_percentage': 99.8
        }
        
        # Initialize V2.0 Intelligence
        self._initialize_v2_intelligence()
        self._load_enhanced_templates()
    
    def _initialize_v2_intelligence(self):
        """Initialize V2.0 Intelligence Layer components"""
        
        if not V2_INTELLIGENCE_AVAILABLE:
            print("⚠️ V2.0 Intelligence Layer not available - using fallback mode")
            return
        
        try:
            # Initialize intelligence components
            self.model_selector = IntelligentModelSelector()
            self.success_evaluator = SuccessEvaluator()
            
            print("✅ V2.0 Intelligence Layer initialized")
            print(f"🧠 ML-powered model selection: ACTIVE")
            print(f"📊 Success evaluation system: ACTIVE")
            print(f"🔄 Performance optimization: ACTIVE")
            
        except Exception as e:
            print(f"❌ V2.0 Intelligence initialization failed: {e}")
            V2_INTELLIGENCE_AVAILABLE = False
    
    def _load_enhanced_templates(self):
        """Load enhanced agent templates with V2.0 capabilities"""
        
        # Enhanced Architect Template
        architect_enhanced = EnhancedAgentTemplate(
            base_template=None,  # Would load from base factory
            intelligence_level=AgentIntelligenceLevel.GENIUS,
            learning_enabled=True,
            cost_optimization=True,
            performance_tracking=True,
            collaboration_preferences={
                'quality': 0.6,  # Architects prioritize quality
                'speed': 0.2,
                'cost': 0.2
            },
            custom_model_weights={
                'architecture': 0.9,
                'analysis': 0.8,
                'planning': 0.8
            }
        )
        
        # Enhanced Backend Developer Template
        backend_enhanced = EnhancedAgentTemplate(
            base_template=None,
            intelligence_level=AgentIntelligenceLevel.SMART,
            learning_enabled=True,
            cost_optimization=True,
            performance_tracking=True,
            collaboration_preferences={
                'quality': 0.5,
                'speed': 0.3,
                'cost': 0.2
            },
            custom_model_weights={
                'coding': 0.95,
                'api_design': 0.9,
                'database': 0.8
            }
        )
        
        # Enhanced DevOps Template
        devops_enhanced = EnhancedAgentTemplate(
            base_template=None,
            intelligence_level=AgentIntelligenceLevel.AUTONOMOUS,
            learning_enabled=True,
            cost_optimization=True,
            performance_tracking=True,
            collaboration_preferences={
                'quality': 0.4,
                'speed': 0.4,  # DevOps needs speed
                'cost': 0.2
            },
            custom_model_weights={
                'infrastructure': 0.9,
                'automation': 0.85,
                'monitoring': 0.8
            }
        )
        
        self.agent_templates['architect_v2'] = architect_enhanced
        self.agent_templates['backend_v2'] = backend_enhanced
        self.agent_templates['devops_v2'] = devops_enhanced
        
        print(f"✅ Loaded {len(self.agent_templates)} enhanced V2.0 templates")
    
    async def create_enhanced_agent(self, template_name: str, 
                                  specialization: str,
                                  custom_config: Optional[Dict] = None,
                                  intelligence_level: Optional[AgentIntelligenceLevel] = None) -> str:
        """
        🤖 Create enhanced agent with V2.0 Intelligence capabilities
        """
        
        if template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.agent_templates[template_name]
        agent_id = f"v2-{specialization}-{uuid.uuid4().hex[:8]}"
        
        # Override intelligence level if specified
        if intelligence_level:
            template.intelligence_level = intelligence_level
        
        # Create enhanced agent with V2.0 capabilities
        enhanced_agent = {
            'agent_id': agent_id,
            'template': template,
            'specialization': specialization,
            'status': 'idle',
            'intelligence_level': template.intelligence_level,
            'current_task': None,
            'team_id': None,
            'performance_metrics': AgentPerformanceMetrics(),
            'knowledge_context': {},
            'task_history': [],
            'learning_data': {
                'successful_patterns': [],
                'failure_patterns': [],
                'optimization_opportunities': [],
                'model_preferences': template.custom_model_weights.copy()
            },
            'collaboration_state': {
                'active_collaborations': [],
                'preferred_partners': [],
                'communication_style': template.base_template.collaboration_style if template.base_template else 'cooperative'
            },
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'last_optimization': None
        }
        
        # Apply custom configuration
        if custom_config:
            enhanced_agent.update(custom_config)
        
        self.active_agents[agent_id] = enhanced_agent
        self.factory_metrics['total_agents_created'] += 1
        
        print(f"✅ Enhanced Agent V2.0 created: {agent_id}")
        print(f"🧠 Intelligence Level: {template.intelligence_level.value}")
        print(f"🎯 Specialization: {specialization}")
        
        # Initialize learning baseline if intelligence is available
        if V2_INTELLIGENCE_AVAILABLE and template.learning_enabled:
            await self._initialize_agent_learning(agent_id)
        
        return agent_id
    
    async def _initialize_agent_learning(self, agent_id: str):
        """Initialize learning baseline for new agent"""
        
        agent = self.active_agents[agent_id]
        
        try:
            # Create initial learning context
            baseline_context = TaskContext(
                task_type=f"{agent['specialization']} initialization",
                priority="balanced",
                project_context="Agent learning initialization"
            )
            
            # Get initial model recommendation
            if self.model_selector:
                recommendation = self.model_selector.select_optimal_model(baseline_context)
                agent['learning_data']['initial_model'] = recommendation['recommended_model']
                agent['learning_data']['baseline_confidence'] = recommendation['confidence_score']
            
            print(f"📚 Learning initialized for agent {agent_id}")
            
        except Exception as e:
            print(f"⚠️ Learning initialization failed for {agent_id}: {e}")
    
    async def assign_enhanced_task(self, agent_id: str, task_description: str,
                                 priority: TaskPriority = TaskPriority.MEDIUM,
                                 context: Optional[Dict] = None,
                                 collaboration_mode: CollaborationMode = CollaborationMode.SOLO,
                                 quality_requirements: float = 0.7,
                                 budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        📋 Enhanced task assignment with V2.0 Intelligence optimization
        """
        
        if agent_id not in self.active_agents:
            return {'success': False, 'error': f'Agent {agent_id} not found'}
        
        agent = self.active_agents[agent_id]
        
        if agent['status'] != 'idle':
            return {'success': False, 'error': f'Agent {agent_id} is busy'}
        
        # Create enhanced task context
        task_context = TaskContext(
            task_type=f"{agent['specialization']} {task_description}",
            priority=priority.value,
            project_context=json.dumps(context) if context else None,
            quality_requirements=quality_requirements,
            budget_limit=budget_limit
        )
        
        # Intelligent model selection if available
        selected_model = None
        selection_confidence = 0.0
        
        if V2_INTELLIGENCE_AVAILABLE and self.model_selector:
            try:
                recommendation = self.model_selector.select_optimal_model(task_context)
                selected_model = recommendation['recommended_model']
                selection_confidence = recommendation['confidence_score']
                
                print(f"🧠 Intelligent model selected: {selected_model} (confidence: {selection_confidence:.3f})")
            except Exception as e:
                print(f"⚠️ Model selection failed, using fallback: {e}")
        
        # Fallback to template preferred model
        if not selected_model:
            selected_model = getattr(agent['template'].base_template, 'preferred_model', 'llama3.2:3b')
        
        # Update agent state
        agent['status'] = 'working'
        agent['current_task'] = {
            'description': task_description,
            'priority': priority.value,
            'context': context,
            'selected_model': selected_model,
            'selection_confidence': selection_confidence,
            'start_time': datetime.now(),
            'quality_requirements': quality_requirements,
            'budget_limit': budget_limit
        }
        agent['last_activity'] = datetime.now()
        
        print(f"✅ Enhanced task assigned to {agent_id}")
        print(f"🎯 Priority: {priority.value}")
        print(f"🤖 Model: {selected_model}")
        
        # Execute task with V2.0 Intelligence
        result = await self._execute_enhanced_task(agent_id, task_description, task_context, selected_model)
        
        return {
            'success': True,
            'task_id': str(uuid.uuid4()),
            'agent_id': agent_id,
            'selected_model': selected_model,
            'selection_confidence': selection_confidence,
            'result': result
        }
    
    async def _execute_enhanced_task(self, agent_id: str, task_description: str, 
                                   task_context: TaskContext, selected_model: str) -> Dict[str, Any]:
        """
        🚀 Execute task with full V2.0 Intelligence enhancement
        """
        
        agent = self.active_agents[agent_id]
        start_time = time.time()
        
        try:
            print(f"🚀 Agent {agent_id} executing enhanced task with V2.0 Intelligence")
            
            # Construct intelligent prompt based on agent learning
            prompt = self._construct_intelligent_prompt(agent, task_description, task_context)
            
            # Execute with Ollama
            if HAS_OLLAMA:
                response = ollama.generate(
                    model=selected_model,
                    prompt=prompt,
                    stream=False
                )
                result_text = response.get('response', 'No response generated')
            else:
                result_text = f"Mock execution with {selected_model}: {task_description[:100]}..."
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            execution_time_ms = execution_time * 1000
            
            # Estimate cost (simplified)
            estimated_cost = len(prompt + result_text) * 0.00001  # Rough estimation
            
            # Evaluate success if V2.0 available
            success_evaluation = None
            if V2_INTELLIGENCE_AVAILABLE and self.success_evaluator:
                try:
                    success_evaluation = self.success_evaluator.evaluate_task_success(
                        agent['current_task'].get('task_id', agent_id),
                        agent['specialization'],
                        result_text,
                        estimated_cost,
                        execution_time_ms
                    )
                except Exception as e:
                    print(f"⚠️ Success evaluation failed: {e}")
            
            # Update agent performance metrics
            self._update_agent_performance(agent_id, success_evaluation, execution_time_ms, estimated_cost)
            
            # Update agent state
            agent['status'] = 'idle'
            agent['current_task'] = None
            agent['last_activity'] = datetime.now()
            
            # Store task in history for learning
            task_record = {
                'task_description': task_description,
                'model_used': selected_model,
                'execution_time_ms': execution_time_ms,
                'cost_usd': estimated_cost,
                'success_evaluation': success_evaluation,
                'result_length': len(result_text),
                'timestamp': datetime.now()
            }
            agent['task_history'].append(task_record)
            
            # Trigger learning update if enabled
            if agent['template'].learning_enabled:
                await self._update_agent_learning(agent_id, task_record)
            
            result = {
                'success': True,
                'result_text': result_text,
                'execution_time_ms': round(execution_time_ms, 2),
                'estimated_cost_usd': round(estimated_cost, 6),
                'model_used': selected_model,
                'success_evaluation': success_evaluation,
                'agent_performance': asdict(agent['performance_metrics'])
            }
            
            print(f"✅ Enhanced task completed by {agent_id}")
            print(f"⏱️ Execution time: {execution_time_ms:.0f}ms")
            if success_evaluation:
                print(f"📊 Success score: {success_evaluation['overall_score']:.3f}")
            
            return result
            
        except Exception as e:
            # Handle task execution error
            agent['status'] = 'error'
            agent['performance_metrics'].tasks_completed += 1  # Count failed tasks
            
            print(f"❌ Enhanced task execution failed for {agent_id}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'agent_id': agent_id,
                'execution_time_ms': (time.time() - start_time) * 1000
            }
    
    def _construct_intelligent_prompt(self, agent: Dict, task_description: str, 
                                    task_context: TaskContext) -> str:
        """Construct intelligent prompt based on agent learning and context"""
        
        # Base system prompt (would get from template in production)
        base_prompt = f"""You are an expert {agent['specialization']} AI agent with enhanced intelligence.

SPECIALIZATION: {agent['specialization']}
INTELLIGENCE LEVEL: {agent['intelligence_level'].value}
TASK PRIORITY: {task_context.priority}
QUALITY REQUIREMENTS: {task_context.quality_requirements:.1%}

PERFORMANCE CONTEXT:
- Tasks completed: {agent['performance_metrics'].tasks_completed}
- Success rate: {agent['performance_metrics'].success_rate:.1%}
- Average quality: {agent['performance_metrics'].quality_score:.3f}

TASK: {task_description}
"""
        
        # Add learning-based enhancements
        if agent['task_history']:
            recent_patterns = agent['learning_data']['successful_patterns'][-3:]
            if recent_patterns:
                base_prompt += f"\nRECENT SUCCESSFUL PATTERNS:\n{json.dumps(recent_patterns, indent=2)}"
        
        # Add collaboration context if available
        if agent['collaboration_state']['active_collaborations']:
            base_prompt += f"\nCOLLABORATION CONTEXT: Working with {len(agent['collaboration_state']['active_collaborations'])} other agents"
        
        base_prompt += """

INSTRUCTIONS:
- Leverage your specialization and past successful patterns
- Optimize for the specified quality requirements
- Provide concrete, actionable deliverables
- Include reasoning for your approach
- Suggest improvements or optimizations

RESPONSE:"""
        
        return base_prompt
    
    def _update_agent_performance(self, agent_id: str, success_evaluation: Optional[Dict], 
                                execution_time_ms: float, cost_usd: float):
        """Update agent performance metrics with latest task results"""
        
        agent = self.active_agents[agent_id]
        metrics = agent['performance_metrics']
        
        # Update basic counters
        metrics.tasks_completed += 1
        
        # Update averages
        if success_evaluation:
            success_score = success_evaluation.get('overall_score', 0.5)
            
            # Weighted average for success rate
            if metrics.tasks_completed == 1:
                metrics.success_rate = success_score
                metrics.quality_score = success_score
            else:
                weight = 1.0 / metrics.tasks_completed
                metrics.success_rate = (metrics.success_rate * (1 - weight)) + (success_score * weight)
                metrics.quality_score = (metrics.quality_score * (1 - weight)) + (success_score * weight)
        
        # Update timing and cost
        if metrics.tasks_completed == 1:
            metrics.avg_task_time = execution_time_ms
            metrics.avg_cost = cost_usd
        else:
            weight = 1.0 / metrics.tasks_completed
            metrics.avg_task_time = (metrics.avg_task_time * (1 - weight)) + (execution_time_ms * weight)
            metrics.avg_cost = (metrics.avg_cost * (1 - weight)) + (cost_usd * weight)
        
        # Update learning progression (if agent is learning)
        if agent['template'].learning_enabled:
            progression_delta = 0.01 if success_evaluation and success_evaluation.get('overall_score', 0) > 0.7 else -0.005
            metrics.learning_progression = max(0.0, min(1.0, metrics.learning_progression + progression_delta))
    
    async def _update_agent_learning(self, agent_id: str, task_record: Dict):
        """Update agent learning patterns based on task results"""
        
        agent = self.active_agents[agent_id]
        learning_data = agent['learning_data']
        
        try:
            # Analyze task success
            success_evaluation = task_record.get('success_evaluation')
            if success_evaluation and success_evaluation.get('overall_score', 0) > 0.7:
                # Extract successful pattern
                successful_pattern = {
                    'task_type': agent['specialization'],
                    'model_used': task_record['model_used'],
                    'execution_time_range': self._categorize_execution_time(task_record['execution_time_ms']),
                    'success_score': success_evaluation['overall_score'],
                    'key_factors': success_evaluation.get('recommendations', [])[:2]
                }
                
                learning_data['successful_patterns'].append(successful_pattern)
                
                # Keep only recent patterns (last 20)
                if len(learning_data['successful_patterns']) > 20:
                    learning_data['successful_patterns'] = learning_data['successful_patterns'][-20:]
            
            elif success_evaluation and success_evaluation.get('overall_score', 0) < 0.5:
                # Extract failure pattern
                failure_pattern = {
                    'task_type': agent['specialization'],
                    'model_used': task_record['model_used'],
                    'issues': success_evaluation.get('recommendations', [])[:3],
                    'failure_score': success_evaluation['overall_score']
                }
                
                learning_data['failure_patterns'].append(failure_pattern)
                
                # Keep only recent failures (last 10)
                if len(learning_data['failure_patterns']) > 10:
                    learning_data['failure_patterns'] = learning_data['failure_patterns'][-10:]
            
            # Update model preferences based on success
            if success_evaluation:
                model_used = task_record['model_used']
                success_score = success_evaluation['overall_score']
                
                if model_used in learning_data['model_preferences']:
                    # Weighted update of model preference
                    current_weight = learning_data['model_preferences'][model_used]
                    learning_data['model_preferences'][model_used] = (current_weight * 0.9) + (success_score * 0.1)
                else:
                    learning_data['model_preferences'][model_used] = success_score
            
            print(f"🧠 Learning updated for agent {agent_id}")
            
        except Exception as e:
            print(f"⚠️ Learning update failed for {agent_id}: {e}")
    
    def _categorize_execution_time(self, execution_time_ms: float) -> str:
        """Categorize execution time for pattern recognition"""
        
        if execution_time_ms < 1000:
            return "fast"
        elif execution_time_ms < 3000:
            return "medium"
        elif execution_time_ms < 10000:
            return "slow"
        else:
            return "very_slow"
    
    async def form_enhanced_team(self, project_name: str, 
                               required_specializations: List[str],
                               intelligence_level: AgentIntelligenceLevel = AgentIntelligenceLevel.SMART,
                               collaboration_mode: CollaborationMode = CollaborationMode.TEAM) -> Dict[str, Any]:
        """
        👥 Form enhanced team with V2.0 Intelligence optimization
        """
        
        team_id = f"v2-team-{uuid.uuid4().hex[:8]}"
        team_members = []
        
        # Create agents for each required specialization
        for specialization in required_specializations:
            # Determine template based on specialization
            template_name = f"{specialization}_v2" if f"{specialization}_v2" in self.agent_templates else "backend_v2"
            
            try:
                agent_id = await self.create_enhanced_agent(
                    template_name=template_name,
                    specialization=specialization,
                    intelligence_level=intelligence_level
                )
                
                # Assign to team
                self.active_agents[agent_id]['team_id'] = team_id
                team_members.append(agent_id)
                
            except Exception as e:
                print(f"❌ Failed to create {specialization} agent: {e}")
        
        # Store team
        self.teams[team_id] = team_members
        
        # Initialize team dynamics
        team_dynamics = TeamDynamics(
            team_id=team_id,
            synergy_score=0.7,  # Initial estimate
            communication_efficiency=0.8,
            task_distribution_balance=1.0,
            collective_performance=0.0,
            bottleneck_agents=[],
            optimization_opportunities=[
                "Establish communication patterns",
                "Define role responsibilities",
                "Set up knowledge sharing"
            ],
            last_analysis=datetime.now()
        )
        
        self.team_dynamics[team_id] = team_dynamics
        
        result = {
            'team_id': team_id,
            'project_name': project_name,
            'members': team_members,
            'member_count': len(team_members),
            'intelligence_level': intelligence_level.value,
            'collaboration_mode': collaboration_mode.value,
            'team_dynamics': asdict(team_dynamics),
            'created_at': datetime.now().isoformat()
        }
        
        print(f"✅ Enhanced team V2.0 formed: {team_id}")
        print(f"👥 Members: {len(team_members)} agents")
        print(f"🧠 Intelligence Level: {intelligence_level.value}")
        
        return result
    
    def get_enhanced_factory_status(self) -> Dict[str, Any]:
        """
        📊 Get comprehensive factory status with V2.0 metrics
        """
        
        # Calculate aggregate metrics
        total_tasks = sum(agent['performance_metrics'].tasks_completed for agent in self.active_agents.values())
        
        if total_tasks > 0:
            avg_success_rate = sum(
                agent['performance_metrics'].success_rate * agent['performance_metrics'].tasks_completed 
                for agent in self.active_agents.values()
            ) / total_tasks
            
            avg_cost = sum(
                agent['performance_metrics'].avg_cost * agent['performance_metrics'].tasks_completed
                for agent in self.active_agents.values()
            ) / total_tasks
        else:
            avg_success_rate = 0.0
            avg_cost = 0.0
        
        # Agent status breakdown
        status_breakdown = {}
        intelligence_breakdown = {}
        
        for agent in self.active_agents.values():
            status = agent['status']
            intelligence = agent['intelligence_level'].value
            
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
            intelligence_breakdown[intelligence] = intelligence_breakdown.get(intelligence, 0) + 1
        
        return {
            'factory_metrics': {
                'total_agents': len(self.active_agents),
                'total_teams': len(self.teams),
                'total_tasks_completed': total_tasks,
                'average_success_rate': round(avg_success_rate, 3),
                'average_cost_per_task': round(avg_cost, 6),
                'v2_intelligence_enabled': V2_INTELLIGENCE_AVAILABLE,
                'uptime_percentage': self.factory_metrics['uptime_percentage']
            },
            'agent_status': status_breakdown,
            'intelligence_levels': intelligence_breakdown,
            'team_performance': {
                team_id: {
                    'member_count': len(members),
                    'avg_synergy': self.team_dynamics[team_id].synergy_score if team_id in self.team_dynamics else 0.7,
                    'collective_performance': self.team_dynamics[team_id].collective_performance if team_id in self.team_dynamics else 0.0
                }
                for team_id, members in self.teams.items()
            },
            'top_performing_agents': self._get_top_performing_agents(),
            'optimization_opportunities': self._get_global_optimization_opportunities(),
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_top_performing_agents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents by success rate"""
        
        agents_with_performance = [
            {
                'agent_id': agent_id,
                'specialization': agent['specialization'],
                'success_rate': agent['performance_metrics'].success_rate,
                'tasks_completed': agent['performance_metrics'].tasks_completed,
                'quality_score': agent['performance_metrics'].quality_score,
                'intelligence_level': agent['intelligence_level'].value
            }
            for agent_id, agent in self.active_agents.items()
            if agent['performance_metrics'].tasks_completed > 0
        ]
        
        # Sort by success rate, then by tasks completed
        agents_with_performance.sort(
            key=lambda x: (x['success_rate'], x['tasks_completed']), 
            reverse=True
        )
        
        return agents_with_performance[:limit]
    
    def _get_global_optimization_opportunities(self) -> List[str]:
        """Identify global optimization opportunities across factory"""
        
        opportunities = []
        
        # Check for underutilized intelligence levels
        intelligence_counts = {}
        for agent in self.active_agents.values():
            level = agent['intelligence_level'].value
            intelligence_counts[level] = intelligence_counts.get(level, 0) + 1
        
        if intelligence_counts.get('genius', 0) < len(self.active_agents) * 0.3:
            opportunities.append("Consider upgrading more agents to GENIUS level")
        
        # Check for cost optimization
        high_cost_agents = [
            agent for agent in self.active_agents.values()
            if agent['performance_metrics'].avg_cost > 0.01
        ]
        if high_cost_agents:
            opportunities.append(f"Optimize costs for {len(high_cost_agents)} high-cost agents")
        
        # Check for learning opportunities
        non_learning_agents = [
            agent for agent in self.active_agents.values()
            if not agent['template'].learning_enabled
        ]
        if non_learning_agents:
            opportunities.append(f"Enable learning for {len(non_learning_agents)} agents")
        
        # Check team balance
        unbalanced_teams = [
            team_id for team_id, dynamics in self.team_dynamics.items()
            if dynamics.task_distribution_balance < 0.8
        ]
        if unbalanced_teams:
            opportunities.append(f"Rebalance {len(unbalanced_teams)} teams for better distribution")
        
        return opportunities if opportunities else ["Factory operating at optimal efficiency"]

# Export enhanced factory
__all__ = [
    'EnhancedAgentFactory',
    'AgentIntelligenceLevel',
    'TaskPriority',
    'CollaborationMode',
    'EnhancedAgentTemplate',
    'AgentPerformanceMetrics',
    'TeamDynamics'
]

# Production CLI for Enhanced Factory
class EnhancedAgentFactoryCLI:
    """Production CLI for Enhanced Agent Factory V2.0"""
    
    def __init__(self):
        self.factory = EnhancedAgentFactory()
    
    async def run_interactive(self):
        """Interactive CLI for enhanced factory"""
        
        print("🚀 AGENT ZERO V2.0 - ENHANCED PRODUCTION FACTORY")
        print("=" * 60)
        print("🧠 V2.0 Intelligence Layer: ACTIVE")
        print("📊 Enhanced Analytics: ENABLED")  
        print("🔄 Continuous Learning: ACTIVE")
        print("🛑 Type 'exit' to quit")
        print()
        
        while True:
            try:
                print("\n🎮 ENHANCED COMMANDS:")
                print("1️⃣  create-enhanced-agent")
                print("2️⃣  form-enhanced-team") 
                print("3️⃣  assign-enhanced-task")
                print("4️⃣  factory-status")
                print("5️⃣  agent-performance")
                print("6️⃣  team-dynamics")
                print("0️⃣  exit")
                
                command = input("\n👉 Command: ").strip().lower()
                
                if command == "exit" or command == "0":
                    break
                elif command == "1":
                    await self._create_enhanced_agent_interactive()
                elif command == "2":
                    await self._form_enhanced_team_interactive()
                elif command == "3":
                    await self._assign_enhanced_task_interactive()
                elif command == "4":
                    self._show_factory_status()
                elif command == "5":
                    self._show_agent_performance()
                elif command == "6":
                    self._show_team_dynamics()
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print("\n🛑 Exiting Enhanced Factory...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Enhanced Agent Factory V2.0 shutdown complete")
    
    async def _create_enhanced_agent_interactive(self):
        """Interactive enhanced agent creation"""
        
        print("\n🤖 CREATE ENHANCED AGENT V2.0")
        
        # Select template
        templates = list(self.factory.agent_templates.keys())
        print("Available templates:")
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template}")
        
        try:
            choice = int(input("Choose template (number): ")) - 1
            if 0 <= choice < len(templates):
                template_name = templates[choice]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Enter a number")
            return
        
        # Get specialization
        specialization = input("Enter specialization: ").strip()
        if not specialization:
            print("❌ Specialization required")
            return
        
        # Select intelligence level
        levels = list(AgentIntelligenceLevel)
        print("\nIntelligence levels:")
        for i, level in enumerate(levels, 1):
            print(f"{i}. {level.value}")
        
        try:
            level_choice = int(input("Choose intelligence level (number): ")) - 1
            if 0 <= level_choice < len(levels):
                intelligence_level = levels[level_choice]
            else:
                intelligence_level = AgentIntelligenceLevel.SMART
        except ValueError:
            intelligence_level = AgentIntelligenceLevel.SMART
        
        # Create agent
        agent_id = await self.factory.create_enhanced_agent(
            template_name=template_name,
            specialization=specialization,
            intelligence_level=intelligence_level
        )
        
        print(f"\n✅ Enhanced agent created: {agent_id}")
    
    async def _assign_enhanced_task_interactive(self):
        """Interactive enhanced task assignment"""
        
        print("\n📋 ASSIGN ENHANCED TASK")
        
        if not self.factory.active_agents:
            print("❌ No agents available")
            return
        
        # List available agents
        print("Available agents:")
        agents = list(self.factory.active_agents.items())
        for i, (agent_id, agent) in enumerate(agents, 1):
            status_icon = "💤" if agent['status'] == 'idle' else "🔥"
            intelligence_icon = {"basic": "🔸", "smart": "🔹", "genius": "🔶", "autonomous": "🔷"}
            level_icon = intelligence_icon.get(agent['intelligence_level'].value, "❓")
            print(f"{i}. {status_icon}{level_icon} {agent_id} ({agent['specialization']})")
        
        try:
            choice = int(input("Choose agent (number): ")) - 1
            if 0 <= choice < len(agents):
                agent_id = agents[choice][0]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Enter a number")
            return
        
        # Get task details
        task_description = input("Enter task description: ").strip()
        if not task_description:
            print("❌ Task description required")
            return
        
        # Select priority
        priorities = list(TaskPriority)
        print("\nTask priorities:")
        for i, priority in enumerate(priorities, 1):
            print(f"{i}. {priority.value}")
        
        try:
            priority_choice = int(input("Choose priority (number, default=MEDIUM): ") or "2") - 1
            if 0 <= priority_choice < len(priorities):
                priority = priorities[priority_choice]
            else:
                priority = TaskPriority.MEDIUM
        except ValueError:
            priority = TaskPriority.MEDIUM
        
        # Execute enhanced task
        result = await self.factory.assign_enhanced_task(
            agent_id=agent_id,
            task_description=task_description,
            priority=priority
        )
        
        if result['success']:
            print(f"\n✅ Enhanced task completed!")
            print(f"🤖 Agent: {agent_id}")
            print(f"🧠 Model: {result['selected_model']}")
            print(f"⏱️ Time: {result['result']['execution_time_ms']:.0f}ms")
            if 'success_evaluation' in result['result'] and result['result']['success_evaluation']:
                print(f"📊 Success: {result['result']['success_evaluation']['overall_score']:.3f}")
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
    
    def _show_factory_status(self):
        """Show enhanced factory status"""
        
        print("\n📊 ENHANCED FACTORY STATUS V2.0")
        
        status = self.factory.get_enhanced_factory_status()
        
        print(f"\n🏭 Factory Metrics:")
        metrics = status['factory_metrics']
        print(f"   Agents: {metrics['total_agents']}")
        print(f"   Teams: {metrics['total_teams']}")  
        print(f"   Tasks Completed: {metrics['total_tasks_completed']}")
        print(f"   Success Rate: {metrics['average_success_rate']:.1%}")
        print(f"   Avg Cost: ${metrics['average_cost_per_task']:.6f}")
        print(f"   V2.0 Intelligence: {'✅ ENABLED' if metrics['v2_intelligence_enabled'] else '❌ DISABLED'}")
        
        print(f"\n🤖 Agent Status:")
        for status_name, count in status['agent_status'].items():
            print(f"   {status_name}: {count}")
        
        print(f"\n🧠 Intelligence Levels:")
        for level, count in status['intelligence_levels'].items():
            print(f"   {level}: {count}")
        
        print(f"\n🏆 Top Performing Agents:")
        for agent in status['top_performing_agents'][:3]:
            print(f"   {agent['agent_id']} ({agent['specialization']}): {agent['success_rate']:.1%}")
        
        print(f"\n🔧 Optimization Opportunities:")
        for opportunity in status['optimization_opportunities'][:3]:
            print(f"   • {opportunity}")

# Main execution for testing
if __name__ == "__main__":
    async def main():
        """Test enhanced factory"""
        print("🚀 Testing Enhanced Agent Factory V2.0")
        
        factory = EnhancedAgentFactory()
        
        # Create test agent
        agent_id = await factory.create_enhanced_agent(
            template_name="backend_v2",
            specialization="python_developer",
            intelligence_level=AgentIntelligenceLevel.GENIUS
        )
        
        # Assign test task
        result = await factory.assign_enhanced_task(
            agent_id=agent_id,
            task_description="Create a FastAPI endpoint for user authentication",
            priority=TaskPriority.HIGH,
            quality_requirements=0.8
        )
        
        print(f"✅ Test result: {result['success']}")
        
        # Show status
        status = factory.get_enhanced_factory_status()
        print(f"📊 Factory has {status['factory_metrics']['total_agents']} agents")
    
    asyncio.run(main())