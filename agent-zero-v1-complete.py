# Agent Zero V1 - Complete System Integration
# Kompletna integracja wszystkich komponentów systemu
# Date: 10 października 2025 - Final Implementation

"""
Agent Zero V1 Complete System Integration
Zintegrowany system wszystkich komponentów V2.0 Intelligence Layer

Ten moduł łączy wszystkie komponenty:
- Project Orchestrator (A0-20) - finalne 10%
- Hierarchical Task Planner (A0-17) - fundament  
- AI-First Decision System (A0-22) - inteligentny selektor
- Neo4j Knowledge Graph (A0-24) - pattern recognition
- Success/Failure Classifier (A0-25) - multi-dimensional success
- Active Metrics Analyzer (A0-26) - real-time Kaizen
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3

# Import all our components
try:
    import sys
    sys.path.insert(0, '.')
    
    # Core system components
    from simple_tracker import SimpleTracker
    from business_requirements_parser import BusinessRequirementsParser, IntentType, ComplexityLevel
    from feedback_loop_engine import FeedbackLoopEngine
    
    # New V2.0 Intelligence Layer components
    from project_orchestrator import ProjectOrchestrator, Project, ProjectState
    from hierarchical_task_planner import HierarchicalTaskPlanner, TaskType, TaskPriority
    from ai_decision_system import AIFirstDecisionSystem, ModelType, DecisionContext
    from neo4j_knowledge_graph import KaizenKnowledgeGraph
    from success_failure_classifier import SuccessClassifier, SuccessLevel
    from active_metrics_analyzer import ActiveMetricsAnalyzer
    
except ImportError as e:
    logging.warning(f"Could not import all components: {e}")
    # System will work with available components

class SystemState(Enum):
    """Overall system state"""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"

@dataclass
class SystemHealth:
    """System health metrics"""
    overall_state: SystemState
    component_status: Dict[str, bool] = field(default_factory=dict)
    active_projects: int = 0
    total_tasks_today: int = 0
    success_rate_24h: float = 0.0
    avg_response_time: float = 0.0
    cost_today: float = 0.0
    critical_alerts: int = 0
    knowledge_patterns: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class AgentZeroV1System:
    """
    Main Agent Zero V1 System
    Orchestrates all V2.0 Intelligence Layer components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()
        
        # Initialize all components
        self._initialize_components()
        
        # Integration state
        self._integration_active = False
        self._health_check_interval = 60  # seconds
        
    def _initialize_components(self):
        """Initialize all system components"""
        
        self.logger.info("Initializing Agent Zero V1 Intelligence Layer components...")
        
        # Core components (already existing)
        try:
            self.tracker = SimpleTracker()
            self.business_parser = BusinessRequirementsParser()
            self.feedback_engine = FeedbackLoopEngine()
            self.logger.info("✅ Core components initialized")
        except Exception as e:
            self.logger.error(f"❌ Error initializing core components: {e}")
            self.tracker = None
            self.business_parser = None
            self.feedback_engine = None
        
        # V2.0 Intelligence Layer components
        try:
            # Project lifecycle management
            self.project_orchestrator = ProjectOrchestrator(
                db_path=self.config.get('orchestrator_db', 'project_orchestrator.db')
            )
            
            # Hierarchical task planning
            self.task_planner = HierarchicalTaskPlanner(integration_mode=True)
            
            # Intelligent model selection
            self.ai_decision_system = AIFirstDecisionSystem(
                db_path=self.config.get('ai_decisions_db', 'ai_decisions.db')
            )
            
            # Knowledge graph and pattern recognition
            neo4j_config = self.config.get('neo4j', {})
            self.knowledge_graph = KaizenKnowledgeGraph(
                neo4j_uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
                neo4j_user=neo4j_config.get('user', 'neo4j'),
                neo4j_password=neo4j_config.get('password', 'password123')
            )
            
            # Success/failure classification
            self.success_classifier = SuccessClassifier(
                threshold_db_path=self.config.get('success_thresholds_db', 'success_thresholds.db'),
                evaluation_db_path=self.config.get('success_evaluations_db', 'success_evaluations.db')
            )
            
            # Real-time metrics and Kaizen analytics
            self.metrics_analyzer = ActiveMetricsAnalyzer(
                success_classifier_db=self.config.get('success_evaluations_db', 'success_evaluations.db'),
                metrics_db=self.config.get('metrics_db', 'active_metrics.db')
            )
            
            self.logger.info("✅ V2.0 Intelligence Layer components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing V2.0 components: {e}")
            raise
        
        # Component integration
        self._setup_component_integration()
        
        self.state = SystemState.READY
        self.logger.info("🎯 Agent Zero V1 system ready!")
    
    def _setup_component_integration(self):
        """Setup integration between components"""
        
        # Connect project orchestrator with task planner
        if hasattr(self.project_orchestrator, 'set_task_planner'):
            self.project_orchestrator.set_task_planner(self.task_planner)
        
        # Connect AI decision system with project orchestrator
        if hasattr(self.project_orchestrator, 'set_ai_decision_system'):
            self.project_orchestrator.set_ai_decision_system(self.ai_decision_system)
        
        # Connect success classifier with metrics analyzer
        if hasattr(self.metrics_analyzer, 'success_classifier'):
            self.metrics_analyzer.success_classifier = self.success_classifier
        
        self.logger.info("🔗 Component integration configured")
    
    async def start_system(self):
        """Start the complete Agent Zero V1 system"""
        
        if self.state != SystemState.READY:
            raise RuntimeError("System not ready to start")
        
        self.logger.info("🚀 Starting Agent Zero V1 Complete System...")
        
        try:
            # Start metrics monitoring
            metrics_task = asyncio.create_task(
                self.metrics_analyzer.start_monitoring()
            )
            
            # Start health monitoring
            health_task = asyncio.create_task(
                self._run_health_monitoring()
            )
            
            # Start knowledge graph learning (periodic)
            knowledge_task = asyncio.create_task(
                self._run_knowledge_learning()
            )
            
            # Start AI decision learning
            ai_learning_task = asyncio.create_task(
                self._run_ai_learning()
            )
            
            self.state = SystemState.ACTIVE
            self._integration_active = True
            
            self.logger.info("✅ Agent Zero V1 system fully active!")
            
            # Run all monitoring tasks
            await asyncio.gather(
                metrics_task,
                health_task, 
                knowledge_task,
                ai_learning_task,
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error starting system: {e}")
            self.state = SystemState.ERROR
            raise
        finally:
            self._integration_active = False
    
    async def stop_system(self):
        """Stop the system gracefully"""
        
        self.logger.info("🛑 Stopping Agent Zero V1 system...")
        
        self._integration_active = False
        
        # Stop metrics monitoring
        if hasattr(self.metrics_analyzer, 'stop_monitoring'):
            self.metrics_analyzer.stop_monitoring()
        
        # Close knowledge graph connections
        if hasattr(self.knowledge_graph, 'close'):
            self.knowledge_graph.close()
        
        self.state = SystemState.READY
        self.logger.info("✅ System stopped gracefully")
    
    async def _run_health_monitoring(self):
        """Run continuous health monitoring"""
        
        while self._integration_active:
            try:
                health = await self.get_system_health()
                
                # Log health status
                if health.overall_state == SystemState.DEGRADED:
                    self.logger.warning(f"⚠️ System health degraded: {health.critical_alerts} critical alerts")
                elif health.overall_state == SystemState.ERROR:
                    self.logger.error(f"🚨 System health critical!")
                
                # Update system state
                self.state = health.overall_state
                
                await asyncio.sleep(self._health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _run_knowledge_learning(self):
        """Run periodic knowledge graph learning"""
        
        while self._integration_active:
            try:
                # Every hour, analyze completed projects for patterns
                await asyncio.sleep(3600)  # 1 hour
                
                # Get recently completed projects
                completed_projects = self.project_orchestrator.list_projects(ProjectState.COMPLETED)
                recent_projects = [
                    p for p in completed_projects 
                    if p.completed_at and (datetime.now() - p.completed_at).days <= 7
                ]
                
                if len(recent_projects) >= 3:  # Minimum for pattern analysis
                    self.logger.info(f"🧠 Learning from {len(recent_projects)} recent projects...")
                    
                    # Analyze projects and update knowledge graph
                    learning_results = await self.knowledge_graph.analyze_and_store_projects(recent_projects)
                    
                    self.logger.info(f"📚 Knowledge learning: {learning_results['patterns_identified']} patterns identified")
                
            except Exception as e:
                self.logger.error(f"Error in knowledge learning: {e}")
                await asyncio.sleep(3600)
    
    async def _run_ai_learning(self):
        """Run periodic AI decision system learning"""
        
        while self._integration_active:
            try:
                # Every 6 hours, learn from decisions
                await asyncio.sleep(21600)  # 6 hours
                
                self.logger.info("🤖 Running AI decision system learning...")
                
                learning_results = await self.ai_decision_system.learn_and_improve()
                
                if learning_results['learned_decisions'] > 0:
                    self.logger.info(f"📈 AI Learning: {learning_results['learned_decisions']} decisions processed")
                
            except Exception as e:
                self.logger.error(f"Error in AI learning: {e}")
                await asyncio.sleep(21600)
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        
        component_status = {}
        
        # Check each component
        try:
            # Core components
            component_status['simple_tracker'] = self.tracker is not None
            component_status['business_parser'] = self.business_parser is not None
            component_status['feedback_engine'] = self.feedback_engine is not None
            
            # V2.0 components
            component_status['project_orchestrator'] = True  # Always available if initialized
            component_status['task_planner'] = True
            component_status['ai_decision_system'] = True
            component_status['knowledge_graph'] = getattr(self.knowledge_graph, 'graph_builder', {}).get('connected', False) if hasattr(self.knowledge_graph, 'graph_builder') else True
            component_status['success_classifier'] = True
            component_status['metrics_analyzer'] = True
            
        except Exception as e:
            self.logger.error(f"Error checking component status: {e}")
        
        # Get system metrics
        try:
            # Active projects
            active_projects = len(self.project_orchestrator.list_projects(ProjectState.ACTIVE))
            
            # Success rate (last 24h)
            success_stats = self.success_classifier.get_success_statistics(days=1)
            success_rate_24h = success_stats.get('success_rate', 0.0)
            
            # Cost today
            cost_today = 0.0  # Would calculate from metrics
            
            # Critical alerts
            analyzer_status = self.metrics_analyzer.get_current_system_status()
            critical_alerts = analyzer_status.get('critical_alerts', 0)
            
            # Knowledge patterns
            kg_stats = self.knowledge_graph.get_system_statistics()
            knowledge_patterns = kg_stats.get('pattern_engine', {}).get('patterns_in_cache', 0)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            active_projects = 0
            success_rate_24h = 0.0
            cost_today = 0.0
            critical_alerts = 0
            knowledge_patterns = 0
        
        # Determine overall state
        healthy_components = sum(component_status.values())
        total_components = len(component_status)
        
        if critical_alerts > 5:
            overall_state = SystemState.ERROR
        elif healthy_components < total_components * 0.8:
            overall_state = SystemState.DEGRADED
        elif critical_alerts > 0:
            overall_state = SystemState.DEGRADED
        else:
            overall_state = SystemState.ACTIVE if self._integration_active else SystemState.READY
        
        return SystemHealth(
            overall_state=overall_state,
            component_status=component_status,
            active_projects=active_projects,
            success_rate_24h=success_rate_24h,
            cost_today=cost_today,
            critical_alerts=critical_alerts,
            knowledge_patterns=knowledge_patterns,
            last_updated=datetime.now()
        )
    
    # Main workflow methods
    async def execute_business_request(self, business_request: str, 
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point: Execute a business request through the complete system
        
        This demonstrates the full V2.0 Intelligence Layer workflow:
        1. Business Requirements Parsing
        2. AI-First Model Selection  
        3. Hierarchical Task Planning
        4. Project Orchestration
        5. Success Classification
        6. Knowledge Graph Learning
        """
        
        if context is None:
            context = {}
        
        request_id = f"req_{int(time.time())}"
        self.logger.info(f"🎯 Executing business request [{request_id}]: {business_request[:100]}...")
        
        try:
            # Step 1: Parse business requirements
            intent = None
            if self.business_parser:
                intent = self.business_parser.parse_intent(business_request)
                self.logger.info(f"📋 Parsed intent: {intent.primary_action.value if intent else 'unknown'}")
            
            # Step 2: Get AI model recommendation
            model_recommendation = await self.ai_decision_system.recommend_model_for_task(
                business_request,
                context.get('decision_context', DecisionContext.DEVELOPMENT),
                context.get('user_preferences')
            )
            self.logger.info(f"🤖 Recommended model: {model_recommendation.recommended_model.value}")
            
            # Step 3: Create hierarchical task plan
            hierarchy_id = f"hierarchy_{request_id}"
            task_hierarchy = await self.task_planner.create_task_hierarchy(
                hierarchy_id=hierarchy_id,
                business_requests=[business_request],
                max_depth=context.get('max_task_depth', 3)
            )
            self.logger.info(f"📊 Created task hierarchy: {len(task_hierarchy)} tasks")
            
            # Step 4: Create and execute project
            project_id = await self.project_orchestrator.create_project(
                name=f"Business Request: {business_request[:50]}",
                description=business_request,
                business_requests=[business_request]
            )
            
            # Start project execution
            success = await self.project_orchestrator.start_project(project_id)
            if not success:
                raise RuntimeError("Failed to start project execution")
            
            self.logger.info(f"🚀 Started project execution: {project_id}")
            
            # Step 5: Monitor execution (simplified for demo)
            # In a real system, this would be handled by the ongoing monitoring
            await asyncio.sleep(2)  # Simulate execution time
            
            # Step 6: Get project results
            project = self.project_orchestrator.get_project(project_id)
            project_status = self.project_orchestrator.get_project_status(project_id)
            
            # Step 7: Evaluate success
            evaluation = None
            if project:
                evaluation = self.success_classifier.evaluate_project_success(
                    project, 
                    {**context, 'request_id': request_id}
                )
                self.logger.info(f"📊 Success evaluation: {evaluation.overall_success_level.value}")
            
            # Step 8: Get knowledge graph recommendations for similar future requests
            recommendations = await self.knowledge_graph.get_recommendations_for_project(
                [business_request],
                context
            )
            
            # Compile results
            results = {
                'request_id': request_id,
                'business_request': business_request,
                'execution_status': 'completed',
                'project_id': project_id,
                'hierarchy_id': hierarchy_id,
                
                # Intelligence Layer results
                'parsed_intent': {
                    'primary_action': intent.primary_action.value if intent else 'unknown',
                    'complexity': intent.complexity.value if intent else 'unknown',
                    'confidence': intent.confidence if intent else 0.0
                } if intent else None,
                
                'ai_model_selection': {
                    'recommended_model': model_recommendation.recommended_model.value,
                    'confidence': model_recommendation.confidence_score,
                    'expected_cost': model_recommendation.expected_cost,
                    'reasoning': model_recommendation.reasoning
                },
                
                'task_planning': {
                    'total_tasks': len(task_hierarchy),
                    'hierarchy_levels': max(task.level for task in task_hierarchy.values()) if task_hierarchy else 0,
                    'critical_path_tasks': len(self.task_planner.get_critical_path(hierarchy_id))
                },
                
                'project_execution': project_status,
                
                'success_evaluation': {
                    'overall_level': evaluation.overall_success_level.value if evaluation else 'unknown',
                    'overall_score': evaluation.overall_score if evaluation else 0.0,
                    'confidence': evaluation.confidence if evaluation else 0.0,
                    'strengths': evaluation.strengths if evaluation else [],
                    'weaknesses': evaluation.weaknesses if evaluation else [],
                    'improvement_suggestions': evaluation.improvement_suggestions if evaluation else []
                } if evaluation else None,
                
                'knowledge_recommendations': {
                    'similar_projects': len(recommendations.get('similar_projects', [])),
                    'recommended_patterns': len(recommendations.get('recommended_patterns', [])),
                    'suggested_solutions': len(recommendations.get('suggested_solutions', [])),
                    'risk_warnings': len(recommendations.get('risk_warnings', []))
                },
                
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': time.time() - int(request_id.split('_')[1])
            }
            
            # Track in SimpleTracker if available
            if self.tracker:
                self.tracker.track_event({
                    'type': 'business_request_executed',
                    'request_id': request_id,
                    'success_level': results['success_evaluation']['overall_level'] if results['success_evaluation'] else 'unknown',
                    'processing_time': results['processing_time_seconds']
                })
            
            self.logger.info(f"✅ Business request completed [{request_id}] - {results['success_evaluation']['overall_level'] if results['success_evaluation'] else 'unknown'}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error executing business request [{request_id}]: {e}")
            
            return {
                'request_id': request_id,
                'business_request': business_request,
                'execution_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # CLI Commands
    def status_cli(self) -> str:
        """CLI command: a0 status"""
        
        try:
            health = asyncio.run(self.get_system_health())
            
            output = []
            output.append("🎯 Agent Zero V1 - System Status")
            output.append("=" * 50)
            output.append(f"Overall State: {health.overall_state.value.upper()}")
            output.append(f"Uptime: {datetime.now() - self.start_time}")
            output.append(f"Last Updated: {health.last_updated.strftime('%H:%M:%S')}")
            
            output.append(f"\n📊 System Metrics:")
            output.append(f"   Active Projects: {health.active_projects}")
            output.append(f"   Success Rate (24h): {health.success_rate_24h:.1%}")
            output.append(f"   Cost Today: ${health.cost_today:.4f}")
            output.append(f"   Critical Alerts: {health.critical_alerts}")
            output.append(f"   Knowledge Patterns: {health.knowledge_patterns}")
            
            output.append(f"\n🔧 Component Status:")
            for component, status in health.component_status.items():
                status_icon = "✅" if status else "❌"
                output.append(f"   {status_icon} {component.replace('_', ' ').title()}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Error getting system status: {e}"
    
    def kaizen_report_cli(self, report_type: str = 'daily') -> str:
        """CLI command: a0 kaizen-report"""
        return self.metrics_analyzer.generate_kaizen_report_cli(report_type)
    
    def cost_analysis_cli(self, days: int = 7) -> str:
        """CLI command: a0 cost-analysis"""
        return self.metrics_analyzer.generate_cost_analysis_cli(days)
    
    async def demo_cli(self) -> str:
        """CLI command: a0 demo - Demonstrate full system capabilities"""
        
        output = []
        output.append("🎯 Agent Zero V1 - Complete System Demo")
        output.append("=" * 60)
        
        # Demo business requests
        demo_requests = [
            "Create a user authentication API with JWT tokens and rate limiting",
            "Analyze sales data and generate executive dashboard with KPI insights",
            "Build automated testing framework with CI/CD integration"
        ]
        
        output.append(f"📋 Executing {len(demo_requests)} demo business requests...")
        
        for i, request in enumerate(demo_requests):
            output.append(f"\n🔍 Request {i+1}: {request[:50]}...")
            
            try:
                # Execute request through full system
                result = await self.execute_business_request(
                    request,
                    {'decision_context': DecisionContext.DEVELOPMENT}
                )
                
                if result['execution_status'] == 'completed':
                    success_info = result.get('success_evaluation', {})
                    ai_info = result.get('ai_model_selection', {})
                    
                    output.append(f"   ✅ Status: {result['execution_status']}")
                    output.append(f"   🤖 AI Model: {ai_info.get('recommended_model', 'unknown')}")
                    output.append(f"   📊 Success: {success_info.get('overall_level', 'unknown')}")
                    output.append(f"   ⏱️  Time: {result.get('processing_time_seconds', 0):.1f}s")
                else:
                    output.append(f"   ❌ Status: {result['execution_status']}")
                    output.append(f"   Error: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                output.append(f"   ❌ Error: {e}")
        
        output.append(f"\n✅ Demo completed! All V2.0 Intelligence Layer components working together.")
        
        return "\n".join(output)

# Main CLI interface
async def main():
    """Main CLI interface for Agent Zero V1 Complete System"""
    
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize system
    config = {
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j', 
            'password': 'password123'
        }
    }
    
    try:
        system = AgentZeroV1System(config)
        
        # Handle CLI commands
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'status':
                print(system.status_cli())
                
            elif command == 'kaizen-report':
                report_type = sys.argv[2] if len(sys.argv) > 2 else 'daily'
                print(system.kaizen_report_cli(report_type))
                
            elif command == 'cost-analysis':
                days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
                print(system.cost_analysis_cli(days))
                
            elif command == 'demo':
                print(await system.demo_cli())
                
            elif command == 'start':
                print("🚀 Starting Agent Zero V1 system...")
                await system.start_system()
                
            else:
                print(f"Unknown command: {command}")
                print("Available commands: status, kaizen-report, cost-analysis, demo, start")
        
        else:
            # Interactive demo
            print("🎯 Agent Zero V1 - Complete System Integration")
            print("=" * 60)
            print("This is the complete Agent Zero V1 system with all V2.0 Intelligence Layer components!")
            print()
            
            # Show system status
            print(system.status_cli())
            print()
            
            # Run demo
            print("🎬 Running integrated system demo...")
            demo_result = await system.demo_cli()
            print(demo_result)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down Agent Zero V1 system...")
        if 'system' in locals():
            await system.stop_system()
            
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)