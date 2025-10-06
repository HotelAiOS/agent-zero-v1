# Agent Zero v1 - Phase 2: Interactive Control Integration
# Integrated system combining LiveMonitor, QualityAnalyzer, and PerformanceOptimizer

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Import our Phase 2 components
from shared.monitoring.livemonitor import LiveMonitor, AgentUpdate, AgentStatus
from shared.monitoring.checkpoint_manager import CheckpointManager, ProjectCheckpoint
from shared.quality.qualityanalyzer import QualityAnalyzer, QualityReport, QualityLevel
from shared.performance.optimizer import PerformanceOptimizer, LLMPerformanceData, OptimizationSuggestion


@dataclass
class InteractiveControlSession:
    """Interactive control session state"""
    session_id: str
    project_name: str
    start_time: datetime
    live_monitor: LiveMonitor
    quality_analyzer: QualityAnalyzer
    performance_optimizer: PerformanceOptimizer
    checkpoint_manager: CheckpointManager
    
    # Session metrics
    total_tasks_completed: int = 0
    quality_gates_passed: bool = False
    optimization_suggestions: List[OptimizationSuggestion] = None
    
    def __post_init__(self):
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []


class InteractiveControlSystem:
    """
    Integrated Interactive Control System for Agent Zero v1
    
    Combines:
    - LiveMonitor: Real-time agent monitoring and user interaction
    - QualityAnalyzer: Code quality analysis and security scanning
    - PerformanceOptimizer: System optimization and bottleneck detection
    """
    
    def __init__(self, 
                 llm_client=None,
                 quality_threshold: float = 70.0,
                 checkpoint_dir: str = "./checkpoints",
                 reports_dir: str = "./reports"):
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.live_monitor = LiveMonitor(self.checkpoint_manager)
        self.quality_analyzer = QualityAnalyzer(quality_threshold)
        self.performance_optimizer = PerformanceOptimizer(llm_client)
        
        # Configuration
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Active sessions
        self.active_sessions: Dict[str, InteractiveControlSession] = {}
        
        # Integration callbacks
        self._setup_component_integration()
        
    def _setup_component_integration(self):
        """Setup integration between components"""
        
        # Subscribe to live monitor updates for performance tracking
        self.live_monitor.subscribe_to_updates(self._on_agent_update)
        
        # Quality analyzer integration (triggered on project completion)
        # Performance optimizer integration (continuous monitoring)
        
    def _on_agent_update(self, update: AgentUpdate):
        """Handle agent updates from LiveMonitor"""
        
        # Convert to LLM performance data for optimizer
        if update.tokens_generated > 0 and update.time_elapsed > 0:
            perf_data = LLMPerformanceData(
                agent_type=update.agent_type,
                model_name="unknown",  # Would be provided by actual integration
                prompt_length=800,  # Would be actual prompt length
                response_length=update.tokens_generated,
                execution_time=update.time_elapsed,
                tokens_per_second=update.tokens_generated / max(update.time_elapsed, 0.1),
                cost_estimate=0.01,  # Would be calculated based on model
                success=update.status == AgentStatus.COMPLETED,
                memory_peak_mb=update.memory_usage_mb
            )
            
            self.performance_optimizer.llm_tracker.record_llm_execution(perf_data)
            
    async def start_interactive_session(self, 
                                      project_name: str,
                                      project_path: str,
                                      requirements: str) -> InteractiveControlSession:
        """Start a new interactive control session"""
        
        session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Create session
        session = InteractiveControlSession(
            session_id=session_id,
            project_name=project_name,
            start_time=datetime.now(),
            live_monitor=self.live_monitor,
            quality_analyzer=self.quality_analyzer,
            performance_optimizer=self.performance_optimizer,
            checkpoint_manager=self.checkpoint_manager
        )
        
        self.active_sessions[session_id] = session
        
        # Start performance monitoring
        await self.performance_optimizer.start_optimization_monitoring()
        
        # Create initial checkpoint
        await session.checkpoint_manager.save_checkpoint({
            "project_name": project_name,
            "checkpoint_id": "initial",
            "timestamp": session.start_time,
            "completed_tasks": [],
            "current_phase": "initialization",
            "agent_states": {},
            "execution_context": {
                "requirements": requirements,
                "project_path": project_path
            }
        })
        
        print(f"🚀 Interactive Control Session Started: {session_id}")
        print(f"   Project: {project_name}")
        print(f"   Path: {project_path}")
        print(f"   Session Features:")
        print(f"   ├── 📺 Live monitoring with real-time dashboard")
        print(f"   ├── 🔍 Continuous quality analysis")
        print(f"   ├── ⚡ Performance optimization")
        print(f"   └── 💾 Automatic checkpoint creation")
        
        return session
        
    async def execute_project_with_control(self,
                                         session: InteractiveControlSession,
                                         project_orchestrator,
                                         requirements: str) -> Dict[str, Any]:
        """Execute project with full interactive control"""
        
        print(f"\n🤖 Starting Project Execution with Interactive Control")
        print(f"📊 Dashboard Controls: [S]top [P]ause [C]ontinue [R]etry [?]Status [H]elp")
        
        # Start live monitoring
        monitor_task = asyncio.create_task(
            session.live_monitor.start_monitoring(session.project_name)
        )
        
        # Start quality monitoring
        quality_task = asyncio.create_task(
            self._continuous_quality_monitoring(session)
        )
        
        try:
            # Execute project with monitoring integration
            result = await self._execute_with_monitoring(
                session, 
                project_orchestrator, 
                requirements
            )
            
            # Final quality analysis
            final_quality_report = await self._perform_final_quality_analysis(session)
            
            # Performance optimization suggestions
            optimization_suggestions = await self.performance_optimizer.analyze_system_performance()
            session.optimization_suggestions = optimization_suggestions
            
            # Generate comprehensive session report
            session_report = await self._generate_session_report(session, result, final_quality_report)
            
            return {
                'execution_result': result,
                'quality_report': final_quality_report,
                'optimization_suggestions': optimization_suggestions,
                'session_report': session_report,
                'session_id': session.session_id
            }
            
        finally:
            # Cleanup
            monitor_task.cancel()
            quality_task.cancel()
            await session.live_monitor.stop_monitoring()
            
    async def _execute_with_monitoring(self,
                                     session: InteractiveControlSession,
                                     orchestrator,
                                     requirements: str) -> Any:
        """Execute project with integrated monitoring"""
        
        # This would integrate with the actual ProjectOrchestrator
        # For demonstration, we'll simulate the execution process
        
        print("🔄 Integrating with ProjectOrchestrator...")
        
        # Simulate task execution phases
        phases = [
            ("Analysis", "Analyzing requirements and architecture", 15),
            ("Design", "Creating system design and database schema", 25), 
            ("Implementation", "Generating code components", 45),
            ("Testing", "Running tests and validation", 10),
            ("Documentation", "Creating project documentation", 5)
        ]
        
        completed_tasks = []
        
        for i, (phase_name, phase_desc, duration) in enumerate(phases):
            print(f"\n📍 Phase {i+1}/5: {phase_name}")
            
            # Create checkpoint before phase
            checkpoint_id = f"phase_{i+1}_{phase_name.lower()}"
            await session.checkpoint_manager.save_checkpoint({
                "project_name": session.project_name,
                "checkpoint_id": checkpoint_id,
                "timestamp": datetime.now(),
                "completed_tasks": completed_tasks.copy(),
                "current_phase": phase_name,
                "agent_states": {},
                "execution_context": {"phase_description": phase_desc}
            })
            
            # Simulate agent execution with real-time updates
            await self._simulate_phase_execution(session, phase_name, phase_desc, duration)
            
            completed_tasks.append(phase_name)
            session.total_tasks_completed += 1
            
            # Check for user intervention
            if session.live_monitor.should_stop:
                print("🛑 Execution stopped by user")
                break
                
            while session.live_monitor.is_paused:
                await asyncio.sleep(1)
                
        return {
            'status': 'completed' if not session.live_monitor.should_stop else 'stopped',
            'completed_phases': len(completed_tasks),
            'total_phases': len(phases),
            'completed_tasks': completed_tasks,
            'final_checkpoint': checkpoint_id if 'checkpoint_id' in locals() else None
        }
        
    async def _simulate_phase_execution(self,
                                      session: InteractiveControlSession,
                                      phase_name: str,
                                      phase_desc: str,
                                      duration: int):
        """Simulate phase execution with agent updates"""
        
        # Simulate different agents working on the phase
        agents = [
            ("architect", "Designing system architecture"),
            ("backend", "Implementing backend services"),
            ("database", "Creating database schema"),
            ("frontend", "Building user interface"),
            ("tester", "Writing and running tests")
        ]
        
        for agent_type, task_desc in agents:
            if session.live_monitor.should_stop:
                break
                
            # Send agent thinking update
            await session.live_monitor.send_agent_update(AgentUpdate(
                agent_id=f"{agent_type}_001",
                agent_type=agent_type,
                task_id=f"{phase_name.lower()}_task",
                project_name=session.project_name,
                status=AgentStatus.THINKING,
                thinking_text=f"Analyzing {task_desc.lower()}...",
                progress_percent=10.0,
                current_step=task_desc,
                tokens_generated=0
            ))
            
            await asyncio.sleep(2)
            
            # Send agent executing update
            for progress in range(20, 101, 20):
                if session.live_monitor.should_stop:
                    break
                    
                await session.live_monitor.send_agent_update(AgentUpdate(
                    agent_id=f"{agent_type}_001",
                    agent_type=agent_type,
                    task_id=f"{phase_name.lower()}_task",
                    project_name=session.project_name,
                    status=AgentStatus.EXECUTING,
                    progress_percent=progress,
                    current_step=task_desc,
                    tokens_generated=progress * 3,  # Simulate token generation
                    time_elapsed=progress * 0.1,
                    memory_usage_mb=256 + progress * 2
                ))
                
                await asyncio.sleep(1)
                
                while session.live_monitor.is_paused:
                    await asyncio.sleep(0.5)
                    
            # Send completion update
            if not session.live_monitor.should_stop:
                await session.live_monitor.send_agent_update(AgentUpdate(
                    agent_id=f"{agent_type}_001",
                    agent_type=agent_type,
                    task_id=f"{phase_name.lower()}_task",
                    project_name=session.project_name,
                    status=AgentStatus.COMPLETED,
                    progress_percent=100.0,
                    current_step="Completed",
                    tokens_generated=300,
                    time_elapsed=duration * 0.2
                ))
                
    async def _continuous_quality_monitoring(self, session: InteractiveControlSession):
        """Continuous quality monitoring during execution"""
        
        while not session.live_monitor.should_stop:
            try:
                # Perform periodic quality checks
                # In real implementation, this would analyze generated code in real-time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Simulate quality check
                print("🔍 Performing quality check...")
                
                # Check if quality gates are still passing
                # This would integrate with actual code being generated
                
            except Exception as e:
                print(f"[QUALITY MONITOR ERROR] {e}")
                await asyncio.sleep(60)
                
    async def _perform_final_quality_analysis(self, session: InteractiveControlSession) -> QualityReport:
        """Perform comprehensive quality analysis at project completion"""
        
        print("\n🔍 Performing Final Quality Analysis...")
        
        # In real implementation, this would analyze the actual generated project
        # For demo, we'll create a simulated quality report
        
        sample_report = QualityReport(
            project_name=session.project_name,
            analysis_timestamp=datetime.now().isoformat(),
            overall_score=78.5,
            quality_level=QualityLevel.GOOD,
            code_quality={},
            security_issues=[],
            best_practices={},
            performance_metrics={},
            maintainability={},
            language_analysis={},
            total_files_analyzed=15,
            total_lines_of_code=2847,
            test_coverage_percent=72.3,
            quality_gates_passed=True,
            blocking_issues=[],
            priority_fixes=[],
            improvement_suggestions=[
                "Add more comprehensive error handling",
                "Improve test coverage for edge cases",
                "Consider adding API documentation"
            ]
        )
        
        session.quality_gates_passed = sample_report.quality_gates_passed
        
        # Export quality report
        report_file = await self.quality_analyzer.export_report(
            sample_report,
            str(self.reports_dir),
            "html"
        )
        
        print(f"✅ Quality Analysis Complete:")
        print(f"   ├── Overall Score: {sample_report.overall_score:.1f}%")
        print(f"   ├── Quality Level: {sample_report.quality_level.value}")
        print(f"   ├── Files Analyzed: {sample_report.total_files_analyzed}")
        print(f"   ├── Lines of Code: {sample_report.total_lines_of_code:,}")
        print(f"   ├── Quality Gates: {'✅ PASSED' if sample_report.quality_gates_passed else '❌ FAILED'}")
        print(f"   └── Report: {report_file}")
        
        return sample_report
        
    async def _generate_session_report(self,
                                     session: InteractiveControlSession,
                                     execution_result: Dict[str, Any],
                                     quality_report: QualityReport) -> Dict[str, Any]:
        """Generate comprehensive session report"""
        
        session_duration = (datetime.now() - session.start_time).total_seconds()
        
        # Get performance metrics
        performance_report = await self.performance_optimizer.generate_optimization_report()
        
        report = {
            'session_info': {
                'session_id': session.session_id,
                'project_name': session.project_name,
                'start_time': session.start_time.isoformat(),
                'duration_seconds': session_duration,
                'duration_formatted': f"{session_duration//60:.0f}m {session_duration%60:.0f}s"
            },
            'execution_summary': {
                'status': execution_result['status'],
                'completed_tasks': session.total_tasks_completed,
                'quality_gates_passed': session.quality_gates_passed,
                'final_checkpoint': execution_result.get('final_checkpoint')
            },
            'quality_summary': {
                'overall_score': quality_report.overall_score,
                'quality_level': quality_report.quality_level.value,
                'files_analyzed': quality_report.total_files_analyzed,
                'lines_of_code': quality_report.total_lines_of_code,
                'security_issues': len(quality_report.security_issues),
                'improvement_suggestions': len(quality_report.improvement_suggestions)
            },
            'performance_summary': performance_report,
            'optimization_suggestions': [
                asdict(suggestion) for suggestion in session.optimization_suggestions[:5]
            ],
            'session_metrics': {
                'user_interventions': 0,  # Would track actual interventions
                'checkpoints_created': len(await session.checkpoint_manager.list_checkpoints(session.project_name)),
                'quality_checks_performed': 1,  # Final quality check
                'optimization_suggestions_generated': len(session.optimization_suggestions)
            }
        }
        
        # Save session report
        report_file = self.reports_dir / f"session_report_{session.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\n📋 Session Report Generated: {report_file}")
        
        return report
        
    async def restore_session_from_checkpoint(self,
                                            project_name: str,
                                            checkpoint_id: str) -> Optional[InteractiveControlSession]:
        """Restore session from checkpoint"""
        
        checkpoint = await self.checkpoint_manager.load_checkpoint(project_name, checkpoint_id)
        
        if not checkpoint:
            print(f"❌ Checkpoint not found: {checkpoint_id}")
            return None
            
        # Create new session from checkpoint
        session = InteractiveControlSession(
            session_id=f"restored_{int(datetime.now().timestamp())}",
            project_name=project_name,
            start_time=datetime.now(),
            live_monitor=self.live_monitor,
            quality_analyzer=self.quality_analyzer,
            performance_optimizer=self.performance_optimizer,
            checkpoint_manager=self.checkpoint_manager
        )
        
        print(f"✅ Session Restored from Checkpoint:")
        print(f"   ├── Project: {project_name}")
        print(f"   ├── Checkpoint: {checkpoint_id}")
        print(f"   ├── Original Time: {checkpoint.timestamp}")
        print(f"   ├── Completed Tasks: {len(checkpoint.completed_tasks)}")
        print(f"   └── Current Phase: {checkpoint.current_phase}")
        
        return session
        
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active sessions"""
        
        sessions_info = {}
        
        for session_id, session in self.active_sessions.items():
            duration = (datetime.now() - session.start_time).total_seconds()
            
            sessions_info[session_id] = {
                'project_name': session.project_name,
                'start_time': session.start_time.isoformat(),
                'duration_seconds': duration,
                'tasks_completed': session.total_tasks_completed,
                'quality_gates_passed': session.quality_gates_passed,
                'optimization_suggestions': len(session.optimization_suggestions)
            }
            
        return sessions_info
        
    async def shutdown(self):
        """Gracefully shutdown the interactive control system"""
        
        print("🔄 Shutting down Interactive Control System...")
        
        # Stop all active monitoring
        for session in self.active_sessions.values():
            await session.live_monitor.stop_monitoring()
            
        # Stop performance monitoring
        self.performance_optimizer.optimization_enabled = False
        
        # Save final state
        for session_id, session in self.active_sessions.items():
            await session.checkpoint_manager.save_checkpoint({
                "project_name": session.project_name,
                "checkpoint_id": f"shutdown_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now(),
                "completed_tasks": [],
                "current_phase": "shutdown",
                "agent_states": {},
                "execution_context": {"shutdown_reason": "system_shutdown"}
            })
            
        print("✅ Interactive Control System shutdown complete")


# Example usage and integration with existing Agent Zero v1 architecture
if __name__ == "__main__":
    
    async def demo_interactive_control():
        """Demonstrate the integrated Interactive Control System"""
        
        # Initialize the integrated system
        control_system = InteractiveControlSystem(
            quality_threshold=70.0,
            checkpoint_dir="./demo_checkpoints",
            reports_dir="./demo_reports"
        )
        
        print("🤖 Agent Zero v1 - Interactive Control System Demo")
        print("="*60)
        
        # Start interactive session
        session = await control_system.start_interactive_session(
            project_name="demo_api_project",
            project_path="./demo_output",
            requirements="Build a FastAPI REST API for user management with authentication"
        )
        
        # Simulate project execution (would integrate with real ProjectOrchestrator)
        class MockOrchestrator:
            async def execute_project(self, requirements, project_name):
                return {"status": "completed", "artifacts": ["main.py", "models.py", "api.py"]}
                
        mock_orchestrator = MockOrchestrator()
        
        # Execute with full interactive control
        result = await control_system.execute_project_with_control(
            session=session,
            project_orchestrator=mock_orchestrator,
            requirements="Build a FastAPI REST API for user management with authentication"
        )
        
        # Display results
        print("\n🎉 Project Execution Complete!")
        print("="*60)
        print(f"📊 Execution Status: {result['execution_result']['status']}")
        print(f"🔍 Quality Score: {result['quality_report'].overall_score:.1f}%")
        print(f"⚡ Optimization Suggestions: {len(result['optimization_suggestions'])}")
        
        # Show top optimization suggestions
        if result['optimization_suggestions']:
            print("\n💡 Top Optimization Suggestions:")
            for i, suggestion in enumerate(result['optimization_suggestions'][:3], 1):
                print(f"{i}. {suggestion.title} (Priority: {suggestion.priority})")
                
        # List available checkpoints
        checkpoints = await session.checkpoint_manager.list_checkpoints(session.project_name)
        if checkpoints:
            print(f"\n💾 Available Checkpoints: {len(checkpoints)}")
            for checkpoint in checkpoints[-3:]:  # Show last 3
                print(f"   └── {checkpoint}")
                
        # Cleanup
        await control_system.shutdown()
        
    # Run the demo
    asyncio.run(demo_interactive_control())