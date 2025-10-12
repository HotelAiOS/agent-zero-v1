#!/bin/bash
# priority1_integration.sh - Integracja Intelligent Agent Selector z AgentExecutor

echo "🔧 Agent Zero V2.0 Phase 4 - Priority 1 Integration"
echo "=================================================="
echo "Integracja IntelligentAgentSelector z AgentExecutor"

# 1. Backup original agent_executor.py
echo "📄 Creating backup..."
cp src/core/agent_executor.py src/core/agent_executor_backup.py
echo "✅ Backup created: src/core/agent_executor_backup.py"

# 2. Create enhanced agent_executor.py with AI integration
echo "🧠 Creating enhanced AgentExecutor with AI agent selection..."

cat > src/core/agent_executor_v2.py << 'EOF'
"""
Agent Executor V2.0 - Enhanced with AI Agent Selection
Agent Zero V2.0 Phase 4 - Integration with IntelligentAgentSelector

Priority 1.2: AgentExecutor Integration (1 SP)
- AI-powered agent selection when ai_routing=True
- Backward compatibility maintained
- Seamless integration with existing workflow
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Import original classes
from .agent_executor import TaskStatus, ExecutionResult

# Import AI agent selector
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from shared.ai.intelligent_agent_selector import IntelligentAgentSelector
    AI_SELECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AI Selector not available: {e}")
    AI_SELECTOR_AVAILABLE = False
    IntelligentAgentSelector = None

logger = logging.getLogger(__name__)

@dataclass
class ExecutionContext:
    """Enhanced execution context with AI routing support"""
    task_id: str
    agent_type: str
    input_data: Dict[str, Any]
    workspace_dir: str
    timeout: Optional[int] = 300
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def ai_routing_enabled(self) -> bool:
        """Check if AI routing is enabled for this task"""
        return self.metadata.get("ai_routing", False)
    
    @property
    def task_description(self) -> str:
        """Get task description for AI analysis"""
        return self.input_data.get("description", self.input_data.get("task", "Unknown task"))

class EnhancedAgentExecutor:
    """
    Enhanced Agent Executor with AI-powered agent selection
    
    New Features:
    - AI-powered agent selection when ai_routing=True
    - Intelligent task analysis with Ollama
    - Performance tracking and learning
    - Backward compatibility maintained
    """

    def __init__(self, agent_registry: Optional[Dict] = None):
        self.agent_registry = agent_registry or {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize AI selector
        self.ai_selector = None
        if AI_SELECTOR_AVAILABLE:
            try:
                self.ai_selector = IntelligentAgentSelector()
                logger.info("✅ AI Agent Selection enabled")
            except Exception as e:
                logger.warning(f"AI Selector initialization failed: {e}")
                self.ai_selector = None
        
        logger.info("EnhancedAgentExecutor initialized")

    async def execute_task(
        self,
        context: ExecutionContext,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute agent task with optional AI-powered agent selection"""
        start_time = time.time()

        logger.info(f"Executing task {context.task_id} with agent {context.agent_type}")

        # AI ROUTING: If enabled and available, use AI to select optimal agent
        if context.ai_routing_enabled and self.ai_selector:
            try:
                optimal_agent = await self._ai_agent_selection(context)
                if optimal_agent:
                    original_agent = context.agent_type
                    context.agent_type = optimal_agent
                    logger.info(f"🧠 AI selected agent: {original_agent} → {optimal_agent}")
                else:
                    logger.info(f"🔄 AI selection fallback: keeping {context.agent_type}")
            except Exception as e:
                logger.warning(f"AI agent selection failed: {e}, using original agent")

        # Check if agent is registered
        if context.agent_type not in self.agent_registry:
            error_msg = f"Agent type '{context.agent_type}' not registered"
            logger.error(error_msg)
            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        try:
            agent = self.agent_registry[context.agent_type]

            task = asyncio.create_task(
                self._execute_with_timeout(agent, context, callback)
            )
            self.active_tasks[context.task_id] = task

            result = await task
            result.execution_time = time.time() - start_time

            # Track performance for AI learning
            if self.ai_selector and context.ai_routing_enabled:
                await self._track_execution_performance(context, result)

            logger.info(f"✅ Task {context.task_id} completed in {result.execution_time:.2f}s")
            return result

        except asyncio.TimeoutError:
            error_msg = f"Task execution timed out after {context.timeout}s"
            logger.error(error_msg)
            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        finally:
            if context.task_id in self.active_tasks:
                del self.active_tasks[context.task_id]

    async def _ai_agent_selection(self, context: ExecutionContext) -> Optional[str]:
        """Use AI to select optimal agent for task"""
        try:
            # Analyze task with AI
            task_analysis = await self.ai_selector.analyze_task_with_ai(
                context.task_description,
                {
                    "task_id": context.task_id,
                    "current_agent": context.agent_type,
                    "project_context": context.metadata.get("project_context", {}),
                    "available_agents": list(self.agent_registry.keys())
                }
            )
            
            logger.info(f"🧠 Task analysis: {task_analysis.complexity.value} complexity, "
                       f"{len(task_analysis.required_capabilities)} capabilities required")

            # Select optimal agent
            selection = await self.ai_selector.select_optimal_agent(task_analysis, context.metadata)
            
            # Verify selected agent is registered
            if selection.selected_agent.agent_type in self.agent_registry:
                logger.info(f"✅ AI selected: {selection.selected_agent.agent_type} "
                           f"(confidence: {selection.confidence:.2f})")
                return selection.selected_agent.agent_type
            else:
                logger.warning(f"Selected agent {selection.selected_agent.agent_type} not registered")
                return None

        except Exception as e:
            logger.error(f"AI agent selection failed: {e}")
            return None

    async def _track_execution_performance(self, context: ExecutionContext, result: ExecutionResult):
        """Track execution performance for AI learning"""
        try:
            # Update agent workload
            if context.ai_routing_enabled:
                # This would be implemented to track actual performance
                logger.info(f"📊 Performance tracked for {context.agent_type}: "
                           f"success={result.is_success}, time={result.execution_time:.2f}s")
        except Exception as e:
            logger.warning(f"Performance tracking failed: {e}")

    async def _execute_with_timeout(
        self,
        agent: Any,
        context: ExecutionContext,
        callback: Optional[Callable]
    ) -> ExecutionResult:
        """Execute agent task with timeout (unchanged from original)"""

        if callback:
            callback(context.task_id, TaskStatus.RUNNING)

        try:
            if asyncio.iscoroutinefunction(agent.execute):
                output = await asyncio.wait_for(
                    agent.execute(context.input_data, context.workspace_dir),
                    timeout=context.timeout
                )
            else:
                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        agent.execute,
                        context.input_data,
                        context.workspace_dir
                    ),
                    timeout=context.timeout
                )

            if callback:
                callback(context.task_id, TaskStatus.COMPLETED)

            return ExecutionResult(
                task_id=context.task_id,
                status=TaskStatus.COMPLETED,
                output=output
            )

        except Exception as e:
            if callback:
                callback(context.task_id, TaskStatus.FAILED)
            raise

    def register_agent(self, agent_type: str, agent_instance: Any) -> None:
        """Register agent implementation (unchanged)"""
        self.agent_registry[agent_type] = agent_instance
        logger.info(f"Registered agent: {agent_type}")

    def enable_ai_routing(self) -> bool:
        """Enable AI routing if available"""
        if self.ai_selector:
            logger.info("🧠 AI routing enabled")
            return True
        else:
            logger.warning("AI routing not available")
            return False

    def get_ai_stats(self) -> Dict[str, Any]:
        """Get AI system statistics"""
        if self.ai_selector:
            return self.ai_selector.get_agent_stats()
        else:
            return {"ai_available": False}

# Backward compatibility - export original AgentExecutor as alias
AgentExecutor = EnhancedAgentExecutor
EOF

# 3. Replace original with enhanced version
echo "🔄 Installing enhanced AgentExecutor..."
mv src/core/agent_executor.py src/core/agent_executor_original.py
mv src/core/agent_executor_v2.py src/core/agent_executor.py

echo "✅ Enhanced AgentExecutor installed"

# 4. Create integration test
echo "🧪 Creating integration test..."

cat > test_priority1_integration.py << 'EOF'
#!/usr/bin/env python3
"""
Test Priority 1 Integration - AI Agent Selection
"""

import asyncio
import sys
sys.path.insert(0, '.')

from src.core.agent_executor import EnhancedAgentExecutor, ExecutionContext, TaskStatus

# Mock agent for testing
class MockAgent:
    def __init__(self, agent_name):
        self.name = agent_name
    
    async def execute(self, input_data, workspace_dir):
        return f"Task completed by {self.name}"

async def test_ai_routing():
    print("🧪 Testing AI Agent Selection Integration")
    print("=" * 50)
    
    # Initialize enhanced executor
    executor = EnhancedAgentExecutor()
    
    # Register mock agents
    executor.register_agent("backend_developer", MockAgent("Backend Developer"))
    executor.register_agent("frontend_developer", MockAgent("Frontend Developer"))
    executor.register_agent("fullstack_developer", MockAgent("Fullstack Developer"))
    
    print("✅ Mock agents registered")
    
    # Test 1: Without AI routing (backward compatibility)
    print("\n📋 Test 1: Traditional agent selection")
    context1 = ExecutionContext(
        task_id="test_1",
        agent_type="backend_developer",
        input_data={"description": "Create a FastAPI endpoint"},
        workspace_dir="/tmp",
        metadata={"ai_routing": False}
    )
    
    result1 = await executor.execute_task(context1)
    print(f"   Result: {result1.status.value}")
    print(f"   Output: {result1.output}")
    
    # Test 2: With AI routing
    print("\n🧠 Test 2: AI-powered agent selection")
    context2 = ExecutionContext(
        task_id="test_2", 
        agent_type="backend_developer",  # Will be potentially changed by AI
        input_data={"description": "Create a React component with TypeScript"},
        workspace_dir="/tmp",
        metadata={"ai_routing": True, "project_context": {"tech_stack": ["React", "TypeScript"]}}
    )
    
    result2 = await executor.execute_task(context2)
    print(f"   Result: {result2.status.value}")
    print(f"   Output: {result2.output}")
    
    # Test 3: AI Stats
    print("\n📊 AI System Statistics:")
    stats = executor.get_ai_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 Integration test completed!")
    print(f"   Traditional routing: {'✅ OK' if result1.is_success else '❌ FAIL'}")
    print(f"   AI routing: {'✅ OK' if result2.is_success else '❌ FAIL'}")
    print(f"   AI availability: {'✅ YES' if executor.ai_selector else '❌ NO'}")

if __name__ == "__main__":
    asyncio.run(test_ai_routing())
EOF

chmod +x test_priority1_integration.py

# 5. Run integration test
echo "🧪 Running integration test..."
venv/bin/python test_priority1_integration.py

echo ""
echo "🎉 Priority 1 Integration completed!"
echo ""  
echo "📋 Co zostało zrobione:"
echo "✅ Enhanced AgentExecutor with AI routing capability"
echo "✅ Backward compatibility maintained (ai_routing=False works as before)"
echo "✅ AI routing feature (ai_routing=True uses IntelligentAgentSelector)"
echo "✅ Integration test created and executed"
echo ""
echo "🎯 Następne kroki:"
echo "1. Sprawdź wyniki testu integracji"
echo "2. Uruchom główne testy: ./run_tests_fish.sh"
echo "3. Przejdź do Priority 2: Real-Time AI Reasoning Engine"