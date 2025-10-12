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
