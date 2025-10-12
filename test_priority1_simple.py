#!/usr/bin/env python3
"""Test Priority 1 Integration - Fixed"""

import asyncio
import sys
sys.path.insert(0, ".")

from src.core.agent_executor import AgentExecutor, ExecutionContext, TaskStatus
from shared.ai.intelligent_agent_selector import IntelligentAgentSelector

class MockAgent:
    def __init__(self, agent_name):
        self.name = agent_name
    
    async def execute(self, input_data, workspace_dir):
        return f"Task completed by {self.name}"

async def test_integration():
    print("🧪 Testing AI Agent Selection - Fish Shell Compatible")
    print("=" * 60)
    
    # Test 1: Original AgentExecutor works
    print("📋 Test 1: Original AgentExecutor")
    executor = AgentExecutor()
    executor.register_agent("backend_developer", MockAgent("Backend Dev"))
    
    context = ExecutionContext(
        task_id="test_1",
        agent_type="backend_developer",
        input_data={"description": "Create API endpoint"},
        workspace_dir="/tmp"
    )
    
    result = await executor.execute_task(context)
    print(f"   Status: {result.status.value}")
    print(f"   Success: ✅ OK" if result.is_success else "   Success: ❌ FAIL")
    
    # Test 2: AI Selector works
    print("\n🧠 Test 2: AI Agent Selection System")
    try:
        ai_selector = IntelligentAgentSelector()
        stats = ai_selector.get_stats()
        print(f"   AI Stats: {stats}")
        print("   Status: ✅ AI Selector operational")
        ai_ok = True
    except Exception as e:
        print(f"   Status: ❌ AI Selector failed: {e}")
        ai_ok = False
    
    return result.is_success and ai_ok

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    print(f"\n🎯 Integration test: ✅ PASSED" if success else "\n🎯 Integration test: ❌ FAILED")

