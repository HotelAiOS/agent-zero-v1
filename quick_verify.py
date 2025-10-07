#!/usr/bin/env python3
"""
Agent Zero V1 - Reality Check Script
Sprawdza, które komponenty działają naprawdę, a które to tylko szkielety.
"""

import asyncio
from pathlib import Path

# 1. Infrastructure imports
def test_infrastructure():
    try:
        import shared.llm.ollama_client
        import shared.knowledge.neo4j_client
        import shared.messaging.bus
        import shared.execution.agent_executor
        print("✅ Infrastructure imports OK")
        return True
    except Exception as e:
        print(f"❌ Infrastructure import error: {e}")
        return False

# 2. LLM generation test
def test_llm():
    try:
        from shared.llm.ollama_client import OllamaClient
        client = OllamaClient()
        prompt = "def add(a, b): return a + b"
        response = client.chat([{'role':'user','content':prompt}], 'backend')
        if "def add" in response:
            print("✅ LLM code generation OK")
            return True
        else:
            print("❌ LLM response not code-like")
            return False
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return False

# 3. Database test
def test_database():
    try:
        from shared.knowledge.neo4j_client import Neo4jClient
        client = Neo4jClient()
        res = client.execute_query("RETURN 1 AS value")
        print(f"✅ Neo4j query result: {res}")
        return True
    except Exception as e:
        print(f"❌ Neo4j error: {e}")
        return False

# 4. Agent execution test
async def test_agent_execution():
    try:
        from shared.execution.agent_executor import AgentExecutor
        from shared.llm.llm_factory import LLMFactory
        # Minimal task and agent stubs
        class SimpleTask:
            def __init__(self):
                self.id = 'tv1'
                self.name = 'Hello'
                self.description = 'Write Hello World'
        class SimpleAgent:
            def __init__(self):
                self.id = 'a1'
                self.agent_type = 'backend'
        executor = AgentExecutor(LLMFactory.create())
        out_dir = Path("verify_out")
        out_dir.mkdir(exist_ok=True)
        result = await executor.execute_task(SimpleAgent(), SimpleTask(), out_dir)
        files = list(out_dir.glob("*"))
        if files:
            print(f"✅ AgentExecutor created {len(files)} files")
            return True
        else:
            print("❌ No files created")
            return False
    except Exception as e:
        print(f"❌ AgentExecutor error: {e}")
        return False

# 5. Messaging test
def test_messaging():
    try:
        from shared.messaging.bus import MessageBus
        bus = MessageBus()
        ok = bus.connect()
        bus.disconnect()
        if ok:
            print("✅ RabbitMQ connection OK")
            return True
        else:
            print("❌ RabbitMQ connection failed")
            return False
    except Exception as e:
        print(f"❌ Messaging error: {e}")
        return False

async def main():
    print("🔍 Starting quick verification...")
    results = []
    results.append(test_infrastructure())
    results.append(test_llm())
    results.append(test_database())
    results.append(await test_agent_execution())
    results.append(test_messaging())
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"📊 Passed {passed}/{total} tests")
    if passed >= 4:
        print("🎉 Project foundation is solid!")
    elif passed >= 2:
        print("⚠️ Mixed results, significant work needed.")
    else:
        print("❌ Major issues detected.")

if __name__ == "__main__":
    asyncio.run(main())
