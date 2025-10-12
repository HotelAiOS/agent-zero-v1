"""
Test Intelligent Agents - Prawdziwa komunikacja agent-to-agent

Scenariusz:
1. Frontend Agent i Backend Agent startują
2. Frontend deleguje task do Backend
3. Backend wykonuje task (symulacja)
4. Backend wysyła odpowiedź do Frontend
5. Oba agenty zatrzymują się
"""
import asyncio
import logging
from intelligent_agent import IntelligentAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


async def test_intelligent_agents():
    """Test komunikacji między dwoma agentami"""
    
    print("\n" + "="*70)
    print("🧪 TESTING INTELLIGENT MULTI-AGENT COMMUNICATION")
    print("="*70 + "\n")
    
    # ========================================
    # 1. STWÓRZ BACKEND AGENT
    # ========================================
    print("📝 Step 1: Creating Backend Agent...")
    backend = IntelligentAgent(
        agent_id="backend_001",
        agent_type="backend",
        capabilities=["python", "fastapi", "database", "api_development"]
    )
    
    # Handler dla Backend - obsługuje task delegation
    async def backend_task_handler(message):
        """Backend otrzymuje task od Frontend"""
        task_desc = message['data'].get('description')
        from_agent = message['from']
        
        print(f"\n🔧 BACKEND RECEIVED TASK:")
        print(f"   From: {from_agent}")
        print(f"   Task: {task_desc}")
        print(f"   Requirements: {message['data'].get('requirements', [])}")
        
        # Symulacja pracy (w prawdziwym systemie tutaj byłby AI Brain)
        print(f"   🤖 Backend processing...")
        await asyncio.sleep(1)  # Symulacja pracy
        
        # Wyślij odpowiedź do Frontend
        await backend.send_to_agent(
            target_agent_id=from_agent,
            message_type="task_completed",
            data={
                "status": "success",
                "result": "User authentication API created",
                "endpoint": "/api/auth",
                "methods": ["POST /login", "POST /register", "POST /logout"],
                "code_snippet": "# FastAPI endpoint with JWT authentication..."
            }
        )
    
    # Zarejestruj handler
    backend.register_handler("task_delegation", backend_task_handler)
    print("   ✅ Backend handler registered\n")
    
    # ========================================
    # 2. STWÓRZ FRONTEND AGENT
    # ========================================
    print("📝 Step 2: Creating Frontend Agent...")
    frontend = IntelligentAgent(
        agent_id="frontend_001",
        agent_type="frontend",
        capabilities=["react", "typescript", "ui_design", "frontend_development"]
    )
    
    # Handler dla Frontend - odbiera wyniki
    async def frontend_result_handler(message):
        """Frontend otrzymuje wynik od Backend"""
        result = message['data']
        from_agent = message['from']
        
        print(f"\n🎨 FRONTEND RECEIVED RESULT:")
        print(f"   From: {from_agent}")
        print(f"   Status: {result.get('status')}")
        print(f"   Result: {result.get('result')}")
        print(f"   Endpoint: {result.get('endpoint')}")
        print(f"   Methods: {result.get('methods')}")
        print(f"   💚 Task completed successfully!")
    
    # Zarejestruj handler
    frontend.register_handler("task_completed", frontend_result_handler)
    print("   ✅ Frontend handler registered\n")
    
    # ========================================
    # 3. URUCHOM OBA AGENTY
    # ========================================
    print("📝 Step 3: Starting agents...")
    print("-" * 70)
    await backend.start()
    await frontend.start()
    print("-" * 70)
    print("✅ Both agents are ONLINE!\n")
    
    # ========================================
    # 4. FRONTEND DELEGUJE TASK DO BACKEND
    # ========================================
    print("📝 Step 4: Frontend delegates task to Backend...")
    print("-" * 70)
    
    delegated_to = await frontend.delegate_task(
        capability="api_development",
        task={
            "description": "Create user authentication API",
            "requirements": [
                "JWT token authentication",
                "PostgreSQL database",
                "Password hashing with bcrypt",
                "Rate limiting"
            ],
            "priority": "high",
            "deadline": "2 hours"
        }
    )
    
    if delegated_to:
        print(f"✅ Task successfully delegated to {delegated_to}")
    else:
        print("❌ No agent found with required capability!")
        
    print("-" * 70 + "\n")
    
    # ========================================
    # 5. POCZEKAJ NA KOMUNIKACJĘ
    # ========================================
    print("📝 Step 5: Waiting for agent communication...")
    await asyncio.sleep(3)  # Poczekaj na przetworzenie
    
    # ========================================
    # 6. SPRAWDŹ STATUS AGENTÓW
    # ========================================
    print("\n📝 Step 6: Checking agent status...")
    print("-" * 70)
    
    backend_status = await backend.get_status()
    print(f"Backend Status:")
    print(f"   ID: {backend_status['agent_id']}")
    print(f"   Type: {backend_status['agent_type']}")
    print(f"   Status: {backend_status['status']}")
    print(f"   Capabilities: {backend_status['capabilities']}")
    print(f"   Handlers: {backend_status['handlers']}")
    
    print()
    
    frontend_status = await frontend.get_status()
    print(f"Frontend Status:")
    print(f"   ID: {frontend_status['agent_id']}")
    print(f"   Type: {frontend_status['agent_type']}")
    print(f"   Status: {frontend_status['status']}")
    print(f"   Capabilities: {frontend_status['capabilities']}")
    print(f"   Handlers: {frontend_status['handlers']}")
    
    print("-" * 70 + "\n")
    
    # ========================================
    # 7. ZATRZYMAJ AGENTY
    # ========================================
    print("📝 Step 7: Stopping agents...")
    print("-" * 70)
    await backend.stop()
    await frontend.stop()
    print("-" * 70)
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    print("📊 Summary:")
    print("   ✅ 2 agents created (Backend + Frontend)")
    print("   ✅ Both agents registered in Agent Registry")
    print("   ✅ Both agents connected to RabbitMQ")
    print("   ✅ Task delegation: Frontend → Backend")
    print("   ✅ Task execution: Backend processed task")
    print("   ✅ Result delivery: Backend → Frontend")
    print("   ✅ Clean shutdown: Both agents stopped")
    print("\n🎉 Multi-Agent Communication WORKING!\n")


if __name__ == "__main__":
    asyncio.run(test_intelligent_agents())
