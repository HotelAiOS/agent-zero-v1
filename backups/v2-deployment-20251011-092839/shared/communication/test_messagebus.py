import asyncio
import logging
from messagebus import message_bus, publish_agent_message

# Setup logging
logging.basicConfig(level=logging.INFO)

async def backend_handler(data):
    """Handler for backend agent messages"""
    print(f"🔧 Backend received: {data}")

async def frontend_handler(data):
    """Handler for frontend agent messages"""
    print(f"🎨 Frontend received: {data}")

async def test_communication():
    """Test agent-to-agent communication"""
    print("🚀 Starting Multi-Agent Communication Test...")
    
    # Connect to message bus
    await message_bus.connect()
    
    # Subscribe handlers (fixed routing keys)
    await message_bus.subscribe("agent.backend.*", backend_handler)
    await message_bus.subscribe("agent.frontend.*", frontend_handler)
    
    # Wait a moment for subscriptions
    await asyncio.sleep(1)
    
    # Test messages
    await publish_agent_message("backend", "collaboration_request", {
        "from": "frontend_agent",
        "task": "Create REST API endpoint",
        "priority": "high",
        "details": "Need /api/users CRUD operations"
    })
    
    await publish_agent_message("frontend", "task_completed", {
        "from": "backend_agent", 
        "task": "API endpoint ready",
        "endpoint": "/api/users",
        "methods": ["GET", "POST", "PUT", "DELETE"]
    })
    
    # Keep alive to see messages
    print("⏳ Waiting for messages...")
    await asyncio.sleep(3)
    
    await message_bus.close()
    print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_communication())
