from ai_brain import ai_brain
from agent_factory import agent_factory
import asyncio

class SmartAgent:
    def __init__(self, template_name, agent_name=None):
        self.config = agent_factory.create_agent(template_name, agent_name)
        self.ai_brain = ai_brain
        
    async def work_on_task(self, task):
        agent_name = self.config['name']
        task_preview = task[:50] + '...' if len(task) > 50 else task
        print(f"🤖 {agent_name} working on: {task_preview}")
        
        # AI Brain thinks about the task
        result = await self.ai_brain.think(task)
        
        print(f"✅ Completed in {result.processing_time:.1f}s with {result.model_used}")
        return result

async def demo():
    print("🚀 DEMO: Smart Agent with AI Brain...")
    
    # Create intelligent agents
    backend_agent = SmartAgent("backend_developer", "AI_Backend_Alice")
    frontend_agent = SmartAgent("frontend_developer", "AI_Frontend_Bob")
    
    # Real AI work
    print("\n📋 Backend Task:")
    backend_result = await backend_agent.work_on_task("Create REST API for user authentication")
    
    print("\n📋 Frontend Task:")
    frontend_result = await frontend_agent.work_on_task("Design React login component")
    
    print("\n📊 SUMMARY:")
    print(f"Backend AI: {backend_result.model_used} - {len(backend_result.response)} chars response")
    print(f"Frontend AI: {frontend_result.model_used} - {len(frontend_result.response)} chars response")
    
    # Performance stats
    stats = ai_brain.get_stats()
    print(f"Total AI thoughts: {stats['total_thoughts']}")
    
    print("\n🎉 SMART AGENT DEMO COMPLETED!")

if __name__ == "__main__":
    asyncio.run(demo())
