import asyncio
from ai_brain import ai_brain
from agent_factory import agent_factory

class EcommerceTeam:
    def __init__(self):
        # Tworzymy specialized team
        self.backend_agent = agent_factory.create_agent("backend_developer", "Senior_Backend_Alice")
        self.frontend_agent = agent_factory.create_agent("frontend_developer", "Senior_Frontend_Bob")
        
        print("🏗️ E-commerce Development Team Assembled:")
        print(f"  👩‍💻 Backend: {self.backend_agent['name']}")
        print(f"  👨‍💻 Frontend: {self.frontend_agent['name']}")
        print()
    
    async def develop_ecommerce_platform(self):
        print("🛒 REAL PROJECT: E-commerce Platform Development")
        print("=" * 60)
        
        # BACKEND TASKS
        backend_tasks = [
            "Create SQLAlchemy models for Product, Order, User tables with relationships",
            "Build FastAPI REST endpoints for products CRUD operations with authentication",
            "Implement JWT authentication system with login and registration endpoints"
        ]
        
        # FRONTEND TASKS  
        frontend_tasks = [
            "Create React product listing component with search and filters",
            "Build shopping cart component with add/remove/update functionality",
            "Design user authentication forms with login and registration"
        ]
        
        backend_results = []
        frontend_results = []
        
        # BACKEND DEVELOPMENT
        print("\n🔧 BACKEND DEVELOPMENT PHASE:")
        print("-" * 40)
        
        for i, task in enumerate(backend_tasks, 1):
            print(f"\n📋 Backend Task {i}: {task[:50]}...")
            
            # AI Brain automatically selects deepseek-coder:33b for code tasks
            result = await ai_brain.think(task)
            
            print(f"  🤖 Model: {result.model_used}")
            print(f"  ⏱️ Time: {result.processing_time:.1f}s")
            print(f"  📊 Confidence: {result.confidence:.2f}")
            print(f"  📝 Code generated: {len(result.response)} characters")
            
            # Show first few lines of generated code
            lines = result.response.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"    {line[:60]}...")
                    break
            
            backend_results.append(result)
        
        # FRONTEND DEVELOPMENT
        print("\n🎨 FRONTEND DEVELOPMENT PHASE:")
        print("-" * 40)
        
        for i, task in enumerate(frontend_tasks, 1):
            print(f"\n📋 Frontend Task {i}: {task[:50]}...")
            
            # AI Brain selects appropriate model for frontend work
            result = await ai_brain.think(task)
            
            print(f"  🤖 Model: {result.model_used}")
            print(f"  ⏱️ Time: {result.processing_time:.1f}s")
            print(f"  📊 Confidence: {result.confidence:.2f}")
            print(f"  📝 Code generated: {len(result.response)} characters")
            
            # Show component structure
            lines = result.response.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"    {line[:60]}...")
                    break
            
            frontend_results.append(result)
        
        # PROJECT SUMMARY
        await self.project_summary(backend_results, frontend_results)
    
    async def project_summary(self, backend_results, frontend_results):
        print("\n" + "=" * 60)
        print("📊 E-COMMERCE PROJECT SUMMARY")
        print("=" * 60)
        
        total_backend_time = sum(r.processing_time for r in backend_results)
        total_frontend_time = sum(r.processing_time for r in frontend_results)
        total_code_chars = sum(len(r.response) for r in backend_results + frontend_results)
        
        avg_backend_confidence = sum(r.confidence for r in backend_results) / len(backend_results)
        avg_frontend_confidence = sum(r.confidence for r in frontend_results) / len(frontend_results)
        
        print(f"🔧 Backend Development:")
        print(f"  ⏱️ Total time: {total_backend_time:.1f}s")
        print(f"  📊 Avg confidence: {avg_backend_confidence:.2f}")
        print(f"  📁 Tasks completed: {len(backend_results)}")
        
        print(f"\n🎨 Frontend Development:")
        print(f"  ⏱️ Total time: {total_frontend_time:.1f}s")
        print(f"  📊 Avg confidence: {avg_frontend_confidence:.2f}")
        print(f"  📁 Tasks completed: {len(frontend_results)}")
        
        print(f"\n🏆 PROJECT TOTALS:")
        print(f"  ⏱️ Total development time: {total_backend_time + total_frontend_time:.1f}s")
        print(f"  📝 Lines of code generated: ~{total_code_chars // 50} lines")
        print(f"  🤖 AI models utilized: {len(set(r.model_used for r in backend_results + frontend_results))}")
        print(f"  👥 Team members: 2 specialized agents")
        
        # AI Brain performance stats
        stats = ai_brain.get_stats()
        print(f"\n🧠 AI Brain Performance:")
        print(f"  💭 Total thoughts: {stats.get('total_thoughts', 0)}")
        print(f"  ⚡ Models used: {', '.join(stats.get('model_usage', {}).keys())}")
        
        print(f"\n✅ E-COMMERCE PLATFORM: DEVELOPMENT COMPLETED!")
        print("🚀 Ready for: Testing, Deployment, Production!")

async def main():
    print("🎯 REAL WORLD APPLICATION: Multi-Agent E-commerce Development")
    print()
    
    # Create development team
    team = EcommerceTeam()
    
    # Build complete e-commerce platform
    await team.develop_ecommerce_platform()
    
    print(f"\n🎉 DEMO COMPLETED AT: {asyncio.get_event_loop().time():.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
