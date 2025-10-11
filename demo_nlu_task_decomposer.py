#!/usr/bin/env python3
'''
NLU Task Decomposer Demo Script
Agent Zero V1 - Week 43 Implementation
'''

import asyncio
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent / "shared" / "orchestration"))

async def main():
    print("🎯 NLU Task Decomposer Demo")
    print("=" * 50)
    
    try:
        from nlp_enhanced_task_decomposer import NLUTaskDecomposer, DomainContext
        
        decomposer = NLUTaskDecomposer()
        
        # Demo task
        task = "Create a user management system with JWT authentication"
        context = DomainContext(
            tech_stack=["FastAPI", "React", "PostgreSQL"],
            project_type="fullstack_web_app"
        )
        
        print(f"Task: {task}")
        print(f"Tech Stack: {', '.join(context.tech_stack)}")
        print("\nAnalyzing...")
        
        result = await decomposer.enhanced_decompose(task, context)
        
        print(f"\n✅ Analysis Complete!")
        print(f"Intent: {result.main_intent.primary_intent}")
        print(f"Subtasks: {len(result.subtasks)}")
        print(f"Complexity: {result.estimated_complexity:.2f}")
        
        print("\n🎉 Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to run the setup script first")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
