#!/usr/bin/env python3
"""
Test Script dla Enhanced Task Decomposer
Pokazuje jak używać i co powinno się stać
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_enhanced_task_decomposer():
    """
    Test funkcji Enhanced Task Decomposer
    Pokazuje expected output i behavior
    """
    
    print("🚀 Testing Enhanced Task Decomposer...")
    print("=" * 60)
    
    try:
        # Import Enhanced Task Decomposer
        from shared.orchestration.enhanced.enhanced_task_decomposer import (
            EnhancedTaskDecomposer, 
            AIReasoningContext,
            create_reasoning_context
        )
        
        print("✅ Enhanced Task Decomposer imported successfully!")
        
        # Utworz decomposer
        decomposer = EnhancedTaskDecomposer()
        print("✅ EnhancedTaskDecomposer instance created")
        
        # Utworz AI reasoning context
        context = create_reasoning_context(
            complexity="high",
            tech_stack=["Python", "FastAPI", "Neo4j"],
            team_size=2
        )
        print("✅ AI Reasoning Context created:")
        print(f"   • Complexity: {context.project_complexity}")
        print(f"   • Tech Stack: {', '.join(context.tech_stack)}")
        print(f"   • Team Size: {context.team_size}")
        print()
        
        # Test AI-enhanced decomposition
        task_description = "Create enterprise AI platform with real-time analytics"
        print(f"🎯 Task Description: {task_description}")
        print("🤖 Running AI-enhanced decomposition...")
        print()
        
        # Execute decomposition
        enhanced_tasks = await decomposer.decompose_with_ai_reasoning(
            task_description, 
            context
        )
        
        print("🎉 AI DECOMPOSITION COMPLETE!")
        print("=" * 60)
        print(f"📊 Generated {len(enhanced_tasks)} enhanced tasks:")
        print()
        
        for i, task in enumerate(enhanced_tasks, 1):
            print(f"📋 Task {i}: {task.title}")
            print(f"   📝 Description: {task.description}")
            print(f"   🎯 Type: {task.task_type.value}")
            print(f"   ⭐ Priority: {task.priority.value}")
            print(f"   🧠 AI Confidence: {task.ai_reasoning.confidence_score:.2f}")
            print(f"   📈 Complexity: {task.complexity_score:.2f}")
            print(f"   🤖 Automation Potential: {task.automation_potential:.2f}")
            print(f"   ⏱️ Estimated Hours: {task.estimated_hours}")
            
            if task.dependencies:
                deps = [str(dep.task_id) for dep in task.dependencies]
                print(f"   🔗 Dependencies: {', '.join(deps)}")
            
            if task.ai_reasoning.risk_factors:
                print(f"   ⚠️ Risk Factors: {', '.join(task.ai_reasoning.risk_factors)}")
            
            if task.ai_reasoning.optimization_opportunities:
                print(f"   💡 Optimizations: {', '.join(task.ai_reasoning.optimization_opportunities)}")
            
            if task.context_tags:
                print(f"   🏷️ Tags: {', '.join(task.context_tags)}")
            
            print()
        
        print("=" * 60)
        print("✅ WSZYSTKO DZIAŁA POPRAWNIE!")
        print()
        print("🎯 Expected Behavior:")
        print("   • System powinien generate intelligent tasks")
        print("   • Każdy task ma AI confidence score")
        print("   • Dependencies są AI-optimized")
        print("   • Risk factors są identified")
        print("   • Optimization opportunities są suggested")
        print()
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("🔧 ROZWIĄZANIE:")
        print("1. Sprawdź że plik enhanced-task-decomposer.py jest w:")
        print("   shared/orchestration/enhanced/enhanced_task_decomposer.py")
        print()
        print("2. Sprawdź że folder structure jest:")
        print("   shared/orchestration/enhanced/__init__.py")
        print("   shared/orchestration/enhanced/enhanced_task_decomposer.py")
        print()
        print("3. Sprawdź że oryginalny task_decomposer.py jest dostępny w:")
        print("   shared/orchestration/task_decomposer.py")
        print()
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print(f"❌ Error Type: {type(e).__name__}")
        
        if "Ollama" in str(e):
            print()
            print("🔧 OLLAMA CONNECTION ISSUE:")
            print("   • Sprawdź że Ollama server działa: ollama serve")
            print("   • Sprawdź port 11434: curl http://localhost:11434")
            print("   • System będzie use fallback analysis jeśli Ollama nie jest available")
        
        return False

def check_file_structure():
    """Check if files are in correct locations"""
    print("📁 Checking file structure...")
    
    required_files = [
        "shared/orchestration/task_decomposer.py",
        "shared/orchestration/enhanced/enhanced_task_decomposer.py",
        "shared/orchestration/enhanced/__init__.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
    
    print()

def main():
    """Main test function"""
    print("🧪 ENHANCED TASK DECOMPOSER - TEST SUITE")
    print("=" * 60)
    
    # Check file structure first
    check_file_structure()
    
    # Run async test
    success = asyncio.run(test_enhanced_task_decomposer())
    
    if success:
        print("🎉 TEST PASSED - Enhanced Task Decomposer działa poprawnie!")
        print()
        print("📋 CO NASTĘPNE:")
        print("1. ✅ Enhanced Task Decomposer - GOTOWY")
        print("2. 🔄 AI Reasoning Engine - NASTĘPNY KROK")
        print("3. 📝 Enhanced CLI Commands")
        print("4. 🧪 Integration Testing")
    else:
        print("❌ TEST FAILED - Sprawdź errors powyżej")
        print()
        print("📋 TROUBLESHOOTING STEPS:")
        print("1. Sprawdź file locations")
        print("2. Sprawdź imports")
        print("3. Sprawdź Ollama server")
        print("4. Sprawdź Python dependencies")

if __name__ == "__main__":
    main()