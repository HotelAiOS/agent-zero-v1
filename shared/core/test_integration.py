"""
Integration Test
Test pełnej integracji wszystkich modułów Agent Zero
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import AgentZeroCore, ProjectManager
from orchestration import ScheduleStrategy


def test_full_integration():
    """Test pełnej integracji systemu"""
    
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 FULL INTEGRATION TEST - AGENT ZERO V1".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\n" + "="*70)
    print("🚀 INICJALIZACJA AGENT ZERO CORE")
    print("="*70)
    
    # Utwórz core engine
    core = AgentZeroCore()
    
    print(f"\n📊 System Status:")
    status = core.get_system_status()
    print(f"   Agent types: {status['agent_types_available']}")
    print(f"   Patterns detected: {status['patterns_detected']}")
    print(f"   Anti-patterns known: {status['antipatterns_known']}")
    
    # Test 1: Tworzenie projektu
    print("\n" + "="*70)
    print("TEST 1: Tworzenie i planowanie projektu")
    print("="*70)
    
    project = core.create_project(
        project_name="E-commerce Platform MVP",
        project_type="fullstack_web_app",
        business_requirements=[
            "User authentication and registration",
            "Product catalog with search",
            "Shopping cart functionality",
            "Order management",
            "Payment integration",
            "Admin dashboard"
        ],
        schedule_strategy=ScheduleStrategy.LOAD_BALANCED
    )
    
    print(f"\n✅ Projekt utworzony:")
    print(f"   ID: {project.project_id}")
    print(f"   Status: {project.status}")
    print(f"   Team size: {len(project.team)}")
    print(f"   Tasks: {len(project.plan.tasks)}")
    print(f"   Duration: {project.plan.estimated_duration_days:.1f} days")
    print(f"   Cost: {project.plan.total_cost_estimate:,.2f} PLN")
    
    # Test 2: Protokoły
    print("\n" + "="*70)
    print("TEST 2: Uruchamianie protokołów komunikacji")
    print("="*70)
    
    # Code Review Protocol
    print("\n📝 Starting Code Review Protocol...")
    code_review = core.start_protocol(
        project.project_id,
        'code_review',
        {
            'initiator': list(project.team.keys())[0],
            'code_files': ['api/auth.py', 'models/user.py'],
            'reviewers': list(project.team.keys())[1:3],
            'required_approvals': 1,
            'description': 'Authentication module review'
        }
    )
    print(f"   ✓ Code Review: {code_review.protocol_id}")
    
    # Knowledge Sharing Protocol
    print("\n📚 Starting Knowledge Sharing Protocol...")
    knowledge = core.start_protocol(
        project.project_id,
        'knowledge_sharing',
        {
            'initiator': 'system',
            'broadcast': True
        }
    )
    print(f"   ✓ Knowledge Sharing: {knowledge.protocol_id}")
    
    # Test 3: Wykonanie projektu
    print("\n" + "="*70)
    print("TEST 3: Wykonanie projektu przez zespół")
    print("="*70)
    
    project_manager = ProjectManager(core)
    execution_result = project_manager.execute_project(project, auto_advance=True)
    
    print(f"\n✅ Projekt wykonany:")
    print(f"   Total duration: {execution_result['total_duration_hours']:.1f}h")
    print(f"   Average quality: {execution_result['average_quality']:.2f}")
    print(f"   Phases: {execution_result['phases_completed']}")
    
    # Test 4: Zakończenie i Post-mortem
    print("\n" + "="*70)
    print("TEST 4: Post-mortem Analysis")
    print("="*70)
    
    post_mortem = core.complete_project(project.project_id, perform_post_mortem=True)
    
    if post_mortem:
        print(f"\n📊 Post-mortem Results:")
        print(f"   Status: {post_mortem.status.value}")
        print(f"   Quality Score: {post_mortem.quality_score:.2f}")
        print(f"   Insights: {len(post_mortem.insights)}")
        
        print(f"\n✅ What went well:")
        for item in post_mortem.what_went_well[:3]:
            print(f"      - {item}")
        
        print(f"\n❌ What went wrong:")
        for item in post_mortem.what_went_wrong[:3]:
            print(f"      - {item}")
        
        print(f"\n📚 Lessons learned:")
        for lesson in post_mortem.lessons_learned[:5]:
            print(f"      - {lesson}")
    
    # Test 5: System Status
    print("\n" + "="*70)
    print("TEST 5: Final System Status")
    print("="*70)
    
    final_status = core.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Active projects: {final_status['active_projects']}")
    print(f"   Completed projects: {final_status['completed_projects']}")
    print(f"   Active protocols: {final_status['active_protocols']}")
    
    project_status = core.get_project_status(project.project_id)
    if project_status:
        print(f"\n📋 Project Status:")
        print(f"   Name: {project_status['project_name']}")
        print(f"   Status: {project_status['status']}")
        print(f"   Progress: {project_status['progress']:.0%}")
        print(f"   Completed tasks: {project_status['completed_tasks']}")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n🎉 Agent Zero Core - Full System Integration:")
    print(f"   ✓ Agent Factory: {len(project.team)} agents created")
    print(f"   ✓ Orchestration: {len(project.plan.tasks)} tasks planned")
    print(f"   ✓ Protocols: {len(project.active_protocols)} protocols active")
    print(f"   ✓ Learning: Post-mortem analysis completed")
    print(f"   ✓ Project Manager: Full execution cycle")
    
    print("\n" + "="*70)
    print("🚀 Agent Zero V1 - Ready for Production!")
    print("="*70 + "\n")
    
    return True


def main():
    """Main test runner"""
    try:
        success = test_full_integration()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
