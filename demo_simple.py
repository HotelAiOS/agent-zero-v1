"""Agent Zero V1 - Demo"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'shared'))

from core import AgentZeroCore, ProjectManager
from orchestration import ScheduleStrategy
import time

def main():
    print("\n" + "█"*70)
    print("█  🎬 AGENT ZERO V1 - DEMO".center(70) + "█")
    print("█"*70)
    
    start = time.time()
    
    # Init
    print("\n📍 KROK 1: Inicjalizacja")
    core = AgentZeroCore()
    pm = ProjectManager(core)
    print(f"✅ System ready: {len(core.agent_factory.templates)} typów agentów")
    
    # Projekt
    print("\n📍 KROK 2: Projekt TODO API")
    requirements = [
        "REST API do zarządzania zadaniami",
        "GET /tasks - lista",
        "POST /tasks - dodaj",
        "PUT /tasks/{id} - aktualizuj",
        "DELETE /tasks/{id} - usuń",
        "SQLite database"
    ]
    
    for i, r in enumerate(requirements, 1):
        print(f"   {i}. {r}")
    
    # Create
    print("\n📍 KROK 3: Tworzenie projektu")
    execution = core.create_project(
        project_name="TODO API",
        project_type="api_backend",
        business_requirements=requirements,
        schedule_strategy=ScheduleStrategy.LOAD_BALANCED
    )
    print(f"✅ Projekt: {execution.project_id}")
    
    # Plan
    print("\n📍 KROK 4: Plan")
    if execution.plan:
        print(f"📊 Zadań: {len(execution.plan.tasks)}")
        print(f"⏱️  Czas: {execution.plan.estimated_duration_days:.1f} dni")
        print(f"💰 Koszt: {execution.plan.total_cost_estimate:,.0f} PLN")
    
    # Symulacja
    print("\n📍 KROK 5: Symulacja wykonania")
    phases = ['Planning', 'Design', 'Implementation', 'Testing', 'Review', 'Deployment']
    for num, phase in enumerate(phases, 1):
        execution.current_phase = num
        execution.progress = num / 6
        print(f"   {num}/6: {phase} ({execution.progress:.0%})")
        time.sleep(0.2)
    
    execution.status = 'completed'
    execution.progress = 1.0
    print("✅ Wykonane!")
    
    # Summary
    elapsed = time.time() - start
    print("\n" + "="*70)
    print("✅ DEMO ZAKOŃCZONE!")
    print("="*70)
    print(f"\n🎉 Projekt TODO API:")
    print(f"   ✅ {len(execution.plan.tasks) if execution.plan else 0} zadań")
    print(f"   ✅ Status: {execution.status}")
    print(f"   ⏱️  Czas demo: {elapsed:.2f}s")
    print("\n🚀 Agent Zero V1 - Production Ready!\n")
    
    return True

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF
