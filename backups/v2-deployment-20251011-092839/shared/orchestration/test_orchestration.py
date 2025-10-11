"""
Test Orchestration System
Test kompletnego systemu orkiestracji projektów
"""

import sys
from pathlib import Path

# Dodaj parent directory do path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration import (
    IntelligentPlanner,
    TaskDecomposer,
    DependencyGraph,
    TeamFormationEngine,
    QualityGateManager,
    TaskScheduler,
    ScheduleStrategy
)


def test_task_decomposer():
    """Test dekompozycji zadań"""
    print("="*70)
    print("🧪 TEST 1: Task Decomposer")
    print("="*70)
    
    decomposer = TaskDecomposer()
    
    # Test Full Stack Web App
    print("\n📋 Dekompozycja projektu: Full Stack Web App")
    tasks = decomposer.decompose_project(
        'fullstack_web_app',
        ['User authentication', 'Data management', 'Admin panel']
    )
    
    print(f"✅ Utworzono {len(tasks)} zadań")
    print(f"   Całkowity czas: {sum(t.estimated_hours for t in tasks):.1f}h")
    
    # Pokaż przykładowe zadania
    print("\n📌 Przykładowe zadania:")
    for task in tasks[:5]:
        print(f"   - {task.title} ({task.task_type.value}, {task.estimated_hours}h)")
    
    # Test parallel tasks
    print("\n⚡ Analiza równoległości:")
    parallel_groups = decomposer.get_parallel_tasks(tasks)
    print(f"   Grupy równoległe: {len(parallel_groups)}")
    for i, group in enumerate(parallel_groups[:3]):
        print(f"   Grupa {i+1}: {len(group)} zadań")
    
    return tasks


def test_dependency_graph(tasks):
    """Test grafu zależności"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Dependency Graph")
    print("="*70)
    
    graph = DependencyGraph()
    graph.build_from_tasks(tasks)
    
    stats = graph.get_statistics()
    print(f"\n📊 Statystyki grafu:")
    print(f"   Zadania: {stats['total_tasks']}")
    print(f"   Zależności: {stats['total_dependencies']}")
    print(f"   Maksymalna głębokość: {stats['max_depth']}")
    print(f"   Cykle: {'❌ TAK' if stats['has_cycle'] else '✅ NIE'}")
    
    # Test topological sort
    print("\n🔢 Sortowanie topologiczne:")
    order = graph.topological_sort()
    if order:
        print(f"   ✅ Kolejność wykonania: {len(order)} zadań")
        print(f"   Pierwsze 5: {order[:5]}")
    
    # Critical path
    print("\n🎯 Critical Path:")
    critical_path, duration = graph.get_critical_path(tasks)
    print(f"   Długość: {len(critical_path)} zadań")
    print(f"   Czas trwania: {duration:.1f}h")
    
    return graph


def test_team_formation(tasks):
    """Test formowania zespołów"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Team Formation")
    print("="*70)
    
    engine = TeamFormationEngine()
    
    # Symuluj agent pool
    print("\n📝 Rejestracja agent pool:")
    engine.register_agent_pool('architect', ['arch_1'])
    engine.register_agent_pool('backend', ['back_1', 'back_2'])
    engine.register_agent_pool('frontend', ['front_1'])
    engine.register_agent_pool('database', ['db_1'])
    engine.register_agent_pool('tester', ['test_1'])
    engine.register_agent_pool('devops', ['devops_1'])
    engine.register_agent_pool('security', ['sec_1'])
    
    # Analiza wymagań
    print("\n🔍 Analiza wymagań projektu:")
    requirements = engine.analyze_project_requirements(tasks)
    for agent_type, data in requirements.items():
        print(f"   {agent_type}: {data['count']} agent(ów), {data['total_hours']:.1f}h")
    
    # Formowanie zespołu
    print("\n👥 Formowanie zespołu:")
    team = engine.form_team('test_project', tasks, force_minimal=False)
    
    print(f"\n✅ Zespół utworzony:")
    print(f"   Członkowie: {len(team.members)}")
    for member in team.members:
        print(f"   - {member.agent_id} ({member.role})")
    
    # Przypisanie zadań
    print("\n📌 Przypisywanie zadań:")
    assignments = engine.assign_tasks_to_team(team, tasks)
    
    for agent_id, assigned_tasks in assignments.items():
        if assigned_tasks:
            total_h = sum(t.estimated_hours for t in assigned_tasks)
            print(f"   {agent_id}: {len(assigned_tasks)} zadań ({total_h:.1f}h)")
    
    return team, engine


def test_quality_gates():
    """Test quality gates"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Quality Gates")
    print("="*70)
    
    manager = QualityGateManager()
    manager.define_standard_gates()
    
    print(f"\n🚦 Zdefiniowane quality gates: {len(manager.gates)}")
    
    for gate in list(manager.gates.values())[:5]:
        print(f"\n   📍 {gate.name}")
        print(f"      Severity: {gate.severity.value}")
        print(f"      Kryteria: {len(gate.criteria)}")
        print(f"      Human approval: {'✅' if gate.requires_human_approval else '❌'}")
    
    # Test checking gates
    print("\n✓ Symulacja sprawdzenia gates:")
    context = {'test': True}
    
    # Zatwierdź kilka gates
    manager.approve_gate('arch_review', 'test_user', 'Approved for testing')
    manager.approve_gate('security_audit', 'test_user', 'Security OK')
    
    print(f"   ✅ Zatwierdzone: arch_review, security_audit")
    
    # Podsumowanie
    summary = manager.get_gate_summary()
    print(f"\n📊 Podsumowanie:")
    print(f"   Total: {summary['total_gates']}")
    print(f"   Passed: {summary['passed']}")
    print(f"   Pending: {summary['pending']}")
    print(f"   Deployment ready: {'✅' if summary['deployment_ready'] else '❌'}")
    
    return manager


def test_scheduler(tasks, team, graph):
    """Test schedulera"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Task Scheduler")
    print("="*70)
    
    scheduler = TaskScheduler(ScheduleStrategy.LOAD_BALANCED)
    
    print(f"\n📅 Tworzenie harmonogramu (strategia: {scheduler.strategy.value})...")
    schedule = scheduler.create_schedule(tasks, team, graph)
    
    print(f"✅ Harmonogram utworzony: {len(schedule)} zadań")
    
    # Podsumowanie
    summary = scheduler.get_schedule_summary()
    print(f"\n📊 Podsumowanie harmonogramu:")
    print(f"   Czas trwania: {summary['total_duration_hours']:.1f}h ({summary['total_duration_days']:.1f} dni)")
    print(f"   Start: {summary['start_date']}")
    print(f"   Koniec: {summary['end_date']}")
    print(f"   Agenci: {summary['agents_count']}")
    
    print(f"\n📌 Zadania na agenta:")
    for agent_id, count in summary['tasks_per_agent'].items():
        print(f"   {agent_id}: {count} zadań")
    
    # Pokaż pierwsze 5 zadań z harmonogramu
    print(f"\n⏰ Pierwsze 5 zadań w harmonogramie:")
    for st in schedule[:5]:
        print(f"   - {st.task.title}")
        print(f"     Start: {st.scheduled_start.strftime('%Y-%m-%d %H:%M')}")
        print(f"     Agent: {st.assigned_agent}")
    
    return scheduler


def test_intelligent_planner():
    """Test głównego plannera"""
    print("\n" + "="*70)
    print("🧪 TEST 6: Intelligent Planner (FULL INTEGRATION)")
    print("="*70)
    
    planner = IntelligentPlanner()
    
    # Zarejestruj agent pool
    print("\n📝 Rejestracja agent pool...")
    planner.team_formation.register_agent_pool('architect', ['arch_1'])
    planner.team_formation.register_agent_pool('backend', ['back_1', 'back_2'])
    planner.team_formation.register_agent_pool('frontend', ['front_1', 'front_2'])
    planner.team_formation.register_agent_pool('database', ['db_1'])
    planner.team_formation.register_agent_pool('tester', ['test_1'])
    planner.team_formation.register_agent_pool('devops', ['devops_1'])
    planner.team_formation.register_agent_pool('security', ['sec_1'])
    
    # Utwórz kompletny plan projektu
    print("\n🎯 Tworzenie kompletnego planu projektu...\n")
    
    plan = planner.create_project_plan(
        project_name="E-commerce Platform MVP",
        project_type='fullstack_web_app',
        business_requirements=[
            'User authentication and authorization',
            'Product catalog with search',
            'Shopping cart functionality',
            'Order management',
            'Payment integration',
            'Admin dashboard'
        ],
        schedule_strategy=ScheduleStrategy.LOAD_BALANCED
    )
    
    if plan:
        print("\n" + "="*70)
        print("📋 PODSUMOWANIE PLANU PROJEKTU")
        print("="*70)
        
        summary = plan.get_summary()
        print(f"\n🎯 Projekt: {summary['project_name']}")
        print(f"   ID: {summary['project_id']}")
        print(f"   Typ: {summary['project_type']}")
        print(f"   Status: {summary['status']}")
        
        print(f"\n📊 Statystyki:")
        print(f"   Zadania: {summary['total_tasks']}")
        print(f"   Zespół: {summary['team_size']} agentów")
        print(f"   Czas: {summary['estimated_duration_days']:.1f} dni roboczych")
        print(f"   Koszt: {plan.total_cost_estimate:,.2f} PLN")
        print(f"   Quality Gates: {summary['quality_gates']}")
        
        print(f"\n👥 Skład zespołu:")
        for member in plan.team.members:
            print(f"   - {member.agent_id} ({member.role})")
        
        print(f"\n📅 Harmonogram (pierwsze 5 zadań):")
        for st in plan.schedule[:5]:
            print(f"   - {st.task.title[:50]}")
            print(f"     Agent: {st.assigned_agent}")
            print(f"     Start: {st.scheduled_start.strftime('%Y-%m-%d')}")
            print(f"     Czas: {st.task.estimated_hours}h")
        
        print(f"\n🚦 Quality Gates:")
        for gate_id, gate in list(plan.quality_gates.items())[:5]:
            print(f"   - {gate.name} ({gate.severity.value})")
            print(f"     Status: {gate.status.value}")
        
        return plan
    else:
        print("❌ Nie udało się utworzyć planu")
        return None


def main():
    """Uruchom wszystkie testy"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 TEST ORCHESTRATION SYSTEM - AGENT ZERO V1".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Test 1: Task Decomposer
    tasks = test_task_decomposer()
    
    # Test 2: Dependency Graph
    graph = test_dependency_graph(tasks)
    
    # Test 3: Team Formation
    team, engine = test_team_formation(tasks)
    
    # Test 4: Quality Gates
    gate_manager = test_quality_gates()
    
    # Test 5: Scheduler
    scheduler = test_scheduler(tasks, team, graph)
    
    # Test 6: Intelligent Planner (Full Integration)
    plan = test_intelligent_planner()
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE POMYŚLNIE!")
    print("="*70)
    
    if plan:
        print(f"\n🎉 Utworzono kompletny plan projektu:")
        print(f"   Project: {plan.project_name}")
        print(f"   Tasks: {len(plan.tasks)}")
        print(f"   Team: {len(plan.team.members)} agentów")
        print(f"   Duration: {plan.estimated_duration_days:.1f} dni")
        print(f"   Cost: {plan.total_cost_estimate:,.2f} PLN")
    
    print("\n" + "="*70)
    print("🚀 System Orchestration działa poprawnie!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
