"""
Test Orchestration MVP - Task Decomposer + Team Builder
"""
import sys
from pathlib import Path
import asyncio
import json

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from orchestration.task_decomposer import TaskDecomposer, TaskDependency
from orchestration.team_builder import TeamBuilder


async def main():
    print("🚀 Testing Orchestration MVP: Task Decomposer + Team Builder\n")
    
    # === 1. Task Decomposition ===
    print("=" * 60)
    print("1️⃣ TASK DECOMPOSITION")
    print("=" * 60)
    
    requirements = """
    Build a REST API for user management with the following features:
    - User registration and authentication (JWT tokens)
    - User profile management (CRUD operations)
    - PostgreSQL database for data storage
    - Input validation and error handling
    - Unit tests and integration tests
    - Docker deployment setup
    """
    
    decomposer = TaskDecomposer()
    
    print("\n📋 Requirements:")
    print(requirements)
    
    print("\n⏳ Decomposing project into tasks...")
    tasks = await decomposer.decompose_project(
        requirements=requirements,
        project_type="api"
    )
    
    print(f"\n✅ Created {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. [{task.task_id}] {task.title}")
        print(f"   Agent: {task.agent_type}")
        print(f"   Priority: {task.priority.name}")
        print(f"   Complexity: {task.complexity}/5")
        print(f"   Est. Duration: {task.estimated_duration_hours}h")
        if task.depends_on:
            print(f"   Depends on: {', '.join(task.depends_on)}")
    
    # === 2. Dependency Graph ===
    print("\n" + "=" * 60)
    print("2️⃣ DEPENDENCY GRAPH & PARALLEL EXECUTION")
    print("=" * 60)
    
    dep_graph = decomposer.build_dependency_graph(tasks)
    execution_levels = dep_graph.get_execution_order()
    
    print(f"\n📊 Execution Plan ({len(execution_levels)} phases):\n")
    for level_num, level_tasks in enumerate(execution_levels, 1):
        print(f"Phase {level_num} (parallel execution possible):")
        for task_id in level_tasks:
            task = next(t for t in tasks if t.task_id == task_id)
            print(f"  - {task_id}: {task.title} ({task.agent_type})")
    
    # === 3. Team Building ===
    print("\n" + "=" * 60)
    print("3️⃣ TEAM BUILDING")
    print("=" * 60)
    
    print("\n🏗️ Building team for project...")
    team_builder = TeamBuilder()
    team = team_builder.build_team(tasks, project_id="user_management_api")
    
    print(f"\n✅ Team assembled!")
    summary = team.get_team_summary()
    
    print(f"\n👥 Team Size: {summary['team_size']} agents")
    print(f"📋 Total Tasks: {summary['total_tasks']}")
    
    print("\n🤖 Team Members:")
    for agent_info in summary['agents']:
        print(f"\n  Agent: {agent_info['agent_id']}")
        print(f"  Type: {agent_info['agent_type']}")
        print(f"  Assigned Tasks: {agent_info['assigned_tasks']}")
        print(f"  Workload: {agent_info['workload_hours']}h")
    
    # === 4. Task Assignments ===
    print("\n" + "=" * 60)
    print("4️⃣ TASK ASSIGNMENTS")
    print("=" * 60)
    
    print("\n📌 Task → Agent Mapping:\n")
    for task in tasks:
        assigned_agent = team.get_agent_for_task(task.task_id)
        if assigned_agent:
            print(f"  {task.task_id}: {task.title[:50]}")
            print(f"    → {assigned_agent} ({task.agent_type})")
        else:
            print(f"  {task.task_id}: ⚠️ NOT ASSIGNED")
    
    # === 5. Summary Statistics ===
    print("\n" + "=" * 60)
    print("5️⃣ SUMMARY STATISTICS")
    print("=" * 60)
    
    total_duration = sum(t.estimated_duration_hours for t in tasks)
    critical_tasks = len([t for t in tasks if t.priority.name == 'CRITICAL'])
    high_tasks = len([t for t in tasks if t.priority.name == 'HIGH'])
    
    print(f"\n📊 Project Statistics:")
    print(f"  Total Tasks: {len(tasks)}")
    print(f"  Critical Priority: {critical_tasks}")
    print(f"  High Priority: {high_tasks}")
    print(f"  Total Estimated Duration: {total_duration}h")
    print(f"  Execution Phases: {len(execution_levels)}")
    print(f"  Team Size: {summary['team_size']}")
    
    # Agent type distribution
    agent_types = {}
    for task in tasks:
        agent_types[task.agent_type] = agent_types.get(task.agent_type, 0) + 1
    
    print(f"\n🎯 Tasks by Agent Type:")
    for agent_type, count in sorted(agent_types.items()):
        print(f"  {agent_type}: {count} tasks")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
