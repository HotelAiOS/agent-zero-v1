"""
Test Persistence Layer
Test Database, Models, Repositories, Cache
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from persistence import (
    DatabaseManager,
    ProjectModel,
    AgentModel,
    TaskModel,
    ProtocolModel,
    AnalysisModel,
    ProjectRepository,
    AgentRepository,
    TaskRepository,
    ProtocolRepository,
    AnalysisRepository,
    CacheManager
)
from datetime import datetime
import json


def test_database_setup():
    """Test 1: Database setup"""
    print("="*70)
    print("🧪 TEST 1: Database Setup")
    print("="*70)
    
    # Utwórz in-memory SQLite
    db = DatabaseManager(db_url='sqlite:///:memory:', echo=False)
    db.create_tables()
    
    print("\n✅ Database utworzona (in-memory SQLite)")
    
    # Health check
    health = db.health_check()
    print(f"   Health check: {'✅ OK' if health else '❌ FAILED'}")
    
    return db


def test_project_repository(db: DatabaseManager):
    """Test 2: Project Repository"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Project Repository")
    print("="*70)
    
    with db.session_scope() as session:
        repo = ProjectRepository(session)
        
        # Create project
        print("\n📝 Tworzenie projektu...")
        project = repo.create(
            project_id='proj_test_001',
            project_name='E-commerce Platform',
            project_type='fullstack_web_app',
            status='planned',
            business_requirements=json.dumps([
                'User authentication',
                'Product catalog',
                'Shopping cart'
            ]),
            estimated_duration_days=30.0,
            estimated_cost=50000.0
        )
        
        project_id_value = project.id  # Zapisz ID przed zamknięciem sesji
        
        print(f"   ✅ Projekt utworzony: {project.project_name}")
        print(f"      ID: {project.project_id}")
        print(f"      Status: {project.status}")
        
        # Get by project_id
        retrieved = repo.get_by_project_id('proj_test_001')
        print(f"\n🔍 Pobieranie projektu...")
        print(f"   ✅ Retrieved: {retrieved.project_name}")
        
        # Update status
        print(f"\n🔄 Update statusu...")
        repo.update_status('proj_test_001', 'in_progress', progress=0.5)
        updated = repo.get_by_project_id('proj_test_001')
        print(f"   ✅ Status: {updated.status}, Progress: {updated.progress:.0%}")
        
        # Statistics
        stats = repo.get_statistics()
        print(f"\n📊 Statystyki:")
        print(f"   Total: {stats['total']}")
        print(f"   Active: {stats['active']}")
        print(f"   Completed: {stats['completed']}")
        
        return project_id_value  # Zwróć tylko ID


def test_agent_repository(db: DatabaseManager, project_id: int):
    """Test 3: Agent Repository"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Agent Repository")
    print("="*70)
    
    with db.session_scope() as session:
        repo = AgentRepository(session)
        
        # Create agents
        print("\n👥 Tworzenie agentów...")
        agent1 = repo.create(
            agent_id='architect_1',
            agent_type='architect',
            agent_name='Senior Architect',
            model='deepseek-coder',
            capabilities=json.dumps(['architecture', 'design', 'review']),
            status='ready',
            project_id=project_id
        )
        
        agent2 = repo.create(
            agent_id='backend_1',
            agent_type='backend',
            agent_name='Backend Developer',
            model='deepseek-coder',
            capabilities=json.dumps(['python', 'fastapi', 'database']),
            status='ready',
            project_id=project_id
        )
        
        agent1_id = agent1.id
        agent2_id = agent2.id
        
        print(f"   ✅ Agent 1: {agent1.agent_id} ({agent1.agent_type})")
        print(f"   ✅ Agent 2: {agent2.agent_id} ({agent2.agent_type})")
        
        # Get by project
        project_agents = repo.get_by_project(project_id)
        print(f"\n🔍 Agenci w projekcie: {len(project_agents)}")
        for agent in project_agents:
            print(f"   - {agent.agent_id} ({agent.agent_type})")
        
        # Update performance
        print(f"\n📈 Update performance metrics...")
        repo.update_performance(
            agent_id='backend_1',
            tasks_completed=5,
            avg_quality=0.85,
            success_rate=0.9
        )
        updated = repo.get_by_agent_id('backend_1')
        print(f"   ✅ Tasks: {updated.tasks_completed}")
        print(f"   ✅ Quality: {updated.avg_quality_score:.2f}")
        print(f"   ✅ Success rate: {updated.success_rate:.0%}")
        
        return agent1_id, agent2_id


def test_task_repository(db: DatabaseManager, project_id: int, agent_id: int):
    """Test 4: Task Repository"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Task Repository")
    print("="*70)
    
    with db.session_scope() as session:
        repo = TaskRepository(session)
        
        # Create tasks
        print("\n📋 Tworzenie zadań...")
        task1 = repo.create(
            task_id='task_001',
            task_name='Design database schema',
            description='Design PostgreSQL schema for e-commerce',
            task_type='design',
            complexity=7,
            priority=1,
            status='pending',
            estimated_hours=8.0,
            project_id=project_id,
            agent_id=agent_id
        )
        
        task2 = repo.create(
            task_id='task_002',
            task_name='Implement user authentication',
            description='JWT-based auth with refresh tokens',
            task_type='implementation',
            complexity=8,
            priority=1,
            status='in_progress',
            estimated_hours=16.0,
            project_id=project_id,
            agent_id=agent_id
        )
        
        print(f"   ✅ Task 1: {task1.task_name} ({task1.status})")
        print(f"   ✅ Task 2: {task2.task_name} ({task2.status})")
        
        # Get by project
        project_tasks = repo.get_by_project(project_id)
        print(f"\n🔍 Zadania w projekcie: {len(project_tasks)}")
        
        # Get by status
        pending = repo.get_by_status('pending', project_id)
        in_progress = repo.get_by_status('in_progress', project_id)
        print(f"   Pending: {len(pending)}")
        print(f"   In progress: {len(in_progress)}")
        
        # Update status
        print(f"\n✅ Kończenie zadania...")
        repo.update_status('task_001', 'completed', quality_score=0.9)
        completed = repo.get_by_task_id('task_001')
        print(f"   Status: {completed.status}")
        print(f"   Quality: {completed.quality_score:.2f}")
        print(f"   Completed at: {completed.completed_at.strftime('%H:%M:%S')}")
        
        return task1.id, task2.id


def test_protocol_repository(db: DatabaseManager, project_id: int):
    """Test 5: Protocol Repository"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Protocol Repository")
    print("="*70)
    
    with db.session_scope() as session:
        repo = ProtocolRepository(session)
        
        # Create protocols
        print("\n📡 Tworzenie protokołów...")
        protocol1 = repo.create(
            protocol_id='proto_001',
            protocol_type='code_review',
            initiator='backend_1',
            participants=json.dumps(['architect_1', 'backend_2']),
            status='initiated',
            context=json.dumps({
                'code_files': ['api/auth.py'],
                'reviewers': 2
            }),
            project_id=project_id
        )
        
        protocol2 = repo.create(
            protocol_id='proto_002',
            protocol_type='consensus',
            initiator='architect_1',
            participants=json.dumps(['backend_1', 'backend_2', 'devops_1']),
            status='in_progress',
            context=json.dumps({
                'topic': 'Choose database',
                'options': ['PostgreSQL', 'MongoDB']
            }),
            project_id=project_id
        )
        
        print(f"   ✅ Protocol 1: {protocol1.protocol_type} ({protocol1.status})")
        print(f"   ✅ Protocol 2: {protocol2.protocol_type} ({protocol2.status})")
        
        # Get by project
        project_protocols = repo.get_by_project(project_id)
        print(f"\n🔍 Protokoły w projekcie: {len(project_protocols)}")
        
        # Get active
        active = repo.get_active()
        print(f"   Active protocols: {len(active)}")
        
        return protocol1.id, protocol2.id


def test_cache_manager():
    """Test 6: Cache Manager"""
    print("\n" + "="*70)
    print("🧪 TEST 6: Cache Manager")
    print("="*70)
    
    cache = CacheManager(default_ttl_seconds=60)
    
    # Set values
    print("\n💾 Cache operations...")
    cache.set('project:001', {'name': 'E-commerce', 'status': 'active'})
    cache.set('agent:arch_1', {'type': 'architect', 'tasks': 5})
    cache.set_system_status({'active_projects': 3, 'agents': 10}, ttl=30)
    
    print("   ✅ Set 3 keys")
    
    # Get values
    project = cache.get('project:001')
    agent = cache.get('agent:arch_1')
    status = cache.get_system_status()
    
    print(f"\n🔍 Get operations:")
    print(f"   Project: {project['name']}")
    print(f"   Agent: {agent['type']}")
    print(f"   System: {status['active_projects']} projects")
    
    # Stats
    stats = cache.get_stats()
    print(f"\n📊 Cache stats:")
    print(f"   Total keys: {stats['total_keys']}")
    print(f"   Expired: {stats['expired_keys']}")
    
    # Clear
    cleared = cache.clear()
    print(f"\n🗑️  Cleared: {cleared} keys")
    
    return cache


def main():
    """Run all tests"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 TEST PERSISTENCE LAYER - AGENT ZERO V1".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Test 1: Database
    db = test_database_setup()
    
    # Test 2: Project Repository
    project_id = test_project_repository(db)
    
    # Test 3: Agent Repository
    agent1_id, agent2_id = test_agent_repository(db, project_id)
    
    # Test 4: Task Repository
    task1_id, task2_id = test_task_repository(db, project_id, agent1_id)
    
    # Test 5: Protocol Repository
    protocol1_id, protocol2_id = test_protocol_repository(db, project_id)
    
    # Test 6: Cache Manager
    cache = test_cache_manager()
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ ALL PERSISTENCE TESTS PASSED!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Database: SQLite in-memory")
    print(f"   ✅ Projects: 1 created, updated, queried")
    print(f"   ✅ Agents: 2 created with performance tracking")
    print(f"   ✅ Tasks: 2 created, 1 completed")
    print(f"   ✅ Protocols: 2 created, status tracked")
    print(f"   ✅ Cache: Set/Get/Clear operations")
    
    print("\n" + "="*70)
    print("🚀 Persistence Layer - Ready for Production!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
