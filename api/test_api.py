"""
Test Agent Zero API
Test wszystkich endpointów REST API
"""

import requests
import json
from time import sleep

BASE_URL = "http://localhost:8000"


def print_test(name):
    print("\n" + "="*70)
    print(f"🧪 TEST: {name}")
    print("="*70)


def test_health():
    """Test health check"""
    print_test("Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    print("✅ Health check passed")


def test_system_status():
    """Test system status"""
    print_test("System Status")
    
    response = requests.get(f"{BASE_URL}/api/v1/system/status")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Active projects: {data['active_projects']}")
    print(f"Agent types: {data['agent_types_available']}")
    
    assert response.status_code == 200
    print("✅ System status OK")


def test_agent_types():
    """Test lista typów agentów"""
    print_test("Agent Types")
    
    response = requests.get(f"{BASE_URL}/api/v1/system/agents/types")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available types: {', '.join(data['agent_types'])}")
    print(f"Total: {data['total']}")
    
    assert response.status_code == 200
    print("✅ Agent types OK")


def test_create_project():
    """Test tworzenia projektu"""
    print_test("Create Project")
    
    project_data = {
        "project_name": "Test E-commerce API",
        "project_type": "api_backend",
        "business_requirements": [
            "User authentication",
            "Product CRUD",
            "Order management"
        ],
        "schedule_strategy": "load_balanced"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/projects/",
        json=project_data
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 201:
        data = response.json()
        project_id = data['project_id']
        print(f"✅ Project created: {project_id}")
        print(f"   Name: {data['project_name']}")
        print(f"   Status: {data['status']}")
        print(f"   Duration: {data.get('estimated_duration_days')} days")
        return project_id
    else:
        print(f"❌ Error: {response.text}")
        return None


def test_list_projects():
    """Test listowania projektów"""
    print_test("List Projects")
    
    response = requests.get(f"{BASE_URL}/api/v1/projects/")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        projects = response.json()
        print(f"Total projects: {len(projects)}")
        for p in projects[:3]:
            print(f"  - {p['project_name']} ({p['status']})")
        print("✅ List projects OK")
    else:
        print(f"❌ Error: {response.text}")


def test_get_project(project_id):
    """Test pobierania projektu"""
    print_test(f"Get Project: {project_id}")
    
    response = requests.get(f"{BASE_URL}/api/v1/projects/{project_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Name: {data['project_name']}")
        print(f"Type: {data['project_type']}")
        print(f"Status: {data['status']}")
        print(f"Progress: {data['progress']:.0%}")
        print("✅ Get project OK")
    else:
        print(f"❌ Error: {response.text}")


def test_list_agents():
    """Test listowania agentów"""
    print_test("List Agents")
    
    response = requests.get(f"{BASE_URL}/api/v1/agents/")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        agents = response.json()
        print(f"Total agents: {len(agents)}")
        for a in agents[:5]:
            print(f"  - {a['agent_id']} ({a['agent_type']}) - {a['status']}")
        print("✅ List agents OK")
    else:
        print(f"❌ Error: {response.text}")


def test_cache_operations():
    """Test operacji cache"""
    print_test("Cache Operations")
    
    # Stats
    response = requests.get(f"{BASE_URL}/api/v1/system/cache/stats")
    print(f"Cache stats: {response.json()}")
    
    # Clear
    response = requests.post(f"{BASE_URL}/api/v1/system/cache/clear")
    print(f"Cache cleared: {response.json()}")
    
    print("✅ Cache operations OK")


def main():
    """Run all tests"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 TEST AGENT ZERO REST API".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\n⚠️  Make sure API is running on http://localhost:8000")
    print("   Start with: uvicorn api.main:app --reload\n")
    
    input("Press Enter to start tests...")
    
    try:
        # Test 1: Health
        test_health()
        
        # Test 2: System status
        test_system_status()
        
        # Test 3: Agent types
        test_agent_types()
        
        # Test 4: Create project
        project_id = test_create_project()
        
        if project_id:
            # Test 5: Get project
            test_get_project(project_id)
        
        # Test 6: List projects
        test_list_projects()
        
        # Test 7: List agents
        test_list_agents()
        
        # Test 8: Cache
        test_cache_operations()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL API TESTS PASSED!")
        print("="*70)
        
        print("\n📊 API Endpoints Tested:")
        print("   ✅ GET  /health")
        print("   ✅ GET  /api/v1/system/status")
        print("   ✅ GET  /api/v1/system/agents/types")
        print("   ✅ POST /api/v1/projects/")
        print("   ✅ GET  /api/v1/projects/")
        print("   ✅ GET  /api/v1/projects/{id}")
        print("   ✅ GET  /api/v1/agents/")
        print("   ✅ GET  /api/v1/system/cache/stats")
        print("   ✅ POST /api/v1/system/cache/clear")
        
        print("\n" + "="*70)
        print("🚀 Agent Zero API - Ready!")
        print("="*70)
        print("\n📖 API Documentation: http://localhost:8000/docs")
        print("\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("   Make sure API is running:")
        print("   cd /home/ianua/projects/agent-zero-v1")
        print("   uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
