#!/usr/bin/env python3
"""
Agent Zero V1 - System Test for Virtual Environment
==================================================
Test systemowy przystosowany do środowiska venv w Arch Linux.
"""

import os
import sys
import socket
from pathlib import Path

def print_banner():
    """Display test banner."""
    print("🧪 Agent Zero V1 - System Test (Virtual Environment)")
    print("====================================================")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check if in virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    print(f"🌟 Virtual environment: {'✅ Active' if in_venv else '❌ Not active'}")
    print()

def check_dependency(module_name, package_name=None):
    """Check if Python module is available."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError:
        pkg = package_name or module_name
        print(f"❌ {module_name} - MISSING")
        print(f"   Install with: pip install {pkg}")
        return False

def check_service_port(port, service_name):
    """Check if service is running on specified port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"✅ {service_name} (port {port}) - RUNNING")
            return True
        else:
            print(f"❌ {service_name} (port {port}) - NOT RUNNING")
            return False
    except Exception as e:
        print(f"❌ {service_name} (port {port}) - ERROR: {e}")
        return False

def test_agent_executor():
    """Test AgentExecutor with proper error handling."""
    try:
        # Add current directory to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Try to import AgentExecutor
        from shared.execution.agent_executor import AgentExecutor
        
        # Create test instance
        executor = AgentExecutor()
        
        # Mock agent
        class MockAgent:
            def __init__(self):
                self.agent_type = "test"
                self.capabilities = ["test"]
                self.id = "test-agent-001"
        
        # Test task
        test_task = {
            "id": "test-task-001",
            "type": "system_test",
            "description": "Test task for signature verification",
            "parameters": {}
        }
        
        # Create test directory
        test_dir = "/tmp/agent_zero_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test execute_task method signature
        mock_agent = MockAgent()
        result = executor.execute_task(mock_agent, test_task, test_dir)
        
        print("✅ AgentExecutor.execute_task - OK")
        print(f"   Result: {type(result).__name__}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except ImportError as e:
        print(f"❌ AgentExecutor import - ERROR: {e}")
        print("   Check if shared/execution/agent_executor.py exists")
        return False
    except Exception as e:
        print(f"❌ AgentExecutor test - ERROR: {e}")
        return False

def check_file_structure():
    """Check critical project files."""
    print("\n📁 File Structure Check...")
    
    critical_files = [
        "shared/execution/agent_executor.py",
        "shared/execution/project_orchestrator.py", 
        "shared/monitoring/websocket_monitor.py",
        "scripts/websocket_monitor_minimal.py",
    ]
    
    found_files = 0
    total_files = len(critical_files)
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
            found_files += 1
        else:
            print(f"❌ {file_path} - MISSING")
    
    return found_files, total_files

def main():
    """Main test execution."""
    print_banner()
    
    # Track results
    tests_passed = 0
    total_tests = 0
    
    # 1. Python Dependencies Check
    print("📦 1. Python Dependencies Check...")
    dependencies = [
        ("docker", "docker"),
        ("aiohttp_cors", "aiohttp-cors"),
        ("neo4j", "neo4j"),
        ("pytest", "pytest"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("websockets", "websockets"),
        ("aiohttp", "aiohttp"),
    ]
    
    for module, package in dependencies:
        total_tests += 1
        if check_dependency(module, package):
            tests_passed += 1
    
    # 2. Services Check
    print("\n🔌 2. Services Check...")
    services = [
        (7474, "Neo4j HTTP"),
        (7687, "Neo4j Bolt"),
        (5672, "RabbitMQ"),
        (6379, "Redis"),
    ]
    
    for port, service in services:
        total_tests += 1
        if check_service_port(port, service):
            tests_passed += 1
    
    # 3. Core Components Test
    print("\n⚙️  3. Core Components Test...")
    total_tests += 1
    if test_agent_executor():
        tests_passed += 1
    
    # 4. File Structure Check
    found_files, total_files = check_file_structure()
    tests_passed += found_files
    total_tests += total_files
    
    # Final Results
    print("\n" + "="*60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} passed")
    print(f"📊 Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - System fully operational!")
        success_rate = 100
    elif tests_passed >= total_tests * 0.8:
        print("✅ MOST TESTS PASSED - System mostly operational")
        success_rate = 80
    else:
        print("❌ SOME TESTS FAILED - System needs attention")
        success_rate = 50
    
    print("\n🔧 Next Steps:")
    if success_rate == 100:
        print("   → System ready for development!")
        print("   → Start WebSocket: ./scripts/start_websocket_venv.sh")
    elif success_rate >= 80:
        print("   → Start missing services (Neo4j, RabbitMQ, Redis)")
    else:
        print("   → Check file structure and dependencies")
    
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())