#!/usr/bin/env python3
"""
Agent Zero V1 - Comprehensive System Test
Tests all currently working components (11/124 completed tasks)

Usage: python agent_zero_system_test.py
Path: /home/ianua/projects/agent-zero-v1
"""

import os
import sys
import time
import json
import docker
import requests
import subprocess
from datetime import datetime
from pathlib import Path
import importlib.util

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class AgentZeroSystemTest:
    def __init__(self, project_path="/home/ianua/projects/agent-zero-v1"):
        self.project_path = Path(project_path)
        self.test_results = {}
        self.critical_blockers = []
        self.working_components = []
        
        # Change to project directory
        os.chdir(self.project_path)
        
        # Add project path to Python path
        sys.path.insert(0, str(self.project_path))
        
        print(f"{Colors.BOLD}🚀 Agent Zero V1 - System Test{Colors.END}")
        print(f"📁 Project Path: {self.project_path}")
        print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S CEST')}")
        print(f"🎯 Testing 11 completed components + 4 critical blockers")
        print("=" * 60)

    def log_test(self, component_name, status, message="", details=""):
        """Log test result with color coding"""
        if status == "PASS":
            icon = "✅"
            color = Colors.GREEN
            self.working_components.append(component_name)
        elif status == "FAIL":
            icon = "❌"
            color = Colors.RED
            self.critical_blockers.append(component_name)
        elif status == "WARN":
            icon = "⚠️"
            color = Colors.YELLOW
        else:
            icon = "ℹ️"
            color = Colors.BLUE
            
        print(f"{icon} {color}{component_name}: {status}{Colors.END}")
        if message:
            print(f"   📝 {message}")
        if details:
            print(f"   🔍 {details}")
            
        self.test_results[component_name] = {
            "status": status,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def test_docker_services(self):
        """Test Docker Compose services status"""
        print(f"\n{Colors.BOLD}🐳 Docker Services Test{Colors.END}")
        
        try:
            client = docker.from_env()
            
            # Test Neo4j (CRITICAL BLOCKER)
            try:
                neo4j_container = client.containers.get("agent-zero-v1_neo4j_1")
                if neo4j_container.status == "running":
                    # Test connection
                    try:
                        response = requests.get("http://localhost:7474", timeout=5)
                        if response.status_code == 200:
                            self.log_test("Neo4j Service", "PASS", "Container running, port accessible")
                        else:
                            self.log_test("Neo4j Service", "FAIL", f"Port not accessible: {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        self.log_test("Neo4j Service", "FAIL", f"Connection failed: {str(e)}")
                else:
                    self.log_test("Neo4j Service", "FAIL", f"Container status: {neo4j_container.status}")
            except docker.errors.NotFound:
                self.log_test("Neo4j Service", "FAIL", "Container not found - run: docker-compose up -d neo4j")
            
            # Test RabbitMQ (WORKING)
            try:
                rabbitmq_container = client.containers.get("agent-zero-v1_rabbitmq_1")
                if rabbitmq_container.status == "running":
                    try:
                        response = requests.get("http://localhost:15672", timeout=5)
                        if response.status_code == 200:
                            self.log_test("RabbitMQ Service", "PASS", "Container running, management UI accessible")
                        else:
                            self.log_test("RabbitMQ Service", "WARN", "Container running but management UI not accessible")
                    except requests.exceptions.RequestException:
                        self.log_test("RabbitMQ Service", "WARN", "Container running but management UI not accessible")
                else:
                    self.log_test("RabbitMQ Service", "FAIL", f"Container status: {rabbitmq_container.status}")
            except docker.errors.NotFound:
                self.log_test("RabbitMQ Service", "FAIL", "Container not found")
                
            # Test Ollama (LLM Factory)
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=10)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model["name"] for model in models]
                    if "deepseek-coder:33b" in model_names:
                        self.log_test("Ollama LLM Service", "PASS", f"Running with {len(models)} models including deepseek-coder:33b")
                    else:
                        self.log_test("Ollama LLM Service", "WARN", f"Running with {len(models)} models, but deepseek-coder:33b not found")
                else:
                    self.log_test("Ollama LLM Service", "FAIL", f"API not accessible: {response.status_code}")
            except requests.exceptions.RequestException as e:
                self.log_test("Ollama LLM Service", "FAIL", f"Connection failed: {str(e)}")
                
        except Exception as e:
            self.log_test("Docker Client", "FAIL", f"Docker connection failed: {str(e)}")

    def test_python_imports(self):
        """Test Python module imports for working components"""
        print(f"\n{Colors.BOLD}🐍 Python Module Import Test{Colors.END}")
        
        # Working components from PDF
        modules_to_test = [
            ("shared.llm.llm_factory", "LLM Factory"),
            ("shared.llm.ollama_client", "Ollama Client"),
            ("shared.agent_factory.factory", "Agent Factory"),
            ("shared.messaging.bus", "Message Bus"),
            ("shared.messaging.agent_comm", "Agent Communication"),
            ("shared.orchestration.team_builder", "Team Builder"),
            ("shared.agent_factory.lifecycle", "Agent Registry"),
            ("shared.orchestration.project_orchestrator", "Project Orchestrator"),
            ("shared.monitoring.livemonitor", "Live Monitor"),
        ]
        
        for module_path, component_name in modules_to_test:
            try:
                spec = importlib.util.spec_from_file_location(
                    component_name, 
                    self.project_path / f"{module_path.replace('.', '/')}.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.log_test(f"{component_name} Import", "PASS", f"Module loaded successfully")
                else:
                    self.log_test(f"{component_name} Import", "FAIL", f"Module spec not found")
            except Exception as e:
                self.log_test(f"{component_name} Import", "FAIL", f"Import error: {str(e)}")

    def test_critical_blockers(self):
        """Test the 4 critical blockers that need fixing"""
        print(f"\n{Colors.BOLD}🚨 Critical Blockers Test{Colors.END}")
        
        # 1. Neo4j Connection Test (Already covered in docker test)
        
        # 2. AgentExecutor Method Signature Test
        try:
            test_file = self.project_path / "test_full_integration.py"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    content = f.read()
                    if "execute_task(agent, zadanie, katalog_wyjściowy)" in content:
                        self.log_test("AgentExecutor Signature", "PASS", "Method signature is correct")
                    elif "execute_task(zadanie, zespół)" in content:
                        self.log_test("AgentExecutor Signature", "FAIL", "Method signature needs fix on line 129")
                    else:
                        self.log_test("AgentExecutor Signature", "WARN", "Cannot determine method signature")
            else:
                self.log_test("AgentExecutor Signature", "FAIL", "test_full_integration.py not found")
        except Exception as e:
            self.log_test("AgentExecutor Signature", "FAIL", f"Error checking signature: {str(e)}")
        
        # 3. Task Decomposer JSON Parsing Test
        try:
            decomposer_file = self.project_path / "shared/orchestration/task_decomposer.py"
            if decomposer_file.exists():
                # Try to import and test basic functionality
                from shared.orchestration.task_decomposer import TaskDecomposer
                decomposer = TaskDecomposer()
                # This would fail if JSON parsing is broken
                self.log_test("Task Decomposer JSON", "WARN", "Module imports but JSON parsing needs testing")
            else:
                self.log_test("Task Decomposer JSON", "FAIL", "task_decomposer.py not found")
        except Exception as e:
            self.log_test("Task Decomposer JSON", "FAIL", f"JSON parsing broken: {str(e)}")
        
        # 4. WebSocket Frontend Test
        try:
            response = requests.get("http://localhost:8000", timeout=5)
            if response.status_code == 200:
                if "WebSocket" in response.text or "ws://" in response.text:
                    self.log_test("WebSocket Frontend", "PASS", "Frontend loads with WebSocket code")
                else:
                    self.log_test("WebSocket Frontend", "WARN", "Frontend loads but WebSocket code unclear")
            else:
                self.log_test("WebSocket Frontend", "FAIL", f"Frontend not accessible: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.log_test("WebSocket Frontend", "FAIL", f"WebSocket server not responding: {str(e)}")

    def run_integration_test(self):
        """Run the full integration test if possible"""
        print(f"\n{Colors.BOLD}🔧 Integration Test{Colors.END}")
        
        try:
            test_file = self.project_path / "test_full_integration.py"
            if test_file.exists():
                # Try to run the integration test
                result = subprocess.run(
                    [sys.executable, "test_full_integration.py"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.log_test("Full Integration Test", "PASS", "All components integrated successfully")
                else:
                    self.log_test("Full Integration Test", "FAIL", 
                                f"Integration test failed: {result.stderr[:200]}")
            else:
                self.log_test("Full Integration Test", "FAIL", "test_full_integration.py not found")
                
        except subprocess.TimeoutExpired:
            self.log_test("Full Integration Test", "FAIL", "Integration test timed out (>60s)")
        except Exception as e:
            self.log_test("Full Integration Test", "FAIL", f"Integration test error: {str(e)}")

    def test_file_structure(self):
        """Test if required files and directories exist"""
        print(f"\n{Colors.BOLD}📁 File Structure Test{Colors.END}")
        
        required_paths = [
            "shared/llm/llm_factory.py",
            "shared/agent_factory/factory.py",
            "shared/messaging/bus.py",
            "shared/orchestration/project_orchestrator.py",
            "docker-compose.yml",
            "shared/agent_factory/templates/",
        ]
        
        for path in required_paths:
            full_path = self.project_path / path
            if full_path.exists():
                self.log_test(f"File: {path}", "PASS", "Exists")
            else:
                self.log_test(f"File: {path}", "FAIL", "Missing")

    def test_agent_factory(self):
        """Test Agent Factory with 8 specialized agents"""
        print(f"\n{Colors.BOLD}🤖 Agent Factory Test{Colors.END}")
        
        try:
            from shared.agent_factory.factory import AgentFactory
            factory = AgentFactory()
            
            # Test agent creation
            agent_types = ["CodeAgent", "TestAgent", "DevOpsAgent", "ArchitectAgent", 
                          "SecurityAgent", "UIAgent", "ReviewAgent", "DocumentationAgent"]
            
            created_agents = 0
            for agent_type in agent_types:
                try:
                    agent = factory.create_agent(agent_type)
                    if agent:
                        created_agents += 1
                except Exception as e:
                    self.log_test(f"Agent Creation: {agent_type}", "FAIL", f"Creation failed: {str(e)}")
            
            if created_agents == 8:
                self.log_test("Agent Factory", "PASS", f"All 8 agent types created successfully")
            elif created_agents > 0:
                self.log_test("Agent Factory", "WARN", f"Only {created_agents}/8 agent types working")
            else:
                self.log_test("Agent Factory", "FAIL", "No agents could be created")
                
        except Exception as e:
            self.log_test("Agent Factory", "FAIL", f"Factory import/init failed: {str(e)}")

    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n{Colors.BOLD}📊 Test Report Summary{Colors.END}")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "FAIL"])
        warning_tests = len([r for r in self.test_results.values() if r["status"] == "WARN"])
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"⚠️  Warnings: {warning_tests} ({warning_tests/total_tests*100:.1f}%)")
        
        if self.working_components:
            print(f"\n{Colors.GREEN}✅ Working Components ({len(self.working_components)}):{Colors.END}")
            for component in self.working_components:
                print(f"   • {component}")
        
        if self.critical_blockers:
            print(f"\n{Colors.RED}🚨 Critical Blockers ({len(self.critical_blockers)}):{Colors.END}")
            for blocker in self.critical_blockers:
                print(f"   • {blocker}")
        
        # System Health Score
        health_score = (passed_tests / total_tests) * 100
        if health_score >= 80:
            health_color = Colors.GREEN
            health_status = "EXCELLENT"
        elif health_score >= 60:
            health_color = Colors.YELLOW
            health_status = "GOOD"
        else:
            health_color = Colors.RED
            health_status = "NEEDS ATTENTION"
        
        print(f"\n🏥 System Health: {health_color}{health_score:.1f}% - {health_status}{Colors.END}")
        
        # Next Actions
        print(f"\n{Colors.BOLD}🎯 Recommended Actions:{Colors.END}")
        if failed_tests > 0:
            print("1. 🚨 Fix critical blockers immediately (see Critical Actions Tracker in Notion)")
            print("2. 🔧 Run individual component tests for failed items")
            print("3. 📖 Check logs and error messages above")
        else:
            print("1. 🚀 All tests passing - ready for V2.0 development!")
            print("2. 📈 Consider implementing next sprint features")
        
        # Save detailed results to JSON
        report_file = self.project_path / "test_results.json"
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "warnings": warning_tests,
                    "health_score": health_score
                },
                "working_components": self.working_components,
                "critical_blockers": self.critical_blockers,
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {report_file}")

    def run_all_tests(self):
        """Run all system tests"""
        start_time = time.time()
        
        try:
            self.test_docker_services()
            self.test_python_imports()
            self.test_file_structure()
            self.test_agent_factory()
            self.test_critical_blockers()
            self.run_integration_test()
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️ Tests interrupted by user{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Unexpected error during testing: {str(e)}{Colors.END}")
        finally:
            end_time = time.time()
            test_duration = end_time - start_time
            
            self.generate_report()
            print(f"\n⏱️ Test Duration: {test_duration:.2f} seconds")
            print(f"🏁 Testing completed at {datetime.now().strftime('%H:%M:%S CEST')}")

if __name__ == "__main__":
    # Allow custom project path as command line argument
    project_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ianua/projects/agent-zero-v1"
    
    # Run comprehensive system test
    tester = AgentZeroSystemTest(project_path)
    tester.run_all_tests()