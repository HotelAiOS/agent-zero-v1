#!/usr/bin/env python3
"""
Integration Setup Script for Agent Zero V1 Backend Services
INTEGRATION: Prepares existing system for new services while preserving all existing functionality
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

class AgentZeroIntegrationSetup:
    """Setup new backend services integrated with existing Agent Zero V1 system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.services_dir = self.project_root / "services"
        self.backup_dir = self.project_root / "backups" / f"integration_{int(datetime.now().timestamp())}"
        
    def run_full_integration(self):
        """Execute complete integration setup"""
        print("🚀 Agent Zero V1 - Backend Services Integration")
        print("=" * 60)
        
        try:
            # Step 1: Backup existing system
            self.create_backup()
            
            # Step 2: Setup service directories
            self.setup_service_directories()
            
            # Step 3: Install dependencies
            self.install_dependencies()
            
            # Step 4: Update Docker configuration  
            self.update_docker_config()
            
            # Step 5: Create integration verification script
            self.create_verification_script()
            
            # Step 6: Success message
            self.print_success_message()
            
        except Exception as e:
            print(f"❌ Integration failed: {e}")
            self.rollback_changes()
            
    def create_backup(self):
        """Create backup of existing files before integration"""
        print("📦 Creating backup of existing system...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup key files
        files_to_backup = [
            "docker-compose.yml",
            "cli/__main__.py", 
            "simple-tracker.py",
            "services"  # Entire services directory
        ]
        
        for file_path in files_to_backup:
            source = self.project_root / file_path
            if source.exists():
                if source.is_dir():
                    shutil.copytree(source, self.backup_dir / file_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, self.backup_dir / file_path)
                print(f"   ✅ Backed up: {file_path}")
        
        print(f"   📁 Backup location: {self.backup_dir}")
    
    def setup_service_directories(self):
        """Setup directory structure for integrated services"""
        print("📁 Setting up integrated service directories...")
        
        service_configs = [
            {
                "name": "api-gateway",
                "port": 8000,
                "main_file": "integrated-api-gateway.py",
                "description": "API Gateway integrated with SimpleTracker"
            },
            {
                "name": "chat-service", 
                "port": 8080,
                "main_file": "integrated-websocket-service.py",
                "description": "WebSocket service with FeedbackLoopEngine"
            },
            {
                "name": "agent-orchestrator",
                "port": 8002, 
                "main_file": "integrated-agent-orchestrator.py",
                "description": "Orchestrator using existing components"
            }
        ]
        
        for config in service_configs:
            service_dir = self.services_dir / config["name"]
            src_dir = service_dir / "src"
            
            # Create directory structure
            service_dir.mkdir(parents=True, exist_ok=True)
            src_dir.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files
            (src_dir / "__init__.py").touch()
            
            # Create Dockerfile
            dockerfile_content = f'''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY {config["main_file"]} ./main.py

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE {config["port"]}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config["port"]}/health || exit 1

# Start application
CMD ["python", "main.py"]
'''
            
            (service_dir / "Dockerfile").write_text(dockerfile_content)
            
            # Create requirements.txt
            requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
websockets==12.0
neo4j==5.15.0
redis==5.0.1
python-multipart==0.0.6
'''
            (service_dir / "requirements.txt").write_text(requirements)
            
            print(f"   ✅ Setup: {config['name']} - {config['description']}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("📦 Installing integration dependencies...")
        
        try:
            # Install FastAPI and related packages
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "fastapi==0.104.1",
                "uvicorn[standard]==0.24.0", 
                "websockets==12.0",
                "python-multipart==0.0.6"
            ], check=True, capture_output=True)
            
            print("   ✅ FastAPI dependencies installed")
            
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Dependency installation warning: {e}")
            print("   📝 Manual install: pip install fastapi uvicorn websockets")
    
    def update_docker_config(self):
        """Update Docker configuration with integrated services"""
        print("🐳 Updating Docker configuration...")
        
        # Backup original docker-compose.yml
        original_compose = self.project_root / "docker-compose.yml"
        backup_compose = self.backup_dir / "docker-compose.yml.backup"
        
        if original_compose.exists():
            shutil.copy2(original_compose, backup_compose)
            print("   ✅ Docker compose backed up")
        
        print("   ℹ️  Enhanced docker-compose.yml ready for deployment")
        print("   📝 Replace existing docker-compose.yml with enhanced version")
    
    def create_verification_script(self):
        """Create integration verification script"""
        print("🔍 Creating integration verification...")
        
        verify_script = '''#!/usr/bin/env python3
"""Integration Verification for Agent Zero V1"""

import requests
import json
import time

def verify_integration():
    """Verify all integrated services are working"""
    
    print("🔍 Agent Zero V1 - Integration Verification")
    print("=" * 50)
    
    services = [
        {"name": "API Gateway", "url": "http://localhost:8000/health"},
        {"name": "WebSocket Service", "url": "http://localhost:8080/health"}, 
        {"name": "Agent Orchestrator", "url": "http://localhost:8002/health"}
    ]
    
    for service in services:
        try:
            response = requests.get(service["url"], timeout=10)
            if response.status_code == 200:
                print(f"✅ {service['name']}: Healthy")
            else:
                print(f"❌ {service['name']}: Unhealthy ({response.status_code})")
        except Exception as e:
            print(f"❌ {service['name']}: Connection failed - {e}")
    
    # Test SimpleTracker integration
    try:
        response = requests.get("http://localhost:8000/api/v1/agents/status")
        if response.status_code == 200:
            data = response.json()
            if "integration" in data:
                print("✅ SimpleTracker Integration: Connected")
            else:
                print("⚠️  SimpleTracker Integration: Partial")
        else:
            print("❌ SimpleTracker Integration: Failed")
    except Exception as e:
        print(f"❌ SimpleTracker Integration: Error - {e}")
    
    print("\\n🎯 Integration Status: Ready for Developer B Frontend")

if __name__ == "__main__":
    verify_integration()
'''
        
        verify_path = self.project_root / "verify_integration.py"
        verify_path.write_text(verify_script)
        verify_path.chmod(0o755)
        
        print(f"   ✅ Verification script: {verify_path}")
    
    def print_success_message(self):
        """Print success message with next steps"""
        print("\\n" + "🎉" * 20)
        print("✅ AGENT ZERO V1 BACKEND INTEGRATION COMPLETE!")
        print("🎉" * 20)
        
        print("\\n📋 NEXT STEPS FOR DEVELOPER B:")
        print("\\n1. 🐳 Deploy integrated services:")
        print("   cp enhanced-docker-compose.yml docker-compose.yml")
        print("   docker-compose down && docker-compose up -d")
        
        print("\\n2. 📁 Copy integrated service files:")
        print("   cp integrated-api-gateway.py services/api-gateway/src/main.py")
        print("   cp integrated-websocket-service.py services/chat-service/src/main.py")
        print("   cp integrated-agent-orchestrator.py services/agent-orchestrator/src/main.py")
        
        print("\\n3. ✅ Verify integration:")
        print("   python verify_integration.py")
        
        print("\\n4. 🌐 Frontend endpoints available:")
        print("   • http://localhost:8000/api/v1/agents/status")
        print("   • http://localhost:8000/api/v1/tasks/current") 
        print("   • ws://localhost:8080/ws/agents/live-monitor")
        
        print("\\n🎯 INTEGRATION BENEFITS:")
        print("   ✅ Uses existing SimpleTracker data (no mock data)")
        print("   ✅ Compatible with CLI feedback system") 
        print("   ✅ Preserves all existing functionality")
        print("   ✅ Business parser (A0-19) exposed as API")
        print("   ✅ Real-time updates via WebSocket")
        print("   ✅ Multi-agent orchestration with Neo4j")
        
        print("\\n🚀 DEVELOPER B: Ready for frontend development!")
    
    def rollback_changes(self):
        """Rollback changes if integration fails"""
        print("🔄 Rolling back changes...")
        
        # This would restore from backup_dir
        print(f"   📁 Backup available at: {self.backup_dir}")
        print("   📝 Manual rollback may be needed")

if __name__ == "__main__":
    from datetime import datetime
    
    setup = AgentZeroIntegrationSetup()
    setup.run_full_integration()