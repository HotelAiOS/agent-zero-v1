#!/usr/bin/env python3
"""
Neo4j Connection Fix - Agent Zero V1
Simple and reliable Neo4j startup script
Compatible with Arch Linux + Fish Shell

Save this file as: /home/ianua/projects/agent-zero-v1/scripts/neo4j_fix.py
Then run: python scripts/neo4j_fix.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Configuration
BOLT_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "agent_zero_neo4j_dev"
PROJECT_ROOT = Path("/home/ianua/projects/agent-zero-v1")

def print_header():
    print("=" * 50)
    print("🚀 Agent Zero V1 - Neo4j Connection Fix")
    print("=" * 50)

def ensure_directories():
    """Create necessary directories for Neo4j data."""
    print("📁 Creating required directories...")
    
    directories = [
        "data/neo4j",
        "logs/neo4j", 
        "import",
        "plugins"
    ]
    
    for dir_path in directories:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path}")

def check_docker():
    """Check if Docker is running."""
    print("🐳 Checking Docker status...")
    
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ Docker is available")
            return True
    except FileNotFoundError:
        print("   ❌ Docker is not installed")
        return False
    
    return False

def stop_existing_neo4j():
    """Stop any existing Neo4j containers."""
    print("🛑 Stopping existing Neo4j containers...")
    
    try:
        # Stop container if running
        subprocess.run(["docker", "stop", "agent-zero-neo4j"], 
                      capture_output=True)
        
        # Remove container
        subprocess.run(["docker", "rm", "agent-zero-neo4j"], 
                      capture_output=True)
        
        print("   ✓ Cleaned up existing containers")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

def start_neo4j():
    """Start Neo4j using docker-compose."""
    print("🚀 Starting Neo4j service...")
    
    os.chdir(PROJECT_ROOT)
    
    try:
        result = subprocess.run([
            "docker-compose", "up", "-d", "neo4j"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✓ Neo4j container started")
            return True
        else:
            print(f"   ❌ Failed to start Neo4j: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error starting Neo4j: {e}")
        return False

def wait_for_neo4j(timeout=60):
    """Wait for Neo4j to be ready."""
    print(f"⏳ Waiting for Neo4j to be ready (timeout: {timeout}s)")
    
    start_time = time.time()
    print("   ", end="", flush=True)
    
    while time.time() - start_time < timeout:
        try:
            # Test HTTP endpoint
            result = subprocess.run([
                "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", 
                "http://localhost:7474"
            ], capture_output=True, text=True, timeout=3)
            
            if result.stdout == "200":
                print("\n   ✅ Neo4j is ready!")
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print(".", end="", flush=True)
        time.sleep(2)
    
    print(f"\n   ❌ Neo4j not ready after {timeout}s")
    return False

def test_connection():
    """Test Neo4j connection."""
    print("🔗 Testing Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            BOLT_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message")
            record = result.single()
            
            if record:
                print(f"   ✅ {record['message']}")
                return True
            
        driver.close()
        
    except ImportError:
        print("   ⚠️  neo4j package not available - skipping connection test")
        print("   ℹ️  Install with: pip install neo4j")
        return True  # Don't fail if package not available
        
    except Exception as e:
        print(f"   ❌ Connection test failed: {e}")
        return False
    
    return False

def show_access_info():
    """Show Neo4j access information."""
    print("\n" + "=" * 50)
    print("🎉 Neo4j Fix Completed Successfully!")
    print("=" * 50)
    print("📋 Access Information:")
    print(f"   • Neo4j Browser: http://localhost:7474")
    print(f"   • Bolt URI: {BOLT_URI}")
    print(f"   • Username: {NEO4J_USER}")
    print(f"   • Password: {NEO4J_PASSWORD}")
    print("\n📝 Next Steps:")
    print("   1. Open Neo4j Browser in your web browser")
    print("   2. Login with the credentials above")
    print("   3. Run your Agent Zero application tests")
    print("   4. Check container status: docker ps")
    print("=" * 50)

def main():
    """Main execution function."""
    print_header()
    
    # Change to project directory
    if not PROJECT_ROOT.exists():
        print(f"❌ Project directory not found: {PROJECT_ROOT}")
        print("Please make sure the Agent Zero project is cloned to the correct location")
        sys.exit(1)
    
    # Run fix procedure
    steps = [
        ("Checking Docker", check_docker),
        ("Creating directories", ensure_directories),
        ("Stopping existing containers", stop_existing_neo4j),
        ("Starting Neo4j", start_neo4j),
        ("Waiting for Neo4j ready", lambda: wait_for_neo4j()),
        ("Testing connection", test_connection)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Failed at step: {step_name}")
            sys.exit(1)
    
    show_access_info()

if __name__ == "__main__":
    main()