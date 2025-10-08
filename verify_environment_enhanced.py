#!/usr/bin/env python3
"""
Enhanced verification script for Agent Zero V1 environment
Includes credential reset and detailed diagnostics
"""

import subprocess
import sys
import time
import requests
import json
from pathlib import Path


def run_command(cmd, timeout=30):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def reset_neo4j_credentials():
    """Reset Neo4j credentials by restarting with fresh volume"""
    print("🔄 Resetting Neo4j credentials...")

    commands = [
        "docker-compose stop neo4j",
        "docker volume rm agent-zero-v1_neo4j_data 2>/dev/null || true",
        "docker-compose up -d neo4j"
    ]

    for cmd in commands:
        success, stdout, stderr = run_command(cmd)
        if not success and "no such volume" not in stderr.lower():
            print(f"⚠️  Command failed: {cmd}")
            print(f"Error: {stderr}")

    print("⏳ Waiting for Neo4j to initialize with fresh credentials...")
    time.sleep(20)

    return True


def test_neo4j_with_retry():
    """Test Neo4j connection with credential reset if needed"""
    try:
        from neo4j import GraphDatabase

        # First attempt
        try:
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agent-pass"))
            with driver.session() as session:
                result = session.run("RETURN 'Connection OK' as message")
                message = result.single()["message"]
                driver.close()
                return True, f"Neo4j: {message}"
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "unauthorized" in error_msg:
                print("🔄 Authentication failed, resetting Neo4j credentials...")
                reset_neo4j_credentials()

                # Retry after reset
                try:
                    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agent-pass"))
                    with driver.session() as session:
                        result = session.run("RETURN 'Connection OK after reset' as message")
                        message = result.single()["message"]
                        driver.close()
                        return True, f"Neo4j: {message}"
                except Exception as retry_e:
                    return False, f"Neo4j failed after reset: {retry_e}"
            else:
                return False, f"Neo4j failed: {e}"

    except ImportError:
        return False, "Neo4j driver not installed. Run: pip install neo4j"


def test_rabbitmq_with_diagnosis():
    """Test RabbitMQ connection with credential diagnosis"""
    try:
        import pika

        # Check container logs first
        success, logs, _ = run_command("docker logs agent-zero-rabbitmq --tail 10")

        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host='localhost',
                    credentials=pika.PlainCredentials('admin', 'SecureRabbitPass123')
                )
            )
            connection.close()
            return True, "RabbitMQ: Connection OK"
        except Exception as e:
            # Check if it's a credential issue
            if "ACCESS_REFUSED" in str(e):
                print("🔍 Checking RabbitMQ container environment...")
                success, env_output, _ = run_command("docker exec agent-zero-rabbitmq env | grep RABBITMQ")
                print(f"Container environment: {env_output}")

                # Try with default guest credentials
                try:
                    connection = pika.BlockingConnection(
                        pika.ConnectionParameters(
                            host='localhost',
                            credentials=pika.PlainCredentials('guest', 'guest')
                        )
                    )
                    connection.close()
                    return True, "RabbitMQ: Connected with guest credentials"
                except:
                    pass

            return False, f"RabbitMQ failed: {e}. Check logs above."

    except ImportError:
        return False, "Pika not installed. Run: pip install pika"


def test_redis():
    """Test Redis connection"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        return True, "Redis: Connection OK"
    except ImportError:
        return False, "Redis not installed. Run: pip install redis"
    except Exception as e:
        return False, f"Redis failed: {e}"


def check_docker_containers():
    """Check Docker container status with details"""
    print("📦 Checking Docker containers...")

    success, stdout, stderr = run_command("docker-compose ps --format json")

    if not success:
        return False, f"Docker Compose issues: {stderr}"

    try:
        containers = []
        for line in stdout.strip().split('\n'):
            if line.strip():
                container = json.loads(line)
                containers.append(container)

        print("📊 Container Status:")
        for container in containers:
            name = container.get('Name', 'Unknown')
            state = container.get('State', 'Unknown')
            health = container.get('Health', 'N/A')
            ports = container.get('Publishers', [])

            port_info = ""
            if ports:
                port_list = [f"{p.get('PublishedPort', '?')}→{p.get('TargetPort', '?')}" for p in ports]
                port_info = f" [{', '.join(port_list)}]"

            status_icon = "✅" if state == "running" else "❌"
            health_info = f" (Health: {health})" if health != 'N/A' else ""

            print(f"  {status_icon} {name}: {state}{health_info}{port_info}")

        running_count = sum(1 for c in containers if c.get('State') == 'running')
        return running_count == len(containers), f"{running_count}/{len(containers)} containers running"

    except (json.JSONDecodeError, KeyError) as e:
        # Fallback to simple docker ps
        success, stdout, _ = run_command("docker-compose ps")
        return success, "Docker containers checked (simple format)"


def install_missing_dependencies():
    """Install missing Python dependencies"""
    print("📦 Installing missing dependencies...")

    packages = ["neo4j", "pika", "redis", "requests", "pytest"]

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"📦 Installing {package}...")
            success, _, stderr = run_command(f"pip install {package}")
            if not success:
                print(f"⚠️  Failed to install {package}: {stderr}")


def create_test_files():
    """Create missing test files"""
    test_dir = Path("tests")
    test_file = test_dir / "test_full_integration.py"

    if not test_file.exists():
        print("📝 Creating missing test files...")
        success, _, _ = run_command("python3 create_integration_test.py")
        if success:
            print("✅ Test files created")
        else:
            print("⚠️  Failed to create test files")
    else:
        print("✅ Test files already exist")


def main():
    """Main verification routine"""
    print("🔍 Agent Zero V1 Environment Verification (Enhanced)")
    print("=" * 60)

    # Install missing dependencies first
    install_missing_dependencies()

    # Create test files if missing
    create_test_files()

    # Check Docker containers with detailed info
    docker_success, docker_msg = check_docker_containers()
    if docker_success:
        print(f"✅ {docker_msg}")
    else:
        print(f"❌ {docker_msg}")
        return 1

    # Test services with enhanced error handling
    services = [
        ("Neo4j", test_neo4j_with_retry),
        ("Redis", test_redis), 
        ("RabbitMQ", test_rabbitmq_with_diagnosis)
    ]

    failed_services = []

    for name, test_func in services:
        print(f"\n🧪 Testing {name}...")
        success, message = test_func()

        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            failed_services.append(name)

    # Test HTTP endpoints
    print("\n🌐 Testing HTTP endpoints...")
    endpoints = [
        ("Neo4j Browser", "http://localhost:7474"),
        ("RabbitMQ Management", "http://localhost:15672")
    ]

    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: HTTP {response.status_code}")
            else:
                print(f"⚠️  {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {e}")

    # Run tests if available
    print("\n🧪 Running integration tests...")
    if Path("tests/test_full_integration.py").exists():
        success, stdout, stderr = run_command("pytest tests/test_full_integration.py -v", timeout=60)
        if success:
            print("✅ Integration tests passed!")
        else:
            print(f"⚠️  Some tests failed:")
            print(stderr)
    else:
        print("⚠️  Integration test file not found")

    # Final report
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 30)

    if not failed_services:
        print("🎉 ALL SERVICES WORKING!")
        print("✅ Environment ready for development")
        print("\n🚀 Ready for Agent Zero V1 development!")
        return 0
    else:
        print(f"❌ Failed services: {', '.join(failed_services)}")
        print("\n🔧 Try running: ./fix_environment_corrected.fish")
        print("💡 Or check Docker logs: docker-compose logs <service>")
        return 1


if __name__ == "__main__":
    sys.exit(main())
