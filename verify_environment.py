#!/usr/bin/env python3
"""
Quick verification script for Agent Zero V1 environment
"""

import subprocess
import sys
import time
import requests
from neo4j import GraphDatabase
import redis
import pika


def run_command(cmd):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def test_neo4j():
    """Test Neo4j connection"""
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "agent-pass"))
        with driver.session() as session:
            result = session.run("RETURN 'Connection OK' as message")
            message = result.single()["message"]
            driver.close()
            return True, f"Neo4j: {message}"
    except Exception as e:
        return False, f"Neo4j failed: {e}"


def test_redis():
    """Test Redis connection"""
    try:
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        return True, "Redis: Connection OK"
    except Exception as e:
        return False, f"Redis failed: {e}"


def test_rabbitmq():
    """Test RabbitMQ connection"""
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
        return False, f"RabbitMQ failed: {e}"


def main():
    """Main verification routine"""
    print("🔍 Agent Zero V1 Environment Verification")
    print("=" * 50)

    # Check Docker containers
    print("📦 Checking Docker containers...")
    success, stdout, stderr = run_command("docker-compose ps --format json")

    if success:
        print("✅ Docker Compose running")
    else:
        print(f"❌ Docker Compose issues: {stderr}")
        return 1

    # Test services
    services = [
        ("Neo4j", test_neo4j),
        ("Redis", test_redis), 
        ("RabbitMQ", test_rabbitmq)
    ]

    failed_services = []

    for name, test_func in services:
        print(f"🧪 Testing {name}...")
        success, message = test_func()

        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            failed_services.append(name)

    # Test HTTP endpoints
    print("🌐 Testing HTTP endpoints...")
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

    # Final report
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 30)

    if not failed_services:
        print("🎉 ALL SERVICES WORKING!")
        print("✅ Environment ready for development")
        return 0
    else:
        print(f"❌ Failed services: {', '.join(failed_services)}")
        print("🔧 Run fix_environment.fish to resolve issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
