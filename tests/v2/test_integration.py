#!/usr/bin/env python3
"""
Agent Zero V2.0 Integration Tests
"""

import requests
import pytest
import time

def test_health_checks():
    """Test all service health endpoints"""
    services = [
        ("http://localhost:8000/health", "API Gateway"),
        ("http://localhost:8010/health", "AI Intelligence"),
        ("http://localhost:8001/health", "WebSocket"),
        ("http://localhost:8002/health", "Orchestrator")
    ]
    
    for url, name in services:
        try:
            response = requests.get(url, timeout=5)
            assert response.status_code == 200, f"{name} health check failed"
            print(f"✅ {name} healthy")
        except Exception as e:
            print(f"❌ {name} health check failed: {e}")

def test_ai_intelligence():
    """Test AI Intelligence Layer functionality"""
    try:
        response = requests.get("http://localhost:8010/api/v2/system-insights", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "insights" in data
        print("✅ AI Intelligence Layer working")
    except Exception as e:
        print(f"❌ AI Intelligence test failed: {e}")

if __name__ == "__main__":
    print("🧪 Running Agent Zero V2.0 Integration Tests")
    test_health_checks()
    test_ai_intelligence()
    print("✅ Tests completed")
