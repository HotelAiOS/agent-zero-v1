#!/usr/bin/env python3
"""Working test for Agent Zero V1"""

import requests
import json
import time

def test_system():
    print("🧪 AGENT ZERO V1 - SYSTEM TEST")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Basic connection
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"📡 Basic connection: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Test 2: Health endpoint variations
    health_endpoints = [
        "/health", 
        "/api/health", 
        "/api/v1/health",
        "/status"
    ]
    
    for endpoint in health_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=3)
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK")
                print(f"   Response: {response.json()}")
                break
        except:
            print(f"⚠️ {endpoint}: Not found")
    
    # Test 3: List available endpoints
    try:
        response = requests.get(f"{base_url}/docs", timeout=3)
        if response.status_code == 200:
            print("✅ FastAPI docs available at /docs")
        else:
            print("⚠️ No FastAPI docs")
    except:
        print("⚠️ Docs endpoint not accessible")
    
    print("🎯 Test completed!")

if __name__ == "__main__":
    test_system()
