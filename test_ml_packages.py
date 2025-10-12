#!/usr/bin/env python3
"""
Test ML packages availability
"""
import sys
import os

print("🧪 Testing ML packages availability...")
print(f"Python: {sys.executable}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

packages = ['joblib', 'sklearn', 'pandas', 'numpy', 'neo4j']
results = []

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package}: OK")
        results.append(True)
    except ImportError as e:
        print(f"❌ {package}: FAIL - {e}")
        results.append(False)

success_rate = sum(results) / len(results) * 100
print(f"\n📊 Package availability: {success_rate:.1f}%")

if success_rate >= 80:
    print("🎉 ML packages ready for Agent Zero V2.0!")
    sys.exit(0)
else:
    print("⚠️  Some packages missing - install manually")
    sys.exit(1)
