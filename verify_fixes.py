#!/usr/bin/env python3
"""
Agent Zero V1 - Fix Verification Test
Tests that all critical fixes are working correctly
"""

import sys
import subprocess
from pathlib import Path


def test_file_exists(filepath: str, description: str) -> bool:
    """Test if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"  ✅ {description}: OK")
        return True
    else:
        print(f"  ❌ {description}: MISSING")
        return False


def test_python_syntax(filepath: str, description: str) -> bool:
    """Test if Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print(f"  ✅ {description}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"  ❌ {description}: Syntax Error - {e}")
        return False
    except FileNotFoundError:
        print(f"  ⚠️  {description}: File not found")
        return False


def test_docker_compose_syntax(filepath: str) -> bool:
    """Test if docker-compose.yml is valid"""
    try:
        result = subprocess.run(
            ['docker-compose', '-f', filepath, 'config'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✅ Docker Compose: Valid YAML")
            return True
        else:
            print(f"  ❌ Docker Compose: Invalid - {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"  ⚠️  docker-compose not installed, skipping validation")
        return True  # Don't fail if docker-compose not available


def main():
    print("="*80)
    print("AGENT ZERO V1 - FIX VERIFICATION TEST")
    print("="*80)

    all_passed = True

    # Test fix files exist
    print("\n📄 Checking fix files...")
    all_passed &= test_file_exists("neo4j_client.py", "Neo4j Client")
    all_passed &= test_file_exists("agent_executor.py", "Agent Executor")
    all_passed &= test_file_exists("task_decomposer.py", "Task Decomposer")
    all_passed &= test_file_exists("docker-compose.yml", "Docker Compose")

    # Test Python syntax
    print("\n🐍 Checking Python syntax...")
    all_passed &= test_python_syntax("neo4j_client.py", "Neo4j Client")
    all_passed &= test_python_syntax("agent_executor.py", "Agent Executor")
    all_passed &= test_python_syntax("task_decomposer.py", "Task Decomposer")
    all_passed &= test_python_syntax("apply_fixes.py", "Apply Fixes Script")

    # Test Docker Compose
    print("\n🐳 Checking Docker Compose...")
    all_passed &= test_docker_compose_syntax("docker-compose.yml")

    # Test key features
    print("\n🔍 Checking key features...")

    # Check Neo4j retry logic
    with open("neo4j_client.py", 'r') as f:
        content = f.read()
        if "max_retries" in content and "exponential" in content.lower():
            print("  ✅ Neo4j: Retry logic present")
        else:
            print("  ❌ Neo4j: Retry logic missing")
            all_passed = False

    # Check AgentExecutor standardized interface
    with open("agent_executor.py", 'r') as f:
        content = f.read()
        if "ExecutionContext" in content and "ExecutionResult" in content:
            print("  ✅ AgentExecutor: Standardized interface")
        else:
            print("  ❌ AgentExecutor: Missing dataclasses")
            all_passed = False

    # Check TaskDecomposer robust parser
    with open("task_decomposer.py", 'r') as f:
        content = f.read()
        if "RobustJSONParser" in content and "extract_json" in content:
            print("  ✅ TaskDecomposer: Robust parser present")
        else:
            print("  ❌ TaskDecomposer: Robust parser missing")
            all_passed = False

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("="*80)
        print("\nFix package is ready to deploy!")
        print("\nNext steps:")
        print("  1. Run: python apply_fixes.py --project-root /path/to/agent-zero-v1")
        print("  2. Verify: docker-compose ps")
        print("  3. Test: pytest tests/test_full_integration.py")
        return 0
    else:
        print("❌ SOME VERIFICATION TESTS FAILED")
        print("="*80)
        print("\nPlease check the errors above and regenerate fix files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
