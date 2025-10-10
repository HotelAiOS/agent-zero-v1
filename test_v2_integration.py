#!/usr/bin/env python3
"""
Agent Zero V1 - V2.0 Integration Test
End-to-end testing for Week 43 implementation
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path

def test_directory_structure():
    """Test that all required directories exist"""
    required_dirs = [
        "shared/kaizen",
        "shared/knowledge", 
        "shared/utils",
        "cli"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0, missing_dirs

def test_file_creation():
    """Test that all required files were created"""
    required_files = [
        "shared/kaizen/__init__.py",
        "shared/knowledge/__init__.py",
        "shared/utils/simple_tracker.py",
        "cli/__main__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def test_imports():
    """Test that imports work correctly"""
    try:
        sys.path.append('.')
        from shared.utils.simple_tracker import SimpleTracker
        from shared.kaizen import get_intelligent_model_recommendation
        from shared.knowledge import sync_tracker_to_graph_cli
        return True, None
    except ImportError as e:
        return False, str(e)

def test_simple_tracker():
    """Test enhanced SimpleTracker functionality"""
    try:
        from shared.utils.simple_tracker import SimpleTracker
        
        tracker = SimpleTracker()
        test_task_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Test tracking
        tracker.track_task(
            task_id=test_task_id,
            task_type="chat",
            model_used="llama3.2-3b",
            model_recommended="llama3.2-3b", 
            cost=0.0,
            latency=800,
            context={"test": True}
        )
        
        # Test feedback
        tracker.record_feedback(test_task_id, 4, "Test feedback")
        
        # Test model comparison
        comparison = tracker.get_model_comparison(days=1)
        
        tracker.close()
        return True, len(comparison)
    except Exception as e:
        return False, str(e)

def test_cli_commands():
    """Test that CLI commands work"""
    try:
        result = subprocess.run([
            sys.executable, "-m", "cli", "status"
        ], capture_output=True, text=True, cwd=".")
        
        return result.returncode == 0, result.stderr if result.returncode != 0 else None
    except Exception as e:
        return False, str(e)

def main():
    """Run all tests"""
    print("🧪 AGENT ZERO V1 - V2.0 INTEGRATION TEST")
    print("="*50)
    print(f"Test Date: {datetime.now()}")
    print("="*50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("File Creation", test_file_creation), 
        ("Import System", test_imports),
        ("SimpleTracker Enhanced", test_simple_tracker),
        ("CLI Commands", test_cli_commands)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            success, details = test_func()
            if success:
                print(f"   ✅ {test_name}: PASSED")
                if details:
                    print(f"      Details: {details}")
                passed += 1
            else:
                print(f"   ❌ {test_name}: FAILED")
                if details:
                    print(f"      Error: {details}")
        except Exception as e:
            print(f"   ❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")  
    print(f"{'='*50}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ V2.0 Intelligence Layer integration successful")
        print("🚀 System ready for production use")
        return 0
    else:
        print(f"\n⚠️ {total-passed} TESTS FAILED")
        print("❌ Integration needs attention before production")
        return 1

if __name__ == "__main__":
    sys.exit(main())
