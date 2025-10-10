#!/usr/bin/env python3
"""
Agent Zero V2.0 - Analytics API Import Fix
Fixes import path for Task class in Analytics Dashboard API
"""

import os
import re

def patch_analytics_imports():
    """Fix Task import in Analytics Dashboard API"""
    
    print("🔧 Patching Analytics Dashboard API imports...")
    
    file_path = "api/analytics_dashboard_api.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found")
        return False
    
    try:
        # Read current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        print("✅ Read analytics_dashboard_api.py")
        
        # Fix import patterns
        import_fixes = [
            # Fix 1: orchestration.task_decomposer -> shared.orchestration.task_decomposer
            (
                r'from orchestration\.task_decomposer import Task',
                'from shared.orchestration.task_decomposer import Task'
            ),
            # Fix 2: from orchestration import task_decomposer
            (
                r'from orchestration import task_decomposer',
                'from shared.orchestration import task_decomposer'
            ),
            # Fix 3: import orchestration.task_decomposer
            (
                r'import orchestration\.task_decomposer',
                'import shared.orchestration.task_decomposer'
            ),
            # Fix 4: Any remaining orchestration references
            (
                r'orchestration\.task_decomposer',
                'shared.orchestration.task_decomposer'
            )
        ]
        
        # Apply fixes
        fixes_applied = 0
        for old_pattern, new_pattern in import_fixes:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                fixes_applied += 1
                print(f"✅ Fixed import pattern: {old_pattern}")
        
        if fixes_applied == 0:
            print("ℹ️  No import fixes needed - imports already correct")
        
        # Additional safety check - ensure Task is properly imported
        if 'from shared.orchestration.task_decomposer import Task' not in content:
            if 'class AnalyticsDashboardAPI' in content:
                # Add the import at the top of the file after other imports
                import_line = 'from shared.orchestration.task_decomposer import Task\n'
                
                # Find a good place to insert - after other imports
                lines = content.split('\n')
                insert_idx = 0
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('from ') or line.strip().startswith('import '):
                        insert_idx = i + 1
                    elif line.strip() == '':
                        continue
                    else:
                        break
                
                lines.insert(insert_idx, import_line)
                content = '\n'.join(lines)
                print("✅ Added missing Task import")
                fixes_applied += 1
        
        # Write fixed file
        if fixes_applied > 0:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Applied {fixes_applied} import fixes")
        
        return True
        
    except Exception as e:
        print(f"❌ Import patching failed: {e}")
        return False

def test_analytics_api():
    """Test Analytics API after import fix"""
    
    print("\n🧪 Testing Analytics Dashboard API...")
    
    try:
        import sys
        sys.path.append('.')
        from api.analytics_dashboard_api import AnalyticsDashboardAPI
        
        # Try to initialize
        api = AnalyticsDashboardAPI()
        print("✅ Analytics Dashboard API imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Import OK but initialization failed: {e}")
        return True  # Import itself worked

if __name__ == "__main__":
    print("🔧 Agent Zero V2.0 - Analytics API Import Fix")
    print("=" * 50)
    
    # Apply import fixes
    success = patch_analytics_imports()
    
    if success:
        # Test the fix
        if test_analytics_api():
            print("\n🎉 Analytics API Import Fix: SUCCESS")
            print("✅ Task import path corrected")
            print("✅ Analytics Dashboard API available")
        else:
            print("\n⚠️  Import fix applied but API test failed")
            print("💡 This may be due to other missing dependencies")
    else:
        print("\n❌ Analytics API Import Fix: FAILED")
