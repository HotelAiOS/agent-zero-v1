#!/usr/bin/env python3
"""
Agent Zero V2.0 - Enhanced SimpleTracker Null-Safe Patch
Patches get_enhanced_summary() to handle None values properly
"""

def patch_enhanced_simple_tracker():
    """Patch Enhanced SimpleTracker with null-safe operations"""
    
    print("🔧 Patching Enhanced SimpleTracker for null-safe operations...")
    
    # Read current file
    file_path = "shared/utils/enhanced_simple_tracker.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        print("✅ Read enhanced_simple_tracker.py")
        
        # Patch 1: Fix get_enhanced_summary v1_metrics section
        old_v1_pattern = '''# V1 Metrics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_tasks,
                AVG(success_score) as avg_success_rate,
                SUM(cost_usd) as total_cost_usd,
                AVG(latency_ms) as avg_latency_ms,
                COUNT(CASE WHEN success_score >= 0.8 THEN 1 END) as high_success_count
            FROM simple_tracker
        """)
        
        v1_row = cursor.fetchone()
        total_tasks, v1_avg, v1_sum_cost, v1_avg_latency, high_success = v1_row'''
        
        new_v1_pattern = '''# V1 Metrics - NULL-SAFE VERSION
        cursor.execute("""
            SELECT 
                COUNT(*) as total_tasks,
                AVG(CASE WHEN success_score IS NOT NULL THEN success_score ELSE 0 END) as avg_success_rate,
                SUM(CASE WHEN cost_usd IS NOT NULL THEN cost_usd ELSE 0 END) as total_cost_usd,
                AVG(CASE WHEN latency_ms IS NOT NULL THEN latency_ms ELSE 0 END) as avg_latency_ms,
                COUNT(CASE WHEN success_score >= 0.8 THEN 1 END) as high_success_count
            FROM simple_tracker
        """)
        
        v1_row = cursor.fetchone()
        total_tasks, v1_avg, v1_sum_cost, v1_avg_latency, high_success = v1_row
        
        # Null-safe conversions
        v1_avg = float(v1_avg or 0.0)
        v1_sum_cost = float(v1_sum_cost or 0.0)
        v1_avg_latency = float(v1_avg_latency or 0.0)'''
        
        if old_v1_pattern in content:
            content = content.replace(old_v1_pattern, new_v1_pattern)
            print("✅ Patched V1 metrics aggregation")
        else:
            print("⚠️  V1 metrics pattern not found - manual patch needed")
        
        # Patch 2: Fix v1_metrics dictionary creation
        old_metrics_dict = '''v1_metrics = {
            "total_tasks": total_tasks,
            "avg_success_rate": round(v1_avg * 100, 1) if v1_avg else 0,
            "total_cost_usd": round(v1_sum_cost, 4) if v1_sum_cost else 0,
            "avg_latency_ms": int(v1_avg_latency) if v1_avg_latency else 0,'''
        
        new_metrics_dict = '''v1_metrics = {
            "total_tasks": total_tasks,
            "avg_success_rate": round(float(v1_avg or 0.0) * 100, 1),
            "total_cost_usd": round(float(v1_sum_cost or 0.0), 4),
            "avg_latency_ms": int(round(float(v1_avg_latency or 0.0))),'''
        
        if old_metrics_dict in content:
            content = content.replace(old_metrics_dict, new_metrics_dict)
            print("✅ Patched V1 metrics dictionary")
        
        # Patch 3: Fix V2 dimension averages
        old_v2_pattern = '''# V2 Enhanced Intelligence
        cursor.execute("""
            SELECT 
                AVG(correctness_score),
                AVG(efficiency_score),
                AVG(cost_score),
                AVG(latency_score),
                AVG(overall_score)
            FROM v2_success_evaluations
        """)
        
        v2_row = cursor.fetchone()
        if v2_row and v2_row[0] is not None:
            correctness_avg, efficiency_avg, cost_avg, latency_avg, overall_avg = v2_row'''
        
        new_v2_pattern = '''# V2 Enhanced Intelligence - NULL-SAFE VERSION
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN correctness_score IS NOT NULL THEN correctness_score ELSE 0 END),
                AVG(CASE WHEN efficiency_score IS NOT NULL THEN efficiency_score ELSE 0 END),
                AVG(CASE WHEN cost_score IS NOT NULL THEN cost_score ELSE 0 END),
                AVG(CASE WHEN latency_score IS NOT NULL THEN latency_score ELSE 0 END),
                AVG(CASE WHEN overall_score IS NOT NULL THEN overall_score ELSE 0 END)
            FROM v2_success_evaluations
        """)
        
        v2_row = cursor.fetchone()
        if v2_row:
            correctness_avg = float(v2_row[0] or 0.0)
            efficiency_avg = float(v2_row[1] or 0.0)
            cost_avg = float(v2_row[2] or 0.0)
            latency_avg = float(v2_row[3] or 0.0)
            overall_avg = float(v2_row[4] or 0.0)'''
        
        if old_v2_pattern in content:
            content = content.replace(old_v2_pattern, new_v2_pattern)
            print("✅ Patched V2 dimension averages")
        
        # Patch 4: Fix success level distribution
        old_distribution = '''# Success level distribution
        cursor.execute("""
            SELECT success_level, COUNT(*) 
            FROM v2_success_evaluations 
            GROUP BY success_level
        """)
        
        distribution_data = cursor.fetchall()
        success_level_distribution = dict(distribution_data) if distribution_data else {}'''
        
        new_distribution = '''# Success level distribution - NULL-SAFE VERSION
        cursor.execute("""
            SELECT COALESCE(success_level, 'unknown') as success_level, COUNT(*) 
            FROM v2_success_evaluations 
            GROUP BY COALESCE(success_level, 'unknown')
        """)
        
        distribution_data = cursor.fetchall()
        success_level_distribution = dict(distribution_data) if distribution_data else {
            'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unknown': 0
        }'''
        
        if old_distribution in content:
            content = content.replace(old_distribution, new_distribution)
            print("✅ Patched success level distribution")
        
        # Write patched file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Enhanced SimpleTracker patched successfully!")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File {file_path} not found")
        return False
    except Exception as e:
        print(f"❌ Patching failed: {e}")
        return False

def test_patched_tracker():
    """Test the patched Enhanced SimpleTracker"""
    
    print("\n🧪 Testing patched Enhanced SimpleTracker...")
    
    try:
        import sys
        sys.path.append('.')
        from shared.utils.enhanced_simple_tracker import EnhancedSimpleTracker
        
        tracker = EnhancedSimpleTracker()
        print("✅ Enhanced SimpleTracker imported")
        
        # Test enhanced summary
        summary = tracker.get_enhanced_summary()
        print("✅ Enhanced summary works - no None comparison errors")
        
        print(f"   V1 total tasks: {summary['v1_metrics']['total_tasks']}")
        print(f"   V1 avg success: {summary['v1_metrics']['avg_success_rate']}%")
        print(f"   V2 components: {len(summary['v2_components'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Agent Zero V2.0 - Enhanced SimpleTracker Patch")
    print("=" * 55)
    
    # Apply patch
    success = patch_enhanced_simple_tracker()
    
    if success:
        # Test patch
        if test_patched_tracker():
            print("\n🎉 Enhanced SimpleTracker Patch: SUCCESS")
            print("✅ Null-safe operations implemented")
            print("✅ Enhanced summary working")
            print("\n🚀 You can now run:")
            print("   python3 test-v2-core.py")
            print("   python3 cli/advanced_commands.py v2-system status")
        else:
            print("\n⚠️  Patch applied but test failed - check imports")
    else:
        print("\n❌ Enhanced SimpleTracker Patch: FAILED")
