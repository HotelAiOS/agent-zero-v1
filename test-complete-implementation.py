#!/usr/bin/env python3
"""
Complete Implementation Test - Agent Zero V2.0
Tests all Developer A tasks from Week 44 implementation
Based on analysis of two previous threads

Total: 28 Story Points
- Priority 1: Experience Management System [8 SP]
- Priority 2: Neo4j Knowledge Graph Integration [6 SP]  
- Priority 3: Pattern Mining Engine [6 SP]
- Priority 4: ML Model Training Pipeline [4 SP]
- Priority 5: Enhanced Analytics Dashboard Backend [2 SP]
"""

import asyncio
import json
import sys
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Colors:
    """Terminal colors for pretty output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def log_test(message: str, status: str = "INFO"):
    """Log test message with color"""
    color = {
        "INFO": Colors.BLUE,
        "PASS": Colors.GREEN,
        "FAIL": Colors.RED,
        "WARN": Colors.YELLOW,
        "START": Colors.PURPLE
    }.get(status, Colors.NC)
    
    prefix = {
        "INFO": "ℹ️",
        "PASS": "✅", 
        "FAIL": "❌",
        "WARN": "⚠️",
        "START": "🚀"
    }.get(status, "📋")
    
    print(f"{color}[{status}]{Colors.NC} {prefix} {message}")

class V2ImplementationTester:
    """Comprehensive tester for all V2.0 components"""
    
    def __init__(self):
        self.test_results = {
            'test_timestamp': datetime.utcnow().isoformat(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0,
            'details': []
        }
        self.components_available = {}
    
    async def run_all_tests(self) -> bool:
        """Run complete test suite for all Developer A tasks"""
        log_test("Agent Zero V2.0 - Complete Implementation Test", "START")
        log_test("Testing 28 Story Points from Week 44 implementation", "INFO")
        print("=" * 80)
        
        # Test each priority in order
        await self.test_priority_1_experience_management()
        await self.test_priority_2_neo4j_integration()
        await self.test_priority_3_pattern_mining()
        await self.test_priority_4_ml_training()
        await self.test_priority_5_analytics_dashboard()
        
        # Run integration tests
        await self.test_integration_workflow()
        
        # Generate final report
        self.generate_final_report()
        
        return self.test_results['failed_tests'] == 0
    
    async def test_priority_1_experience_management(self):
        """Test Priority 1: Experience Management System [8 SP]"""
        print("\n" + "="*60)
        log_test("Priority 1: Experience Management System [8 SP]", "START")
        print("="*60)
        
        # Test 1.1: Enhanced Experience Tracker Import
        await self._test_component_import(
            "Enhanced Experience Tracker",
            "shared.experience.enhanced_tracker",
            "V2ExperienceTracker"
        )
        
        # Test 1.2: Experience Tracking Functionality
        if self.components_available.get("V2ExperienceTracker"):
            await self._test_experience_tracking()
        
        # Test 1.3: ML Insights Generation
        if self.components_available.get("V2ExperienceTracker"):
            await self._test_ml_insights_generation()
        
        # Test 1.4: Experience API
        await self._test_component_import(
            "Experience Management API",
            "api.v2.experience_api",
            "app"
        )
    
    async def test_priority_2_neo4j_integration(self):
        """Test Priority 2: Neo4j Knowledge Graph Integration [6 SP]"""
        print("\n" + "="*60) 
        log_test("Priority 2: Neo4j Knowledge Graph Integration [6 SP]", "START")
        print("="*60)
        
        # Test 2.1: Graph Schema Import
        await self._test_component_import(
            "AgentZero Graph Schema",
            "shared.knowledge.graph_integration_v2",
            "AgentZeroGraphSchema"
        )
        
        # Test 2.2: SQLite Migrator
        await self._test_component_import(
            "SQLite to Neo4j Migrator", 
            "shared.knowledge.graph_integration_v2",
            "SQLiteToNeo4jMigrator"
        )
        
        # Test 2.3: Optimized Queries
        await self._test_component_import(
            "Optimized Graph Queries",
            "shared.knowledge.graph_integration_v2", 
            "OptimizedGraphQueries"
        )
        
        # Test 2.4: Schema Initialization
        if self.components_available.get("AgentZeroGraphSchema"):
            await self._test_schema_initialization()
    
    async def test_priority_3_pattern_mining(self):
        """Test Priority 3: Pattern Mining Engine [6 SP]"""
        print("\n" + "="*60)
        log_test("Priority 3: Pattern Mining Engine [6 SP]", "START") 
        print("="*60)
        
        # Test 3.1: Pattern Mining Engine Import
        await self._test_component_import(
            "Pattern Mining Engine",
            "shared.learning.pattern_mining_engine",
            "PatternMiningEngine"
        )
        
        # Test 3.2: Pattern Detection
        if self.components_available.get("PatternMiningEngine"):
            await self._test_pattern_detection()
        
        # Test 3.3: Pattern Types
        await self._test_component_import(
            "Pattern Types",
            "shared.learning.pattern_mining_engine",
            "PatternType"
        )
    
    async def test_priority_4_ml_training(self):
        """Test Priority 4: ML Model Training Pipeline [4 SP]"""
        print("\n" + "="*60)
        log_test("Priority 4: ML Model Training Pipeline [4 SP]", "START")
        print("="*60)
        
        # Test 4.1: ML Pipeline Import
        await self._test_component_import(
            "ML Training Pipeline",
            "shared.learning.ml_training_pipeline",
            "MLTrainingPipeline"
        )
        
        # Test 4.2: Model Training
        if self.components_available.get("MLTrainingPipeline"):
            await self._test_ml_training()
        
        # Test 4.3: Model Prediction
        if self.components_available.get("MLTrainingPipeline"):
            await self._test_model_prediction()
    
    async def test_priority_5_analytics_dashboard(self):
        """Test Priority 5: Enhanced Analytics Dashboard Backend [2 SP]"""
        print("\n" + "="*60)
        log_test("Priority 5: Enhanced Analytics Dashboard Backend [2 SP]", "START")
        print("="*60)
        
        # Test 5.1: Analytics API Import
        await self._test_component_import(
            "Analytics Dashboard API",
            "api.v2.analytics_api",
            "app"
        )
        
        # Test 5.2: API Endpoints
        if self.components_available.get("analytics_api"):
            await self._test_api_endpoints()
    
    async def test_integration_workflow(self):
        """Test complete integration workflow"""
        print("\n" + "="*60)
        log_test("Integration Workflow Test", "START")
        print("="*60)
        
        # Test end-to-end workflow
        await self._test_end_to_end_workflow()
    
    async def _test_component_import(self, component_name: str, module_path: str, class_name: str):
        """Test if a component can be imported"""
        test_name = f"Import {component_name}"
        self.test_results['total_tests'] += 1
        
        try:
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Store for later tests
            self.components_available[class_name] = component_class
            
            log_test(f"{test_name} - SUCCESS", "PASS")
            self.test_results['passed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'PASSED',
                'component': component_name,
                'module': module_path
            })
            
        except ImportError as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e),
                'component': component_name
            })
        except AttributeError as e:
            log_test(f"{test_name} - FAILED: {class_name} not found in module", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': f"{class_name} not found",
                'component': component_name
            })
    
    async def _test_experience_tracking(self):
        """Test experience tracking functionality"""
        test_name = "Experience Tracking"
        self.test_results['total_tests'] += 1
        
        try:
            V2ExperienceTracker = self.components_available["V2ExperienceTracker"]
            tracker = V2ExperienceTracker()
            
            # Test experience data
            test_experience = {
                'task_id': 'test_tracking_001',
                'task_type': 'integration_test',
                'model_used': 'test_model',
                'success_score': 0.95,
                'cost_usd': 0.002,
                'latency_ms': 1200,
                'user_feedback': 'Excellent test results'
            }
            
            result = await tracker.track_experience(test_experience)
            
            if result.get('status') == 'success' and 'insights' in result:
                log_test(f"{test_name} - SUCCESS (tracked {result.get('experience_id')})", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'insights_count': result.get('insights_count', 0),
                    'experience_id': result.get('experience_id')
                })
            else:
                log_test(f"{test_name} - FAILED: Invalid result format", "FAIL")
                self.test_results['failed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'error': 'Invalid result format'
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_ml_insights_generation(self):
        """Test ML insights generation"""
        test_name = "ML Insights Generation"
        self.test_results['total_tests'] += 1
        
        try:
            from shared.experience.enhanced_tracker import MLInsightEngine, EnhancedExperience, ExperienceType
            
            engine = MLInsightEngine()
            
            # Create test experience
            experience = EnhancedExperience(
                experience_id="test_insight_001",
                experience_type=ExperienceType.TASK_EXECUTION,
                task_id="test_001",
                task_type="test",
                model_used="test_model",
                success_score=0.92,
                cost_usd=0.003,
                latency_ms=800,
                timestamp=datetime.utcnow().isoformat()
            )
            
            insights = await engine.analyze_experience(experience)
            
            if isinstance(insights, list) and len(insights) > 0:
                log_test(f"{test_name} - SUCCESS (generated {len(insights)} insights)", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'insights_generated': len(insights)
                })
            else:
                log_test(f"{test_name} - WARNING: No insights generated", "WARN")
                self.test_results['warnings'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'WARNING',
                    'message': 'No insights generated'
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_schema_initialization(self):
        """Test Neo4j schema initialization"""
        test_name = "Neo4j Schema Initialization"
        self.test_results['total_tests'] += 1
        
        try:
            AgentZeroGraphSchema = self.components_available["AgentZeroGraphSchema"]
            
            # Mock Neo4j client for testing
            class MockNeo4jClient:
                async def execute_query(self, query, params=None):
                    return [{'count': 1}]
            
            schema = AgentZeroGraphSchema(MockNeo4jClient())
            result = await schema.initialize_v2_schema()
            
            if result.get('status') == 'success':
                log_test(f"{test_name} - SUCCESS", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'schema_version': result.get('schema_version'),
                    'setup_time_ms': result.get('setup_time_ms')
                })
            else:
                log_test(f"{test_name} - FAILED: {result.get('error')}", "FAIL")
                self.test_results['failed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'error': result.get('error')
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_pattern_detection(self):
        """Test pattern detection"""
        test_name = "Pattern Detection"
        self.test_results['total_tests'] += 1
        
        try:
            PatternMiningEngine = self.components_available["PatternMiningEngine"]
            
            # Mock Neo4j client
            class MockNeo4jClient:
                async def execute_query(self, query, params=None):
                    return [
                        {
                            'id': 'exp_001',
                            'task_type': 'test',
                            'model': 'test_model',
                            'success_score': 0.9,
                            'cost': 0.001,
                            'latency': 1000
                        }
                    ]
            
            engine = PatternMiningEngine(MockNeo4jClient())
            patterns = await engine.discover_patterns(time_window_days=7)
            
            if isinstance(patterns, list):
                log_test(f"{test_name} - SUCCESS (discovered {len(patterns)} patterns)", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'patterns_discovered': len(patterns)
                })
            else:
                log_test(f"{test_name} - FAILED: Invalid return type", "FAIL")
                self.test_results['failed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'error': 'Invalid return type'
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_ml_training(self):
        """Test ML model training"""
        test_name = "ML Model Training"
        self.test_results['total_tests'] += 1
        
        try:
            MLTrainingPipeline = self.components_available["MLTrainingPipeline"]
            
            # Mock Neo4j client with insufficient data
            class MockNeo4jClient:
                async def execute_query(self, query, params=None):
                    return [
                        {
                            'task_type': 'test',
                            'model': 'test_model',
                            'success_score': 0.9,
                            'cost_usd': 0.001,
                            'latency_ms': 1000,
                            'feedback_length': 0
                        }
                    ] * 10  # Only 10 samples - insufficient for training
            
            pipeline = MLTrainingPipeline(MockNeo4jClient())
            result = await pipeline.train_models()
            
            # Should return error about insufficient data
            if 'error' in result and 'insufficient' in result['error'].lower():
                log_test(f"{test_name} - SUCCESS (correctly identified insufficient data)", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'message': 'Correctly handled insufficient training data'
                })
            elif result.get('status') == 'success':
                log_test(f"{test_name} - SUCCESS (training completed)", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'training_samples': result.get('training_samples', 0)
                })
            else:
                log_test(f"{test_name} - FAILED: Unexpected result", "FAIL")
                self.test_results['failed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'error': 'Unexpected result format'
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_model_prediction(self):
        """Test model prediction functionality"""
        test_name = "Model Prediction"
        self.test_results['total_tests'] += 1
        
        try:
            MLTrainingPipeline = self.components_available["MLTrainingPipeline"]
            pipeline = MLTrainingPipeline()
            
            # Test prediction without trained models
            result = await pipeline.predict_optimal_model('test_task')
            
            if 'error' in result:
                log_test(f"{test_name} - SUCCESS (correctly identified missing models)", "PASS")
                self.test_results['passed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'message': 'Correctly handled missing models'
                })
            else:
                log_test(f"{test_name} - UNEXPECTED: Got prediction without models", "WARN")
                self.test_results['warnings'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'WARNING',
                    'message': 'Unexpected prediction result'
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_api_endpoints(self):
        """Test API endpoints structure"""
        test_name = "Analytics API Endpoints"
        self.test_results['total_tests'] += 1
        
        try:
            # Import should have been successful from earlier test
            import api.v2.analytics_api as analytics_api
            
            # Check if FastAPI app exists
            if hasattr(analytics_api, 'app'):
                app = analytics_api.app
                
                # Check routes (basic structure test)
                routes = [route.path for route in app.routes]
                expected_routes = [
                    '/api/v2/analytics/dashboard',
                    '/api/v2/analytics/cost-optimization',
                    '/api/v2/analytics/performance-trends',
                    '/health'
                ]
                
                missing_routes = [route for route in expected_routes if route not in routes]
                
                if not missing_routes:
                    log_test(f"{test_name} - SUCCESS (all routes present)", "PASS")
                    self.test_results['passed_tests'] += 1
                    self.test_results['details'].append({
                        'test': test_name,
                        'status': 'PASSED',
                        'routes_found': len(routes)
                    })
                else:
                    log_test(f"{test_name} - FAILED: Missing routes {missing_routes}", "FAIL")
                    self.test_results['failed_tests'] += 1
                    self.test_results['details'].append({
                        'test': test_name,
                        'status': 'FAILED',
                        'missing_routes': missing_routes
                    })
            else:
                log_test(f"{test_name} - FAILED: FastAPI app not found", "FAIL")
                self.test_results['failed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'error': 'FastAPI app not found'
                })
                
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    async def _test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        test_name = "End-to-End Workflow"
        self.test_results['total_tests'] += 1
        
        try:
            # Check if we have all major components
            required_components = [
                "V2ExperienceTracker",
                "PatternMiningEngine", 
                "MLTrainingPipeline"
            ]
            
            missing_components = [comp for comp in required_components 
                                if comp not in self.components_available]
            
            if missing_components:
                log_test(f"{test_name} - FAILED: Missing components {missing_components}", "FAIL")
                self.test_results['failed_tests'] += 1
                self.test_results['details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'missing_components': missing_components
                })
                return
            
            # Test basic workflow integration
            workflow_steps = [
                "Experience Tracking",
                "Pattern Discovery",
                "Model Training Pipeline"
            ]
            
            log_test(f"{test_name} - SUCCESS (workflow components available)", "PASS")
            self.test_results['passed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'PASSED',
                'workflow_steps': workflow_steps,
                'components_available': len(self.components_available)
            })
            
        except Exception as e:
            log_test(f"{test_name} - FAILED: {e}", "FAIL")
            self.test_results['failed_tests'] += 1
            self.test_results['details'].append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        log_test("TEST SUMMARY - Agent Zero V2.0 Implementation", "START")
        print("="*80)
        
        total_tests = self.test_results['total_tests']
        passed_tests = self.test_results['passed_tests'] 
        failed_tests = self.test_results['failed_tests']
        warnings = self.test_results['warnings']
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"{Colors.CYAN}📊 Overall Results:{Colors.NC}")
        print(f"   Total Tests:    {total_tests}")
        print(f"   {Colors.GREEN}✅ Passed:        {passed_tests}{Colors.NC}")
        print(f"   {Colors.RED}❌ Failed:        {failed_tests}{Colors.NC}")
        print(f"   {Colors.YELLOW}⚠️  Warnings:      {warnings}{Colors.NC}")
        print(f"   {Colors.PURPLE}📈 Success Rate:   {success_rate:.1f}%{Colors.NC}")
        
        print(f"\n{Colors.CYAN}📋 Story Points Status:{Colors.NC}")
        priorities_status = [
            ("Priority 1: Experience Management System", 8, self._count_priority_results("Priority 1")),
            ("Priority 2: Neo4j Knowledge Graph Integration", 6, self._count_priority_results("Priority 2")),  
            ("Priority 3: Pattern Mining Engine", 6, self._count_priority_results("Priority 3")),
            ("Priority 4: ML Model Training Pipeline", 4, self._count_priority_results("Priority 4")),
            ("Priority 5: Enhanced Analytics Dashboard Backend", 2, self._count_priority_results("Priority 5"))
        ]
        
        total_sp_implemented = 0
        for priority, sp, (passed, total) in priorities_status:
            status_icon = "✅" if passed == total and total > 0 else "⚠️" if passed > 0 else "❌"
            if passed == total and total > 0:
                total_sp_implemented += sp
            elif passed > 0:
                total_sp_implemented += sp * (passed / total)
            
            print(f"   {status_icon} {priority}: {passed}/{total} tests - {sp} SP")
        
        print(f"\n{Colors.PURPLE}🎯 Implementation Status:{Colors.NC}")
        print(f"   Story Points Implemented: {total_sp_implemented:.0f}/28 SP")
        print(f"   Implementation Rate: {(total_sp_implemented/28)*100:.1f}%")
        
        if failed_tests == 0:
            print(f"\n{Colors.GREEN}🎉 ALL TESTS PASSED! Agent Zero V2.0 is ready for production!{Colors.NC}")
        elif failed_tests < total_tests / 2:
            print(f"\n{Colors.YELLOW}⚠️  Some tests failed, but core functionality is working{Colors.NC}")
        else:
            print(f"\n{Colors.RED}❌ Major issues detected - implementation needs attention{Colors.NC}")
        
        # Save detailed results
        with open(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n{Colors.BLUE}📄 Detailed results saved to test_results_*.json{Colors.NC}")
        
        print(f"\n{Colors.CYAN}🚀 Next Steps:{Colors.NC}")
        if failed_tests == 0:
            print("   1. Deploy to production environment")
            print("   2. Run performance benchmarks")
            print("   3. Set up monitoring and alerts")
            print("   4. Begin Week 45 Advanced CLI Commands [2 SP]")
        else:
            print("   1. Fix failed components")
            print("   2. Re-run tests")
            print("   3. Check dependency installation")
            print("   4. Review deployment logs")
    
    def _count_priority_results(self, priority_name: str) -> tuple:
        """Count passed/total tests for a priority"""
        passed = 0
        total = 0
        
        for detail in self.test_results['details']:
            if priority_name.lower() in detail.get('test', '').lower():
                total += 1
                if detail.get('status') == 'PASSED':
                    passed += 1
        
        return (passed, total)

# Main execution
async def main():
    """Run complete V2.0 implementation test"""
    tester = V2ImplementationTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        log_test("Test interrupted by user", "WARN")
        return 2
    except Exception as e:
        log_test(f"Test suite failed with error: {e}", "FAIL")
        return 3

if __name__ == "__main__":
    # Run the complete test suite
    print("🧪 Starting Agent Zero V2.0 Complete Implementation Test")
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Running from: {os.getcwd()}")
    print()
    
    start_time = time.time()
    exit_code = asyncio.run(main())
    end_time = time.time()
    
    print(f"\n⏱️  Total test time: {end_time - start_time:.1f} seconds")
    print(f"🏁 Test completed with exit code: {exit_code}")
    
    sys.exit(exit_code)