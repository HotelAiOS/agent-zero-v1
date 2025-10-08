#!/usr/bin/env python3
"""
Agent Zero V1 - Performance Benchmarking Suite
Comprehensive performance testing for all system components
"""

import asyncio
import time
import json
import statistics
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmarking and monitoring system"""
    
    def __init__(self):
        self.results = {}
        self.system_metrics = {}
        self.start_time = None
        self.end_time = None
        
    def start_monitoring(self):
        """Start system monitoring"""
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.end_time = time.time()
        
    def _monitor_system(self):
        """Monitor system resources"""
        metrics = []
        while self.end_time is None:
            metric = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                'network_io': psutil.net_io_counters()._asdict()
            }
            metrics.append(metric)
            time.sleep(1)
        
        self.system_metrics = {
            'duration': self.end_time - self.start_time,
            'samples': len(metrics),
            'avg_cpu': statistics.mean([m['cpu_percent'] for m in metrics]),
            'max_cpu': max([m['cpu_percent'] for m in metrics]),
            'avg_memory': statistics.mean([m['memory_percent'] for m in metrics]),
            'max_memory': max([m['memory_percent'] for m in metrics])
        }

class WebSocketPerformanceTest:
    """WebSocket connection and messaging performance tests"""
    
    async def test_connection_speed(self, iterations: int = 10) -> Dict[str, Any]:
        """Test WebSocket connection establishment speed"""
        import websockets
        
        connection_times = []
        
        for i in range(iterations):
            start = time.time()
            try:
                async with websockets.connect('ws://localhost:8000/ws') as websocket:
                    await websocket.recv()  # Wait for welcome message
                    connection_times.append(time.time() - start)
            except Exception as e:
                logger.error(f"WebSocket connection {i+1} failed: {e}")
                connection_times.append(float('inf'))
        
        successful_connections = [t for t in connection_times if t != float('inf')]
        
        return {
            'total_attempts': iterations,
            'successful_connections': len(successful_connections),
            'failure_rate': (iterations - len(successful_connections)) / iterations * 100,
            'avg_connection_time': statistics.mean(successful_connections) if successful_connections else 0,
            'min_connection_time': min(successful_connections) if successful_connections else 0,
            'max_connection_time': max(successful_connections) if successful_connections else 0,
            'connection_times': connection_times
        }
    
    async def test_message_throughput(self, messages: int = 100, concurrent: int = 5) -> Dict[str, Any]:
        """Test message throughput and response times"""
        import websockets
        
        async def send_messages(websocket, count):
            times = []
            for i in range(count):
                start = time.time()
                await websocket.send(json.dumps({
                    'type': 'performance_test',
                    'message_id': i,
                    'timestamp': start
                }))
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    times.append(time.time() - start)
                except asyncio.TimeoutError:
                    times.append(float('inf'))
            return times
        
        start_time = time.time()
        
        try:
            async with websockets.connect('ws://localhost:8000/ws') as websocket:
                # Wait for connection
                await websocket.recv()
                
                # Send messages concurrently
                messages_per_worker = messages // concurrent
                tasks = [send_messages(websocket, messages_per_worker) for _ in range(concurrent)]
                
                results = await asyncio.gather(*tasks)
                all_times = [time for worker_times in results for time in worker_times]
                
        except Exception as e:
            logger.error(f"Message throughput test failed: {e}")
            all_times = []
        
        total_time = time.time() - start_time
        successful_messages = [t for t in all_times if t != float('inf')]
        
        return {
            'total_messages': messages,
            'successful_messages': len(successful_messages),
            'total_time': total_time,
            'messages_per_second': len(successful_messages) / total_time if total_time > 0 else 0,
            'avg_response_time': statistics.mean(successful_messages) if successful_messages else 0,
            'min_response_time': min(successful_messages) if successful_messages else 0,
            'max_response_time': max(successful_messages) if successful_messages else 0,
            'p95_response_time': statistics.quantiles(successful_messages, n=20)[18] if len(successful_messages) >= 20 else 0,
            'failure_rate': (messages - len(successful_messages)) / messages * 100
        }

class DatabasePerformanceTest:
    """Database connection and query performance tests"""
    
    def test_neo4j_connection(self, iterations: int = 50) -> Dict[str, Any]:
        """Test Neo4j connection performance"""
        try:
            from neo4j import GraphDatabase
            
            connection_times = []
            query_times = []
            
            for i in range(iterations):
                # Test connection
                start = time.time()
                try:
                    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'agentzerov1'))
                    connection_times.append(time.time() - start)
                    
                    # Test simple query
                    start = time.time()
                    with driver.session() as session:
                        result = session.run('RETURN 1 as test')
                        record = result.single()
                    query_times.append(time.time() - start)
                    
                    driver.close()
                    
                except Exception as e:
                    logger.error(f"Neo4j test {i+1} failed: {e}")
                    connection_times.append(float('inf'))
                    query_times.append(float('inf'))
            
            successful_connections = [t for t in connection_times if t != float('inf')]
            successful_queries = [t for t in query_times if t != float('inf')]
            
            return {
                'total_attempts': iterations,
                'successful_connections': len(successful_connections),
                'successful_queries': len(successful_queries),
                'avg_connection_time': statistics.mean(successful_connections) if successful_connections else 0,
                'avg_query_time': statistics.mean(successful_queries) if successful_queries else 0,
                'max_connection_time': max(successful_connections) if successful_connections else 0,
                'max_query_time': max(successful_queries) if successful_queries else 0,
                'connection_success_rate': len(successful_connections) / iterations * 100,
                'query_success_rate': len(successful_queries) / iterations * 100
            }
            
        except ImportError:
            return {'error': 'neo4j driver not available'}
    
    def test_redis_performance(self, operations: int = 1000) -> Dict[str, Any]:
        """Test Redis operations performance"""
        try:
            import redis
            
            r = redis.Redis(host='localhost', port=6379, db=0)
            
            # Test connection
            start = time.time()
            r.ping()
            connection_time = time.time() - start
            
            # Test SET operations
            set_times = []
            for i in range(operations):
                start = time.time()
                r.set(f'test_key_{i}', f'test_value_{i}')
                set_times.append(time.time() - start)
            
            # Test GET operations
            get_times = []
            for i in range(operations):
                start = time.time()
                r.get(f'test_key_{i}')
                get_times.append(time.time() - start)
            
            # Cleanup
            for i in range(operations):
                r.delete(f'test_key_{i}')
            
            return {
                'connection_time': connection_time,
                'total_operations': operations * 2,  # SET + GET
                'avg_set_time': statistics.mean(set_times),
                'avg_get_time': statistics.mean(get_times),
                'max_set_time': max(set_times),
                'max_get_time': max(get_times),
                'sets_per_second': operations / sum(set_times),
                'gets_per_second': operations / sum(get_times)
            }
            
        except ImportError:
            return {'error': 'redis library not available'}
        except Exception as e:
            return {'error': f'Redis test failed: {e}'}

class TaskDecomposerPerformanceTest:
    """Task Decomposer performance and accuracy tests"""
    
    def test_json_parsing_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """Test JSON parsing robustness and speed"""
        try:
            import sys
            sys.path.append('shared/orchestration')
            from task_decomposer import TaskDecomposer
            
            decomposer = TaskDecomposer()
            
            test_cases = [
                # Clean JSON
                '{"subtasks": [{"id": 1, "title": "Test", "description": "Test task"}]}',
                # JSON with markdown
                '```json\n{"subtasks": [{"id": 1, "title": "Test", "description": "Test task"}]}\n```',
                # JSON with extra text
                'Here is the response: {"subtasks": [{"id": 1, "title": "Test", "description": "Test task"}]} Hope this helps!',
                # Malformed but recoverable
                '{\n  "subtasks": [\n    {\n      "id": 1,\n      "title": "Test",\n      "description": "Test task"\n    }\n  ]\n}',
                # Complex nested
                '{"subtasks": [{"id": 1, "title": "Complex", "description": "Complex task", "dependencies": [2, 3]}, {"id": 2, "title": "Sub1"}, {"id": 3, "title": "Sub2"}]}'
            ]
            
            parsing_times = []
            success_counts = [0] * len(test_cases)
            
            for iteration in range(iterations):
                for case_idx, test_case in enumerate(test_cases):
                    start = time.time()
                    result = decomposer.safe_parse_llm_response(test_case)
                    parsing_times.append(time.time() - start)
                    
                    if result and 'subtasks' in result:
                        success_counts[case_idx] += 1
            
            return {
                'total_tests': iterations * len(test_cases),
                'test_cases': len(test_cases),
                'iterations_per_case': iterations,
                'avg_parsing_time': statistics.mean(parsing_times),
                'max_parsing_time': max(parsing_times),
                'min_parsing_time': min(parsing_times),
                'success_rates': [count / iterations * 100 for count in success_counts],
                'overall_success_rate': sum(success_counts) / (iterations * len(test_cases)) * 100,
                'parses_per_second': len(parsing_times) / sum(parsing_times) if sum(parsing_times) > 0 else 0
            }
            
        except ImportError:
            return {'error': 'TaskDecomposer not available'}
        except Exception as e:
            return {'error': f'Task decomposer test failed: {e}'}
    
    def test_decomposition_performance(self, tasks: int = 20) -> Dict[str, Any]:
        """Test full task decomposition performance"""
        try:
            import sys
            sys.path.append('shared/orchestration')
            from task_decomposer import TaskDecomposer
            
            decomposer = TaskDecomposer()
            
            test_tasks = [
                "Create a web application",
                "Build a REST API",
                "Design a database schema",
                "Implement authentication system",
                "Create unit tests",
                "Setup CI/CD pipeline",
                "Write documentation",
                "Implement caching layer",
                "Add monitoring and logging",
                "Create mobile application"
            ]
            
            decomposition_times = []
            subtask_counts = []
            success_count = 0
            
            for i in range(tasks):
                task = test_tasks[i % len(test_tasks)] + f" (iteration {i+1})"
                
                start = time.time()
                result = decomposer.decompose_task(task)
                decomposition_time = time.time() - start
                
                decomposition_times.append(decomposition_time)
                
                if result and 'subtasks' in result:
                    success_count += 1
                    subtask_counts.append(len(result['subtasks']))
                else:
                    subtask_counts.append(0)
            
            return {
                'total_decompositions': tasks,
                'successful_decompositions': success_count,
                'success_rate': success_count / tasks * 100,
                'avg_decomposition_time': statistics.mean(decomposition_times),
                'max_decomposition_time': max(decomposition_times),
                'min_decomposition_time': min(decomposition_times),
                'avg_subtasks_per_task': statistics.mean(subtask_counts) if subtask_counts else 0,
                'max_subtasks': max(subtask_counts) if subtask_counts else 0,
                'decompositions_per_second': tasks / sum(decomposition_times) if sum(decomposition_times) > 0 else 0
            }
            
        except Exception as e:
            return {'error': f'Decomposition test failed: {e}'}

class AgentExecutorPerformanceTest:
    """Agent Executor performance tests"""
    
    def test_execution_performance(self, executions: int = 50) -> Dict[str, Any]:
        """Test agent execution performance"""
        try:
            # Mock AgentExecutor for performance testing
            class MockAgentExecutor:
                def execute_task(self, agent, task, output_dir):
                    # Simulate work
                    time.sleep(0.1)  # 100ms simulated work
                    return {
                        'status': 'success',
                        'agent': agent,
                        'task': task,
                        'output_dir': output_dir,
                        'result': f'Executed {task}'
                    }
            
            executor = MockAgentExecutor()
            execution_times = []
            success_count = 0
            
            for i in range(executions):
                start = time.time()
                try:
                    result = executor.execute_task(f'agent_{i}', f'task_{i}', '/tmp')
                    execution_times.append(time.time() - start)
                    if result and result.get('status') == 'success':
                        success_count += 1
                except Exception as e:
                    logger.error(f"Execution {i+1} failed: {e}")
                    execution_times.append(float('inf'))
            
            successful_executions = [t for t in execution_times if t != float('inf')]
            
            return {
                'total_executions': executions,
                'successful_executions': success_count,
                'success_rate': success_count / executions * 100,
                'avg_execution_time': statistics.mean(successful_executions) if successful_executions else 0,
                'max_execution_time': max(successful_executions) if successful_executions else 0,
                'min_execution_time': min(successful_executions) if successful_executions else 0,
                'executions_per_second': len(successful_executions) / sum(successful_executions) if sum(successful_executions) > 0 else 0,
                'throughput': success_count / sum(execution_times) if sum(execution_times) > 0 else 0
            }
            
        except Exception as e:
            return {'error': f'Executor test failed: {e}'}

async def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark"""
    print("🚀 Starting Agent Zero V1 Performance Benchmark...")
    
    benchmark = PerformanceBenchmark()
    benchmark.start_monitoring()
    
    results = {}
    
    # WebSocket Performance Tests
    print("🌐 Testing WebSocket Performance...")
    ws_test = WebSocketPerformanceTest()
    
    try:
        results['websocket_connection'] = await ws_test.test_connection_speed(20)
        results['websocket_throughput'] = await ws_test.test_message_throughput(200, 5)
        print("✅ WebSocket tests completed")
    except Exception as e:
        print(f"❌ WebSocket tests failed: {e}")
        results['websocket_connection'] = {'error': str(e)}
        results['websocket_throughput'] = {'error': str(e)}
    
    # Database Performance Tests
    print("💾 Testing Database Performance...")
    db_test = DatabasePerformanceTest()
    
    results['neo4j_performance'] = db_test.test_neo4j_connection(30)
    results['redis_performance'] = db_test.test_redis_performance(500)
    print("✅ Database tests completed")
    
    # Task Decomposer Performance Tests
    print("🧠 Testing Task Decomposer Performance...")
    decomposer_test = TaskDecomposerPerformanceTest()
    
    results['json_parsing'] = decomposer_test.test_json_parsing_performance(50)
    results['task_decomposition'] = decomposer_test.test_decomposition_performance(15)
    print("✅ Task Decomposer tests completed")
    
    # Agent Executor Performance Tests
    print("⚡ Testing Agent Executor Performance...")
    executor_test = AgentExecutorPerformanceTest()
    
    results['agent_execution'] = executor_test.test_execution_performance(30)
    print("✅ Agent Executor tests completed")
    
    # Stop monitoring
    benchmark.stop_monitoring()
    
    # Add system metrics to results
    results['system_metrics'] = benchmark.system_metrics
    
    return results

def generate_performance_report(results: Dict[str, Any]) -> str:
    """Generate HTML performance report"""
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Zero V1 - Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ border: 1px solid #ddd; border-radius: 10px; padding: 20px; background: #f8f9fa; }}
            .metric-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 15px; color: #333; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
            .metric-unit {{ font-size: 0.8em; color: #666; }}
            .success {{ color: #28a745; }}
            .warning {{ color: #ffc107; }}
            .error {{ color: #dc3545; }}
            .performance-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            .performance-table th, .performance-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .performance-table th {{ background-color: #f2f2f2; }}
            .summary {{ background: #e7f3ff; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Agent Zero V1 - Performance Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <p>Performance benchmark completed for all Agent Zero V1 components. Key findings:</p>
            <ul>
    """
    
    # Add summary points based on results
    if 'websocket_connection' in results and 'avg_connection_time' in results['websocket_connection']:
        conn_time = results['websocket_connection']['avg_connection_time']
        if conn_time < 0.5:
            report_html += f"<li class='success'>✅ WebSocket connections: Excellent ({conn_time:.3f}s avg)</li>"
        elif conn_time < 1.0:
            report_html += f"<li class='warning'>⚠️ WebSocket connections: Good ({conn_time:.3f}s avg)</li>"
        else:
            report_html += f"<li class='error'>❌ WebSocket connections: Needs improvement ({conn_time:.3f}s avg)</li>"
    
    if 'neo4j_performance' in results and 'avg_query_time' in results['neo4j_performance']:
        query_time = results['neo4j_performance']['avg_query_time']
        if query_time < 0.1:
            report_html += f"<li class='success'>✅ Neo4j queries: Excellent ({query_time:.3f}s avg)</li>"
        elif query_time < 0.5:
            report_html += f"<li class='warning'>⚠️ Neo4j queries: Good ({query_time:.3f}s avg)</li>"
        else:
            report_html += f"<li class='error'>❌ Neo4j queries: Needs improvement ({query_time:.3f}s avg)</li>"
    
    report_html += """
            </ul>
        </div>
        
        <div class="metric-grid">
    """
    
    # WebSocket Metrics
    if 'websocket_connection' in results:
        ws_data = results['websocket_connection']
        report_html += f"""
            <div class="metric-card">
                <div class="metric-title">🌐 WebSocket Performance</div>
                <div class="metric-value">{ws_data.get('avg_connection_time', 0):.3f}<span class="metric-unit">s</span></div>
                <p>Average connection time</p>
                <p>Success rate: {ws_data.get('successful_connections', 0)}/{ws_data.get('total_attempts', 0)} ({100 - ws_data.get('failure_rate', 0):.1f}%)</p>
            </div>
        """
    
    # Database Metrics
    if 'neo4j_performance' in results:
        neo4j_data = results['neo4j_performance']
        report_html += f"""
            <div class="metric-card">
                <div class="metric-title">💾 Neo4j Performance</div>
                <div class="metric-value">{neo4j_data.get('avg_query_time', 0):.3f}<span class="metric-unit">s</span></div>
                <p>Average query time</p>
                <p>Query success rate: {neo4j_data.get('query_success_rate', 0):.1f}%</p>
            </div>
        """
    
    # Task Decomposer Metrics
    if 'json_parsing' in results:
        json_data = results['json_parsing']
        report_html += f"""
            <div class="metric-card">
                <div class="metric-title">🧠 JSON Parsing</div>
                <div class="metric-value">{json_data.get('parses_per_second', 0):.0f}<span class="metric-unit">/s</span></div>
                <p>Parses per second</p>
                <p>Success rate: {json_data.get('overall_success_rate', 0):.1f}%</p>
            </div>
        """
    
    # System Metrics
    if 'system_metrics' in results:
        sys_data = results['system_metrics']
        report_html += f"""
            <div class="metric-card">
                <div class="metric-title">🖥️ System Resources</div>
                <div class="metric-value">{sys_data.get('avg_cpu', 0):.1f}<span class="metric-unit">%</span></div>
                <p>Average CPU usage</p>
                <p>Memory: {sys_data.get('avg_memory', 0):.1f}% avg, {sys_data.get('max_memory', 0):.1f}% peak</p>
            </div>
        """
    
    report_html += """
        </div>
        
        <h2>📋 Detailed Results</h2>
    """
    
    # Add detailed tables
    for test_name, test_results in results.items():
        if isinstance(test_results, dict) and 'error' not in test_results:
            report_html += f"""
                <h3>{test_name.replace('_', ' ').title()}</h3>
                <table class="performance-table">
            """
            
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    if 'time' in key.lower():
                        formatted_value = f"{value:.4f}s"
                    elif 'rate' in key.lower() or 'percent' in key.lower():
                        formatted_value = f"{value:.2f}%"
                    elif 'per_second' in key.lower():
                        formatted_value = f"{value:.2f}/s"
                    else:
                        formatted_value = f"{value:.2f}"
                    
                    report_html += f"""
                        <tr>
                            <td>{key.replace('_', ' ').title()}</td>
                            <td>{formatted_value}</td>
                        </tr>
                    """
            
            report_html += "</table>"
    
    report_html += """
        <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>📈 Recommendations</h3>
            <ul>
                <li><strong>WebSocket:</strong> Consider connection pooling if average connection time > 1s</li>
                <li><strong>Database:</strong> Monitor Neo4j query performance and add indexes if needed</li>
                <li><strong>Task Decomposer:</strong> JSON parsing success rate should be > 95%</li>
                <li><strong>System:</strong> CPU usage should remain < 80% under normal load</li>
            </ul>
        </div>
        
    </body>
    </html>
    """
    
    return report_html

if __name__ == "__main__":
    print("🧮 Agent Zero V1 Performance Benchmark Suite")
    print("============================================")
    
    # Run benchmark
    results = asyncio.run(run_comprehensive_benchmark())
    
    # Generate and save report
    report_html = generate_performance_report(results)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # JSON results
    json_file = f"performance_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # HTML report
    html_file = f"performance_report_{timestamp}.html"
    with open(html_file, 'w') as f:
        f.write(report_html)
    
    print(f"\n📊 Performance benchmark completed!")
    print(f"📄 Results saved to: {json_file}")
    print(f"📄 Report saved to: {html_file}")
    
    # Print summary to console
    print("\n🎯 Performance Summary:")
    
    if 'websocket_connection' in results:
        ws_data = results['websocket_connection']
        print(f"  WebSocket: {ws_data.get('avg_connection_time', 0):.3f}s avg connection ({100 - ws_data.get('failure_rate', 0):.1f}% success)")
    
    if 'neo4j_performance' in results:
        neo4j_data = results['neo4j_performance']
        print(f"  Neo4j: {neo4j_data.get('avg_query_time', 0):.3f}s avg query ({neo4j_data.get('query_success_rate', 0):.1f}% success)")
    
    if 'json_parsing' in results:
        json_data = results['json_parsing']
        print(f"  JSON Parsing: {json_data.get('parses_per_second', 0):.0f}/s ({json_data.get('overall_success_rate', 0):.1f}% success)")
    
    if 'system_metrics' in results:
        sys_data = results['system_metrics']
        print(f"  System: {sys_data.get('avg_cpu', 0):.1f}% CPU, {sys_data.get('avg_memory', 0):.1f}% Memory")
    
    print(f"\n🌐 View detailed report: open {html_file}")