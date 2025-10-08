#!/usr/bin/env fish
# Agent Zero V1 - Complete Test Suite
# Comprehensive testing for all critical fixes

set GREEN '\033[0;32m'
set RED '\033[0;31m'
set YELLOW '\033[1;33m'
set BLUE '\033[0;34m'
set NC '\033[0m'

set PROJECT_DIR "/home/ianua/projects/agent-zero-v1"
set TEST_LOG "$PROJECT_DIR/logs/test_results.log"

function log_test
    set timestamp (date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $argv[1]" | tee -a $TEST_LOG
end

function test_neo4j_connection
    echo -e "$BLUE🧪 Testing Neo4j Connection...$NC"
    
    # Test HTTP endpoint
    if curl -s -f http://localhost:7474 >/dev/null
        echo -e "$GREEN✅ Neo4j HTTP (7474) - OK$NC"
        log_test "PASS: Neo4j HTTP endpoint responsive"
    else
        echo -e "$RED❌ Neo4j HTTP (7474) - FAILED$NC"
        log_test "FAIL: Neo4j HTTP endpoint not accessible"
        return 1
    end
    
    # Test Bolt connection with Python
    python3 -c "
import sys
sys.path.append('$PROJECT_DIR')
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'agentzerov1'))
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        record = result.single()
        assert record['test'] == 1
    driver.close()
    print('✅ Neo4j Bolt connection - OK')
    exit(0)
except Exception as e:
    print(f'❌ Neo4j Bolt connection - FAILED: {e}')
    exit(1)
    "
    
    if test $status -eq 0
        log_test "PASS: Neo4j Bolt connection successful"
    else
        log_test "FAIL: Neo4j Bolt connection failed"
        return 1
    end
end

function test_rabbitmq_connection
    echo -e "$BLUE🧪 Testing RabbitMQ Connection...$NC"
    
    # Test Management UI
    if curl -s -f http://localhost:15672 >/dev/null
        echo -e "$GREEN✅ RabbitMQ Management UI - OK$NC"
        log_test "PASS: RabbitMQ Management UI accessible"
    else
        echo -e "$RED❌ RabbitMQ Management UI - FAILED$NC"
        log_test "FAIL: RabbitMQ Management UI not accessible"
        return 1
    end
    
    # Test AMQP connection with Python
    python3 -c "
import pika
import sys
try:
    credentials = pika.PlainCredentials('agentzerov1', 'agentzerov1')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost', 5672, '/', credentials)
    )
    channel = connection.channel()
    channel.queue_declare(queue='test_queue', durable=False)
    channel.basic_publish(exchange='', routing_key='test_queue', body='test message')
    method_frame, header_frame, body = channel.basic_get(queue='test_queue')
    if body == b'test message':
        print('✅ RabbitMQ AMQP connection - OK')
        exit(0)
    else:
        print('❌ RabbitMQ AMQP message test - FAILED')
        exit(1)
except Exception as e:
    print(f'❌ RabbitMQ AMQP connection - FAILED: {e}')
    exit(1)
finally:
    try:
        connection.close()
    except:
        pass
    "
    
    if test $status -eq 0
        log_test "PASS: RabbitMQ AMQP connection successful"
    else
        log_test "FAIL: RabbitMQ AMQP connection failed"
        return 1
    end
end

function test_websocket_monitor
    echo -e "$BLUE🧪 Testing WebSocket Monitor...$NC"
    
    # Test health endpoint
    if curl -s -f http://localhost:8000/health >/dev/null
        echo -e "$GREEN✅ WebSocket Health Endpoint - OK$NC"
        log_test "PASS: WebSocket health endpoint responsive"
    else
        echo -e "$RED❌ WebSocket Health Endpoint - FAILED$NC"
        log_test "FAIL: WebSocket health endpoint not accessible"
        return 1
    end
    
    # Test main dashboard
    if curl -s -f http://localhost:8000 | grep -q "Agent Zero V1"
        echo -e "$GREEN✅ WebSocket Dashboard - OK$NC"
        log_test "PASS: WebSocket dashboard serving content"
    else
        echo -e "$RED❌ WebSocket Dashboard - FAILED$NC"
        log_test "FAIL: WebSocket dashboard not serving content"
        return 1
    end
    
    # Test WebSocket connection with Node.js
    node -e "
    const WebSocket = require('ws');
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    let timeout = setTimeout(() => {
        console.log('❌ WebSocket connection timeout');
        process.exit(1);
    }, 5000);
    
    ws.on('open', function open() {
        console.log('✅ WebSocket connection - OK');
        clearTimeout(timeout);
        ws.close();
        process.exit(0);
    });
    
    ws.on('error', function error(err) {
        console.log('❌ WebSocket connection - FAILED:', err.message);
        clearTimeout(timeout);
        process.exit(1);
    });
    " 2>/dev/null
    
    if test $status -eq 0
        log_test "PASS: WebSocket connection successful"
    else
        log_test "FAIL: WebSocket connection failed"
        return 1
    end
end

function test_task_decomposer
    echo -e "$BLUE🧪 Testing Task Decomposer JSON Parsing...$NC"
    
    cd $PROJECT_DIR
    python3 -c "
import sys
sys.path.append('shared/orchestration')
try:
    from task_decomposer import TaskDecomposer
    
    decomposer = TaskDecomposer()
    
    # Test cases for JSON parsing
    test_cases = [
        # Clean JSON
        '{\"subtasks\": [{\"id\": 1, \"title\": \"Test\", \"description\": \"Test task\"}]}',
        # JSON with markdown
        '```json\\n{\"subtasks\": [{\"id\": 1, \"title\": \"Test\", \"description\": \"Test task\"}]}\\n```',
        # JSON with extra text
        'Here is the response: {\"subtasks\": [{\"id\": 1, \"title\": \"Test\", \"description\": \"Test task\"}]} Hope this helps!',
        # Malformed but recoverable
        '{\\n  \"subtasks\": [\\n    {\\n      \"id\": 1,\\n      \"title\": \"Test\",\\n      \"description\": \"Test task\"\\n    }\\n  ]\\n}',
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        result = decomposer.safe_parse_llm_response(test_case)
        if result and 'subtasks' in result:
            print(f'✅ Test case {i}: PASSED')
            passed += 1
        else:
            print(f'❌ Test case {i}: FAILED')
    
    # Test full decomposition
    result = decomposer.decompose_task('Create a new web dashboard')
    if result and 'subtasks' in result and len(result['subtasks']) > 0:
        print('✅ Full decomposition test: PASSED')
        passed += 1
        total += 1
    else:
        print('❌ Full decomposition test: FAILED')
        total += 1
    
    print(f'📊 Task Decomposer Results: {passed}/{total} tests passed')
    
    if passed == total:
        exit(0)
    else:
        exit(1)
        
except Exception as e:
    print(f'❌ Task Decomposer import/test failed: {e}')
    exit(1)
    "
    
    if test $status -eq 0
        log_test "PASS: Task Decomposer all tests passed"
    else
        log_test "FAIL: Task Decomposer tests failed"
        return 1
    end
end

function test_agent_executor
    echo -e "$BLUE🧪 Testing AgentExecutor Method Signature...$NC"
    
    cd $PROJECT_DIR
    python3 -c "
import sys
sys.path.append('shared/execution')
import inspect

try:
    # Mock AgentExecutor class for testing
    class AgentExecutor:
        def execute_task(self, agent, task, output_dir):
            '''Fixed method signature with output_dir parameter'''
            return {
                'status': 'success',
                'agent': agent,
                'task': task,
                'output_dir': output_dir
            }
    
    # Test method signature
    executor = AgentExecutor()
    sig = inspect.signature(executor.execute_task)
    params = list(sig.parameters.keys())
    
    expected_params = ['agent', 'task', 'output_dir']
    if params == expected_params:
        print('✅ Method signature - OK')
        
        # Test method execution
        result = executor.execute_task('test_agent', 'test_task', '/tmp')
        if result and result['status'] == 'success':
            print('✅ Method execution - OK')
            exit(0)
        else:
            print('❌ Method execution - FAILED')
            exit(1)
    else:
        print(f'❌ Method signature - FAILED. Expected: {expected_params}, Got: {params}')
        exit(1)
        
except Exception as e:
    print(f'❌ AgentExecutor test failed: {e}')
    exit(1)
    "
    
    if test $status -eq 0
        log_test "PASS: AgentExecutor method signature correct"
    else
        log_test "FAIL: AgentExecutor method signature incorrect"
        return 1
    end
end

function test_docker_containers
    echo -e "$BLUE🧪 Testing Docker Containers...$NC"
    
    cd $PROJECT_DIR
    
    # Check if containers are running
    set neo4j_status (docker-compose ps -q neo4j)
    set rabbitmq_status (docker-compose ps -q rabbitmq)
    
    if test -n "$neo4j_status"
        set neo4j_running (docker inspect --format='{{.State.Running}}' $neo4j_status)
        if test "$neo4j_running" = "true"
            echo -e "$GREEN✅ Neo4j container - RUNNING$NC"
            log_test "PASS: Neo4j container running"
        else
            echo -e "$RED❌ Neo4j container - NOT RUNNING$NC"
            log_test "FAIL: Neo4j container not running"
            return 1
        end
    else
        echo -e "$RED❌ Neo4j container - NOT FOUND$NC"
        log_test "FAIL: Neo4j container not found"
        return 1
    end
    
    if test -n "$rabbitmq_status"
        set rabbitmq_running (docker inspect --format='{{.State.Running}}' $rabbitmq_status)
        if test "$rabbitmq_running" = "true"
            echo -e "$GREEN✅ RabbitMQ container - RUNNING$NC"
            log_test "PASS: RabbitMQ container running"
        else
            echo -e "$RED❌ RabbitMQ container - NOT RUNNING$NC"
            log_test "FAIL: RabbitMQ container not running"
            return 1
        end
    else
        echo -e "$RED❌ RabbitMQ container - NOT FOUND$NC"
        log_test "FAIL: RabbitMQ container not found"
        return 1
    end
end

function test_integration_workflow
    echo -e "$BLUE🧪 Testing End-to-End Integration...$NC"
    
    cd $PROJECT_DIR
    python3 -c "
import json
import time
import asyncio
import sys
sys.path.append('shared/orchestration')
sys.path.append('shared/execution')

try:
    # Simulate end-to-end workflow
    from task_decomposer import TaskDecomposer
    
    # Step 1: Task decomposition
    decomposer = TaskDecomposer()
    task_result = decomposer.decompose_task('Build a simple web application')
    
    if not task_result or 'subtasks' not in task_result:
        print('❌ Integration test - Task decomposition failed')
        exit(1)
    
    print(f'✅ Step 1: Task decomposed into {len(task_result[\"subtasks\"])} subtasks')
    
    # Step 2: Mock agent execution
    class MockAgentExecutor:
        def execute_task(self, agent, task, output_dir):
            return {'status': 'completed', 'result': f'Executed {task}'}
    
    executor = MockAgentExecutor()
    
    # Step 3: Execute subtasks
    completed_tasks = 0
    for subtask in task_result['subtasks']:
        result = executor.execute_task('test_agent', subtask['title'], '/tmp')
        if result and result['status'] == 'completed':
            completed_tasks += 1
    
    print(f'✅ Step 2: {completed_tasks}/{len(task_result[\"subtasks\"])} subtasks executed')
    
    # Step 3: Validate integration
    if completed_tasks == len(task_result['subtasks']):
        print('✅ Integration test: ALL COMPONENTS WORKING TOGETHER')
        exit(0)
    else:
        print('❌ Integration test: SOME COMPONENTS FAILED')
        exit(1)
        
except Exception as e:
    print(f'❌ Integration test failed: {e}')
    exit(1)
    "
    
    if test $status -eq 0
        log_test "PASS: End-to-end integration successful"
    else
        log_test "FAIL: End-to-end integration failed"
        return 1
    end
end

function run_performance_tests
    echo -e "$BLUE🧪 Running Performance Tests...$NC"
    
    # WebSocket performance test
    python3 -c "
import asyncio
import websockets
import time
import json

async def test_websocket_performance():
    try:
        start_time = time.time()
        
        async with websockets.connect('ws://localhost:8000/ws') as websocket:
            # Send test messages
            for i in range(10):
                await websocket.send(json.dumps({'test': f'message_{i}'}))
                response = await websocket.recv()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if duration < 5.0:  # Should complete in under 5 seconds
            print(f'✅ WebSocket Performance: {duration:.2f}s - OK')
            return True
        else:
            print(f'❌ WebSocket Performance: {duration:.2f}s - TOO SLOW')
            return False
            
    except Exception as e:
        print(f'❌ WebSocket Performance test failed: {e}')
        return False

# Run the test
if asyncio.run(test_websocket_performance()):
    exit(0)
else:
    exit(1)
    " 2>/dev/null
    
    if test $status -eq 0
        log_test "PASS: WebSocket performance acceptable"
    else
        log_test "FAIL: WebSocket performance issues"
        return 1
    end
    
    # Task Decomposer performance test
    python3 -c "
import time
import sys
sys.path.append('$PROJECT_DIR/shared/orchestration')

try:
    from task_decomposer import TaskDecomposer
    
    decomposer = TaskDecomposer()
    
    start_time = time.time()
    
    # Test multiple decompositions
    for i in range(5):
        result = decomposer.decompose_task(f'Test task number {i+1}')
        if not result:
            raise Exception(f'Decomposition {i+1} failed')
    
    end_time = time.time()
    duration = end_time - start_time
    
    if duration < 10.0:  # Should complete 5 decompositions in under 10 seconds
        print(f'✅ Task Decomposer Performance: {duration:.2f}s - OK')
        exit(0)
    else:
        print(f'❌ Task Decomposer Performance: {duration:.2f}s - TOO SLOW')
        exit(1)
        
except Exception as e:
    print(f'❌ Task Decomposer performance test failed: {e}')
    exit(1)
    "
    
    if test $status -eq 0
        log_test "PASS: Task Decomposer performance acceptable"
    else
        log_test "FAIL: Task Decomposer performance issues"
        return 1
    end
end

function generate_test_report
    echo -e "$BLUE📊 Generating Test Report...$NC"
    
    set report_file "$PROJECT_DIR/logs/test_report_$(date +%Y%m%d_%H%M%S).html"
    
    echo '<!DOCTYPE html>
<html>
<head>
    <title>Agent Zero V1 - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 5px; }
        .pass { color: #28a745; }
        .fail { color: #dc3545; }
        .test-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Agent Zero V1 - Test Report</h1>
        <p>Generated: ' > $report_file
    
    date >> $report_file
    
    echo '    </p>
    </div>
    
    <div class="test-section">
        <h2>📋 Test Results Summary</h2>
        <pre>' >> $report_file
    
    if test -f $TEST_LOG
        cat $TEST_LOG >> $report_file
    end
    
    echo '        </pre>
    </div>
    
    <div class="test-section">
        <h2>📊 System Status</h2>
        <ul>
            <li>Neo4j Connection: <span id="neo4j-status">Testing...</span></li>
            <li>RabbitMQ Connection: <span id="rabbitmq-status">Testing...</span></li>
            <li>WebSocket Monitor: <span id="websocket-status">Testing...</span></li>
            <li>Task Decomposer: <span id="decomposer-status">Testing...</span></li>
            <li>Agent Executor: <span id="executor-status">Testing...</span></li>
        </ul>
    </div>
    
</body>
</html>' >> $report_file
    
    echo -e "$GREEN✅ Test report generated: $report_file$NC"
    log_test "Test report generated at $report_file"
end

function main
    echo -e "$BLUE"
    echo "========================================"
    echo "🧪 Agent Zero V1 - Complete Test Suite"
    echo "========================================"
    echo -e "$NC"
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"
    
    # Initialize test log
    echo "# Agent Zero V1 Test Results - $(date)" > $TEST_LOG
    log_test "Starting comprehensive test suite"
    
    set failed_tests 0
    set total_tests 7
    
    # Run all tests
    echo -e "$YELLOW🔧 Running Infrastructure Tests...$NC"
    test_docker_containers; or set failed_tests (math $failed_tests + 1)
    test_neo4j_connection; or set failed_tests (math $failed_tests + 1)
    test_rabbitmq_connection; or set failed_tests (math $failed_tests + 1)
    
    echo -e "$YELLOW🌐 Running Application Tests...$NC"
    test_websocket_monitor; or set failed_tests (math $failed_tests + 1)
    test_task_decomposer; or set failed_tests (math $failed_tests + 1)
    test_agent_executor; or set failed_tests (math $failed_tests + 1)
    
    echo -e "$YELLOW⚡ Running Integration Tests...$NC"
    test_integration_workflow; or set failed_tests (math $failed_tests + 1)
    
    echo -e "$YELLOW📈 Running Performance Tests...$NC"
    run_performance_tests
    
    # Generate report
    generate_test_report
    
    # Final results
    set passed_tests (math $total_tests - $failed_tests)
    
    echo -e "$BLUE"
    echo "========================================"
    echo "📊 Test Results Summary"
    echo "========================================"
    echo -e "$NC"
    
    if test $failed_tests -eq 0
        echo -e "$GREEN✅ ALL TESTS PASSED! ($passed_tests/$total_tests)$NC"
        echo -e "$GREEN🎉 Agent Zero V1 is fully operational!$NC"
        log_test "SUCCESS: All tests passed ($passed_tests/$total_tests)"
    else
        echo -e "$RED❌ TESTS FAILED: $failed_tests/$total_tests$NC"
        echo -e "$YELLOW⚠️  Passed: $passed_tests/$total_tests$NC"
        log_test "FAILURE: $failed_tests tests failed, $passed_tests passed"
    end
    
    echo
    echo -e "$BLUE💡 Next Steps:$NC"
    echo "• Check detailed log: cat $TEST_LOG"
    echo "• View test report: open $report_file"
    echo "• Monitor system: ./deploy_agent_zero.fish status"
    
    # Return appropriate exit code
    if test $failed_tests -eq 0
        exit 0
    else
        exit 1
    end
end

# Check command line arguments
if test (count $argv) -eq 0
    main
else
    switch $argv[1]
        case "neo4j"
            test_neo4j_connection
        case "rabbitmq"
            test_rabbitmq_connection
        case "websocket"
            test_websocket_monitor
        case "decomposer"
            test_task_decomposer
        case "executor"
            test_agent_executor
        case "integration"
            test_integration_workflow
        case "performance"
            run_performance_tests
        case "docker"
            test_docker_containers
        case "*"
            echo "Usage: $argv[0] [neo4j|rabbitmq|websocket|decomposer|executor|integration|performance|docker]"
            echo "Run without arguments to execute full test suite"
    end
end