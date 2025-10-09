#!/bin/bash
# Agent Zero V1 - Quick System Check Script
# Quick version of the comprehensive Python test

echo "🚀 Agent Zero V1 - Quick System Check"
echo "====================================="
echo "📁 Project Path: $(pwd)"
echo "📅 Test Time: $(date '+%Y-%m-%d %H:%M:%S CEST')"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

test_component() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "✅ ${GREEN}$name: PASS${NC}"
        ((PASS_COUNT++))
    else
        echo -e "❌ ${RED}$name: FAIL${NC}"
        ((FAIL_COUNT++))
    fi
}

test_component_warn() {
    local name="$1"
    local command="$2"
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "✅ ${GREEN}$name: PASS${NC}"
        ((PASS_COUNT++))
    else
        echo -e "⚠️  ${YELLOW}$name: WARN${NC}"
        ((WARN_COUNT++))
    fi
}

echo "🐳 Docker Services Test"
echo "----------------------"

# Test Docker is running
test_component "Docker Service" "docker ps" ""

# Test Neo4j (CRITICAL BLOCKER)
if docker ps | grep -q neo4j; then
    if curl -s http://localhost:7474 > /dev/null; then
        echo -e "✅ ${GREEN}Neo4j Service: PASS${NC} - Container running, port accessible"
        ((PASS_COUNT++))
    else
        echo -e "❌ ${RED}Neo4j Service: FAIL${NC} - Container running but port not accessible"
        ((FAIL_COUNT++))
    fi
else
    echo -e "❌ ${RED}Neo4j Service: FAIL${NC} - Container not running (run: docker-compose up -d neo4j)"
    ((FAIL_COUNT++))
fi

# Test RabbitMQ (WORKING)
if docker ps | grep -q rabbitmq; then
    echo -e "✅ ${GREEN}RabbitMQ Service: PASS${NC} - Container running"
    ((PASS_COUNT++))
else
    echo -e "❌ ${RED}RabbitMQ Service: FAIL${NC} - Container not running"
    ((FAIL_COUNT++))
fi

# Test Ollama LLM Service
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "✅ ${GREEN}Ollama LLM Service: PASS${NC} - API responding"
    ((PASS_COUNT++))
else
    echo -e "❌ ${RED}Ollama LLM Service: FAIL${NC} - API not responding"
    ((FAIL_COUNT++))
fi

echo ""
echo "📁 File Structure Test"
echo "----------------------"

# Test required files exist
test_component "LLM Factory File" "[ -f shared/llm/llm_factory.py ]" ""
test_component "Agent Factory File" "[ -f shared/agent_factory/factory.py ]" ""
test_component "Message Bus File" "[ -f shared/messaging/bus.py ]" ""
test_component "Project Orchestrator File" "[ -f shared/orchestration/project_orchestrator.py ]" ""
test_component "Docker Compose File" "[ -f docker-compose.yml ]" ""
test_component "Agent Templates Dir" "[ -d shared/agent_factory/templates ]" ""

echo ""
echo "🚨 Critical Blockers Test"
echo "-------------------------"

# Test AgentExecutor signature (CRITICAL BLOCKER)
if [ -f test_full_integration.py ]; then
    if grep -q "execute_task(agent, zadanie, katalog_wyjściowy)" test_full_integration.py; then
        echo -e "✅ ${GREEN}AgentExecutor Signature: PASS${NC} - Method signature correct"
        ((PASS_COUNT++))
    elif grep -q "execute_task(zadanie, zespół)" test_full_integration.py; then
        echo -e "❌ ${RED}AgentExecutor Signature: FAIL${NC} - Method signature needs fix on line 129"
        ((FAIL_COUNT++))
    else
        echo -e "⚠️  ${YELLOW}AgentExecutor Signature: WARN${NC} - Cannot determine method signature"
        ((WARN_COUNT++))
    fi
else
    echo -e "❌ ${RED}AgentExecutor Signature: FAIL${NC} - test_full_integration.py not found"
    ((FAIL_COUNT++))
fi

# Test Task Decomposer file exists
test_component "Task Decomposer File" "[ -f shared/orchestration/task_decomposer.py ]" ""

# Test WebSocket Frontend (CRITICAL BLOCKER)
if curl -s http://localhost:8000 > /dev/null; then
    echo -e "✅ ${GREEN}WebSocket Frontend: PASS${NC} - Server responding"
    ((PASS_COUNT++))
else
    echo -e "❌ ${RED}WebSocket Frontend: FAIL${NC} - Server not responding"
    ((FAIL_COUNT++))
fi

echo ""
echo "🐍 Python Environment Test"
echo "--------------------------"

# Test Python can import modules
test_component_warn "Python Import Test" "python3 -c 'import sys; sys.path.insert(0, \".\"); import shared.llm.llm_factory'"

echo ""
echo "📊 Test Summary"
echo "==============="

TOTAL_COUNT=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))

echo "📊 Total Tests: $TOTAL_COUNT"
echo -e "✅ Passed: ${GREEN}$PASS_COUNT${NC} ($(( PASS_COUNT * 100 / TOTAL_COUNT ))%)"
echo -e "❌ Failed: ${RED}$FAIL_COUNT${NC} ($(( FAIL_COUNT * 100 / TOTAL_COUNT ))%)"
echo -e "⚠️  Warnings: ${YELLOW}$WARN_COUNT${NC} ($(( WARN_COUNT * 100 / TOTAL_COUNT ))%)"

echo ""
# System Health Score
HEALTH_SCORE=$(( PASS_COUNT * 100 / TOTAL_COUNT ))

if [ $HEALTH_SCORE -ge 80 ]; then
    echo -e "🏥 System Health: ${GREEN}$HEALTH_SCORE% - EXCELLENT${NC}"
elif [ $HEALTH_SCORE -ge 60 ]; then
    echo -e "🏥 System Health: ${YELLOW}$HEALTH_SCORE% - GOOD${NC}"
else
    echo -e "🏥 System Health: ${RED}$HEALTH_SCORE% - NEEDS ATTENTION${NC}"
fi

echo ""
echo "🎯 Recommended Actions:"
if [ $FAIL_COUNT -gt 0 ]; then
    echo "1. 🚨 Fix critical blockers immediately:"
    if ! docker ps | grep -q neo4j; then
        echo "   • Run: docker-compose up -d neo4j"
    fi
    if grep -q "execute_task(zadanie, zespół)" test_full_integration.py 2>/dev/null; then
        echo "   • Fix AgentExecutor signature in test_full_integration.py line 129"
    fi
    if ! curl -s http://localhost:8000 > /dev/null; then
        echo "   • Fix WebSocket frontend at http://localhost:8000"
    fi
    echo "2. 🔧 Run full Python test: python3 agent_zero_system_test.py"
    echo "3. 📖 Check Notion Critical Actions Tracker for detailed instructions"
else
    echo "1. 🚀 All tests passing - ready for V2.0 development!"
    echo "2. 📈 Consider implementing next sprint features"
    echo "3. 🧪 Run full Python test for detailed analysis"
fi

echo ""
echo "⏱️ Quick test completed at $(date '+%H:%M:%S CEST')"
echo "💾 For detailed results, run: python3 agent_zero_system_test.py"