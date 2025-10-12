#!/bin/bash
# 🧪 AGENT ZERO V1 - COMPREHENSIVE SYSTEM TEST
# ============================================
# Final verification test after successful deployment

echo "🧪 AGENT ZERO V1 - COMPREHENSIVE SYSTEM TEST"
echo "============================================="
echo "📅 $(date '+%Y-%m-%d %H:%M:%S CEST')"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Service definitions - ALL 8 MICROSERVICES
declare -A SERVICES=(
    ["basic_ai"]="8000"
    ["agent_selection"]="8002"  
    ["task_prioritization"]="8003"
    ["ai_collaboration"]="8005"
    ["unified_system"]="8006"
    ["experience_management"]="8007"
    ["pattern_mining"]="8008"
    ["enterprise_ai"]="9010"
)

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SERVICE_COUNT=0
OPERATIONAL_SERVICES=0

echo "🔍 PHASE 1: PORT AVAILABILITY CHECK"
echo "==================================="
for service in "${!SERVICES[@]}"; do
    port=${SERVICES[$service]}
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    SERVICE_COUNT=$((SERVICE_COUNT + 1))
    
    echo -n "📡 Checking port $port ($service)... "
    
    if netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; then
        echo -e "${GREEN}✅ PORT ACTIVE${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        OPERATIONAL_SERVICES=$((OPERATIONAL_SERVICES + 1))
    else
        echo -e "${RED}❌ PORT INACTIVE${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
done

echo ""
echo "🌐 PHASE 2: HTTP CONNECTIVITY TEST"
echo "=================================="
for service in "${!SERVICES[@]}"; do
    port=${SERVICES[$service]}
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -n "🔗 Testing HTTP connection to $service (port $port)... "
    
    # Test HTTP connection with timeout
    if curl -s --connect-timeout 3 "http://localhost:$port/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ HTTP RESPONSIVE${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif curl -s --connect-timeout 3 "http://localhost:$port/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ HEALTH CHECK OK${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    elif curl -s --connect-timeout 3 "http://localhost:$port/status" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ STATUS ENDPOINT OK${NC}" 
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${YELLOW}⚠️ NO HTTP RESPONSE${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
done

echo ""
echo "🔍 PHASE 3: PROCESS VERIFICATION"
echo "==============================="
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo -n "🔍 Checking for Agent Zero processes... "

# Count running Agent Zero processes
PROCESS_COUNT=$(ps aux | grep -E "agent.*zero|point[0-9]|dynamic.*task|ultimate.*ai|experience.*fixed|pattern.*mining|integrated.*system" | grep -v grep | wc -l)

if [ $PROCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}✅ $PROCESS_COUNT PROCESSES RUNNING${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    
    # Show running processes
    echo "   Running processes:"
    ps aux | grep -E "agent.*zero|point[0-9]|dynamic.*task|ultimate.*ai|experience.*fixed|pattern.*mining|integrated.*system" | grep -v grep | while read line; do
        echo "   📋 $line"
    done
else
    echo -e "${RED}❌ NO AGENT ZERO PROCESSES FOUND${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

echo ""
echo "🧪 PHASE 4: API ENDPOINT TESTING"
echo "==============================="

# Test key endpoints
declare -A ENDPOINTS=(
    ["basic_ai:8000"]="/"
    ["enterprise_ai:9010"]="/"
    ["task_prioritization:8003"]="/health"
)

for endpoint_key in "${!ENDPOINTS[@]}"; do
    IFS=':' read -r service_name port <<< "$endpoint_key"
    endpoint=${ENDPOINTS[$endpoint_key]}
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -n "🎯 Testing $service_name$endpoint... "
    
    response=$(curl -s --connect-timeout 3 "http://localhost:$port$endpoint" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo -e "${GREEN}✅ API RESPONSIVE${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "   📄 Response preview: ${response:0:100}..."
    else
        echo -e "${YELLOW}⚠️ NO API RESPONSE${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
done

echo ""
echo "🏥 PHASE 5: SYSTEM HEALTH SUMMARY"
echo "================================="

# Calculate percentages
SERVICE_PERCENTAGE=$((OPERATIONAL_SERVICES * 100 / SERVICE_COUNT))
OVERALL_PERCENTAGE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

echo "📊 SERVICE STATUS:"
echo "   🎯 Total Services: $SERVICE_COUNT"
echo "   ✅ Operational: $OPERATIONAL_SERVICES"
echo "   ❌ Non-operational: $((SERVICE_COUNT - OPERATIONAL_SERVICES))"
echo "   📈 Service Success Rate: ${SERVICE_PERCENTAGE}%"
echo ""

echo "📊 OVERALL TEST RESULTS:"
echo "   🧪 Total Tests: $TOTAL_TESTS"
echo "   ✅ Passed: $PASSED_TESTS"
echo "   ❌ Failed: $FAILED_TESTS"
echo "   📈 Overall Success Rate: ${OVERALL_PERCENTAGE}%"
echo ""

# Final status determination
echo "🏆 FINAL SYSTEM STATUS"
echo "====================="

if [ $SERVICE_PERCENTAGE -eq 100 ]; then
    echo -e "${GREEN}🎊 LEGENDARY SUCCESS - ALL SERVICES OPERATIONAL (100%)${NC}"
    echo -e "${GREEN}🏆 AGENT ZERO V1 - WORLD-CLASS DEPLOYMENT ACHIEVED!${NC}"
    EXIT_CODE=0
elif [ $SERVICE_PERCENTAGE -ge 90 ]; then
    echo -e "${GREEN}🌟 EXCELLENT - Nearly Perfect Deployment ($SERVICE_PERCENTAGE%)${NC}"
    echo -e "${GREEN}🎯 Outstanding Achievement!${NC}"
    EXIT_CODE=0
elif [ $SERVICE_PERCENTAGE -ge 75 ]; then
    echo -e "${CYAN}👍 VERY GOOD - Strong System Performance ($SERVICE_PERCENTAGE%)${NC}"
    echo -e "${CYAN}💪 Great Progress Made!${NC}"
    EXIT_CODE=0
elif [ $SERVICE_PERCENTAGE -ge 50 ]; then
    echo -e "${YELLOW}⚠️ PARTIAL SUCCESS - Moderate Performance ($SERVICE_PERCENTAGE%)${NC}"
    echo -e "${YELLOW}🔧 Some Services Need Attention${NC}"
    EXIT_CODE=1
else
    echo -e "${RED}❌ MAJOR ISSUES - Low Performance ($SERVICE_PERCENTAGE%)${NC}"
    echo -e "${RED}🚨 System Needs Significant Fixes${NC}"
    EXIT_CODE=2
fi

echo ""
echo "📝 Test completed: $(date '+%Y-%m-%d %H:%M:%S CEST')"
echo "📄 Detailed logs available in service-specific log files"
echo ""

# Save results to file
TEST_REPORT="agent_zero_test_results_$(date '+%Y%m%d_%H%M%S').log"
{
    echo "AGENT ZERO V1 - TEST RESULTS"
    echo "============================"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S CEST')"
    echo "Services Operational: $OPERATIONAL_SERVICES/$SERVICE_COUNT ($SERVICE_PERCENTAGE%)"
    echo "Overall Tests Passed: $PASSED_TESTS/$TOTAL_TESTS ($OVERALL_PERCENTAGE%)"
    echo ""
    echo "Service Status Details:"
    for service in "${!SERVICES[@]}"; do
        port=${SERVICES[$service]}
        if netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; then
            echo "✅ $service (Port $port): OPERATIONAL"
        else
            echo "❌ $service (Port $port): NOT RUNNING"
        fi
    done
} > "$TEST_REPORT"

echo "📄 Results saved to: $TEST_REPORT"
exit $EXIT_CODE