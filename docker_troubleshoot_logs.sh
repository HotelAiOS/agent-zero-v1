#!/bin/bash
# Agent Zero V1 - Docker Troubleshooting & Logs Analysis
# Check container logs to diagnose startup issues

echo "🔍 DOCKER TROUBLESHOOTING - Analyzing Container Logs"
echo "====================================================="
echo "🎯 Services are restarting - let's check the logs"
echo ""

# Function to check container logs
check_logs() {
    local container=$1
    echo "📋 Checking logs for: $container"
    echo "----------------------------------------"
    docker logs $container --tail 20 2>&1 || echo "❌ Container not found or no logs"
    echo ""
}

# Check each service logs
echo "🔍 Checking Agent Zero service logs..."
check_logs "agent-zero-master"
check_logs "agent-zero-team"
check_logs "agent-zero-analytics"
check_logs "agent-zero-collaboration"
check_logs "agent-zero-predictive"
check_logs "agent-zero-adaptive"
check_logs "agent-zero-quantum"

echo "🔍 Checking infrastructure logs..."
check_logs "agent-zero-gateway"

echo "📊 Current container status:"
docker ps -a --filter "name=agent-zero*" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎯 Common issues to look for:"
echo "   • ImportError: Missing Python modules"
echo "   • FileNotFoundError: Missing required files"
echo "   • PortError: Port already in use"
echo "   • ModuleNotFoundError: Wrong Python path"
echo ""

echo "💡 Quick diagnosis commands:"
echo "   docker logs agent-zero-master --tail 50"
echo "   docker exec -it agent-zero-master /bin/bash"
echo "   docker-compose logs master-integrator"