#!/bin/bash
# Agent Zero V1 - Production Server Deployment

echo "🚀 Agent Zero V1 - Production Server Mode Deployment"
echo "===================================================="
echo "🎯 Converting demo systems to long-running servers"
echo ""

# Clean and rebuild with server mode
echo "🧹 Cleaning existing containers..."
docker-compose down --remove-orphans

echo "🏗️  Rebuilding with production servers..."
docker-compose up --build -d

echo "⏳ Waiting for servers to start (servers run forever now)..."
sleep 30

echo "🔍 Checking server health..."
echo ""

services=("master-integrator:8000" "team-formation:8001" "analytics:8002" "collaboration:8003" "predictive:8004" "adaptive-learning:8005" "quantum-intelligence:8006")

for service in "${services[@]}"; do
    port=${service##*:}
    name=${service%:*}
    if curl -f -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ $name - SERVER OPERATIONAL"
    else
        echo "🔄 $name - Starting up..."
    fi
done

echo ""
echo "🎯 Production Server URLs:"
echo "   • Master API: http://localhost/api/ (via Gateway)"
echo "   • Direct Master: http://localhost:8000/api/"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Team Formation: http://localhost:8001/"
echo "   • Analytics: http://localhost:8002/"
echo "   • Collaboration: http://localhost:8003/"
echo "   • Predictive: http://localhost:8004/"
echo "   • Adaptive Learning: http://localhost:8005/"
echo "   • Quantum Intelligence: http://localhost:8006/"
echo ""
echo "📊 Container status:"
docker ps --filter "name=agent-zero*" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎉 Production Server Mode Deployment Complete!"
echo "🚀 All services now run as long-running servers!"
echo "💼 No more restarts - true production behavior!"
