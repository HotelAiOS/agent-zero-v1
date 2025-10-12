#!/bin/bash
# Agent Zero V1 - Production Deployment Script

echo "🐳 Agent Zero V1 - Production Deployment"
echo "========================================"
echo "🎯 Deploying complete production stack..."
echo ""

# Create necessary directories
mkdir -p data logs nginx monitoring

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
echo ""

services=("master-integrator:8000" "team-formation:8001" "analytics:8002" "collaboration:8003" "predictive:8004" "adaptive-learning:8005" "quantum-intelligence:8006")

for service in "${services[@]}"; do
    if curl -f -s "http://localhost:${service##*:}/health" > /dev/null; then
        echo "✅ $service - HEALTHY"
    else
        echo "❌ $service - UNHEALTHY"
    fi
done

echo ""
echo "🎯 Service URLs:"
echo "   • Main API: http://localhost/api/"
echo "   • API Gateway: http://localhost/"
echo "   • Prometheus: http://localhost:9090/"
echo "   • Grafana: http://localhost:3000/ (admin/admin123)"
echo ""

echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🎉 Agent Zero V1 Production Deployment Complete!"
echo "🚀 Ready for enterprise use and Dev B frontend integration!"
