#!/bin/bash

echo "🧪 Agent Zero v1.0.0 - Smoke Tests"
echo "=================================="

NAMESPACE="a0-dev"
FAILED=0

# Test 1: Wszystkie pody działają
echo ""
echo "Test 1: Pod Health Check"
PODS=$(kubectl get pods -n $NAMESPACE --no-headers | wc -l)
RUNNING=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase=Running --no-headers | wc -l)

if [ "$PODS" -eq "$RUNNING" ]; then
    echo "✅ All $PODS pods are Running"
else
    echo "❌ Only $RUNNING/$PODS pods are Running"
    FAILED=$((FAILED + 1))
fi

# Test 2: AI Router Health
echo ""
echo "Test 2: AI Router Health"
kubectl exec -n $NAMESPACE deployment/ai-router -- curl -sf http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ AI Router is healthy"
else
    echo "❌ AI Router health check failed"
    FAILED=$((FAILED + 1))
fi

# Test 3: Chat Service Health
echo ""
echo "Test 3: Chat Service Health"
kubectl exec -n $NAMESPACE deployment/chat-service -- curl -sf http://localhost:8001/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Chat Service is healthy"
else
    echo "❌ Chat Service health check failed"
    FAILED=$((FAILED + 1))
fi

# Test 4: Agent Orchestrator Health
echo ""
echo "Test 4: Agent Orchestrator Health"
kubectl exec -n $NAMESPACE deployment/agent-orchestrator -- curl -sf http://localhost:8002/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Agent Orchestrator is healthy"
else
    echo "❌ Agent Orchestrator health check failed"
    FAILED=$((FAILED + 1))
fi

# Test 5: Docs Service Health
echo ""
echo "Test 5: Docs Service Health"
kubectl exec -n $NAMESPACE deployment/docs-service -- curl -sf http://localhost:8003/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Docs Service is healthy"
else
    echo "❌ Docs Service health check failed"
    FAILED=$((FAILED + 1))
fi

# Test 6: API Gateway Health
echo ""
echo "Test 6: API Gateway Health"
kubectl exec -n $NAMESPACE deployment/api-gateway -- curl -sf http://localhost:8080/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API Gateway is healthy"
else
    echo "❌ API Gateway health check failed"
    FAILED=$((FAILED + 1))
fi

# Test 7: Database connectivity
echo ""
echo "Test 7: PostgreSQL Connectivity"
kubectl exec -n $NAMESPACE postgresql-0 -- psql -U a0 -d agentzero -c "SELECT 1;" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL is accessible"
else
    echo "❌ PostgreSQL connection failed"
    FAILED=$((FAILED + 1))
fi

# Test 8: Redis connectivity
echo ""
echo "Test 8: Redis Connectivity"
kubectl exec -n $NAMESPACE redis-0 -- redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Redis is accessible"
else
    echo "❌ Redis connection failed"
    FAILED=$((FAILED + 1))
fi

# Test 9: Prometheus
echo ""
echo "Test 9: Prometheus"
kubectl exec -n $NAMESPACE deployment/prometheus -- wget -q -O- http://localhost:9090/-/healthy > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus health check failed"
    FAILED=$((FAILED + 1))
fi

# Test 10: Grafana
echo ""
echo "Test 10: Grafana"
kubectl exec -n $NAMESPACE deployment/grafana -- wget -q -O- http://localhost:3000/api/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana health check failed"
    FAILED=$((FAILED + 1))
fi

# Podsumowanie
echo ""
echo "=================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All smoke tests passed!"
    exit 0
else
    echo "❌ $FAILED test(s) failed"
    exit 1
fi

