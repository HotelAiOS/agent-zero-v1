#!/bin/bash

echo "📊 Agent Zero v1.0.0 - Live Status Dashboard"

while true; do
    clear
    echo "=================================="
    echo "   Agent Zero - Live Monitoring"
    echo "   $(date)"
    echo "=================================="
    
    echo ""
    echo "=== Pods Status ==="
    kubectl get pods -n a0-dev | grep -E "NAME|ai-router|chat-service|agent-orchestrator|docs-service|api-gateway|prometheus|grafana"
    
    echo ""
    echo "=== Resource Usage ==="
    kubectl top pods -n a0-dev 2>/dev/null | grep -E "NAME|ai-router|chat-service|agent-orchestrator" || echo "Metrics not available"
    
    echo ""
    echo "=== Recent Errors (last 10 lines) ==="
    kubectl logs -n a0-dev deployment/ai-router --tail=10 2>/dev/null | grep -i error || echo "No errors"
    
    echo ""
    echo "=== Services ==="
    kubectl get svc -n a0-dev | grep -E "NAME|ai-router|chat-service|agent-orchestrator|api-gateway"
    
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
