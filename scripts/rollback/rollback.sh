#!/bin/bash

echo "⚠️  Agent Zero v1.0.0 - ROLLBACK INITIATED"
echo "=========================================="

NAMESPACE="a0-dev"

read -p "Are you sure you want to rollback? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Rollback cancelled"
    exit 0
fi

echo ""
echo "Step 1: Scaling down new deployments..."
kubectl scale deployment/ai-router --replicas=0 -n $NAMESPACE
kubectl scale deployment/chat-service --replicas=0 -n $NAMESPACE
kubectl scale deployment/agent-orchestrator --replicas=0 -n $NAMESPACE
kubectl scale deployment/docs-service --replicas=0 -n $NAMESPACE
kubectl scale deployment/api-gateway --replicas=0 -n $NAMESPACE

echo ""
echo "Step 2: Waiting for pods to terminate..."
sleep 10

echo ""
echo "Step 3: Database backup..."
kubectl exec -n $NAMESPACE postgresql-0 -- pg_dump -U a0 agentzero > /tmp/rollback-backup-$(date +%Y%m%d-%H%M%S).sql
echo "✅ Backup saved"

echo ""
echo "Step 4: Restore to previous version..."
echo "⚠️  Manual intervention required:"
echo "  1. Restore v0.0.2 Docker Compose"
echo "  2. Restore database backup"
echo "  3. Verify old system"

echo ""
echo "Rollback preparation completed"
echo "Manual steps required to complete rollback"
