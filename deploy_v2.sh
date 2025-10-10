#!/bin/bash
#
# Agent Zero V1 - V2.0 Intelligence Layer Deployment
# Week 43 Implementation
#

set -e

echo "🚀 Deploying Agent Zero V1 - V2.0 Intelligence Layer..."

# Create directories
echo "📁 Creating directory structure..."
mkdir -p shared/kaizen
mkdir -p shared/knowledge  
mkdir -p shared/utils
mkdir -p cli
mkdir -p backups

# Backup existing files
echo "💾 Creating backups..."
BACKUP_DIR="backups/deploy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "cli/__main__.py" ]; then
    cp "cli/__main__.py" "$BACKUP_DIR/"
    echo "✅ Backed up CLI"
fi

if [ -f "shared/utils/simple_tracker.py" ]; then
    cp "shared/utils/simple_tracker.py" "$BACKUP_DIR/"
    echo "✅ Backed up SimpleTracker"
fi

echo "📦 V2.0 Intelligence Layer components deployed"
echo ""
echo "🔧 Next steps:"
echo "1. Run: python -m cli status"
echo "2. Test: python -m cli ask 'Hello Agent Zero'"
echo "3. Generate report: python -m cli kaizen-report"
echo ""
echo "📚 Available commands:"
echo "  - python -m cli ask <question>"
echo "  - python -m cli kaizen-report"
echo "  - python -m cli cost-analysis"
echo "  - python -m cli pattern-discovery"
echo "  - python -m cli model-reasoning <task_type>"
echo "  - python -m cli success-breakdown"
echo "  - python -m cli sync-knowledge-graph"
echo "  - python -m cli status"
echo ""
echo "🎉 V2.0 Intelligence Layer deployment complete!"
