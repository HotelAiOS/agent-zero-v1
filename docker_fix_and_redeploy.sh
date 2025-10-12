#!/bin/bash
# Agent Zero V1 - Docker Fix & Redeploy
# Fix requirements.txt and redeploy

echo "🔧 DOCKER BUILD FIX - Fixing requirements.txt"
echo "=============================================="

# Fix 1: Remove sqlite3 from requirements.txt (it's built-in)
echo "📝 Fixing requirements.txt..."
cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
python-multipart==0.0.6
httpx==0.25.2
prometheus-client==0.19.0
pyyaml==6.0.1
EOF

echo "✅ Fixed requirements.txt (removed sqlite3 - it's built-in)"

# Fix 2: Remove version from docker-compose.yml (deprecated)
echo "📝 Fixing docker-compose.yml version warning..."
sed -i '/^version:/d' docker-compose.yml
echo "✅ Removed deprecated 'version' from docker-compose.yml"

# Fix 3: Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose down --remove-orphans
docker system prune -f
echo "✅ Cleanup complete"

# Fix 4: Redeploy with fixes
echo "🚀 Redeploying with fixes..."
echo ""
./deploy_production.sh