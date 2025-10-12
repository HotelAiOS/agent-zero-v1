#!/bin/bash
# quick_arch_fix.sh - Szybka naprawa dla Agent Zero V2.0 na Arch Linux

echo "🔧 Agent Zero V2.0 - Szybka Naprawa Arch Linux"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Create Virtual Environment and Install Dependencies
log_info "Tworzenie środowiska wirtualnego Python..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    log_success "Utworzono środowisko wirtualne"
else
    log_info "Środowisko wirtualne już istnieje"
fi

# Activate and install dependencies
source venv/bin/activate
log_info "Instalowanie zależności ML..."
pip install --upgrade pip
pip install scikit-learn joblib numpy pandas neo4j fastapi uvicorn pydantic aiofiles

# 2. Start Neo4j with Docker (easiest way)
log_info "Uruchamianie Neo4j przez Docker..."
docker run -d --name neo4j-agent-zero \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=none \
    -e NEO4J_dbms_memory_heap_initial__size=512m \
    neo4j:5.15-community

sleep 10

# 3. Create simple API main
log_info "Tworzenie API main..."
mkdir -p api
cat > api/main.py << 'EOF'
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Agent Zero V2.0", version="2.0.0")

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
async def root():
    return {"message": "Agent Zero V2.0 Intelligence Layer", "docs": "/docs"}
EOF

touch api/__init__.py

# 4. Create Docker requirements
cat > requirements.txt << 'EOF'
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
neo4j>=5.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
aiofiles>=23.0.0
EOF

# 5. Create simple Dockerfile
cat > Dockerfile.ai-intelligence << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# 6. Fix docker-compose.yml
log_info "Naprawianie docker-compose.yml..."
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

cat > docker-compose.yml << 'EOF'
services:
  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=none
      - NEO4J_dbms_memory_heap_initial__size=512m
    networks:
      - agent-network

  api:
    build:
      context: .
      dockerfile: Dockerfile.ai-intelligence
    ports:
      - "8000:8000"
    depends_on:
      - neo4j
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge
EOF

# 7. Create start script  
cat > start_v2.sh << 'EOF'
#!/bin/bash
echo "🚀 Uruchamianie Agent Zero V2.0..."

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo "✅ Aktywowano środowisko wirtualne"
else
    echo "❌ Brak środowiska wirtualnego"
    exit 1
fi

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j działa"
else
    echo "🔄 Uruchamianie Neo4j..."
    docker start neo4j-agent-zero 2>/dev/null || \
    docker run -d --name neo4j-agent-zero \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=none \
        neo4j:5.15-community
    sleep 15
fi

# Start API
echo "🌐 Uruchamianie API serwera..."
echo "📊 Dostęp: http://localhost:8000/docs"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
EOF

chmod +x start_v2.sh

# 8. Create test with venv
cat > test_with_venv.sh << 'EOF'
#!/bin/bash
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    python test-complete-implementation.py "$@"
else
    echo "❌ Brak środowiska wirtualnego. Uruchom ./quick_arch_fix.sh"
    exit 1
fi
EOF

chmod +x test_with_venv.sh

echo ""
log_success "🎉 Szybka naprawa zakończona!"
echo ""
echo "📋 Co zostało naprawione:"
echo "✅ Virtual Environment z ML dependencies"
echo "✅ Neo4j przez Docker"
echo "✅ Podstawowe API"
echo "✅ Naprawiony docker-compose.yml" 
echo "✅ Dockerfile.ai-intelligence"
echo ""
echo "🚀 Następne kroki:"
echo "1. Uruchom: ./start_v2.sh"
echo "2. Testuj: ./test_with_venv.sh"
echo "3. API: http://localhost:8000/docs"
echo ""
echo "🐍 Aktywuj środowisko: source venv/bin/activate"