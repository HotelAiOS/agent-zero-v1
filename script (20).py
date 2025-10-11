# Tworzenie pliku requirements dla produkcji
requirements_production = """# Agent Zero V1 - Production Requirements
# =====================================
# Core FastAPI & Web
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
aiohttp==3.9.1
pydantic==2.5.0

# Database & Cache
neo4j==5.13.0
redis==5.0.1
sqlite3  # Built-in with Python

# Message Queue
pika==1.3.2

# Data Processing
pandas==2.1.3
numpy==1.24.4

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Utilities
python-multipart==0.0.6
python-dotenv==1.0.0
click==8.1.7
rich==13.6.0
"""

# Zapisz requirements
with open("requirements-production.txt", "w", encoding="utf-8") as f:
    f.write(requirements_production)

print("✅ Utworzono requirements-production.txt")

# Tworzenie deployment script
deployment_script = """#!/bin/bash
# 🚀 Agent Zero V1 - Production Deployment Script
# ===============================================
# Kompletny deployment systemu zintegrowanego z AI

set -e

echo "🚀 AGENT ZERO V1 - PRODUCTION DEPLOYMENT"
echo "========================================"

# Sprawdź czy Docker jest uruchomiony
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker nie jest uruchomiony. Uruchom Docker i spróbuj ponownie."
    exit 1
fi

# Sprawdź czy Ollama jest dostępne
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "⚠️ Ollama nie jest dostępne. Uruchamianie Ollama..."
    ollama serve &
    sleep 10
    
    # Pobierz wymagane modele
    echo "📥 Pobieranie modeli AI..."
    ollama pull deepseek-coder:33b
    ollama pull qwen2.5:14b  
    ollama pull qwen2.5:7b
fi

# Utwórz katalog projektu jeśli nie istnieje
mkdir -p agent-zero-v1-integrated
cd agent-zero-v1-integrated

# Skopiuj pliki
cp ../integrated-system-production.py ./integrated-system.py
cp ../requirements-production.txt ./requirements.txt

# Utwórz .env dla produkcji
cat > .env << 'EOF'
# Agent Zero V1 - Production Environment
ENVIRONMENT=production
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=agent-zero-pass
REDIS_URL=redis://localhost:6379
RABBITMQ_URL=amqp://localhost:5672
OLLAMA_URL=http://localhost:11434
LOG_LEVEL=INFO
EOF

# Utwórz Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

EXPOSE 8000

CMD ["python", "integrated-system.py", "--mode", "production"]
EOF

# Utwórz docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  agent-zero-integrated:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - NEO4J_URL=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://rabbitmq:5672
      - OLLAMA_URL=http://host.docker.internal:11434
    depends_on:
      - neo4j
      - redis
      - rabbitmq
    restart: unless-stopped
    networks:
      - agent-zero-network

  neo4j:
    image: neo4j:5.11
    environment:
      NEO4J_AUTH: neo4j/agent-zero-pass
      NEO4J_dbms_security_procedures_unrestricted: "gds.*,apoc.*"
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "agent-zero-pass", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: agent
      RABBITMQ_DEFAULT_PASS: zero-pass
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  neo4j_data:
  redis_data:
  rabbitmq_data:

networks:
  agent-zero-network:
    driver: bridge
EOF

# Utwórz Makefile dla zarządzania
cat > Makefile << 'EOF'
.PHONY: help build run stop clean test logs

help: ## Pokaż pomoc
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

build: ## Zbuduj obrazy Docker
	docker-compose build

run: ## Uruchom wszystkie serwisy
	docker-compose up -d

stop: ## Zatrzymaj wszystkie serwisy
	docker-compose down

clean: ## Wyczyść wszystko (UWAGA: usuwa dane)
	docker-compose down -v
	docker system prune -f

test: ## Uruchom testy
	python integrated-system.py --mode demo

logs: ## Pokaż logi
	docker-compose logs -f

status: ## Sprawdź status serwisów
	docker-compose ps

restart: ## Restartuj serwisy
	docker-compose restart

install: ## Zainstaluj zależności lokalnie
	pip install -r requirements.txt
EOF

echo "✅ DEPLOYMENT SETUP COMPLETE!"
echo ""
echo "📦 Pliki wdrożeniowe utworzone:"
echo "   • integrated-system.py      - Główny system"
echo "   • requirements.txt          - Zależności Python"
echo "   • Dockerfile               - Obraz aplikacji"
echo "   • docker-compose.yml       - Kompletna infrastruktura"
echo "   • .env                     - Konfiguracja środowiska"
echo "   • Makefile                 - Zarządzanie projektem"
echo ""
echo "🚀 INSTRUKCJE URUCHOMIENIA:"
echo "=========================="
echo "1. make build              # Zbuduj obrazy"
echo "2. make run                # Uruchom wszystkie serwisy"
echo "3. make test               # Uruchom demo test"
echo "4. make logs               # Zobacz logi"
echo ""
echo "🌐 ENDPOINTY:"
echo "============="
echo "• API:           http://localhost:8000"
echo "• Health Check:  http://localhost:8000/api/v1/health"
echo "• Neo4j Browser: http://localhost:7474"
echo "• RabbitMQ UI:   http://localhost:15672"
echo ""
echo "🎯 System gotowy do produkcji z pełną integracją AI!"

cd ..
"""

# Zapisz deployment script
with open("deploy-integrated-system.sh", "w", encoding="utf-8") as f:
    f.write(deployment_script)

# Uczyń plik wykonywalnym
import os
os.chmod("deploy-integrated-system.sh", 0o755)

print("✅ Utworzono deploy-integrated-system.sh")
print("🔧 Script jest wykonywalny i gotowy do użycia")