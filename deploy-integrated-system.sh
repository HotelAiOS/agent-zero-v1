#!/bin/bash

# 🚀 Agent Zero V1 - Integrated System Production Deployment
# Author: AI Development Assistant
# Version: 1.0.0

echo "🔥 AGENT ZERO V1 - INTEGRATED SYSTEM DEPLOYMENT"
echo "=============================================="

# Kolory dla output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funkcje pomocnicze
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Sprawdzenie wymagań systemowych
check_requirements() {
    log_info "Sprawdzam wymagania systemowe..."
    
    # Docker
    if command -v docker &> /dev/null; then
        log_success "Docker dostępny: $(docker --version)"
    else
        log_error "Docker nie znaleziony! Zainstaluj Docker Desktop"
        exit 1
    fi
    
    # Docker Compose
    if command -v docker-compose &> /dev/null; then
        log_success "Docker Compose dostępny"
    else
        log_warning "Docker Compose nie znaleziony, używam 'docker compose'"
    fi
    
    # Python 3.11+
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python dostępny: $PYTHON_VERSION"
    else
        log_error "Python 3.11+ wymagany!"
        exit 1
    fi
    
    # Ollama
    if command -v ollama &> /dev/null; then
        log_success "Ollama dostępny"
    else
        log_warning "Ollama nie znaleziony - zostanie zainstalowany"
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
}

# Pobieranie modeli AI
setup_ai_models() {
    log_info "Pobieranie modeli AI..."
    
    # Sprawdź czy Ollama jest uruchomiony
    if ! pgrep -x "ollama" > /dev/null; then
        log_info "Uruchamiam Ollama..."
        ollama serve &
        sleep 5
    fi
    
    # Pobierz modele
    log_info "Pobieranie deepseek-coder:33b..."
    ollama pull deepseek-coder:33b
    
    log_info "Pobieranie qwen2.5:14b..."
    ollama pull qwen2.5:14b
    
    log_info "Pobieranie llama3.2:3b dla testów..."
    ollama pull llama3.2:3b
    
    log_success "Modele AI pobrane pomyślnie"
}

# Utworzenie środowiska wirtualnego
setup_python_env() {
    log_info "Tworzę środowisko Python..."
    
    if [ -d "venv" ]; then
        log_warning "Środowisko już istnieje, używam istniejące"
    else
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-production.txt
    
    log_success "Środowisko Python gotowe"
}

# Konfiguracja Docker
setup_docker_services() {
    log_info "Konfiguracja usług Docker..."
    
    # Sprawdź czy docker-compose.yml istnieje
    if [ ! -f "docker-compose.yml" ]; then
        log_info "Tworzę docker-compose.yml..."
        cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  neo4j:
    image: neo4j:5.13
    container_name: agent_zero_neo4j
    environment:
      NEO4J_AUTH: neo4j/password123
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    networks:
      - agent_zero_network

  redis:
    image: redis:7-alpine
    container_name: agent_zero_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agent_zero_network

  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: agent_zero_rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin123
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent_zero_network

volumes:
  neo4j_data:
  neo4j_logs:
  redis_data:
  rabbitmq_data:

networks:
  agent_zero_network:
    driver: bridge
EOF
    fi
    
    # Uruchom usługi
    log_info "Uruchamiam usługi Docker..."
    docker-compose up -d
    
    # Czekaj na gotowość usług
    log_info "Czekam na gotowość usług (30s)..."
    sleep 30
    
    log_success "Usługi Docker uruchomione"
}

# Testy systemu
run_tests() {
    log_info "Uruchamiam testy systemu..."
    
    source venv/bin/activate
    python test-integrated-system.py
    
    if [ $? -eq 0 ]; then
        log_success "Wszystkie testy przeszły pomyślnie"
    else
        log_error "Niektóre testy nie przeszły - sprawdź logi"
    fi
}

# Uruchomienie systemu
start_system() {
    log_info "Uruchamiam Integrated System..."
    
    source venv/bin/activate
    
    # Uruchom system w tle
    nohup python integrated-system-production.py > system.log 2>&1 &
    SYSTEM_PID=$!
    
    # Uruchom monitoring dashboard
    nohup python monitoring-dashboard.py > dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    
    # Zapisz PID dla późniejszego zatrzymania
    echo $SYSTEM_PID > system.pid
    echo $DASHBOARD_PID > dashboard.pid
    
    log_success "System uruchomiony!"
    log_info "System PID: $SYSTEM_PID"
    log_info "Dashboard PID: $DASHBOARD_PID"
}

# Sprawdzenie health check
health_check() {
    log_info "Sprawdzam health systemu..."
    
    sleep 5  # Daj systemowi czas na start
    
    # Test głównego API
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        log_success "Główne API działa - http://localhost:8000"
    else
        log_error "Główne API nie odpowiada"
    fi
    
    # Test dashboard
    if curl -s http://localhost:8080/health > /dev/null; then
        log_success "Dashboard działa - http://localhost:8080"
    else
        log_warning "Dashboard nie odpowiada"
    fi
    
    # Test Neo4j
    if curl -s http://localhost:7474 > /dev/null; then
        log_success "Neo4j działa - http://localhost:7474"
    else
        log_warning "Neo4j nie odpowiada"
    fi
}

# Wyświetlenie informacji końcowych
show_final_info() {
    echo ""
    echo "🎉 DEPLOYMENT ZAKOŃCZONY POMYŚLNIE!"
    echo "=================================="
    echo ""
    log_success "🌐 Główne API: http://localhost:8000"
    log_success "📊 Dashboard: http://localhost:8080"
    log_success "🗄️  Neo4j Browser: http://localhost:7474 (neo4j/password123)"
    log_success "🐰 RabbitMQ Management: http://localhost:15672 (admin/admin123)"
    echo ""
    echo "📋 Przydatne komendy:"
    echo "• Status: curl http://localhost:8000/api/v1/health"
    echo "• Logi systemu: tail -f system.log"
    echo "• Logi dashboard: tail -f dashboard.log"
    echo "• Zatrzymaj: ./stop-system.sh"
    echo ""
    echo "📖 Dokumentacja: Otwórz QUICK_START_INTEGRATED.md"
    echo ""
}

# Główna funkcja deployment
main() {
    echo "🚀 Rozpoczynam deployment Agent Zero V1 Integrated System..."
    echo ""
    
    check_requirements
    setup_ai_models
    setup_python_env
    setup_docker_services
    run_tests
    start_system
    health_check
    show_final_info
    
    log_success "🎯 SYSTEM GOTOWY DO UŻYCIA!"
}

# Uruchom deployment
main
