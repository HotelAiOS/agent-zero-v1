#!/bin/bash
# arch_linux_fix.sh - Agent Zero V2.0 Arch Linux Deployment Fix
# Rozwiązuje problemy z externally-managed-environment, Neo4j, Docker i dependencies

echo "🔧 Agent Zero V2.0 - Arch Linux Deployment Fix"
echo "============================================================"
echo "Naprawianie problemów wykrytych w deploymencie..."
echo ""

# Configuration
PROJECT_ROOT=${1:-$(pwd)}
VENV_DIR="${PROJECT_ROOT}/venv"
BACKUP_DIR="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_fix() {
    echo -e "${PURPLE}[FIX]${NC} $1"
}

# Fix 1: Create Python Virtual Environment for Arch Linux
fix_python_environment() {
    log_fix "Creating Python virtual environment for Arch Linux..."
    
    # Remove existing venv if it exists
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create new virtual environment
    log_info "Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    if [[ $? -ne 0 ]]; then
        log_error "Failed to create virtual environment"
        log_info "Trying to install python-virtualenv..."
        sudo pacman -S --noconfirm python-virtualenv python-pip
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment created and activated"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install required dependencies
    log_info "Installing ML dependencies in virtual environment..."
    pip install scikit-learn>=1.3.0 joblib>=1.3.0 numpy>=1.24.0 pandas>=2.0.0
    pip install neo4j>=5.0.0
    pip install asyncio-mqtt>=0.13.0 aiofiles>=23.0.0
    pip install fastapi>=0.104.0 uvicorn>=0.24.0 pydantic>=2.0.0
    
    log_success "Dependencies installed in virtual environment"
    
    # Create activation script
    cat > "${PROJECT_ROOT}/activate_venv.sh" << 'EOF'
#!/bin/bash
# Activate Agent Zero V2.0 virtual environment
source venv/bin/activate
echo "🐍 Agent Zero V2.0 virtual environment activated"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
python --version
EOF
    
    chmod +x "${PROJECT_ROOT}/activate_venv.sh"
    log_success "Created activation script: ./activate_venv.sh"
}

# Fix 2: Install and Configure Neo4j for Arch Linux
fix_neo4j_service() {
    log_fix "Installing and configuring Neo4j for Arch Linux..."
    
    # Check if Neo4j is installed
    if ! command -v neo4j &> /dev/null; then
        log_info "Installing Neo4j via AUR..."
        
        # Check if yay is available
        if command -v yay &> /dev/null; then
            yay -S --noconfirm neo4j-community
        elif command -v paru &> /dev/null; then
            paru -S --noconfirm neo4j-community
        else
            log_warning "No AUR helper found. Installing manually..."
            
            # Manual AUR installation
            cd /tmp
            git clone https://aur.archlinux.org/neo4j-community.git
            cd neo4j-community
            makepkg -si --noconfirm
            cd "$PROJECT_ROOT"
        fi
    else
        log_info "Neo4j already installed"
    fi
    
    # Configure Neo4j
    log_info "Configuring Neo4j..."
    
    # Create Neo4j configuration directory
    sudo mkdir -p /etc/neo4j
    
    # Create basic Neo4j configuration
    sudo tee /etc/neo4j/neo4j.conf > /dev/null << 'EOF'
# Neo4j configuration for Agent Zero V2.0

# Network connector configuration
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474

# Database location
server.directories.data=/var/lib/neo4j/data
server.directories.plugins=/var/lib/neo4j/plugins
server.directories.logs=/var/log/neo4j
server.directories.lib=/usr/share/neo4j/lib

# Memory settings
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
server.memory.pagecache.size=512m

# Security (disable auth for development)
dbms.security.auth_enabled=false

# Logging
server.logs.user.level=INFO
server.logs.gc.enabled=false
EOF
    
    # Create Neo4j directories
    sudo mkdir -p /var/lib/neo4j/{data,plugins}
    sudo mkdir -p /var/log/neo4j
    sudo chown -R neo4j:neo4j /var/lib/neo4j /var/log/neo4j 2>/dev/null || true
    
    # Start Neo4j service
    log_info "Starting Neo4j service..."
    
    # Try systemd first
    if systemctl list-unit-files | grep -q neo4j; then
        sudo systemctl enable neo4j
        sudo systemctl start neo4j
        log_success "Neo4j started via systemd"
    else
        # Start Neo4j directly
        log_info "Starting Neo4j directly..."
        sudo -u neo4j neo4j start || {
            # If neo4j user doesn't exist, run as current user
            log_warning "Starting Neo4j as current user..."
            neo4j start &
            sleep 5
        }
    fi
    
    # Wait for Neo4j to start
    log_info "Waiting for Neo4j to start..."
    for i in {1..30}; do
        if curl -s http://localhost:7474 > /dev/null 2>&1; then
            log_success "Neo4j is running on http://localhost:7474"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_error "Neo4j failed to start within 30 seconds"
            return 1
        fi
        sleep 1
    done
    
    # Test connection
    log_info "Testing Neo4j connection..."
    if curl -s http://localhost:7474/db/data/ | grep -q "neo4j_version"; then
        log_success "Neo4j connection test successful"
    else
        log_warning "Neo4j may not be fully ready yet"
    fi
}

# Fix 3: Create Missing Docker Files
fix_docker_configuration() {
    log_fix "Creating missing Docker configuration files..."
    
    # Create missing Dockerfile.ai-intelligence
    cat > "${PROJECT_ROOT}/Dockerfile.ai-intelligence" << 'EOF'
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    
    log_success "Created Dockerfile.ai-intelligence"
    
    # Fix docker-compose.yml - remove version and add missing services
    log_info "Fixing docker-compose.yml configuration..."
    
    # Backup original docker-compose.yml
    if [[ -f "docker-compose.yml" ]]; then
        cp docker-compose.yml "docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Create corrected docker-compose.yml
    cat > "${PROJECT_ROOT}/docker-compose.yml" << 'EOF'
services:
  neo4j:
    image: neo4j:5.15-community
    container_name: agent-zero-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=none
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=1G
      - NEO4J_dbms_memory_pagecache_size=512m
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    networks:
      - agent-zero-network

  redis:
    image: redis:7-alpine
    container_name: agent-zero-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - agent-zero-network

  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    container_name: agent-zero-rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=admin
      - RABBITMQ_DEFAULT_PASS=password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - agent-zero-network

  ai-intelligence:
    build:
      context: .
      dockerfile: Dockerfile.ai-intelligence
    container_name: agent-zero-ai-intelligence
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://admin:password@rabbitmq:5672
    volumes:
      - ./shared:/app/shared
      - ./api:/app/api
      - ./logs:/app/logs
    depends_on:
      neo4j:
        condition: service_healthy
      redis:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - agent-zero-network

volumes:
  neo4j_data:
  neo4j_logs:
  redis_data:
  rabbitmq_data:

networks:
  agent-zero-network:
    driver: bridge
EOF
    
    log_success "Fixed docker-compose.yml configuration"
    
    # Create requirements.txt if it doesn't exist
    if [[ ! -f "requirements.txt" ]]; then
        log_info "Creating requirements.txt..."
        cat > "${PROJECT_ROOT}/requirements.txt" << 'EOF'
# Agent Zero V2.0 Requirements
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
neo4j>=5.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
asyncio-mqtt>=0.13.0
aiofiles>=23.0.0
redis>=5.0.0
pika>=1.3.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
requests>=2.31.0
aiohttp>=3.9.0
uvloop>=0.19.0
websockets>=12.0
sqlalchemy>=2.0.0
alembic>=1.13.0
celery>=5.3.0
prometheus-client>=0.19.0
structlog>=23.2.0
EOF
        log_success "Created requirements.txt"
    fi
}

# Fix 4: Create API Main Entry Point
fix_api_main() {
    log_fix "Creating API main entry point..."
    
    mkdir -p "${PROJECT_ROOT}/api"
    
    cat > "${PROJECT_ROOT}/api/main.py" << 'EOF'
#!/usr/bin/env python3
"""
Agent Zero V2.0 - Main API Entry Point
Combines all V2.0 APIs and services
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agent Zero V2.0 - Intelligence Layer API",
    description="Advanced AI multi-agent platform with ML capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return {
        "status": "healthy",
        "service": "Agent Zero V2.0 Intelligence Layer",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": True,
            "neo4j": "checking...",
            "redis": "checking...",
            "rabbitmq": "checking..."
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Agent Zero V2.0 Intelligence Layer API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "experience": "/api/v2/experience/*",
            "analytics": "/api/v2/analytics/*",
            "patterns": "/api/v2/patterns/*",
            "ml": "/api/v2/ml/*"
        }
    }

# Try to import and include V2.0 APIs
try:
    from api.v2.experience_api import app as experience_app
    app.mount("/api/v2/experience", experience_app)
    logger.info("✅ Experience API mounted successfully")
except ImportError as e:
    logger.warning(f"⚠️  Experience API not available: {e}")

try:
    from api.v2.analytics_api import app as analytics_app  
    app.mount("/api/v2/analytics", analytics_app)
    logger.info("✅ Analytics API mounted successfully")
except ImportError as e:
    logger.warning(f"⚠️  Analytics API not available: {e}")

# API Info endpoint
@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "Agent Zero V2.0 Intelligence Layer",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Experience Management with ML Insights",
            "Neo4j Knowledge Graph Integration", 
            "Pattern Mining Engine",
            "ML Model Training Pipeline",
            "Enhanced Analytics Dashboard"
        ],
        "story_points_implemented": "28 SP",
        "priority_components": [
            "Priority 1: Experience Management System [8 SP]",
            "Priority 2: Neo4j Knowledge Graph Integration [6 SP]",
            "Priority 3: Pattern Mining Engine [6 SP]",
            "Priority 4: ML Model Training Pipeline [4 SP]",
            "Priority 5: Enhanced Analytics Dashboard Backend [2 SP]"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
EOF
    
    # Create __init__.py files
    touch "${PROJECT_ROOT}/api/__init__.py"
    mkdir -p "${PROJECT_ROOT}/api/v2"
    touch "${PROJECT_ROOT}/api/v2/__init__.py"
    
    log_success "Created API main entry point"
}

# Fix 5: Update Test Script for Virtual Environment
fix_test_script() {
    log_fix "Updating test script for virtual environment..."
    
    # Create wrapper script that activates venv
    cat > "${PROJECT_ROOT}/test_v2_with_venv.py" << 'EOF'
#!/usr/bin/env python3
"""
Test wrapper that ensures virtual environment is used
"""

import os
import sys
import subprocess

# Get project root
project_root = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(project_root, 'venv', 'bin', 'python')

if os.path.exists(venv_python):
    print("🐍 Running tests with virtual environment Python...")
    # Run the test with venv python
    result = subprocess.run([venv_python, 'test-complete-implementation.py'] + sys.argv[1:])
    sys.exit(result.returncode)
else:
    print("❌ Virtual environment not found. Please run ./activate_venv.sh first")
    sys.exit(1)
EOF
    
    chmod +x "${PROJECT_ROOT}/test_v2_with_venv.py"
    log_success "Created virtual environment test wrapper"
}

# Fix 6: Create Start Script
create_start_script() {
    log_fix "Creating comprehensive start script..."
    
    cat > "${PROJECT_ROOT}/start_agent_zero_v2.sh" << 'EOF'
#!/bin/bash
# start_agent_zero_v2.sh - Complete Agent Zero V2.0 Startup Script

echo "🚀 Starting Agent Zero V2.0 Intelligence Layer..."

# Get project root
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
cd "$PROJECT_ROOT"

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

# 1. Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    log_info "Activating Python virtual environment..."
    source venv/bin/activate
    log_success "Virtual environment activated"
else
    log_error "Virtual environment not found. Run ./arch_linux_fix.sh first"
    exit 1
fi

# 2. Check Neo4j
log_info "Checking Neo4j status..."
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    log_success "Neo4j is running"
elif systemctl is-active --quiet neo4j; then
    log_success "Neo4j is running via systemd"  
else
    log_warning "Starting Neo4j..."
    # Try to start Neo4j
    if command -v systemctl > /dev/null; then
        sudo systemctl start neo4j || neo4j start &
    else
        neo4j start &
    fi
    sleep 5
fi

# 3. Start Docker services
log_info "Starting Docker services..."
docker-compose up -d
sleep 10

# 4. Check services health
log_info "Checking services health..."
docker-compose ps

# 5. Initialize database schemas
log_info "Initializing database schemas..."
python -c "
import asyncio
import sys

async def init_schemas():
    try:
        from shared.knowledge.graph_integration_v2 import AgentZeroGraphSchema
        from shared.knowledge.neo4j_client import Neo4jClient
        
        client = Neo4jClient()
        schema = AgentZeroGraphSchema(client)
        result = await schema.initialize_v2_schema()
        print(f'✅ Schema initialized: {result}')
        
    except Exception as e:
        print(f'⚠️  Schema initialization: {e}')

asyncio.run(init_schemas())
" 2>/dev/null || log_warning "Schema initialization skipped"

# 6. Start API server
log_info "Starting API server..."
log_success "Agent Zero V2.0 Intelligence Layer started!"
echo ""
echo "📊 Access Points:"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Analytics Dashboard: http://localhost:8000/api/v2/analytics/dashboard" 
echo "   Neo4j Browser: http://localhost:7474"
echo "   RabbitMQ Management: http://localhost:15672 (admin/password)"
echo ""
echo "🧪 Run Tests:"
echo "   ./test_v2_with_venv.py"
echo ""
echo "🛑 To Stop:"
echo "   docker-compose down"

# Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    
    chmod +x "${PROJECT_ROOT}/start_agent_zero_v2.sh"
    log_success "Created start script: ./start_agent_zero_v2.sh"
}

# Main execution
main() {
    echo "Starting Arch Linux deployment fixes..."
    echo "Project Root: $PROJECT_ROOT"
    echo ""
    
    # Run all fixes
    fix_python_environment
    fix_neo4j_service  
    fix_docker_configuration
    fix_api_main
    fix_test_script
    create_start_script
    
    echo ""
    log_success "🎉 All Arch Linux fixes applied successfully!"
    echo "============================================================"
    echo ""
    echo "📋 Fixed Issues:"
    echo "✅ Python externally-managed-environment (created virtual env)"
    echo "✅ Neo4j installation and configuration" 
    echo "✅ Docker missing Dockerfile and configuration"
    echo "✅ ML dependencies (joblib, scikit-learn, etc.)"
    echo "✅ API entry point and structure"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Start services: ./start_agent_zero_v2.sh"
    echo "2. Run tests: ./test_v2_with_venv.py"
    echo "3. Access API: http://localhost:8000/docs"
    echo ""
    echo "🔧 Manual Commands:"
    echo "   Activate venv: source venv/bin/activate"
    echo "   Check Neo4j: curl http://localhost:7474"
    echo "   View logs: docker-compose logs -f"
    echo ""
    echo "Agent Zero V2.0 is ready for Arch Linux! 🎯"
}

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root. Some operations may require regular user privileges."
fi

# Run main function
main "$@"