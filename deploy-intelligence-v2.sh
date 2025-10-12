# deploy-intelligence-v2.sh
#!/bin/bash

"""
Agent Zero V1 - Intelligence V2.0 Deployment Script
Production-ready deployment with full backward compatibility

CRITICAL: Non-disruptive deployment - preserves all existing functionality
"""

set -e  # Exit on any error

echo "🚀 Agent Zero V1 - Intelligence V2.0 Deployment"
echo "=================================================="
echo "📅 Timestamp: $(date)"
echo "🔧 Mode: Production deployment with backward compatibility"
echo "=================================================="

# Configuration
PROJECT_ROOT=$(pwd)
BACKUP_DIR="backups/intelligence-v2-$(date +%Y%m%d_%H%M%S)"
VENV_PATH="${PROJECT_ROOT}/venv"
DOCKER_COMPOSE_FILE="docker-compose.intelligence-v2.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# === STEP 1: ENVIRONMENT VALIDATION ===

print_step "1/10 Validating deployment environment..."

# Check if we're in the right directory
if [[ ! -f "simple-tracker.py" ]] || [[ ! -f "docker-compose.yml" ]]; then
    print_error "Not in Agent Zero V1 project root. Please run from project directory."
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

print_status "Environment validation completed ✅"

# === STEP 2: BACKUP EXISTING SYSTEM ===

print_step "2/10 Creating backup of existing system..."

mkdir -p "${BACKUP_DIR}"

# Backup existing files
cp docker-compose.yml "${BACKUP_DIR}/" 2>/dev/null || true
cp -r services "${BACKUP_DIR}/" 2>/dev/null || true
cp -r shared "${BACKUP_DIR}/" 2>/dev/null || true
cp -r src "${BACKUP_DIR}/" 2>/dev/null || true
cp simple-tracker.py "${BACKUP_DIR}/" 2>/dev/null || true

print_status "Backup created in ${BACKUP_DIR} ✅"

# === STEP 3: VALIDATE EXISTING SERVICES ===

print_step "3/10 Validating existing services..."

# Check if Point 3 service is running
if curl -f http://localhost:8003/health &>/dev/null; then
    print_warning "Point 3 service detected on port 8003. Will run in parallel."
    POINT3_RUNNING=true
else
    print_status "Port 8003 available for compatibility service."
    POINT3_RUNNING=false
fi

# Check existing API service
if curl -f http://localhost:8000/health &>/dev/null; then
    print_warning "Existing API service on port 8000. Will enhance with V2.0 features."
    API_RUNNING=true
else
    print_status "Port 8000 available for enhanced API service."
    API_RUNNING=false
fi

print_status "Service validation completed ✅"

# === STEP 4: INSTALL DEPENDENCIES ===

print_step "4/10 Installing Intelligence V2.0 dependencies..."

# Create virtual environment if it doesn't exist
if [[ ! -d "${VENV_PATH}" ]]; then
    print_status "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}"
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Install requirements
if [[ -f "requirements.intelligence-v2.txt" ]]; then
    print_status "Installing Intelligence V2.0 requirements..."
    pip install -r requirements.intelligence-v2.txt
else
    print_warning "requirements.intelligence-v2.txt not found. Installing minimal requirements..."
    pip install fastapi uvicorn pydantic aiohttp
fi

print_status "Dependencies installed ✅"

# === STEP 5: CREATE INTELLIGENCE V2.0 PACKAGE STRUCTURE ===

print_step "5/10 Creating Intelligence V2.0 package structure..."

# Create directories
mkdir -p intelligence_v2/tests
mkdir -p api/v2
mkdir -p logs
mkdir -p data

# Create __init__.py files if they don't exist
touch intelligence_v2/__init__.py
touch api/__init__.py
touch api/v2/__init__.py

print_status "Package structure created ✅"

# === STEP 6: VALIDATE INTELLIGENCE V2.0 FILES ===

print_step "6/10 Validating Intelligence V2.0 files..."

required_files=(
    "intelligence_v2/__init__.py"
    "intelligence_v2/interfaces.py" 
    "intelligence_v2/prioritization.py"
    "api/v2/intelligence.py"
    "intelligence-v2-main.py"
    "docker-compose.intelligence-v2.yml"
    "Dockerfile.intelligence-v2"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    print_error "Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    print_error "Please ensure all Intelligence V2.0 files are present."
    exit 1
fi

print_status "All required files present ✅"

# === STEP 7: TEST INTELLIGENCE V2.0 COMPONENTS ===

print_step "7/10 Testing Intelligence V2.0 components..."

# Test Python imports
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from intelligence_v2.interfaces import Task, TaskPriority
    from intelligence_v2.prioritization import DynamicTaskPrioritizer
    print('✅ Intelligence V2.0 imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [[ $? -ne 0 ]]; then
    print_error "Intelligence V2.0 component validation failed"
    exit 1
fi

# Test FastAPI application
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from fastapi import FastAPI
    from intelligence_v2.interfaces import Task
    print('✅ FastAPI integration test successful')
except Exception as e:
    print(f'❌ FastAPI test error: {e}')
    sys.exit(1)
"

print_status "Component testing completed ✅"

# === STEP 8: DOCKER BUILD AND PREPARATION ===

print_step "8/10 Building Docker images..."

# Build Intelligence V2.0 Docker image
if [[ -f "Dockerfile.intelligence-v2" ]]; then
    print_status "Building Intelligence V2.0 Docker image..."
    docker build -f Dockerfile.intelligence-v2 -t agent-zero-intelligence-v2:latest .
    
    if [[ $? -eq 0 ]]; then
        print_status "Docker image built successfully ✅"
    else
        print_error "Docker image build failed"
        exit 1
    fi
else
    print_warning "Dockerfile.intelligence-v2 not found. Skipping Docker build."
fi

# === STEP 9: DEPLOYMENT ===

print_step "9/10 Deploying Intelligence V2.0 system..."

# Stop existing services gracefully (if requested)
read -p "Stop existing services before deployment? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Stopping existing services..."
    docker-compose down --remove-orphans 2>/dev/null || true
fi

# Deploy with enhanced Docker Compose
if [[ -f "${DOCKER_COMPOSE_FILE}" ]]; then
    print_status "Deploying Intelligence V2.0 with Docker Compose..."
    
    # Use the new docker-compose file
    cp "${DOCKER_COMPOSE_FILE}" docker-compose.yml
    
    # Start services
    docker-compose up -d
    
    if [[ $? -eq 0 ]]; then
        print_status "Docker services started ✅"
    else
        print_error "Docker deployment failed"
        
        # Restore backup
        print_status "Restoring backup..."
        cp "${BACKUP_DIR}/docker-compose.yml" . 2>/dev/null || true
        exit 1
    fi
else
    print_warning "Enhanced Docker Compose file not found. Manual deployment required."
fi

# === STEP 10: VALIDATION AND HEALTH CHECKS ===

print_step "10/10 Validating deployment..."

print_status "Waiting for services to start..."
sleep 30

# Health check endpoints
endpoints_to_check=(
    "http://localhost:8000/health:Enhanced API Service"
    "http://localhost:8012/health:Intelligence V2.0 Service"
    "http://localhost:8012/api/v2/intelligence/health:Intelligence V2.0 API"
)

successful_endpoints=0
total_endpoints=${#endpoints_to_check[@]}

for endpoint_info in "${endpoints_to_check[@]}"; do
    IFS=':' read -r endpoint description <<< "$endpoint_info"
    
    if curl -f "$endpoint" &>/dev/null; then
        print_status "$description: ✅ Healthy"
        ((successful_endpoints++))
    else
        print_warning "$description: ❌ Not responding"
    fi
done

# === DEPLOYMENT SUMMARY ===

echo ""
echo "=================================================="
echo "🎉 Agent Zero V1 - Intelligence V2.0 Deployment Summary"
echo "=================================================="
echo "📅 Completed: $(date)"
echo "✅ Successful endpoints: $successful_endpoints/$total_endpoints"
echo ""

if [[ $successful_endpoints -eq $total_endpoints ]]; then
    echo -e "${GREEN}🎯 DEPLOYMENT SUCCESSFUL${NC}"
    echo ""
    echo "🌐 Available Services:"
    echo "  - Enhanced API Service: http://localhost:8000"
    echo "  - Intelligence V2.0: http://localhost:8012"
    echo "  - API Documentation: http://localhost:8012/docs"
    echo "  - Health Check: http://localhost:8012/health"
    echo "  - Intelligence API: http://localhost:8012/api/v2/intelligence/"
    echo ""
    echo "🔄 Compatibility:"
    echo "  - Point 3 Legacy: Preserved (can run on port 8003)"
    echo "  - Existing APIs: Enhanced with V2.0 features"
    echo "  - Data Migration: Not required"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Test new Intelligence V2.0 endpoints"
    echo "  2. Verify existing Point 3 compatibility" 
    echo "  3. Monitor system performance"
    echo "  4. Plan gradual migration to V2.0 features"
    
elif [[ $successful_endpoints -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  PARTIAL DEPLOYMENT${NC}"
    echo "Some services are running, but not all endpoints are healthy."
    echo "Check logs: docker-compose logs"
    
else
    echo -e "${RED}❌ DEPLOYMENT FAILED${NC}"
    echo "No services are responding. Rolling back..."
    
    # Restore backup
    cp "${BACKUP_DIR}/docker-compose.yml" . 2>/dev/null
    docker-compose up -d 2>/dev/null
    
    echo "System restored to previous state."
    echo "Check deployment logs for issues."
    exit 1
fi

echo "=================================================="

# Deactivate virtual environment
deactivate 2>/dev/null || true

exit 0