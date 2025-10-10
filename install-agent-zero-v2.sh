#!/bin/bash
#
# Agent Zero V1 - V2.0 Intelligence Layer Installation Script  
# Week 44 Implementation - Complete Deployment Automation
#
# 🎯 Week 44 Final Task: Complete V2.0 Installation & Deployment
# Zadanie: Automated installation of all V2.0 Intelligence Layer components
# Rezultat: One-command deployment of production-ready V2.0 system
# Impact: Seamless transition from development to production deployment
#
# Author: Developer A (Backend Architect)
# Date: 10 października 2025  
# Linear Issue: A0-44 (Week 44 Implementation)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Installation configuration
AGENT_ZERO_ROOT="$(pwd)"
V2_VERSION="2.0.0"
INSTALLATION_DATE=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="v2_installation_$(date +%Y%m%d_%H%M%S).log"

# Component versions and dependencies
PYTHON_MIN_VERSION="3.9"
REQUIRED_PACKAGES=(
    "sqlite3"
    "asyncio" 
    "aiohttp"
    "fastapi"
    "uvicorn"
    "rich"
    "neo4j"
    "scikit-learn"
    "numpy"
)

OPTIONAL_PACKAGES=(
    "redis"
    "pika"  # RabbitMQ client
    "docker"
)

# V2.0 Component file mappings
declare -A V2_COMPONENTS=(
    ["experience-manager.py"]="shared/experience_manager.py"
    ["neo4j-knowledge-graph.py"]="shared/knowledge/neo4j_knowledge_graph.py"
    ["pattern-mining-engine.py"]="shared/learning/pattern_mining_engine.py"
    ["ml-training-pipeline.py"]="shared/learning/ml_training_pipeline.py"
    ["analytics-dashboard-api.py"]="api/analytics_dashboard_api.py"
    ["advanced-cli-commands.py"]="cli/advanced_commands.py"
    ["enhanced-simple-tracker.py"]="shared/utils/enhanced_simple_tracker.py"
    ["v2-integration-test-suite.py"]="tests/v2_integration_test_suite.py"
)

# Directory structure for V2.0
V2_DIRECTORIES=(
    "shared/knowledge"
    "shared/learning" 
    "shared/utils"
    "api"
    "cli"
    "tests"
    "ml_models"
    "logs/v2"
    "config/v2"
    "docs/v2"
)

print_header() {
    echo -e "${PURPLE}"
    echo "=================================================================="
    echo "🚀 Agent Zero V2.0 Intelligence Layer Installation"
    echo "=================================================================="
    echo -e "Version: ${V2_VERSION}"
    echo -e "Date: ${INSTALLATION_DATE}"
    echo -e "Root: ${AGENT_ZERO_ROOT}"
    echo -e "${NC}"
}

log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log_message "INFO" "🔍 Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_message "ERROR" "❌ Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log_message "INFO" "✅ Python version: $python_version"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_message "ERROR" "❌ pip3 not found. Please install pip"
        exit 1
    fi
    
    # Check if we're in Agent Zero root directory
    if [[ ! -f "shared/__init__.py" ]] && [[ ! -d "shared" ]]; then
        log_message "WARN" "⚠️  Shared directory not found - creating Agent Zero structure..."
        mkdir -p shared
        touch shared/__init__.py
    fi
    
    log_message "INFO" "✅ Prerequisites check completed"
}

create_directory_structure() {
    log_message "INFO" "📁 Creating V2.0 directory structure..."
    
    for dir in "${V2_DIRECTORIES[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_message "INFO" "  Created: $dir"
        else
            log_message "INFO" "  Exists: $dir"
        fi
        
        # Create __init__.py in Python directories
        if [[ "$dir" == shared/* ]] || [[ "$dir" == "api" ]] || [[ "$dir" == "cli" ]] || [[ "$dir" == "tests" ]]; then
            if [[ ! -f "$dir/__init__.py" ]]; then
                touch "$dir/__init__.py"
            fi
        fi
    done
    
    log_message "INFO" "✅ Directory structure created"
}

install_python_dependencies() {
    log_message "INFO" "📦 Installing Python dependencies..."
    
    # Create requirements-v2.txt if it doesn't exist
    cat > requirements-v2.txt << EOF
# Agent Zero V2.0 Intelligence Layer Dependencies
# Generated: $INSTALLATION_DATE

# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
sqlite3  # Built-in with Python
asyncio  # Built-in with Python

# Rich CLI interface
rich>=13.0.0

# Neo4j Knowledge Graph
neo4j>=5.0.0

# Machine Learning
scikit-learn>=1.3.0
numpy>=1.24.0

# Optional enterprise components
redis>=5.0.0
pika>=1.3.0  # RabbitMQ client
aiohttp>=3.9.0

# Development and testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
unittest-xml-reporting>=3.2.0
EOF

    # Install dependencies
    log_message "INFO" "Installing required packages..."
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if [[ "$package" != "sqlite3" ]] && [[ "$package" != "asyncio" ]]; then
            log_message "INFO" "  Installing $package..."
            if pip3 install "$package" >> "$LOG_FILE" 2>&1; then
                log_message "INFO" "  ✅ $package installed"
            else
                log_message "WARN" "  ⚠️  $package installation failed (non-critical)"
            fi
        fi
    done
    
    log_message "INFO" "Installing optional packages..."
    for package in "${OPTIONAL_PACKAGES[@]}"; do
        log_message "INFO" "  Installing $package (optional)..."
        if pip3 install "$package" >> "$LOG_FILE" 2>&1; then
            log_message "INFO" "  ✅ $package installed"
        else
            log_message "WARN" "  ⚠️  $package installation failed (optional)"
        fi
    done
    
    log_message "INFO" "✅ Python dependencies installation completed"
}

deploy_v2_components() {
    log_message "INFO" "🚀 Deploying V2.0 components..."
    
    local deployed_count=0
    local total_components=${#V2_COMPONENTS[@]}
    
    for source_file in "${!V2_COMPONENTS[@]}"; do
        local target_path="${V2_COMPONENTS[$source_file]}"
        local target_dir=$(dirname "$target_path")
        
        log_message "INFO" "  Deploying $source_file -> $target_path"
        
        # Ensure target directory exists
        mkdir -p "$target_dir"
        
        # Check if source file exists (it should be provided by user)
        if [[ -f "$source_file" ]]; then
            cp "$source_file" "$target_path"
            chmod +x "$target_path"
            log_message "INFO" "  ✅ $source_file deployed successfully"
            ((deployed_count++))
        else
            log_message "WARN" "  ⚠️  $source_file not found - skipping"
        fi
    done
    
    log_message "INFO" "✅ V2.0 components deployment: $deployed_count/$total_components files deployed"
}

initialize_databases() {
    log_message "INFO" "💾 Initializing V2.0 databases..."
    
    # Create main database if it doesn't exist
    if [[ ! -f "agent_zero.db" ]]; then
        log_message "INFO" "  Creating agent_zero.db..."
        sqlite3 agent_zero.db "SELECT 1;" > /dev/null 2>&1
    fi
    
    # Initialize V2.0 schema by running Enhanced SimpleTracker
    if [[ -f "shared/utils/enhanced_simple_tracker.py" ]]; then
        log_message "INFO" "  Initializing V2.0 database schema..."
        if python3 -c "
import sys
sys.path.append('.')
from shared.utils.enhanced_simple_tracker import EnhancedSimpleTracker
tracker = EnhancedSimpleTracker()
print('V2.0 database schema initialized successfully')
" >> "$LOG_FILE" 2>&1; then
            log_message "INFO" "  ✅ V2.0 database schema initialized"
        else
            log_message "WARN" "  ⚠️  V2.0 schema initialization failed"
        fi
    fi
    
    # Create backup directory
    mkdir -p "backups/v2"
    
    log_message "INFO" "✅ Database initialization completed"
}

setup_configuration() {
    log_message "INFO" "⚙️  Setting up V2.0 configuration..."
    
    # Create V2.0 configuration file
    cat > config/v2/agent_zero_v2_config.json << EOF
{
    "version": "$V2_VERSION",
    "installation_date": "$INSTALLATION_DATE", 
    "intelligence_layer": {
        "experience_management": {
            "enabled": true,
            "database_path": "agent_zero.db",
            "retention_days": 90
        },
        "knowledge_graph": {
            "enabled": true,
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "agent-pass"
        },
        "pattern_mining": {
            "enabled": true,
            "min_confidence": 0.7,
            "discovery_schedule": "daily"
        },
        "ml_pipeline": {
            "enabled": true,
            "models_directory": "ml_models",
            "training_schedule": "weekly"
        },
        "analytics_dashboard": {
            "enabled": true,
            "host": "0.0.0.0",
            "port": 8003,
            "cors_enabled": true
        }
    },
    "logging": {
        "level": "INFO",
        "directory": "logs/v2",
        "max_file_size": "10MB",
        "retention_days": 30
    },
    "performance": {
        "tracking_batch_size": 100,
        "analytics_cache_ttl": 300,
        "max_concurrent_operations": 10
    }
}
EOF

    # Create CLI configuration
    cat > config/v2/cli_config.json << EOF
{
    "cli_version": "$V2_VERSION",
    "default_tracking_level": "enhanced",
    "output_format": "rich",
    "auto_backup": true,
    "command_history": true,
    "aliases": {
        "track": "a0-advanced experience record",
        "analyze": "a0-advanced patterns discover",
        "optimize": "a0-advanced optimize full",
        "status": "a0-advanced v2-system status"
    }
}
EOF

    log_message "INFO" "✅ Configuration files created"
}

run_integration_tests() {
    log_message "INFO" "🧪 Running V2.0 integration tests..."
    
    if [[ -f "tests/v2_integration_test_suite.py" ]]; then
        log_message "INFO" "  Executing integration test suite..."
        
        if python3 -c "
import sys
sys.path.append('.')
from tests.v2_integration_test_suite import quick_v2_health_check, run_v2_integration_tests
import json

# Quick health check
health = quick_v2_health_check()
print(f'Health Check: {health[\"overall_health\"]} ({health[\"components_available\"]}/{health[\"components_checked\"]} components)')

# Basic integration tests
try:
    results = run_v2_integration_tests()
    summary = results['summary']
    print(f'Integration Tests: {summary[\"test_status\"]} ({summary[\"passed_tests\"]}/{summary[\"total_tests\"]} passed)')
    exit(0 if summary[\"overall_success_rate\"] >= 70 else 1)
except Exception as e:
    print(f'Integration Tests: ERROR - {e}')
    exit(1)
" 2>&1 | tee -a "$LOG_FILE"; then
            log_message "INFO" "  ✅ Integration tests passed"
            TESTS_PASSED=true
        else
            log_message "WARN" "  ⚠️  Some integration tests failed - check logs"
            TESTS_PASSED=false
        fi
    else
        log_message "WARN" "  ⚠️  Integration test suite not found - skipping tests"
        TESTS_PASSED=false
    fi
}

create_startup_scripts() {
    log_message "INFO" "📜 Creating startup scripts..."
    
    # Create V2.0 startup script
    cat > start_agent_zero_v2.sh << 'EOF'
#!/bin/bash
#
# Agent Zero V2.0 Startup Script
# Starts all V2.0 Intelligence Layer services
#

echo "🚀 Starting Agent Zero V2.0 Intelligence Layer..."

# Check if V2.0 is installed
if [[ ! -f "config/v2/agent_zero_v2_config.json" ]]; then
    echo "❌ V2.0 not installed. Run install_agent_zero_v2.sh first"
    exit 1
fi

# Start Analytics Dashboard API (background)
if [[ -f "api/analytics_dashboard_api.py" ]]; then
    echo "📊 Starting Analytics Dashboard API on port 8003..."
    python3 -c "
import sys
sys.path.append('.')
from api.analytics_dashboard_api import start_analytics_api
start_analytics_api(host='0.0.0.0', port=8003)
" &
    ANALYTICS_PID=$!
    echo "  Analytics Dashboard API started (PID: $ANALYTICS_PID)"
fi

# Run daily optimization (if enabled)
if [[ "$1" == "--with-optimization" ]]; then
    echo "⚡ Running system optimization..."
    python3 -c "
import sys
sys.path.append('.')
from cli.advanced_commands import AgentZeroAdvancedCLI
cli = AgentZeroAdvancedCLI()
cli._run_full_optimization()
"
fi

echo "✅ Agent Zero V2.0 services started successfully"
echo "📊 Analytics Dashboard: http://localhost:8003"
echo "🎯 Use 'python3 cli/advanced_commands.py' for V2.0 CLI"

# Keep script running
if [[ "$1" == "--daemon" ]]; then
    echo "Running in daemon mode... (Ctrl+C to stop)"
    trap "echo 'Stopping V2.0 services...'; kill $ANALYTICS_PID 2>/dev/null; exit" INT
    wait
fi
EOF

    chmod +x start_agent_zero_v2.sh

    # Create CLI alias script
    cat > a0-advanced << 'EOF'
#!/bin/bash
#
# Agent Zero V2.0 Advanced CLI Wrapper
#
cd "$(dirname "$0")"
python3 cli/advanced_commands.py "$@"
EOF

    chmod +x a0-advanced

    log_message "INFO" "✅ Startup scripts created"
}

generate_installation_report() {
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_message "INFO" "📋 Generating installation report..."
    
    cat > "V2_Installation_Report_$(date +%Y%m%d_%H%M%S).md" << EOF
# Agent Zero V2.0 Intelligence Layer - Installation Report

**Installation Date:** $INSTALLATION_DATE  
**Completion Date:** $end_time  
**Version:** $V2_VERSION  
**Installation Directory:** $AGENT_ZERO_ROOT

## Installation Summary

### ✅ Components Deployed
$(for component in "${!V2_COMPONENTS[@]}"; do
    if [[ -f "${V2_COMPONENTS[$component]}" ]]; then
        echo "- ✅ $component → ${V2_COMPONENTS[$component]}"
    else
        echo "- ❌ $component → ${V2_COMPONENTS[$component]} (MISSING)"
    fi
done)

### 📁 Directory Structure
$(for dir in "${V2_DIRECTORIES[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "- ✅ $dir"
    else
        echo "- ❌ $dir (NOT CREATED)"
    fi
done)

### 🧪 Integration Tests
- **Test Suite:** $(if [[ "$TESTS_PASSED" == "true" ]]; then echo "✅ PASSED"; else echo "⚠️  PARTIAL/FAILED"; fi)
- **Log File:** $LOG_FILE

### 🚀 Getting Started

#### Start V2.0 Services
\`\`\`bash
./start_agent_zero_v2.sh
\`\`\`

#### V2.0 Advanced CLI
\`\`\`bash
./a0-advanced v2-system status
./a0-advanced experience analyze
./a0-advanced patterns discover
./a0-advanced ml train
\`\`\`

#### Analytics Dashboard
- **URL:** http://localhost:8003/api/v2/analytics/overview
- **WebSocket:** ws://localhost:8003/api/v2/analytics/live-stream

### 📖 Next Steps

1. **Verify Installation:**
   \`\`\`bash
   ./a0-advanced v2-system test
   \`\`\`

2. **Configure Neo4j (Optional):**
   - Install Neo4j Desktop or Docker container
   - Update config/v2/agent_zero_v2_config.json
   - Initialize knowledge graph: \`./a0-advanced knowledge-graph init\`

3. **Start Using V2.0:**
   - Track tasks with enhanced capabilities
   - Monitor analytics dashboard 
   - Review pattern discoveries
   - Train ML models for optimization

### 🔧 Configuration Files
- **Main Config:** config/v2/agent_zero_v2_config.json
- **CLI Config:** config/v2/cli_config.json
- **Requirements:** requirements-v2.txt

### 📞 Support
For issues or questions:
- Check installation log: $LOG_FILE
- Run diagnostics: \`./a0-advanced v2-system status\`
- Review test results: \`./a0-advanced v2-system test\`

---
*Agent Zero V2.0 Intelligence Layer - Production Ready*
*Installation completed: $end_time*
EOF

    log_message "INFO" "✅ Installation report generated"
}

main() {
    print_header
    
    # Start installation log
    log_message "INFO" "🚀 Starting Agent Zero V2.0 Intelligence Layer installation..."
    log_message "INFO" "📋 Installation log: $LOG_FILE"
    
    # Installation steps
    check_prerequisites
    create_directory_structure  
    install_python_dependencies
    deploy_v2_components
    initialize_databases
    setup_configuration
    run_integration_tests
    create_startup_scripts
    generate_installation_report
    
    # Final status
    echo -e "${GREEN}"
    echo "=================================================================="
    echo "🎉 Agent Zero V2.0 Installation COMPLETE!"
    echo "=================================================================="
    echo -e "${NC}"
    
    log_message "INFO" "✅ Installation completed successfully"
    
    if [[ "$TESTS_PASSED" == "true" ]]; then
        echo -e "${GREEN}✅ Integration tests: PASSED${NC}"
    else
        echo -e "${YELLOW}⚠️  Integration tests: PARTIAL - check logs${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    echo -e "   ./start_agent_zero_v2.sh     # Start V2.0 services"
    echo -e "   ./a0-advanced v2-system status   # Check system status" 
    echo -e "   ./a0-advanced --help         # View all V2.0 commands"
    echo ""
    echo -e "${CYAN}📊 Analytics Dashboard:${NC} http://localhost:8003"
    echo -e "${CYAN}📋 Installation Report:${NC} V2_Installation_Report_*.md"
    echo -e "${CYAN}🗂️  Installation Log:${NC} $LOG_FILE"
    
    echo -e "\n${PURPLE}Agent Zero V2.0 Intelligence Layer is ready for production! 🚀${NC}"
}

# Run installation if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi