#!/usr/bin/env fish
# Agent Zero V1 - Deployment Script for Arch Linux
# Automated setup and restart script optimized for Fish Shell

# Color definitions for better output
set GREEN '\033[0;32m'
set RED '\033[0;31m'
set YELLOW '\033[1;33m'
set BLUE '\033[0;34m'
set NC '\033[0m' # No Color

# Project configuration
set PROJECT_DIR "/home/ianua/projects/agent-zero-v1"
set DOCKER_COMPOSE_FILE "$PROJECT_DIR/docker-compose.yml"
set PYTHON_ENV "$PROJECT_DIR/venv"
set LOG_FILE "$PROJECT_DIR/logs/deployment.log"

function log_message
    set timestamp (date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $argv[1]" | tee -a $LOG_FILE
end

function print_header
    echo -e "$BLUE"
    echo "=================================="
    echo "🚀 Agent Zero V1 - Deployment"
    echo "=================================="
    echo -e "$NC"
end

function check_dependencies
    log_message "Checking system dependencies..."
    
    # Check if running on Arch Linux
    if not test -f /etc/arch-release
        echo -e "$RED❌ This script is optimized for Arch Linux$NC"
        exit 1
    end
    
    # Required packages for Arch Linux
    set required_packages docker docker-compose python python-pip git
    set missing_packages
    
    for package in $required_packages
        if not pacman -Qi $package >/dev/null 2>&1
            set missing_packages $missing_packages $package
        end
    end
    
    if test (count $missing_packages) -gt 0
        echo -e "$YELLOW⚠️  Installing missing packages: $missing_packages$NC"
        sudo pacman -S --noconfirm $missing_packages
    end
    
    # Check Docker service
    if not systemctl is-active docker >/dev/null 2>&1
        echo -e "$YELLOW⚠️  Starting Docker service...$NC"
        sudo systemctl start docker
        sudo systemctl enable docker
    end
    
    log_message "✅ All dependencies checked"
end

function setup_project_structure
    log_message "Setting up project structure..."
    
    # Create necessary directories
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/shared/monitoring"
    mkdir -p "$PROJECT_DIR/shared/orchestration"
    mkdir -p "$PROJECT_DIR/shared/execution"
    mkdir -p "$PROJECT_DIR/shared/knowledge"
    
    log_message "✅ Project structure ready"
end

function deploy_websocket_fix
    log_message "Deploying WebSocket Monitor fix..."
    
    # Create the fixed WebSocket monitor
    set websocket_file "$PROJECT_DIR/shared/monitoring/websocket_monitor.py"
    
    # Copy the fixed file (assumes it's in current directory)
    if test -f "websocket_monitor_fixed.py"
        cp websocket_monitor_fixed.py $websocket_file
        log_message "✅ WebSocket monitor file deployed"
    else
        log_message "⚠️  websocket_monitor_fixed.py not found in current directory"
    end
end

function deploy_task_decomposer_fix
    log_message "Deploying Task Decomposer fix..."
    
    # Create the fixed Task Decomposer
    set decomposer_file "$PROJECT_DIR/shared/orchestration/task_decomposer.py"
    
    # Copy the fixed file (assumes it's in current directory)
    if test -f "task_decomposer_fixed.py"
        cp task_decomposer_fixed.py $decomposer_file
        log_message "✅ Task Decomposer file deployed"
    else
        log_message "⚠️  task_decomposer_fixed.py not found in current directory"
    end
end

function setup_python_environment
    log_message "Setting up Python virtual environment..."
    
    cd $PROJECT_DIR
    
    # Create virtual environment if it doesn't exist
    if not test -d $PYTHON_ENV
        python -m venv $PYTHON_ENV
        log_message "✅ Python virtual environment created"
    end
    
    # Activate virtual environment (Fish shell syntax)
    source $PYTHON_ENV/bin/activate.fish
    
    # Install/upgrade required packages
    pip install --upgrade pip
    pip install fastapi uvicorn websockets neo4j python-multipart
    
    log_message "✅ Python environment ready"
end

function setup_neo4j_container
    log_message "Setting up Neo4j container..."
    
    cd $PROJECT_DIR
    
    # Create docker-compose.yml if it doesn't exist
    if not test -f $DOCKER_COMPOSE_FILE
        echo 'version: "3.8"
services:
  neo4j:
    image: neo4j:5.13-community
    container_name: agent_zero_neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/agentzerov1
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_import_file_use__neo4j__config=true
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:7474 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: agent_zero_rabbitmq
    ports:
      - "5672:5672"   # AMQP
      - "15672:15672" # Management UI
    environment:
      - RABBITMQ_DEFAULT_USER=agentzerov1
      - RABBITMQ_DEFAULT_PASS=agentzerov1
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped
    healthcheck:
      test: rabbitmq-diagnostics -q ping
      interval: 30s
      timeout: 30s
      retries: 3

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
  rabbitmq_data:' > $DOCKER_COMPOSE_FILE
        
        log_message "✅ Docker Compose configuration created"
    end
    
    # Start containers
    docker-compose up -d
    
    # Wait for Neo4j to be ready
    echo -e "$YELLOW⏳ Waiting for Neo4j to be ready...$NC"
    set max_attempts 30
    set attempt 0
    
    while test $attempt -lt $max_attempts
        set attempt (math $attempt + 1)
        if curl -s http://localhost:7474 >/dev/null 2>&1
            break
        end
        sleep 2
    end
    
    if test $attempt -eq $max_attempts
        echo -e "$RED❌ Neo4j failed to start after $max_attempts attempts$NC"
        exit 1
    end
    
    log_message "✅ Neo4j container is ready"
end

function test_system_health
    log_message "Testing system health..."
    
    # Test Neo4j connection
    if curl -s http://localhost:7474 >/dev/null
        echo -e "$GREEN✅ Neo4j HTTP endpoint (7474) - OK$NC"
    else
        echo -e "$RED❌ Neo4j HTTP endpoint - FAILED$NC"
    end
    
    # Test RabbitMQ connection
    if curl -s http://localhost:15672 >/dev/null
        echo -e "$GREEN✅ RabbitMQ Management UI (15672) - OK$NC"
    else
        echo -e "$RED❌ RabbitMQ Management UI - FAILED$NC"
    end
    
    # Test WebSocket Monitor (if running)
    if curl -s http://localhost:8000/health >/dev/null
        echo -e "$GREEN✅ WebSocket Monitor (8000) - OK$NC"
    else
        echo -e "$YELLOW⚠️  WebSocket Monitor - Not running (start with: python shared/monitoring/websocket_monitor.py)$NC"
    end
    
    log_message "✅ System health check completed"
end

function start_websocket_monitor
    log_message "Starting WebSocket Monitor..."
    
    cd $PROJECT_DIR
    source $PYTHON_ENV/bin/activate.fish
    
    # Start WebSocket monitor in background
    nohup python shared/monitoring/websocket_monitor.py > logs/websocket.log 2>&1 &
    set websocket_pid $last_pid
    
    echo $websocket_pid > logs/websocket.pid
    
    # Wait a moment and check if it started
    sleep 3
    if curl -s http://localhost:8000/health >/dev/null
        echo -e "$GREEN✅ WebSocket Monitor started successfully (PID: $websocket_pid)$NC"
        echo -e "$BLUE🌐 Dashboard available at: http://localhost:8000$NC"
    else
        echo -e "$RED❌ WebSocket Monitor failed to start$NC"
        exit 1
    end
end

function show_status
    echo -e "$BLUE"
    echo "=================================="
    echo "📊 Agent Zero V1 - System Status"
    echo "=================================="
    echo -e "$NC"
    
    # Docker containers status
    echo -e "$YELLOW📦 Docker Containers:$NC"
    docker-compose ps
    echo
    
    # Service endpoints
    echo -e "$YELLOW🌐 Service Endpoints:$NC"
    echo "• Neo4j HTTP: http://localhost:7474 (neo4j/agentzerov1)"
    echo "• Neo4j Bolt: bolt://localhost:7687"
    echo "• RabbitMQ Management: http://localhost:15672 (agentzerov1/agentzerov1)"
    echo "• WebSocket Dashboard: http://localhost:8000"
    echo
    
    # Logs
    echo -e "$YELLOW📋 Recent Logs:$NC"
    if test -f $LOG_FILE
        tail -5 $LOG_FILE
    end
end

function cleanup_old_processes
    log_message "Cleaning up old processes..."
    
    # Stop old WebSocket monitor if running
    if test -f "$PROJECT_DIR/logs/websocket.pid"
        set old_pid (cat "$PROJECT_DIR/logs/websocket.pid")
        if ps -p $old_pid >/dev/null 2>&1
            kill $old_pid
            log_message "✅ Stopped old WebSocket Monitor (PID: $old_pid)"
        end
        rm -f "$PROJECT_DIR/logs/websocket.pid"
    end
end

# Main execution function
function main
    print_header
    log_message "🚀 Starting Agent Zero V1 deployment..."
    
    check_dependencies
    setup_project_structure
    cleanup_old_processes
    deploy_websocket_fix
    deploy_task_decomposer_fix
    setup_python_environment
    setup_neo4j_container
    start_websocket_monitor
    test_system_health
    
    echo -e "$GREEN"
    echo "=================================="
    echo "🎉 DEPLOYMENT COMPLETED!"
    echo "=================================="
    echo -e "$NC"
    
    show_status
    
    echo -e "$BLUE💡 Next Steps:$NC"
    echo "1. Open WebSocket Dashboard: http://localhost:8000"
    echo "2. Check Neo4j Browser: http://localhost:7474"
    echo "3. Monitor logs: tail -f logs/deployment.log"
    echo "4. Test Task Decomposer: python shared/orchestration/task_decomposer.py"
    echo
    echo -e "$GREEN✅ Agent Zero V1 is ready for Phase 2 development!$NC"
end

# Script entry point
if test (count $argv) -eq 0
    main
else
    switch $argv[1]
        case "status"
            show_status
        case "restart"
            log_message "🔄 Restarting services..."
            cleanup_old_processes
            docker-compose restart
            start_websocket_monitor
            show_status
        case "stop"
            log_message "🛑 Stopping services..."
            cleanup_old_processes
            docker-compose down
        case "logs"
            tail -f $LOG_FILE
        case "*"
            echo "Usage: $argv[0] [status|restart|stop|logs]"
            echo "Run without arguments to perform full deployment"
    end
end