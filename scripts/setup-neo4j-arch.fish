#!/usr/bin/env fish

# Agent Zero V1 - Neo4j Service Setup for Arch Linux
# Fish shell script for automated installation and configuration
# Compatible with Arch Linux pacman package manager
# Version: 1.0.0 - Production Ready

set -g SCRIPT_NAME "Agent Zero V1 Neo4j Setup"
set -g VERSION "1.0.0"
set -g AUTHOR "Agent Zero V1 Development Team"

# Color definitions for output
set -g RED '\033[0;31m'
set -g GREEN '\033[0;32m'
set -g YELLOW '\033[1;33m'
set -g BLUE '\033[0;34m'
set -g PURPLE '\033[0;35m'
set -g CYAN '\033[0;36m'
set -g NC '\033[0m' # No Color

# Configuration variables
set -g NEO4J_VERSION_TAG "5.26-community"
set -g PROJECT_ROOT (pwd)
set -g OUTPUT_DIR "/tmp/agent_zero_output"
set -g LOG_FILE "/tmp/agent_zero_setup.log"

function print_header
    echo -e "$CYAN"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Agent Zero V1 Setup                      ║"
    echo "║              Neo4j Service Configuration                     ║"
    echo "║                 Arch Linux + Fish Shell                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "$NC"
    echo "Version: $VERSION"
    echo "Date: "(date)
    echo "Log file: $LOG_FILE"
    echo ""
end

function log_message
    set level $argv[1]
    set message $argv[2..-1]
    set timestamp (date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> $LOG_FILE
    
    switch $level
        case "INFO"
            echo -e "$GREEN[INFO]$NC $message"
        case "WARN"
            echo -e "$YELLOW[WARN]$NC $message"
        case "ERROR"
            echo -e "$RED[ERROR]$NC $message"
        case "DEBUG"
            echo -e "$BLUE[DEBUG]$NC $message"
        case '*'
            echo -e "$PURPLE[$level]$NC $message"
    end
end

function check_arch_linux
    log_message "INFO" "Checking if running on Arch Linux..."
    
    if not test -f /etc/arch-release
        log_message "ERROR" "This script is designed for Arch Linux only!"
        log_message "ERROR" "Please run on an Arch Linux system."
        exit 1
    end
    
    log_message "INFO" "✓ Arch Linux detected"
end

function check_fish_shell
    log_message "INFO" "Checking Fish shell compatibility..."
    
    if test "$SHELL" != "/usr/bin/fish"
        log_message "WARN" "Fish shell is not the default shell"
        log_message "INFO" "Current shell: $SHELL"
    else
        log_message "INFO" "✓ Fish shell is active"
    end
    
    # Check Fish version
    set fish_version (fish --version | string match -r '\d+\.\d+\.\d+')
    log_message "INFO" "Fish version: $fish_version"
end

function update_system
    log_message "INFO" "Updating Arch Linux system packages..."
    
    if sudo pacman -Syu --noconfirm >> $LOG_FILE 2>&1
        log_message "INFO" "✓ System packages updated successfully"
    else
        log_message "ERROR" "Failed to update system packages"
        exit 1
    end
end

function install_dependencies
    log_message "INFO" "Installing required dependencies..."
    
    set dependencies \
        docker \
        docker-compose \
        docker-buildx \
        python \
        python-pip \
        python-virtualenv \
        git \
        curl \
        wget \
        htop \
        net-tools \
        lsof \
        jq \
        yq
    
    log_message "INFO" "Dependencies to install: $dependencies"
    
    if sudo pacman -S --needed --noconfirm $dependencies >> $LOG_FILE 2>&1
        log_message "INFO" "✓ Dependencies installed successfully"
    else
        log_message "ERROR" "Failed to install dependencies"
        exit 1
    end
end

function setup_docker
    log_message "INFO" "Setting up Docker service..."
    
    # Enable and start Docker service
    if sudo systemctl enable docker.service >> $LOG_FILE 2>&1
        log_message "INFO" "✓ Docker service enabled"
    else
        log_message "WARN" "Docker service already enabled or failed to enable"
    end
    
    if sudo systemctl start docker.service >> $LOG_FILE 2>&1
        log_message "INFO" "✓ Docker service started"
    else
        log_message "WARN" "Docker service already running or failed to start"
    end
    
    # Add current user to docker group
    set username (whoami)
    if sudo usermod -aG docker $username >> $LOG_FILE 2>&1
        log_message "INFO" "✓ User $username added to docker group"
        log_message "WARN" "You may need to log out and back in for docker group changes to take effect"
    else
        log_message "WARN" "Failed to add user to docker group or user already in group"
    end
    
    # Check Docker status
    if systemctl is-active --quiet docker.service
        log_message "INFO" "✓ Docker service is running"
    else
        log_message "ERROR" "Docker service is not running"
        exit 1
    end
end

function install_python_dependencies
    log_message "INFO" "Installing Python dependencies for Agent Zero V1..."
    
    # Create virtual environment if it doesn't exist
    if not test -d venv
        log_message "INFO" "Creating Python virtual environment..."
        python -m venv venv >> $LOG_FILE 2>&1
    end
    
    # Activate virtual environment and install dependencies
    log_message "INFO" "Installing Python packages..."
    
    source venv/bin/activate.fish
    
    set python_deps \
        neo4j \
        fastapi \
        uvicorn \
        pydantic \
        requests \
        aiohttp \
        asyncio \
        pytest \
        pytest-asyncio \
        docker \
        redis \
        celery
    
    for dep in $python_deps
        log_message "DEBUG" "Installing $dep..."
        pip install $dep >> $LOG_FILE 2>&1
    end
    
    log_message "INFO" "✓ Python dependencies installed"
end

function create_directories
    log_message "INFO" "Creating project directories..."
    
    set directories \
        "$OUTPUT_DIR" \
        "$PROJECT_ROOT/shared/knowledge" \
        "$PROJECT_ROOT/shared/execution" \
        "$PROJECT_ROOT/configs" \
        "$PROJECT_ROOT/scripts" \
        "$PROJECT_ROOT/tests" \
        "$PROJECT_ROOT/logs" \
        "$PROJECT_ROOT/docker/neo4j"
    
    for dir in $directories
        if not test -d $dir
            mkdir -p $dir >> $LOG_FILE 2>&1
            log_message "DEBUG" "Created directory: $dir"
        end
    end
    
    log_message "INFO" "✓ Project directories created"
end

function setup_neo4j_config
    log_message "INFO" "Setting up Neo4j configuration..."
    
    # Create Neo4j environment file
    set neo4j_env "$PROJECT_ROOT/.env.neo4j"
    
    echo "# Agent Zero V1 - Neo4j Configuration" > $neo4j_env
    echo "NEO4J_URI=bolt://localhost:7687" >> $neo4j_env
    echo "NEO4J_USER=neo4j" >> $neo4j_env
    echo "NEO4J_PASSWORD=agent_zero_2024!" >> $neo4j_env
    echo "NEO4J_DATABASE=neo4j" >> $neo4j_env
    echo "NEO4J_MAX_CONNECTIONS=50" >> $neo4j_env
    echo "NEO4J_CONNECTION_TIMEOUT=30" >> $neo4j_env
    
    log_message "INFO" "✓ Neo4j environment configuration created: $neo4j_env"
    
    # Set appropriate permissions
    chmod 600 $neo4j_env
    log_message "DEBUG" "Set secure permissions for Neo4j environment file"
end

function pull_docker_images
    log_message "INFO" "Pulling required Docker images..."
    
    set images \
        "neo4j:$NEO4J_VERSION_TAG" \
        "rabbitmq:3.13-management" \
        "redis:7-alpine" \
        "python:3.11-slim"
    
    for image in $images
        log_message "DEBUG" "Pulling image: $image"
        if docker pull $image >> $LOG_FILE 2>&1
            log_message "INFO" "✓ Pulled $image"
        else
            log_message "WARN" "Failed to pull $image - will try during compose up"
        end
    end
end

function create_docker_compose
    log_message "INFO" "Validating docker-compose.yml configuration..."
    
    if test -f "$PROJECT_ROOT/docker-compose.yml"
        log_message "INFO" "✓ docker-compose.yml found"
        
        # Validate docker-compose file
        if docker-compose config >> $LOG_FILE 2>&1
            log_message "INFO" "✓ docker-compose.yml is valid"
        else
            log_message "ERROR" "docker-compose.yml configuration is invalid"
            exit 1
        end
    else
        log_message "ERROR" "docker-compose.yml not found in project root"
        log_message "ERROR" "Please ensure the docker-compose.yml file is present"
        exit 1
    end
end

function start_services
    log_message "INFO" "Starting Agent Zero V1 services..."
    
    # Start services in detached mode
    if docker-compose up -d >> $LOG_FILE 2>&1
        log_message "INFO" "✓ Services started successfully"
    else
        log_message "ERROR" "Failed to start services"
        log_message "ERROR" "Check docker-compose logs for details"
        exit 1
    end
    
    log_message "INFO" "Waiting for services to initialize..."
    sleep 10
    
    # Check service status
    set services (docker-compose ps --services)
    for service in $services
        set status (docker-compose ps -q $service | xargs docker inspect --format='{{.State.Status}}')
        if test "$status" = "running"
            log_message "INFO" "✓ $service is running"
        else
            log_message "WARN" "⚠ $service status: $status"
        end
    end
end

function test_neo4j_connection
    log_message "INFO" "Testing Neo4j connection..."
    
    sleep 5  # Additional wait for Neo4j to fully initialize
    
    # Test connection using curl
    set max_attempts 12
    set attempt 1
    
    while test $attempt -le $max_attempts
        log_message "DEBUG" "Connection attempt $attempt/$max_attempts"
        
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:7474 | grep -q "200"
            log_message "INFO" "✓ Neo4j HTTP interface is accessible"
            break
        else
            if test $attempt -eq $max_attempts
                log_message "ERROR" "Neo4j connection test failed after $max_attempts attempts"
                log_message "ERROR" "Check Neo4j container logs: docker-compose logs neo4j"
                exit 1
            end
            
            log_message "DEBUG" "Waiting for Neo4j to start... ($attempt/$max_attempts)"
            sleep 10
            set attempt (math $attempt + 1)
        end
    end
    
    # Test Bolt connection if cypher-shell is available
    if docker exec agent-zero-neo4j /var/lib/neo4j/bin/cypher-shell -u neo4j -p agent_zero_2024! "RETURN 'Connection test successful' as message" >> $LOG_FILE 2>&1
        log_message "INFO" "✓ Neo4j Bolt connection test successful"
    else
        log_message "WARN" "Neo4j Bolt connection test failed - may need manual verification"
    end
end

function create_test_script
    log_message "INFO" "Creating test script..."
    
    set test_script "$PROJECT_ROOT/scripts/test-neo4j-connection.fish"
    
    echo "#!/usr/bin/env fish" > $test_script
    echo "# Agent Zero V1 - Neo4j Connection Test" >> $test_script
    echo "" >> $test_script
    echo "echo 'Testing Neo4j connection...'" >> $test_script
    echo "curl -s http://localhost:7474 || echo 'Neo4j HTTP interface not accessible'" >> $test_script
    echo "docker exec agent-zero-neo4j /var/lib/neo4j/bin/cypher-shell -u neo4j -p agent_zero_2024! 'RETURN \"Test successful\" as result'" >> $test_script
    
    chmod +x $test_script
    log_message "INFO" "✓ Test script created: $test_script"
end

function print_summary
    echo ""
    echo -e "$GREEN"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete!                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "$NC"
    echo ""
    echo "🎉 Agent Zero V1 Neo4j setup completed successfully!"
    echo ""
    echo "📋 Summary:"
    echo "  • Neo4j Graph Database: http://localhost:7474"
    echo "  • Neo4j Bolt Protocol: bolt://localhost:7687"
    echo "  • RabbitMQ Management: http://localhost:15672"
    echo "  • Agent Zero API: http://localhost:8000"
    echo ""
    echo "🔑 Default Credentials:"
    echo "  • Neo4j: neo4j / agent_zero_2024!"
    echo "  • RabbitMQ: agent_zero / agent_zero_2024!"
    echo ""
    echo "📁 Important Files:"
    echo "  • Configuration: $PROJECT_ROOT/.env.neo4j"
    echo "  • Docker Compose: $PROJECT_ROOT/docker-compose.yml"
    echo "  • Test Script: $PROJECT_ROOT/scripts/test-neo4j-connection.fish"
    echo "  • Setup Log: $LOG_FILE"
    echo ""
    echo "🚀 Next Steps:"
    echo "  1. Run: docker-compose ps (check service status)"
    echo "  2. Run: ./scripts/test-neo4j-connection.fish (test connection)"
    echo "  3. Open Neo4j Browser: http://localhost:7474"
    echo "  4. Start developing your Agent Zero V1 application!"
    echo ""
    echo -e "$CYAN"
    echo "For troubleshooting, check the log file: $LOG_FILE"
    echo "Agent Zero V1 setup completed at "(date)
    echo -e "$NC"
end

function cleanup_on_error
    log_message "ERROR" "Setup failed! Cleaning up..."
    
    # Stop any running containers
    docker-compose down >> $LOG_FILE 2>&1
    
    log_message "ERROR" "Setup terminated. Check log file: $LOG_FILE"
    exit 1
end

# Main execution
function main
    # Trap errors and cleanup
    trap cleanup_on_error ERR
    
    print_header
    
    # Initialize log file
    echo "Agent Zero V1 Setup Log - "(date) > $LOG_FILE
    
    # Execute setup steps
    check_arch_linux
    check_fish_shell
    update_system
    install_dependencies
    setup_docker
    install_python_dependencies
    create_directories
    setup_neo4j_config
    pull_docker_images
    create_docker_compose
    start_services
    test_neo4j_connection
    create_test_script
    
    print_summary
end

# Run main function if script is executed directly
if test (basename (status filename)) = (basename (status current-filename))
    main $argv
end