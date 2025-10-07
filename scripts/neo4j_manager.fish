#!/usr/bin/fish
#
# Neo4j Service Manager - Agent Zero V1
# Arch Linux + Fish shell startup script
# Generated: 2025-10-07
#

# Colors for output
set -g RED '\e[0;31m'
set -g GREEN '\e[0;32m'
set -g YELLOW '\e[1;33m'
set -g BLUE '\e[0;34m'
set -g NC '\e[0m' # No Color

# Project configuration
set -g PROJECT_PATH "/home/ianua/projects/agent-zero-v1"
set -g LOG_FILE "/tmp/agent-zero-neo4j.log"

function print_header
    echo -e $BLUE"========================================="$NC
    echo -e $BLUE"   Agent Zero V1 - Neo4j Service Manager"$NC
    echo -e $BLUE"   Arch Linux + Fish Shell Compatible"$NC
    echo -e $BLUE"========================================="$NC
    echo
end

function log_info
    set message $argv[1]
    echo -e $GREEN"[INFO]"$NC" $message"
    echo (date)" [INFO] $message" >> $LOG_FILE
end

function log_error
    set message $argv[1]
    echo -e $RED"[ERROR]"$NC" $message"
    echo (date)" [ERROR] $message" >> $LOG_FILE
end

function log_warning
    set message $argv[1]
    echo -e $YELLOW"[WARNING]"$NC" $message"
    echo (date)" [WARNING] $message" >> $LOG_FILE
end

function check_dependencies
    log_info "Checking system dependencies..."
    
    # Check Docker
    if not command -v docker >/dev/null 2>&1
        log_error "Docker is not installed. Install with: sudo pacman -S docker"
        return 1
    end
    
    # Check Docker Compose
    if not docker compose version >/dev/null 2>&1
        log_error "Docker Compose is not available. Install with: sudo pacman -S docker-compose"
        return 1
    end
    
    # Check if Docker service is running
    if not systemctl is-active --quiet docker
        log_warning "Docker service is not running. Starting it..."
        sudo systemctl start docker
        if test $status -ne 0
            log_error "Failed to start Docker service"
            return 1
        end
    end
    
    # Check project directory
    if not test -d $PROJECT_PATH
        log_error "Project directory not found: $PROJECT_PATH"
        log_info "Please clone the repository first:"
        log_info "git clone https://github.com/HotelAiOS/agent-zero-v1 $PROJECT_PATH"
        return 1
    end
    
    log_info "All dependencies are available"
    return 0
end

function stop_neo4j_service
    log_info "Stopping Neo4j service..."
    
    cd $PROJECT_PATH
    
    # Stop the container
    docker compose stop neo4j
    if test $status -eq 0
        log_info "Neo4j container stopped successfully"
    else
        log_warning "Neo4j container was not running or failed to stop"
    end
    
    # Remove the container
    docker compose rm -f neo4j
    if test $status -eq 0
        log_info "Neo4j container removed successfully"
    end
end

function clean_neo4j_data
    log_info "Cleaning Neo4j data volumes..."
    
    cd $PROJECT_PATH
    
    # Stop and remove volumes
    docker compose down -v
    if test $status -eq 0
        log_info "All volumes cleaned successfully"
    else
        log_warning "Some volumes might not have been cleaned"
    end
end

function start_neo4j_service
    log_info "Starting Neo4j service..."
    
    cd $PROJECT_PATH
    
    # Start Neo4j service
    docker compose up -d neo4j
    if test $status -eq 0
        log_info "Neo4j service started successfully"
        return 0
    else
        log_error "Failed to start Neo4j service"
        return 1
    end
end

function wait_for_neo4j
    log_info "Waiting for Neo4j to be ready..."
    
    set -l max_attempts 30
    set -l attempt 0
    
    while test $attempt -lt $max_attempts
        set attempt (math $attempt + 1)
        
        # Check HTTP endpoint
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:7474 | grep -q "200"
            log_info "Neo4j is ready! (attempt $attempt/$max_attempts)"
            return 0
        end
        
        echo -n "."
        sleep 2
    end
    
    echo
    log_error "Neo4j failed to become ready within timeout"
    return 1
end

function test_neo4j_connection
    log_info "Testing Neo4j connection..."
    
    # Test with curl first
    set http_status (curl -s -o /dev/null -w "%{http_code}" http://localhost:7474)
    if test "$http_status" = "200"
        log_info "✅ Neo4j HTTP endpoint is accessible"
    else
        log_error "❌ Neo4j HTTP endpoint returned: $http_status"
        return 1
    end
    
    # Test Bolt connection if cypher-shell is available
    if command -v cypher-shell >/dev/null 2>&1
        echo 'RETURN "Connection test successful" as message' | cypher-shell -u neo4j -p agent_zero_neo4j_dev >/dev/null 2>&1
        if test $status -eq 0
            log_info "✅ Neo4j Bolt connection is working"
        else
            log_error "❌ Neo4j Bolt connection failed"
            return 1
        end
    else
        log_info "ℹ️  cypher-shell not available, skipping Bolt test"
    end
    
    return 0
end

function show_neo4j_status
    echo
    log_info "Neo4j Service Status:"
    echo -e $BLUE"  • HTTP Interface:"$NC" http://localhost:7474"
    echo -e $BLUE"  • Bolt Protocol:"$NC" bolt://localhost:7687"
    echo -e $BLUE"  • Username:"$NC" neo4j"
    echo -e $BLUE"  • Password:"$NC" agent_zero_neo4j_dev"
    echo -e $BLUE"  • Container:"$NC" agent-zero-neo4j"
    echo
    
    # Show container status
    set container_status (docker inspect --format='{{.State.Status}}' agent-zero-neo4j 2>/dev/null)
    if test $status -eq 0
        echo -e $GREEN"  ✅ Container Status:"$NC" $container_status"
    else
        echo -e $RED"  ❌ Container Status:"$NC" not found"
    end
end

function restart_neo4j
    log_info "Performing complete Neo4j restart..."
    
    stop_neo4j_service
    sleep 2
    
    if start_neo4j_service
        if wait_for_neo4j
            if test_neo4j_connection
                log_info "🎉 Neo4j restart completed successfully!"
                show_neo4j_status
                return 0
            end
        end
    end
    
    log_error "Neo4j restart failed"
    return 1
end

function emergency_fix
    log_info "Performing emergency Neo4j fix..."
    
    stop_neo4j_service
    clean_neo4j_data
    sleep 3
    
    if start_neo4j_service
        if wait_for_neo4j
            if test_neo4j_connection
                log_info "🎉 Emergency fix completed successfully!"
                show_neo4j_status
                return 0
            end
        end
    end
    
    log_error "Emergency fix failed"
    return 1
end

function show_help
    echo "Usage: $argv[0] [COMMAND]"
    echo
    echo "Commands:"
    echo "  start      - Start Neo4j service"
    echo "  stop       - Stop Neo4j service"
    echo "  restart    - Restart Neo4j service"
    echo "  status     - Show Neo4j status"
    echo "  test       - Test Neo4j connection"
    echo "  fix        - Emergency fix (clean data and restart)"
    echo "  logs       - Show Neo4j logs"
    echo "  help       - Show this help message"
    echo
end

function show_logs
    log_info "Showing Neo4j container logs..."
    docker logs agent-zero-neo4j --tail 50 -f
end

# Main function
function main
    print_header
    
    # Check if running with correct permissions
    if not id -nG | grep -qw docker
        log_error "User is not in docker group. Add with: sudo usermod -aG docker $USER"
        log_info "Then logout and login again"
        return 1
    end
    
    if not check_dependencies
        return 1
    end
    
    # Parse command
    set command $argv[1]
    
    switch $command
        case start
            start_neo4j_service
            and wait_for_neo4j
            and test_neo4j_connection
            and show_neo4j_status
            
        case stop
            stop_neo4j_service
            
        case restart
            restart_neo4j
            
        case status
            show_neo4j_status
            
        case test
            test_neo4j_connection
            
        case fix
            emergency_fix
            
        case logs
            show_logs
            
        case help or ""
            show_help
            
        case '*'
            log_error "Unknown command: $command"
            show_help
            return 1
    end
end

# Run main function with all arguments
main $argv