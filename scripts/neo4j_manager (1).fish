#!/usr/bin/fish
#
# Neo4j Manager - Agent Zero V1
# Fish Shell Script for Managing Neo4j Service
#
# Save this file as: /home/ianua/projects/agent-zero-v1/scripts/neo4j_manager.fish
# Make executable: chmod +x scripts/neo4j_manager.fish
# Usage: scripts/neo4j_manager.fish [start|stop|restart|status|test]
#

# Colors
set -g RED '\e[0;31m'
set -g GREEN '\e[0;32m' 
set -g YELLOW '\e[1;33m'
set -g BLUE '\e[0;34m'
set -g NC '\e[0m'

# Configuration
set -g PROJECT_PATH "/home/ianua/projects/agent-zero-v1"

function print_header
    echo -e $BLUE"========================================="$NC
    echo -e $BLUE"   Agent Zero V1 - Neo4j Manager"$NC
    echo -e $BLUE"========================================="$NC
end

function log_info
    echo -e $GREEN"[INFO]"$NC" $argv[1]"
end

function log_error
    echo -e $RED"[ERROR]"$NC" $argv[1]"
end

function log_warning
    echo -e $YELLOW"[WARNING]"$NC" $argv[1]"
end

function start_neo4j
    log_info "Starting Neo4j service..."
    
    cd $PROJECT_PATH
    
    docker-compose up -d neo4j
    if test $status -eq 0
        log_info "Neo4j container started successfully"
        
        # Wait for Neo4j to be ready
        log_info "Waiting for Neo4j to be ready..."
        sleep 10
        
        if test_neo4j_connection
            log_info "✅ Neo4j is ready and accessible!"
            show_access_info
        else
            log_warning "Neo4j started but connection test failed"
        end
    else
        log_error "Failed to start Neo4j container"
        return 1
    end
end

function stop_neo4j
    log_info "Stopping Neo4j service..."
    
    cd $PROJECT_PATH
    
    docker-compose stop neo4j
    if test $status -eq 0
        log_info "Neo4j container stopped successfully"
    else
        log_error "Failed to stop Neo4j container"
        return 1
    end
end

function restart_neo4j
    log_info "Restarting Neo4j service..."
    
    stop_neo4j
    sleep 3
    start_neo4j
end

function test_neo4j_connection
    # Test HTTP endpoint
    set http_response (curl -s -o /dev/null -w "%{http_code}" http://localhost:7474 2>/dev/null)
    
    if test "$http_response" = "200"
        return 0
    else
        return 1
    end
end

function show_status
    log_info "Checking Neo4j status..."
    
    # Check container status
    set container_status (docker inspect --format='{{.State.Status}}' agent-zero-neo4j 2>/dev/null)
    
    if test $status -eq 0
        echo -e "Container Status: "$GREEN"$container_status"$NC
        
        if test "$container_status" = "running"
            # Test connectivity
            if test_neo4j_connection
                echo -e "HTTP Endpoint: "$GREEN"✅ Accessible"$NC
                echo -e "Overall Status: "$GREEN"✅ Healthy"$NC
            else
                echo -e "HTTP Endpoint: "$RED"❌ Not accessible"$NC
                echo -e "Overall Status: "$YELLOW"⚠️  Starting up"$NC
            end
        else
            echo -e "HTTP Endpoint: "$RED"❌ Container not running"$NC
            echo -e "Overall Status: "$RED"❌ Stopped"$NC
        end
    else
        echo -e "Container Status: "$RED"❌ Not found"$NC
        echo -e "Overall Status: "$RED"❌ Not running"$NC
    end
end

function show_access_info
    echo
    echo -e $BLUE"📋 NEO4J ACCESS INFORMATION:"$NC
    echo -e "   • Browser: "$YELLOW"http://localhost:7474"$NC
    echo -e "   • Bolt URI: "$YELLOW"bolt://localhost:7687"$NC  
    echo -e "   • Username: "$YELLOW"neo4j"$NC
    echo -e "   • Password: "$YELLOW"agent_zero_neo4j_dev"$NC
    echo -e "   • Container: "$YELLOW"agent-zero-neo4j"$NC
    echo
end

function show_logs
    log_info "Showing Neo4j container logs (last 50 lines)..."
    docker logs agent-zero-neo4j --tail 50 -f
end

function show_help
    print_header
    echo
    echo "Usage: $argv[0] [COMMAND]"
    echo
    echo "Commands:"
    echo "  start    - Start Neo4j service"
    echo "  stop     - Stop Neo4j service"  
    echo "  restart  - Restart Neo4j service"
    echo "  status   - Show Neo4j status"
    echo "  test     - Test Neo4j connection"
    echo "  logs     - Show Neo4j logs"
    echo "  info     - Show access information"
    echo "  help     - Show this help message"
    echo
end

function main
    # Check if we're in the right directory
    if not test -d $PROJECT_PATH
        log_error "Project directory not found: $PROJECT_PATH"
        return 1
    end
    
    print_header
    
    # Parse command
    set command $argv[1]
    
    switch $command
        case start
            start_neo4j
            
        case stop  
            stop_neo4j
            
        case restart
            restart_neo4j
            
        case status
            show_status
            
        case test
            if test_neo4j_connection
                log_info "✅ Neo4j connection test passed"
                show_access_info
            else
                log_error "❌ Neo4j connection test failed"
                return 1
            end
            
        case logs
            show_logs
            
        case info
            show_access_info
            
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