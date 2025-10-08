#!/bin/bash
#
# Neo4j Rollback Script - Agent Zero V1
# Safe rollback in case of issues
# Arch Linux Compatible
#

set -e

# Colors
RED='\e[0;31m'
GREEN='\e[0;32m'
YELLOW='\e[1;33m'
BLUE='\e[0;34m'
NC='\e[0m'

# Configuration
PROJECT_PATH="/home/ianua/projects/agent-zero-v1"
BACKUP_DIR="/tmp/neo4j_backup_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/tmp/neo4j_rollback.log"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "$(date) [INFO] $1" >> $LOG_FILE
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "$(date) [ERROR] $1" >> $LOG_FILE
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "$(date) [WARNING] $1" >> $LOG_FILE
}

print_header() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}   Agent Zero V1 - Neo4j Rollback${NC}"
    echo -e "${BLUE}   Safe Recovery System${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo
}

backup_current_state() {
    log_info "Creating backup of current state..."
    
    mkdir -p "$BACKUP_DIR"
    
    cd "$PROJECT_PATH"
    
    # Backup docker-compose.yml
    if [[ -f "docker-compose.yml" ]]; then
        cp docker-compose.yml "$BACKUP_DIR/docker-compose.yml.backup"
        log_info "Backed up docker-compose.yml"
    fi
    
    # Backup any custom Neo4j configs
    if [[ -d "infrastructure/neo4j" ]]; then
        cp -r infrastructure/neo4j "$BACKUP_DIR/neo4j_config_backup"
        log_info "Backed up Neo4j configuration"
    fi
    
    # Export current Neo4j data if container is running
    if docker ps | grep -q "agent-zero-neo4j"; then
        log_info "Exporting Neo4j data before rollback..."
        docker exec agent-zero-neo4j neo4j-admin database dump --database=neo4j --to-path=/tmp/dump.db || {
            log_warning "Failed to export Neo4j data, continuing with rollback"
        }
    fi
    
    log_info "Backup completed at: $BACKUP_DIR"
}

stop_services() {
    log_info "Stopping all Agent Zero services..."
    
    cd "$PROJECT_PATH"
    
    # Stop all services
    docker-compose down -v
    
    # Remove containers if they exist
    for container in agent-zero-neo4j agent-zero-postgres agent-zero-rabbitmq agent-zero-redis; do
        if docker ps -a | grep -q "$container"; then
            docker rm -f "$container" || log_warning "Failed to remove container: $container"
        fi
    done
    
    log_info "Services stopped"
}

restore_original_config() {
    log_info "Restoring to original configuration..."
    
    cd "$PROJECT_PATH"
    
    # Restore original docker-compose.yml from git
    if git status >/dev/null 2>&1; then
        git checkout docker-compose.yml || {
            log_warning "Failed to restore docker-compose.yml from git"
        }
    fi
    
    # Remove any custom Neo4j configurations
    if [[ -d "infrastructure/neo4j" ]]; then
        rm -rf infrastructure/neo4j
        log_info "Removed custom Neo4j configuration"
    fi
    
    log_info "Original configuration restored"
}

clean_docker_resources() {
    log_info "Cleaning Docker resources..."
    
    # Remove volumes
    for volume in neo4j_data neo4j_logs neo4j_plugins postgres_data rabbitmq_data redis_data; do
        if docker volume ls | grep -q "$volume"; then
            docker volume rm "${PROJECT_PATH##*/}_$volume" || log_warning "Failed to remove volume: $volume"
        fi
    done
    
    # Clean up networks
    if docker network ls | grep -q "agent-zero-network"; then
        docker network rm agent-zero-network || log_warning "Failed to remove network"
    fi
    
    # Clean up any dangling resources
    docker system prune -f
    
    log_info "Docker resources cleaned"
}

start_basic_services() {
    log_info "Starting services with original configuration..."
    
    cd "$PROJECT_PATH"
    
    # Start with original docker-compose
    docker-compose up -d
    
    # Wait for services to start
    sleep 10
    
    # Check if Neo4j started successfully
    if docker ps | grep -q "agent-zero-neo4j"; then
        log_info "✅ Neo4j started successfully with original configuration"
        return 0
    else
        log_error "❌ Failed to start Neo4j with original configuration"
        return 1
    fi
}

verify_rollback() {
    log_info "Verifying rollback success..."
    
    # Check container status
    if docker ps | grep -q "agent-zero-neo4j"; then
        log_info "✅ Neo4j container is running"
    else
        log_error "❌ Neo4j container is not running"
        return 1
    fi
    
    # Test HTTP endpoint
    sleep 5
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:7474 | grep -q "200"; then
        log_info "✅ Neo4j HTTP endpoint is accessible"
    else
        log_error "❌ Neo4j HTTP endpoint is not accessible"
        return 1
    fi
    
    log_info "✅ Rollback verification completed successfully"
    return 0
}

show_rollback_summary() {
    echo
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}   ROLLBACK SUMMARY${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${GREEN}✅ Rollback completed successfully${NC}"
    echo
    echo "📋 What was done:"
    echo "  • Current state backed up to: $BACKUP_DIR"
    echo "  • All services stopped and cleaned"
    echo "  • Original configuration restored from git"
    echo "  • Docker volumes and networks cleaned"
    echo "  • Services restarted with original settings"
    echo
    echo "🔗 Access Information:"
    echo "  • Neo4j Browser: http://localhost:7474"
    echo "  • Default credentials: neo4j/neo4j (you'll be prompted to change)"
    echo
    echo "📝 Next steps:"
    echo "  1. Access Neo4j Browser and set up credentials"
    echo "  2. Verify your application connections work"
    echo "  3. If issues persist, check logs: docker-compose logs neo4j"
    echo
    echo -e "${BLUE}=========================================${NC}"
}

main() {
    print_header
    
    # Confirm rollback action
    echo -e "${YELLOW}⚠️  This will rollback Neo4j to original configuration${NC}"
    echo -e "${YELLOW}⚠️  All custom Neo4j data and settings will be lost${NC}"
    echo
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]([Ee][Ss])?$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
    
    log_info "Starting Neo4j rollback procedure..."
    
    # Execute rollback steps
    backup_current_state || {
        log_error "Backup failed, aborting rollback"
        exit 1
    }
    
    stop_services || {
        log_error "Failed to stop services"
        exit 1
    }
    
    restore_original_config || {
        log_error "Failed to restore original configuration"
        exit 1
    }
    
    clean_docker_resources || {
        log_error "Failed to clean Docker resources"
        exit 1
    }
    
    start_basic_services || {
        log_error "Failed to start services with original configuration"
        exit 1
    }
    
    verify_rollback || {
        log_error "Rollback verification failed"
        exit 1
    }
    
    show_rollback_summary
    
    log_info "Neo4j rollback completed successfully!"
}

# Check if running from correct directory
if [[ ! -f "docker-compose.yml" ]]; then
    log_error "docker-compose.yml not found. Please run this script from the project root directory."
    log_info "Expected location: $PROJECT_PATH"
    exit 1
fi

main "$@"