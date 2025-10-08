
# GENEROWANIE PLIKU 4: Docker Compose Fix
docker_compose_content = '''# Agent Zero V1 - Fixed Docker Compose Configuration
# Critical Fix A0-5: Neo4j connection and service configuration

version: '3.8'

services:
  # Neo4j Knowledge Graph Database
  neo4j:
    image: neo4j:5.13.0
    container_name: agent-zero-neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/agent-pass
      - NEO4J_dbms_memory_pagecache_size=512M
      - NEO4J_dbms_memory_heap_initial__size=512M
      - NEO4J_dbms_memory_heap_max__size=1G
      - NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687
      - NEO4J_dbms_connector_http_listen__address=0.0.0.0:7474
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_security_procedures_allowlist=apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_plugins:/plugins
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "agent-pass", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  # RabbitMQ Message Broker
  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: agent-zero-rabbitmq
    restart: unless-stopped
    ports:
      - "5672:5672"   # AMQP
      - "15672:15672" # Management UI
    environment:
      - RABBITMQ_DEFAULT_USER=agent
      - RABBITMQ_DEFAULT_PASS=agent-rabbitmq-pass
      - RABBITMQ_DEFAULT_VHOST=/
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache (optional but recommended)
  redis:
    image: redis:7-alpine
    container_name: agent-zero-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # Ollama LLM Server (optional - for local LLM)
  ollama:
    image: ollama/ollama:latest
    container_name: agent-zero-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    networks:
      - agent-zero-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  neo4j_data:
    driver: local
  neo4j_logs:
    driver: local
  neo4j_plugins:
    driver: local
  rabbitmq_data:
    driver: local
  redis_data:
    driver: local
  ollama_models:
    driver: local

networks:
  agent-zero-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
'''

with open('docker-compose_fixed.yml', 'w', encoding='utf-8') as f:
    f.write(docker_compose_content)

print("✅ Generated: docker-compose_fixed.yml")
