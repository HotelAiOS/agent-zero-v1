# Generowanie naprawionego docker-compose.yml
docker_compose_fixed = """services:
  neo4j:
    image: neo4j:5.13
    container_name: agent-zero-neo4j
    environment:
      - NEO4J_AUTH=neo4j/agent-pass
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_PLUGINS='["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:7474 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: agent-zero-rabbitmq
    environment:
      - RABBITMQ_DEFAULT_USER=admin
      - RABBITMQ_DEFAULT_PASS=SecureRabbitPass123
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD-SHELL", "rabbitmq-diagnostics -q ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: agent-zero-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD-SHELL", "redis-cli ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  neo4j_data:
  rabbitmq_data:
  redis_data:

networks:
  agent-zero-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16
"""

# Zapisanie do pliku
with open('docker-compose-fixed.yml', 'w') as f:
    f.write(docker_compose_fixed)
    
print("✅ Created docker-compose-fixed.yml")
print("📋 Key changes:")
print("- Removed obsolete version attribute")  
print("- Fixed Neo4j password: neo4j/agent-pass")
print("- Added health checks for all services")
print("- Fixed network name: agent-zero-network")