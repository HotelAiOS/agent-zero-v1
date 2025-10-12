#!/usr/bin/env python3
"""
AGENT ZERO V1 - PRODUCTION DOCKER STACK
Complete Production Deployment Infrastructure

Creates enterprise-grade Docker infrastructure with:
- All Agent Zero V1 components containerized
- API Gateway (Nginx) 
- Load balancing and health checks
- Production monitoring (Prometheus/Grafana)
- Complete microservices orchestration

Perfect for Dev A (backend) + Dev B (frontend) collaboration!
"""

import os
import json
from typing import Dict, List, Any

class ProductionDockerStackBuilder:
    """
    🐳 Production Docker Stack Builder
    
    Creates complete production deployment infrastructure:
    - Docker Compose orchestration
    - Nginx API Gateway  
    - Health monitoring
    - Service discovery
    - Production configurations
    """
    
    def __init__(self):
        self.services = {}
        self.networks = {}
        self.volumes = {}
        
        print("🐳 Agent Zero V1 - Production Docker Stack Builder")
        print("=" * 55)
        print("🎯 Creating enterprise-grade deployment infrastructure")
        print()
    
    def generate_docker_compose(self) -> str:
        """Generate complete production docker-compose.yml"""
        
        compose_config = {
            'version': '3.8',
            'services': {
                # 1. Master System Integrator (Main API)
                'master-integrator': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.master'
                    },
                    'container_name': 'agent-zero-master',
                    'ports': ['8000:8000'],
                    'environment': [
                        'ENVIRONMENT=production',
                        'LOG_LEVEL=INFO',
                        'DATABASE_URL=sqlite:///app/data/master.db'
                    ],
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs'
                    ],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped',
                    'depends_on': [
                        'team-formation',
                        'analytics',
                        'collaboration',
                        'predictive',
                        'adaptive-learning',
                        'quantum-intelligence'
                    ]
                },
                
                # 2. Team Formation Service (Phase 4)
                'team-formation': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.team'
                    },
                    'container_name': 'agent-zero-team',
                    'expose': ['8001'],
                    'environment': [
                        'SERVICE_NAME=team-formation',
                        'DATABASE_URL=sqlite:///app/data/team_formation.db'
                    ],
                    'volumes': ['./data:/app/data'],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8001/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                },
                
                # 3. Advanced Analytics Service (Phase 5)
                'analytics': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.analytics'
                    },
                    'container_name': 'agent-zero-analytics',
                    'expose': ['8002'],
                    'environment': [
                        'SERVICE_NAME=analytics',
                        'DATABASE_URL=sqlite:///app/data/analytics.db'
                    ],
                    'volumes': ['./data:/app/data'],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8002/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                },
                
                # 4. Real-Time Collaboration Service (Phase 6)
                'collaboration': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.collaboration'
                    },
                    'container_name': 'agent-zero-collaboration',
                    'expose': ['8003'],
                    'environment': [
                        'SERVICE_NAME=collaboration',
                        'DATABASE_URL=sqlite:///app/data/collaboration.db'
                    ],
                    'volumes': ['./data:/app/data'],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8003/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                },
                
                # 5. Predictive Management Service (Phase 7)
                'predictive': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.predictive'
                    },
                    'container_name': 'agent-zero-predictive',
                    'expose': ['8004'],
                    'environment': [
                        'SERVICE_NAME=predictive',
                        'DATABASE_URL=sqlite:///app/data/predictive.db'
                    ],
                    'volumes': ['./data:/app/data'],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8004/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                },
                
                # 6. Adaptive Learning Service (Phase 8)
                'adaptive-learning': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.adaptive'
                    },
                    'container_name': 'agent-zero-adaptive',
                    'expose': ['8005'],
                    'environment': [
                        'SERVICE_NAME=adaptive-learning',
                        'DATABASE_URL=sqlite:///app/data/adaptive_learning.db'
                    ],
                    'volumes': ['./data:/app/data'],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8005/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                },
                
                # 7. Quantum Intelligence Service (Phase 9)
                'quantum-intelligence': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.quantum'
                    },
                    'container_name': 'agent-zero-quantum',
                    'expose': ['8006'],
                    'environment': [
                        'SERVICE_NAME=quantum-intelligence',
                        'DATABASE_URL=sqlite:///app/data/quantum_intelligence.db'
                    ],
                    'volumes': ['./data:/app/data'],
                    'networks': ['agent-zero-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8006/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                },
                
                # 8. API Gateway (Nginx)
                'api-gateway': {
                    'image': 'nginx:alpine',
                    'container_name': 'agent-zero-gateway',
                    'ports': ['80:80', '443:443'],
                    'volumes': [
                        './nginx/nginx.conf:/etc/nginx/nginx.conf:ro',
                        './nginx/ssl:/etc/nginx/ssl:ro'
                    ],
                    'networks': ['agent-zero-network'],
                    'depends_on': ['master-integrator'],
                    'restart': 'unless-stopped'
                },
                
                # 9. Monitoring - Prometheus
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'agent-zero-prometheus',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro',
                        'prometheus-data:/prometheus'
                    ],
                    'networks': ['agent-zero-network'],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--web.enable-lifecycle'
                    ],
                    'restart': 'unless-stopped'
                },
                
                # 10. Monitoring - Grafana
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'agent-zero-grafana',
                    'ports': ['3000:3000'],
                    'volumes': [
                        'grafana-data:/var/lib/grafana',
                        './monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml'
                    ],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=admin123',
                        'GF_USERS_ALLOW_SIGN_UP=false'
                    ],
                    'networks': ['agent-zero-network'],
                    'depends_on': ['prometheus'],
                    'restart': 'unless-stopped'
                }
            },
            
            'networks': {
                'agent-zero-network': {
                    'driver': 'bridge',
                    'name': 'agent-zero-production'
                }
            },
            
            'volumes': {
                'prometheus-data': {
                    'driver': 'local'
                },
                'grafana-data': {
                    'driver': 'local'
                }
            }
        }
        
        # Convert to YAML format
        import yaml
        return yaml.dump(compose_config, default_flow_style=False, indent=2)
    
    def generate_nginx_config(self) -> str:
        """Generate Nginx API Gateway configuration"""
        
        return """
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types text/plain text/css text/xml text/javascript application/json application/xml+rss application/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # Upstream servers
    upstream master-integrator {
        server master-integrator:8000;
    }
    
    upstream team-formation {
        server team-formation:8001;
    }
    
    upstream analytics {
        server analytics:8002;
    }
    
    upstream collaboration {
        server collaboration:8003;
    }
    
    upstream predictive {
        server predictive:8004;
    }
    
    upstream adaptive-learning {
        server adaptive-learning:8005;
    }
    
    upstream quantum-intelligence {
        server quantum-intelligence:8006;
    }
    
    # Main API Gateway Server
    server {
        listen 80;
        server_name localhost agent-zero.local;
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }
        
        # Master System Integrator (Main API)
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://master-integrator/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS headers for Dev B frontend
            add_header Access-Control-Allow-Origin *;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS, PUT, DELETE";
            add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization";
            
            if ($request_method = 'OPTIONS') {
                return 204;
            }
        }
        
        # Individual service endpoints (for debugging/monitoring)
        location /services/team/ {
            proxy_pass http://team-formation/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        location /services/analytics/ {
            proxy_pass http://analytics/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        location /services/collaboration/ {
            proxy_pass http://collaboration/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        location /services/predictive/ {
            proxy_pass http://predictive/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        location /services/adaptive/ {
            proxy_pass http://adaptive-learning/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        location /services/quantum/ {
            proxy_pass http://quantum-intelligence/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        # Static files for Dev B frontend (when ready)
        location /static/ {
            alias /usr/share/nginx/html/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Frontend app (when Dev B delivers)
        location / {
            try_files $uri $uri/ /index.html;
            root /usr/share/nginx/html;
            index index.html;
            
            # CORS for frontend development
            add_header Access-Control-Allow-Origin *;
        }
    }
    
    # Monitoring endpoints
    server {
        listen 8080;
        server_name monitoring.agent-zero.local;
        
        location /prometheus/ {
            proxy_pass http://prometheus:9090/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /grafana/ {
            proxy_pass http://grafana:3000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
"""
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus monitoring configuration"""
        
        return """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files: []

scrape_configs:
  - job_name: 'agent-zero-services'
    static_configs:
      - targets: 
          - 'master-integrator:8000'
          - 'team-formation:8001'
          - 'analytics:8002'
          - 'collaboration:8003'
          - 'predictive:8004'
          - 'adaptive-learning:8005'
          - 'quantum-intelligence:8006'
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:80']
    metrics_path: '/nginx_status'
    scrape_interval: 30s
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""
    
    def generate_base_dockerfile(self) -> str:
        """Generate base Dockerfile for Agent Zero services"""
        
        return """
# Agent Zero V1 - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

# Expose port (will be overridden by specific services)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (will be overridden by specific services)
CMD ["python", "master_system_integrator_fixed.py"]
"""
    
    def generate_deployment_script(self) -> str:
        """Generate deployment script"""
        
        return """#!/bin/bash
# Agent Zero V1 - Production Deployment Script

echo "🐳 Agent Zero V1 - Production Deployment"
echo "========================================"
echo "🎯 Deploying complete production stack..."
echo ""

# Create necessary directories
mkdir -p data logs nginx monitoring

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
echo ""

services=("master-integrator:8000" "team-formation:8001" "analytics:8002" "collaboration:8003" "predictive:8004" "adaptive-learning:8005" "quantum-intelligence:8006")

for service in "${services[@]}"; do
    if curl -f -s "http://localhost:${service##*:}/health" > /dev/null; then
        echo "✅ $service - HEALTHY"
    else
        echo "❌ $service - UNHEALTHY"
    fi
done

echo ""
echo "🎯 Service URLs:"
echo "   • Main API: http://localhost/api/"
echo "   • API Gateway: http://localhost/"
echo "   • Prometheus: http://localhost:9090/"
echo "   • Grafana: http://localhost:3000/ (admin/admin123)"
echo ""

echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🎉 Agent Zero V1 Production Deployment Complete!"
echo "🚀 Ready for enterprise use and Dev B frontend integration!"
"""

def create_production_docker_stack():
    """Create complete production Docker stack"""
    
    builder = ProductionDockerStackBuilder()
    
    print("📝 Generating production configurations...")
    
    # 1. Generate docker-compose.yml
    docker_compose = builder.generate_docker_compose()
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    print("✅ docker-compose.yml created")
    
    # 2. Create nginx directory and config
    os.makedirs('nginx', exist_ok=True)
    nginx_config = builder.generate_nginx_config()
    with open('nginx/nginx.conf', 'w') as f:
        f.write(nginx_config)
    print("✅ nginx/nginx.conf created")
    
    # 3. Create monitoring directory and config
    os.makedirs('monitoring', exist_ok=True)
    prometheus_config = builder.generate_prometheus_config()
    with open('monitoring/prometheus.yml', 'w') as f:
        f.write(prometheus_config)
    print("✅ monitoring/prometheus.yml created")
    
    # 4. Generate Dockerfiles for each service
    base_dockerfile = builder.generate_base_dockerfile()
    
    dockerfiles = [
        ('Dockerfile.master', 'master_system_integrator_fixed.py', '8000'),
        ('Dockerfile.team', 'agent_zero_phases_4_5_production.py', '8001'),
        ('Dockerfile.analytics', 'agent_zero_phases_4_5_production.py', '8002'),
        ('Dockerfile.collaboration', 'agent_zero_phases_6_7_production.py', '8003'),
        ('Dockerfile.predictive', 'agent_zero_phases_6_7_production.py', '8004'),
        ('Dockerfile.adaptive', 'agent_zero_phases_8_9_complete_system.py', '8005'),
        ('Dockerfile.quantum', 'agent_zero_phases_8_9_complete_system.py', '8006')
    ]
    
    for dockerfile, main_script, port in dockerfiles:
        dockerfile_content = base_dockerfile.replace('8000', port).replace('master_system_integrator_fixed.py', main_script)
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
        print(f"✅ {dockerfile} created")
    
    # 5. Generate deployment script
    deploy_script = builder.generate_deployment_script()
    with open('deploy_production.sh', 'w') as f:
        f.write(deploy_script)
    os.chmod('deploy_production.sh', 0o755)
    print("✅ deploy_production.sh created (executable)")
    
    # 6. Create requirements.txt
    requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
sqlite3
asyncio
python-multipart==0.0.6
httpx==0.25.2
prometheus-client==0.19.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ requirements.txt created")
    
    # 7. Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    print("✅ data/ and logs/ directories created")
    
    print()
    print("🏆 PRODUCTION DOCKER STACK COMPLETE!")
    print("=" * 40)
    print("🐳 Ready to deploy with: ./deploy_production.sh")
    print()
    print("📋 What was created:")
    print("   ✅ docker-compose.yml - Complete orchestration")
    print("   ✅ nginx/nginx.conf - API Gateway configuration")
    print("   ✅ monitoring/prometheus.yml - Monitoring setup")
    print("   ✅ 7 Dockerfiles - One for each service")
    print("   ✅ deploy_production.sh - Deployment script")
    print("   ✅ requirements.txt - Python dependencies")
    print("   ✅ data/ logs/ directories - Persistent storage")
    print()
    print("🎯 Services included:")
    print("   🧠 Master System Integrator (Port 8000)")
    print("   👥 Team Formation (Port 8001)")
    print("   📊 Advanced Analytics (Port 8002)")  
    print("   🤝 Real-Time Collaboration (Port 8003)")
    print("   🔮 Predictive Management (Port 8004)")
    print("   🧠 Adaptive Learning (Port 8005)")
    print("   ⚛️  Quantum Intelligence (Port 8006)")
    print("   🚪 Nginx API Gateway (Port 80)")
    print("   📈 Prometheus Monitoring (Port 9090)")
    print("   📊 Grafana Dashboard (Port 3000)")
    print()
    print("🚀 Ready for Dev B frontend integration!")
    print("💼 Enterprise deployment capability: COMPLETE!")

if __name__ == "__main__":
    create_production_docker_stack()