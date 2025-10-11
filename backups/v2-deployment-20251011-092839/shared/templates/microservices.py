"""
Microservices Backend Template
Distributed microservices architecture
"""

from typing import List
from .base import ProjectTemplate, TemplateConfig, TemplateCategory, TemplateRequirement


class MicroservicesTemplate(ProjectTemplate):
    """
    Microservices Backend Template
    
    Distributed microservices with:
    - API Gateway
    - Service discovery
    - Message queue
    - Distributed tracing
    """
    
    def get_config(self) -> TemplateConfig:
        """Get Microservices template configuration"""
        return TemplateConfig(
            template_id='microservices_v1',
            template_name='Microservices Architecture',
            category=TemplateCategory.MICROSERVICES,
            description='Distributed microservices with API gateway, message queue, and monitoring',
            
            # Tech Stack
            tech_stack=[
                'Python', 'FastAPI', 'PostgreSQL', 'MongoDB',
                'RabbitMQ', 'Redis', 'Docker', 'Kubernetes',
                'Kong', 'Prometheus', 'Grafana', 'Jaeger'
            ],
            frameworks=['FastAPI', 'SQLAlchemy', 'Pika', 'Celery'],
            databases=['PostgreSQL', 'MongoDB', 'Redis'],
            
            # Requirements
            requirements=[
                TemplateRequirement(
                    name='API Gateway',
                    description='Kong gateway with routing, auth, rate limiting',
                    priority=1,
                    complexity=8,
                    estimated_hours=24.0
                ),
                TemplateRequirement(
                    name='User Service',
                    description='Authentication, authorization, user management',
                    priority=1,
                    complexity=7,
                    estimated_hours=20.0
                ),
                TemplateRequirement(
                    name='Product Service',
                    description='Product catalog, inventory management',
                    priority=1,
                    complexity=6,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='Order Service',
                    description='Order processing, saga pattern',
                    priority=1,
                    complexity=8,
                    estimated_hours=24.0
                ),
                TemplateRequirement(
                    name='Message Queue',
                    description='RabbitMQ for async communication',
                    priority=1,
                    complexity=7,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='Service Discovery',
                    description='Consul for service registration',
                    priority=2,
                    complexity=6,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='Distributed Tracing',
                    description='Jaeger for request tracing',
                    priority=2,
                    complexity=6,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='Monitoring',
                    description='Prometheus + Grafana dashboards',
                    priority=2,
                    complexity=5,
                    estimated_hours=12.0
                )
            ],
            
            # Team
            required_agents=['architect', 'backend', 'database', 'devops', 'tester', 'performance'],
            team_size=6,
            
            # Estimates
            estimated_duration_days=40.0,
            estimated_cost=100000.0,
            
            # Quality
            min_test_coverage=0.85,
            code_review_required=True,
            
            # Additional
            tags=['microservices', 'distributed', 'kubernetes', 'api-gateway'],
            documentation_url='https://docs.example.com/microservices-template'
        )
    
    def get_business_requirements(self) -> List[str]:
        """Get business requirements"""
        return [
            'API Gateway with Kong for routing and authentication',
            'User Service for authentication and authorization',
            'Product Service with inventory management',
            'Order Service with saga pattern for transactions',
            'RabbitMQ for async inter-service communication',
            'Service discovery with Consul',
            'Distributed tracing with Jaeger',
            'Monitoring with Prometheus and Grafana',
            'Docker containerization for all services',
            'Kubernetes orchestration and deployment'
        ]
