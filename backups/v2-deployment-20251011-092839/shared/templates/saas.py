"""
SaaS Application Template
Multi-tenant SaaS platform with subscriptions
"""

from typing import List
from .base import ProjectTemplate, TemplateConfig, TemplateCategory, TemplateRequirement


class SaaSTemplate(ProjectTemplate):
    """
    SaaS Application Template
    
    Complete multi-tenant SaaS platform with:
    - User authentication & authorization
    - Multi-tenancy (workspace/organization isolation)
    - Subscription management (Stripe integration)
    - Admin dashboard
    - User dashboard
    - API for integrations
    """
    
    def get_config(self) -> TemplateConfig:
        """Get SaaS template configuration"""
        return TemplateConfig(
            template_id='saas_platform_v1',
            template_name='SaaS Platform',
            category=TemplateCategory.SAAS,
            description='Multi-tenant SaaS application with subscriptions, dashboards, and API',
            
            # Tech Stack
            tech_stack=[
                'Python', 'FastAPI', 'PostgreSQL', 'Redis',
                'React', 'TypeScript', 'TailwindCSS',
                'Docker', 'Kubernetes'
            ],
            frameworks=['FastAPI', 'React', 'SQLAlchemy', 'Celery'],
            databases=['PostgreSQL', 'Redis'],
            
            # Requirements
            requirements=[
                TemplateRequirement(
                    name='User Authentication',
                    description='JWT-based auth with email verification, password reset',
                    priority=1,
                    complexity=7,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='Multi-tenancy System',
                    description='Workspace/organization isolation, member management',
                    priority=1,
                    complexity=9,
                    estimated_hours=32.0
                ),
                TemplateRequirement(
                    name='Subscription Management',
                    description='Stripe integration, plans, billing, invoices',
                    priority=1,
                    complexity=8,
                    estimated_hours=24.0
                ),
                TemplateRequirement(
                    name='Admin Dashboard',
                    description='User management, analytics, system monitoring',
                    priority=2,
                    complexity=7,
                    estimated_hours=24.0
                ),
                TemplateRequirement(
                    name='User Dashboard',
                    description='Profile, workspace, team, settings',
                    priority=2,
                    complexity=6,
                    estimated_hours=20.0
                ),
                TemplateRequirement(
                    name='REST API',
                    description='Complete API with documentation, rate limiting',
                    priority=1,
                    complexity=7,
                    estimated_hours=20.0
                ),
                TemplateRequirement(
                    name='Email System',
                    description='Transactional emails (welcome, invoices, notifications)',
                    priority=2,
                    complexity=5,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='Analytics & Reporting',
                    description='Usage analytics, custom reports',
                    priority=3,
                    complexity=6,
                    estimated_hours=16.0
                )
            ],
            
            # Team
            required_agents=['architect', 'backend', 'frontend', 'database', 'devops', 'tester', 'security'],
            team_size=7,
            
            # Estimates
            estimated_duration_days=45.0,
            estimated_cost=120000.0,
            
            # Quality
            min_test_coverage=0.85,
            code_review_required=True,
            
            # Additional
            tags=['saas', 'multi-tenant', 'subscriptions', 'stripe', 'full-stack'],
            documentation_url='https://docs.example.com/saas-template'
        )
    
    def get_business_requirements(self) -> List[str]:
        """Get business requirements"""
        return [
            'User registration and authentication with email verification',
            'Multi-tenant workspace system with role-based access control',
            'Subscription plans (Free, Pro, Enterprise) with Stripe payment',
            'Admin dashboard for user and system management',
            'User dashboard with profile, workspace, and team management',
            'RESTful API with documentation and API keys',
            'Email notifications for important events',
            'Usage analytics and reporting',
            'Responsive design for mobile and desktop',
            'Security: HTTPS, CORS, rate limiting, input validation'
        ]
