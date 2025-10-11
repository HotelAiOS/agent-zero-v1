"""
Mobile App Backend Template
REST API backend for mobile applications
"""

from typing import List
from .base import ProjectTemplate, TemplateConfig, TemplateCategory, TemplateRequirement


class MobileBackendTemplate(ProjectTemplate):
    """
    Mobile App Backend Template
    
    REST API backend for mobile apps with:
    - User authentication
    - Push notifications
    - File uploads
    - Real-time features
    """
    
    def get_config(self) -> TemplateConfig:
        """Get Mobile Backend template configuration"""
        return TemplateConfig(
            template_id='mobile_backend_v1',
            template_name='Mobile App Backend',
            category=TemplateCategory.MOBILE_BACKEND,
            description='REST API backend for mobile applications with push notifications and real-time',
            
            # Tech Stack
            tech_stack=[
                'Python', 'FastAPI', 'PostgreSQL', 'Redis',
                'Firebase', 'WebSocket', 'AWS S3', 'Docker'
            ],
            frameworks=['FastAPI', 'SQLAlchemy', 'Socket.IO', 'Celery'],
            databases=['PostgreSQL', 'Redis'],
            
            # Requirements
            requirements=[
                TemplateRequirement(
                    name='User Authentication',
                    description='JWT auth, OAuth, social login',
                    priority=1,
                    complexity=7,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='Push Notifications',
                    description='Firebase Cloud Messaging integration',
                    priority=1,
                    complexity=6,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='File Upload',
                    description='Image/video upload to S3, thumbnails',
                    priority=1,
                    complexity=5,
                    estimated_hours=10.0
                ),
                TemplateRequirement(
                    name='Real-time Features',
                    description='WebSocket for chat, notifications',
                    priority=2,
                    complexity=7,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='API Documentation',
                    description='OpenAPI/Swagger with examples',
                    priority=2,
                    complexity=4,
                    estimated_hours=8.0
                ),
                TemplateRequirement(
                    name='Rate Limiting',
                    description='API rate limiting and throttling',
                    priority=2,
                    complexity=5,
                    estimated_hours=8.0
                ),
                TemplateRequirement(
                    name='Analytics',
                    description='Usage tracking, crash reporting',
                    priority=3,
                    complexity=5,
                    estimated_hours=10.0
                )
            ],
            
            # Team
            required_agents=['architect', 'backend', 'database', 'devops', 'tester', 'security'],
            team_size=6,
            
            # Estimates
            estimated_duration_days=20.0,
            estimated_cost=50000.0,
            
            # Quality
            min_test_coverage=0.80,
            code_review_required=True,
            
            # Additional
            tags=['mobile', 'backend', 'api', 'push-notifications', 'websocket'],
            documentation_url='https://docs.example.com/mobile-backend-template'
        )
    
    def get_business_requirements(self) -> List[str]:
        """Get business requirements"""
        return [
            'User authentication with JWT and social login',
            'Push notifications via Firebase Cloud Messaging',
            'File upload to AWS S3 with thumbnail generation',
            'Real-time features using WebSocket',
            'RESTful API with comprehensive documentation',
            'Rate limiting and API throttling',
            'Analytics and crash reporting',
            'Offline sync support',
            'Versioning for mobile app compatibility',
            'Security: HTTPS, input validation, encryption'
        ]
