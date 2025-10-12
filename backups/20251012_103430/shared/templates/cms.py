"""
CMS/Blog Template
Content Management System with blog, pages, media
"""

from typing import List
from .base import ProjectTemplate, TemplateConfig, TemplateCategory, TemplateRequirement


class CMSTemplate(ProjectTemplate):
    """
    CMS/Blog Template
    
    Content Management System with:
    - Blog posts with rich editor
    - Static pages
    - Media library
    - Categories & tags
    - Comments
    """
    
    def get_config(self) -> TemplateConfig:
        """Get CMS template configuration"""
        return TemplateConfig(
            template_id='cms_blog_v1',
            template_name='CMS & Blog Platform',
            category=TemplateCategory.CMS,
            description='Content Management System with blog, pages, and media library',
            
            # Tech Stack
            tech_stack=[
                'Python', 'FastAPI', 'PostgreSQL',
                'React', 'Next.js', 'TailwindCSS',
                'TipTap', 'AWS S3'
            ],
            frameworks=['FastAPI', 'Next.js', 'SQLAlchemy'],
            databases=['PostgreSQL'],
            
            # Requirements
            requirements=[
                TemplateRequirement(
                    name='Blog System',
                    description='Posts with rich editor, drafts, publishing',
                    priority=1,
                    complexity=6,
                    estimated_hours=20.0
                ),
                TemplateRequirement(
                    name='Page Builder',
                    description='Static pages, templates, SEO',
                    priority=1,
                    complexity=7,
                    estimated_hours=24.0
                ),
                TemplateRequirement(
                    name='Media Library',
                    description='Image/file upload, organization, CDN',
                    priority=2,
                    complexity=5,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='Categories & Tags',
                    description='Content organization and filtering',
                    priority=2,
                    complexity=4,
                    estimated_hours=8.0
                ),
                TemplateRequirement(
                    name='Comments System',
                    description='User comments, moderation, replies',
                    priority=2,
                    complexity=5,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='User Roles',
                    description='Admin, editor, author permissions',
                    priority=1,
                    complexity=6,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='SEO Tools',
                    description='Meta tags, sitemaps, social sharing',
                    priority=2,
                    complexity=5,
                    estimated_hours=10.0
                )
            ],
            
            # Team
            required_agents=['architect', 'backend', 'frontend', 'database', 'tester'],
            team_size=5,
            
            # Estimates
            estimated_duration_days=25.0,
            estimated_cost=60000.0,
            
            # Quality
            min_test_coverage=0.75,
            code_review_required=True,
            
            # Additional
            tags=['cms', 'blog', 'content', 'publishing'],
            documentation_url='https://docs.example.com/cms-template'
        )
    
    def get_business_requirements(self) -> List[str]:
        """Get business requirements"""
        return [
            'Blog post creation with rich text editor',
            'Static page builder with templates',
            'Media library with image upload and management',
            'Categories and tags for content organization',
            'Comment system with moderation',
            'User roles: Admin, Editor, Author',
            'SEO optimization with meta tags and sitemaps',
            'Responsive design for all devices',
            'Search functionality',
            'RSS feed generation'
        ]

