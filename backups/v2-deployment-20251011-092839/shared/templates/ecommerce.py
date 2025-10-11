"""
E-commerce Platform Template
Online store with products, cart, payments
"""

from typing import List
from .base import ProjectTemplate, TemplateConfig, TemplateCategory, TemplateRequirement


class EcommerceTemplate(ProjectTemplate):
    """
    E-commerce Platform Template
    
    Complete online store with:
    - Product catalog with categories
    - Shopping cart & checkout
    - Payment processing (Stripe)
    - Order management
    - Admin panel
    """
    
    def get_config(self) -> TemplateConfig:
        """Get E-commerce template configuration"""
        return TemplateConfig(
            template_id='ecommerce_store_v1',
            template_name='E-commerce Store',
            category=TemplateCategory.ECOMMERCE,
            description='Complete online store with products, cart, payments, and admin panel',
            
            # Tech Stack
            tech_stack=[
                'Python', 'FastAPI', 'PostgreSQL',
                'React', 'Next.js', 'TailwindCSS',
                'Stripe', 'AWS S3'
            ],
            frameworks=['FastAPI', 'Next.js', 'Prisma'],
            databases=['PostgreSQL', 'Redis'],
            
            # Requirements
            requirements=[
                TemplateRequirement(
                    name='Product Catalog',
                    description='Products, categories, search, filters',
                    priority=1,
                    complexity=6,
                    estimated_hours=20.0
                ),
                TemplateRequirement(
                    name='Shopping Cart',
                    description='Add/remove items, quantities, save for later',
                    priority=1,
                    complexity=5,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='Checkout Process',
                    description='Multi-step checkout, address, shipping',
                    priority=1,
                    complexity=7,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='Payment Integration',
                    description='Stripe payment processing, webhooks',
                    priority=1,
                    complexity=8,
                    estimated_hours=20.0
                ),
                TemplateRequirement(
                    name='Order Management',
                    description='Order tracking, history, status updates',
                    priority=1,
                    complexity=6,
                    estimated_hours=16.0
                ),
                TemplateRequirement(
                    name='Admin Panel',
                    description='Product management, orders, analytics',
                    priority=2,
                    complexity=7,
                    estimated_hours=24.0
                ),
                TemplateRequirement(
                    name='User Accounts',
                    description='Registration, login, profile, order history',
                    priority=2,
                    complexity=5,
                    estimated_hours=12.0
                ),
                TemplateRequirement(
                    name='Email Notifications',
                    description='Order confirmation, shipping updates',
                    priority=2,
                    complexity=4,
                    estimated_hours=8.0
                )
            ],
            
            # Team
            required_agents=['architect', 'backend', 'frontend', 'database', 'devops', 'tester'],
            team_size=6,
            
            # Estimates
            estimated_duration_days=35.0,
            estimated_cost=90000.0,
            
            # Quality
            min_test_coverage=0.80,
            code_review_required=True,
            
            # Additional
            tags=['ecommerce', 'store', 'payments', 'stripe', 'products'],
            documentation_url='https://docs.example.com/ecommerce-template'
        )
    
    def get_business_requirements(self) -> List[str]:
        """Get business requirements"""
        return [
            'Product catalog with categories, search, and filters',
            'Shopping cart with add/remove/update functionality',
            'Multi-step checkout process with address and shipping',
            'Stripe payment integration with secure checkout',
            'Order management with tracking and status updates',
            'Admin panel for products, orders, and customers',
            'User accounts with registration and order history',
            'Email notifications for orders and shipping',
            'Responsive design for mobile shopping',
            'Image upload and management for products'
        ]
