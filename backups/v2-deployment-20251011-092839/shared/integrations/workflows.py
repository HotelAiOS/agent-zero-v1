"""
Workflow Generator
Generowanie GitHub Actions workflows dla CI/CD
"""

from typing import Dict, List, Optional
import yaml


class WorkflowGenerator:
    """
    Workflow Generator
    Generuje GitHub Actions workflows
    """
    
    @staticmethod
    def generate_python_ci(
        python_versions: List[str] = None,
        test_command: str = "pytest",
        coverage_threshold: float = 0.8
    ) -> str:
        """
        Generuj workflow CI dla Python
        
        Args:
            python_versions: Lista wersji Python (default: ["3.9", "3.10", "3.11"])
            test_command: Komenda do testów
            coverage_threshold: Minimalny coverage
        
        Returns:
            YAML workflow jako string
        """
        if python_versions is None:
            python_versions = ["3.9", "3.10", "3.11"]
        
        workflow = {
            'name': 'Python CI',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main', 'develop']
                }
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': python_versions
                        }
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python ${{ matrix.python-version }}',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '${{ matrix.python-version }}'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Run tests',
                            'run': test_command
                        },
                        {
                            'name': 'Check coverage',
                            'run': f'coverage report --fail-under={int(coverage_threshold * 100)}'
                        }
                    ]
                },
                'lint': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.11'
                            }
                        },
                        {
                            'name': 'Install linters',
                            'run': 'pip install flake8 black isort'
                        },
                        {
                            'name': 'Run flake8',
                            'run': 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics'
                        },
                        {
                            'name': 'Check black formatting',
                            'run': 'black --check .'
                        },
                        {
                            'name': 'Check isort',
                            'run': 'isort --check-only .'
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def generate_docker_build(
        docker_registry: str = "ghcr.io",
        image_name: str = "${{ github.repository }}"
    ) -> str:
        """
        Generuj workflow dla Docker build & push
        
        Args:
            docker_registry: Registry (ghcr.io, docker.io, etc.)
            image_name: Nazwa image
        
        Returns:
            YAML workflow jako string
        """
        workflow = {
            'name': 'Docker Build & Push',
            'on': {
                'push': {
                    'branches': ['main'],
                    'tags': ['v*']
                }
            },
            'env': {
                'REGISTRY': docker_registry,
                'IMAGE_NAME': image_name
            },
            'jobs': {
                'build': {
                    'runs-on': 'ubuntu-latest',
                    'permissions': {
                        'contents': 'read',
                        'packages': 'write'
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Log in to Container registry',
                            'uses': 'docker/login-action@v2',
                            'with': {
                                'registry': '${{ env.REGISTRY }}',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v4',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}'
                            }
                        },
                        {
                            'name': 'Build and push Docker image',
                            'uses': 'docker/build-push-action@v4',
                            'with': {
                                'context': '.',
                                'push': True,
                                'tags': '${{ steps.meta.outputs.tags }}',
                                'labels': '${{ steps.meta.outputs.labels }}'
                            }
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def generate_deploy_workflow(
        environment: str = "production",
        deploy_command: str = "kubectl apply -f k8s/"
    ) -> str:
        """
        Generuj workflow deployment
        
        Args:
            environment: Środowisko (production, staging)
            deploy_command: Komenda deployment
        
        Returns:
            YAML workflow jako string
        """
        workflow = {
            'name': f'Deploy to {environment}',
            'on': {
                'push': {
                    'branches': ['main']
                }
            },
            'jobs': {
                'deploy': {
                    'runs-on': 'ubuntu-latest',
                    'environment': environment,
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Setup deployment tools',
                            'run': 'echo "Setup kubectl, helm, etc."'
                        },
                        {
                            'name': 'Deploy',
                            'run': deploy_command,
                            'env': {
                                'KUBECONFIG': '${{ secrets.KUBECONFIG }}'
                            }
                        },
                        {
                            'name': 'Verify deployment',
                            'run': 'kubectl get pods'
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def generate_quality_gates() -> str:
        """
        Generuj workflow quality gates
        
        Returns:
            YAML workflow jako string
        """
        workflow = {
            'name': 'Quality Gates',
            'on': ['pull_request'],
            'jobs': {
                'security': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Run security scan',
                            'uses': 'aquasecurity/trivy-action@master',
                            'with': {
                                'scan-type': 'fs',
                                'scan-ref': '.',
                                'format': 'sarif',
                                'output': 'trivy-results.sarif'
                            }
                        }
                    ]
                },
                'code-quality': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'SonarCloud Scan',
                            'uses': 'SonarSource/sonarcloud-github-action@master',
                            'env': {
                                'GITHUB_TOKEN': '${{ secrets.GITHUB_TOKEN }}',
                                'SONAR_TOKEN': '${{ secrets.SONAR_TOKEN }}'
                            }
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
