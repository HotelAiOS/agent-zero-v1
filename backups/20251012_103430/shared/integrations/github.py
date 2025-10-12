"""
GitHub Integration
Integration z GitHub API dla repository management
"""

from typing import Dict, Any, List, Optional
import requests
import logging
from .base import BaseIntegration, IntegrationType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubIntegration(BaseIntegration):
    """
    GitHub Integration
    
    Funkcjonalności:
    - Repository management
    - Branch operations
    - Pull Request creation
    - Issue tracking
    - CI/CD workflows
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Konfiguracja GitHub
                - api_token: GitHub Personal Access Token
                - username: GitHub username
                - organization: GitHub organization (opcjonalne)
        """
        self.username = config.get('username')
        self.organization = config.get('organization')
        super().__init__(config)
    
    def _validate_config(self):
        """Walidacja konfiguracji"""
        if not self.api_token:
            raise ValueError("GitHub api_token is required")
        if not self.username and not self.organization:
            raise ValueError("Either username or organization is required")
        
        if not self.base_url:
            self.base_url = "https://api.github.com"
    
    def get_integration_type(self) -> IntegrationType:
        """Pobierz typ integracji"""
        return IntegrationType.GITHUB
    
    def _get_headers(self) -> Dict[str, str]:
        """Pobierz headers dla API requests"""
        return {
            'Authorization': f'token {self.api_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def test_connection(self) -> bool:
        """Test połączenia z GitHub API"""
        try:
            response = requests.get(
                f"{self.base_url}/user",
                headers=self._get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GitHub connection test failed: {e}")
            return False
    
    def create_repository(
        self,
        name: str,
        description: str = "",
        private: bool = True,
        auto_init: bool = True,
        gitignore_template: Optional[str] = "Python",
        license_template: Optional[str] = "mit"
    ) -> Dict[str, Any]:
        """
        Utwórz nowe repozytorium
        
        Args:
            name: Nazwa repo
            description: Opis
            private: Czy prywatne
            auto_init: Czy auto-initialize z README
            gitignore_template: Template .gitignore
            license_template: Template licencji
        
        Returns:
            Dict z danymi repo
        """
        logger.info(f"Creating GitHub repository: {name}")
        
        data = {
            'name': name,
            'description': description,
            'private': private,
            'auto_init': auto_init
        }
        
        if gitignore_template:
            data['gitignore_template'] = gitignore_template
        
        if license_template:
            data['license_template'] = license_template
        
        # Create in organization or user account
        if self.organization:
            url = f"{self.base_url}/orgs/{self.organization}/repos"
        else:
            url = f"{self.base_url}/user/repos"
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=data,
                timeout=10
            )
            
            if response.status_code == 201:
                repo_data = response.json()
                logger.info(f"✓ Repository created: {repo_data['html_url']}")
                return repo_data
            else:
                logger.error(f"Failed to create repo: {response.status_code}")
                return {'error': response.json()}
        
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            return {'error': str(e)}
    
    def create_branch(
        self,
        repo_name: str,
        branch_name: str,
        from_branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Utwórz nowy branch
        
        Args:
            repo_name: Nazwa repo
            branch_name: Nazwa nowego brancha
            from_branch: Branch źródłowy
        
        Returns:
            Dict z danymi brancha
        """
        logger.info(f"Creating branch {branch_name} from {from_branch}")
        
        owner = self.organization or self.username
        
        # Get SHA of from_branch
        try:
            ref_response = requests.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/ref/heads/{from_branch}",
                headers=self._get_headers(),
                timeout=10
            )
            
            if ref_response.status_code != 200:
                return {'error': 'Source branch not found'}
            
            sha = ref_response.json()['object']['sha']
            
            # Create new branch
            response = requests.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/refs",
                headers=self._get_headers(),
                json={
                    'ref': f'refs/heads/{branch_name}',
                    'sha': sha
                },
                timeout=10
            )
            
            if response.status_code == 201:
                logger.info(f"✓ Branch created: {branch_name}")
                return response.json()
            else:
                return {'error': response.json()}
        
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return {'error': str(e)}
    
    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
        draft: bool = False
    ) -> Dict[str, Any]:
        """
        Utwórz Pull Request
        
        Args:
            repo_name: Nazwa repo
            title: Tytuł PR
            head: Branch źródłowy
            base: Branch docelowy
            body: Opis PR
            draft: Czy draft PR
        
        Returns:
            Dict z danymi PR
        """
        logger.info(f"Creating PR: {title}")
        
        owner = self.organization or self.username
        
        data = {
            'title': title,
            'head': head,
            'base': base,
            'body': body,
            'draft': draft
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/pulls",
                headers=self._get_headers(),
                json=data,
                timeout=10
            )
            
            if response.status_code == 201:
                pr_data = response.json()
                logger.info(f"✓ PR created: {pr_data['html_url']}")
                return pr_data
            else:
                return {'error': response.json()}
        
        except Exception as e:
            logger.error(f"Error creating PR: {e}")
            return {'error': str(e)}
    
    def create_issue(
        self,
        repo_name: str,
        title: str,
        body: str = "",
        labels: List[str] = None,
        assignees: List[str] = None
    ) -> Dict[str, Any]:
        """
        Utwórz Issue
        
        Args:
            repo_name: Nazwa repo
            title: Tytuł issue
            body: Opis
            labels: Lista labeli
            assignees: Lista assignees
        
        Returns:
            Dict z danymi issue
        """
        logger.info(f"Creating issue: {title}")
        
        owner = self.organization or self.username
        
        data = {
            'title': title,
            'body': body
        }
        
        if labels:
            data['labels'] = labels
        if assignees:
            data['assignees'] = assignees
        
        try:
            response = requests.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/issues",
                headers=self._get_headers(),
                json=data,
                timeout=10
            )
            
            if response.status_code == 201:
                issue_data = response.json()
                logger.info(f"✓ Issue created: {issue_data['html_url']}")
                return issue_data
            else:
                return {'error': response.json()}
        
        except Exception as e:
            logger.error(f"Error creating issue: {e}")
            return {'error': str(e)}
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """Lista repozytoriów"""
        owner = self.organization or self.username
        type_param = 'orgs' if self.organization else 'users'
        
        try:
            response = requests.get(
                f"{self.base_url}/{type_param}/{owner}/repos",
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
        
        except Exception as e:
            logger.error(f"Error listing repos: {e}")
            return []
