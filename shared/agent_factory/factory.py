"""
Agent Factory - Dynamiczne tworzenie agentów z YAML templates
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import sys

# NOWE: Import OllamaClient
sys.path.append(str(Path(__file__).parent.parent))
from llm.ollama_client import OllamaClient

from .capabilities import (
    AgentCapability, 
    CapabilityMatcher, 
    TechStack, 
    SkillLevel
)
from .lifecycle import AgentLifecycleManager, AgentInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentTemplate:
    """Szablon agenta z YAML"""
    name: str
    type: str
    version: str
    description: str
    capabilities: List[Dict[str, Any]]
    personality: Dict[str, str]
    ai_config: Dict[str, Any]
    protocols: Dict[str, bool]
    quality_gates: List[Dict[str, Any]]
    collaboration: Dict[str, List[str]]
    output_formats: List[str]
    raw_data: Dict[str, Any]  # Pełne dane YAML


class AgentFactory:
    """
    Fabryka do dynamicznego tworzenia agentów
    Czyta YAML templates i tworzy instancje agentów
    """
    
    def __init__(
        self, 
        templates_dir: Optional[Path] = None,
        capability_matcher: Optional[CapabilityMatcher] = None,
        lifecycle_manager: Optional[AgentLifecycleManager] = None,
        llm_client: Optional[OllamaClient] = None  # NOWE!
    ):
        """
        Args:
            templates_dir: Katalog z YAML templates (domyślnie ./templates/)
            capability_matcher: Opcjonalny matcher capabilities
            lifecycle_manager: Opcjonalny lifecycle manager
            llm_client: Opcjonalny LLM client (tworzy nowy jeśli None)
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, AgentTemplate] = {}
        self.capability_matcher = capability_matcher or CapabilityMatcher()
        self.lifecycle_manager = lifecycle_manager or AgentLifecycleManager()
        
        # NOWE: Inicjalizuj LLM client
        self.llm_client = llm_client or OllamaClient(
            config_path=str(Path(__file__).parent.parent / "llm" / "config.yaml")
        )
        
        # Załaduj templates przy inicjalizacji
        self._load_templates()
        
        logger.info(
            f"AgentFactory zainicjalizowana z {len(self.templates)} szablonami"
        )
    
    def _load_templates(self):
        """Załaduj wszystkie YAML templates z katalogu"""
        if not self.templates_dir.exists():
            logger.error(f"Katalog templates nie istnieje: {self.templates_dir}")
            return
        
        yaml_files = list(self.templates_dir.glob("*.yaml")) + \
                     list(self.templates_dir.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                self._load_template(yaml_file)
            except Exception as e:
                logger.error(f"Błąd ładowania {yaml_file.name}: {e}")
        
        logger.info(f"Załadowano {len(self.templates)} szablonów agentów")
    
    def _load_template(self, yaml_path: Path):
        """Załaduj pojedynczy YAML template"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        template = AgentTemplate(
            name=data['name'],
            type=data['type'],
            version=data['version'],
            description=data['description'],
            capabilities=data.get('capabilities', []),
            personality=data.get('personality', {}),
            ai_config=data.get('ai_config', {}),
            protocols=data.get('protocols', {}),
            quality_gates=data.get('quality_gates', []),
            collaboration=data.get('collaboration', {}),
            output_formats=data.get('output_formats', []),
            raw_data=data
        )
        
        self.templates[template.type] = template
        logger.info(f"Załadowano szablon: {template.name} (typ: {template.type})")
    
    def _parse_capabilities(
        self, 
        capabilities_data: List[Dict[str, Any]]
    ) -> List[AgentCapability]:
        """Parsuj capabilities z YAML do obiektów AgentCapability"""
        capabilities = []
        
        for cap_data in capabilities_data:
            # Konwersja string tech na enum TechStack
            technologies = []
            for tech_str in cap_data.get('technologies', []):
                try:
                    tech_enum = TechStack[tech_str.upper()]
                    technologies.append(tech_enum)
                except KeyError:
                    # Jeśli tech nie jest w enum, loguj warning
                    logger.warning(
                        f"Nieznana technologia: {tech_str} - pomijam"
                    )
            
            # Konwersja skill level string na enum
            skill_level_str = cap_data.get('skill_level', 'intermediate').upper()
            skill_level = SkillLevel[skill_level_str]
            
            capability = AgentCapability(
                name=cap_data['name'],
                category=cap_data['category'],
                technologies=technologies,
                skill_level=skill_level,
                description=cap_data['description'],
                requires=cap_data.get('requires', [])
            )
            capabilities.append(capability)
        
        return capabilities
    
    def create_agent(
        self, 
        agent_type: str, 
        agent_id: Optional[str] = None
    ) -> Optional[AgentInstance]:
        """
        Utwórz nową instancję agenta z template
        
        Args:
            agent_type: Typ agenta (np. 'backend', 'frontend')
            agent_id: Opcjonalne ID agenta (generowane jeśli None)
        
        Returns:
            AgentInstance lub None jeśli template nie istnieje
        """
        if agent_type not in self.templates:
            logger.error(f"Szablon agenta '{agent_type}' nie istnieje")
            logger.info(f"Dostępne typy: {list(self.templates.keys())}")
            return None
        
        template = self.templates[agent_type]
        
        # Generuj ID jeśli nie podano
        if agent_id is None:
            agent_id = f"{agent_type}_{len(self.lifecycle_manager.agents) + 1}"
        
        # Utwórz instancję w lifecycle manager
        agent = self.lifecycle_manager.create_agent(agent_id, agent_type)
        
        # NOWE: Przypisz LLM client do agenta
        agent.llm_client = self.llm_client
        agent.template = template
        
        # Parsuj i zarejestruj capabilities
        capabilities = self._parse_capabilities(template.capabilities)
        self.capability_matcher.register_agent_capabilities(agent_id, capabilities)
        
        # Inicjalizuj agenta
        self.lifecycle_manager.transition_state(agent_id, agent.state.INITIALIZING)
        
        # Symulacja inicjalizacji (tutaj można dodać prawdziwą inicjalizację AI)
        logger.info(f"Inicjalizacja agenta {agent_id}...")
        logger.info(f"  Model: {template.ai_config.get('model', 'unknown')}")
        logger.info(f"  Capabilities: {len(capabilities)}")
        
        # Oznacz jako gotowy
        self.lifecycle_manager.transition_state(agent_id, agent.state.READY)
        
        logger.info(f"✅ Agent {agent_id} ({template.name}) gotowy do pracy")
        return agent
    
    def create_team(
        self, 
        team_composition: List[str]
    ) -> Dict[str, AgentInstance]:
        """
        Utwórz zespół agentów
        
        Args:
            team_composition: Lista typów agentów, np. ['architect', 'backend', 'tester']
        
        Returns:
            Słownik {agent_id: AgentInstance}
        """
        team = {}
        
        for agent_type in team_composition:
            agent = self.create_agent(agent_type)
            if agent:
                team[agent.agent_id] = agent
        
        logger.info(f"🏗️  Utworzono zespół z {len(team)} agentów")
        return team
    
    def get_template(self, agent_type: str) -> Optional[AgentTemplate]:
        """Pobierz template agenta"""
        return self.templates.get(agent_type)
    
    def list_available_types(self) -> List[str]:
        """Lista dostępnych typów agentów"""
        return list(self.templates.keys())
    
    def get_agent_info(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Pobierz pełne informacje o typie agenta"""
        template = self.get_template(agent_type)
        if not template:
            return None
        
        return {
            'name': template.name,
            'type': template.type,
            'version': template.version,
            'description': template.description,
            'capabilities': [
                {
                    'name': cap['name'],
                    'category': cap['category'],
                    'skill_level': cap['skill_level']
                }
                for cap in template.capabilities
            ],
            'ai_model': template.ai_config.get('model'),
            'output_formats': template.output_formats,
            'collaborates_with': template.collaboration.get('works_with', [])
        }
    
    def recommend_team_for_project(
        self, 
        required_technologies: List[str],
        project_type: str = "fullstack"
    ) -> List[str]:
        """
        Rekomenduj zespół agentów dla projektu
        
        Args:
            required_technologies: Lista wymaganych technologii
            project_type: Typ projektu (fullstack, backend_only, etc.)
        
        Returns:
            Lista typów agentów do utworzenia
        """
        recommended = []
        
        # Podstawowe role dla różnych typów projektów
        project_teams = {
            'fullstack': ['architect', 'backend', 'frontend', 'database', 'tester', 'devops'],
            'backend_only': ['architect', 'backend', 'database', 'tester', 'devops'],
            'api': ['architect', 'backend', 'database', 'security', 'tester'],
            'microservices': ['architect', 'backend', 'database', 'devops', 'tester', 'security']
        }
        
        base_team = project_teams.get(project_type, project_teams['fullstack'])
        recommended.extend(base_team)
        
        # Dodaj specjalistów based on tech stack
        tech_lower = [t.lower() for t in required_technologies]
        
        if any('performance' in t or 'optimization' in t for t in tech_lower):
            if 'performance' not in recommended:
                recommended.append('performance')
        
        if any('security' in t or 'auth' in t for t in tech_lower):
            if 'security' not in recommended:
                recommended.append('security')
        
        logger.info(
            f"Rekomendowany zespół dla {project_type}: {recommended}"
        )
        return recommended


# Funkcja pomocnicza do szybkiego tworzenia fabryki
def create_factory(templates_dir: Optional[Path] = None) -> AgentFactory:
    """
    Utwórz i zainicjalizuj AgentFactory
    
    Args:
        templates_dir: Opcjonalny katalog z templates
    
    Returns:
        Zainicjalizowana AgentFactory
    """
    return AgentFactory(templates_dir=templates_dir)
