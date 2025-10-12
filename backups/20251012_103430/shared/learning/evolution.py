"""
Agent Evolution Engine
Automatyczna ewolucja promptów agentów na podstawie wyników projektów
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Wynik ewolucji agenta"""
    agent_type: str
    version_from: str
    version_to: str
    changes: List[str]
    rationale: str
    expected_improvement: str
    evolved_at: datetime = field(default_factory=datetime.now)


class AgentEvolutionEngine:
    """
    Agent Evolution Engine
    Automatycznie ewoluuje prompty agentów na podstawie:
    - Wzorców sukcesu/porażki
    - Feedback z projektów
    - Lessons learned
    """
    
    def __init__(self):
        self.evolution_history: Dict[str, List[EvolutionResult]] = {}
        logger.info("AgentEvolutionEngine zainicjalizowany")
    
    def evolve_agent(
        self,
        agent_type: str,
        current_prompt: str,
        performance_data: Dict[str, Any],
        lessons_learned: List[str],
        success_patterns: List[Any] = None,
        failure_patterns: List[Any] = None
    ) -> EvolutionResult:
        """
        Ewoluuj agenta na podstawie danych
        
        Args:
            agent_type: Typ agenta (backend, frontend, etc.)
            current_prompt: Obecny prompt
            performance_data: Dane wydajności agenta
            lessons_learned: Wnioski z projektów
            success_patterns: Wzorce sukcesu
            failure_patterns: Wzorce porażki
        
        Returns:
            EvolutionResult
        """
        logger.info(f"Ewolucja agenta: {agent_type}")
        
        changes = []
        rationale_parts = []
        
        # Analiza performance
        success_rate = performance_data.get('success_rate', 0.0)
        avg_quality = performance_data.get('avg_quality_score', 0.0)
        common_issues = performance_data.get('common_issues', [])
        
        # Jeśli niska success rate
        if success_rate < 0.7:
            changes.append("Enhanced error handling and edge case consideration")
            rationale_parts.append(f"Success rate {success_rate:.0%} is below target")
        
        # Jeśli niska jakość
        if avg_quality < 0.75:
            changes.append("Added explicit quality standards and best practices")
            rationale_parts.append(f"Quality score {avg_quality:.2f} needs improvement")
        
        # Analiza common issues
        if common_issues:
            for issue in common_issues[:3]:
                changes.append(f"Addressed common issue: {issue}")
            rationale_parts.append(f"Addressing {len(common_issues)} recurring issues")
        
        # Lessons learned integration
        if lessons_learned:
            relevant_lessons = [l for l in lessons_learned if agent_type.lower() in l.lower()]
            for lesson in relevant_lessons[:2]:
                changes.append(f"Integrated lesson: {lesson}")
            rationale_parts.append(f"Applied {len(relevant_lessons)} relevant lessons")
        
        # Success patterns
        if success_patterns:
            for pattern in success_patterns[:2]:
                if hasattr(pattern, 'description'):
                    changes.append(f"Incorporated success pattern: {pattern.description}")
            rationale_parts.append(f"Applied {len(success_patterns)} success patterns")
        
        # Failure patterns (avoid)
        if failure_patterns:
            for pattern in failure_patterns[:2]:
                if hasattr(pattern, 'description'):
                    changes.append(f"Added safeguard against: {pattern.description}")
            rationale_parts.append(f"Added safeguards for {len(failure_patterns)} failure patterns")
        
        # Określ wersję
        current_version = performance_data.get('version', '1.0.0')
        new_version = self._increment_version(current_version, changes)
        
        # Expected improvement
        if len(changes) >= 5:
            expected_improvement = "significant"
        elif len(changes) >= 3:
            expected_improvement = "moderate"
        else:
            expected_improvement = "minor"
        
        result = EvolutionResult(
            agent_type=agent_type,
            version_from=current_version,
            version_to=new_version,
            changes=changes,
            rationale=" | ".join(rationale_parts) if rationale_parts else "Routine update",
            expected_improvement=expected_improvement
        )
        
        # Zapisz historię
        if agent_type not in self.evolution_history:
            self.evolution_history[agent_type] = []
        self.evolution_history[agent_type].append(result)
        
        logger.info(
            f"Agent {agent_type} evolved: {current_version} → {new_version} "
            f"({len(changes)} changes, {expected_improvement} improvement)"
        )
        
        return result
    
    def _increment_version(self, version: str, changes: List[str]) -> str:
        """Inkrementuj wersję (semantic versioning)"""
        parts = version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Major: breaking changes lub >10 zmian
        if len(changes) > 10:
            major += 1
            minor = 0
            patch = 0
        # Minor: feature additions 3-10 zmian
        elif len(changes) >= 3:
            minor += 1
            patch = 0
        # Patch: bug fixes <3 zmiany
        else:
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def generate_updated_prompt(
        self,
        agent_type: str,
        current_prompt: str,
        evolution_result: EvolutionResult
    ) -> str:
        """
        Wygeneruj zaktualizowany prompt (symulacja - w produkcji użyj LLM)
        
        Args:
            agent_type: Typ agenta
            current_prompt: Obecny prompt
            evolution_result: Wynik ewolucji
        
        Returns:
            Nowy prompt
        """
        # W rzeczywistości użyj LLM do generowania
        # Tutaj prosta symulacja
        
        updated_prompt = current_prompt + "\n\n# Evolution Updates\n"
        updated_prompt += f"# Version: {evolution_result.version_to}\n"
        updated_prompt += f"# Changes:\n"
        
        for change in evolution_result.changes:
            updated_prompt += f"# - {change}\n"
        
        updated_prompt += f"\n# Apply these improvements in your work:\n"
        for i, change in enumerate(evolution_result.changes[:5], 1):
            updated_prompt += f"{i}. {change}\n"
        
        return updated_prompt
    
    def get_evolution_history(self, agent_type: str) -> List[EvolutionResult]:
        """Pobierz historię ewolucji agenta"""
        return self.evolution_history.get(agent_type, [])
    
    def get_current_version(self, agent_type: str) -> str:
        """Pobierz obecną wersję agenta"""
        history = self.evolution_history.get(agent_type, [])
        if history:
            return history[-1].version_to
        return "1.0.0"
    
    def should_evolve(
        self,
        agent_type: str,
        performance_data: Dict[str, Any],
        min_projects: int = 5
    ) -> bool:
        """
        Czy agent powinien ewoluować?
        
        Args:
            agent_type: Typ agenta
            performance_data: Dane wydajności
            min_projects: Minimum projektów do ewolucji
        
        Returns:
            True jeśli powinien ewoluować
        """
        projects_count = performance_data.get('projects_count', 0)
        success_rate = performance_data.get('success_rate', 1.0)
        avg_quality = performance_data.get('avg_quality_score', 1.0)
        
        # Minimum projektów
        if projects_count < min_projects:
            return False
        
        # Niska performance
        if success_rate < 0.7 or avg_quality < 0.75:
            return True
        
        # Regularnie co 10 projektów
        if projects_count % 10 == 0:
            return True
        
        return False
    
    def bulk_evolve(
        self,
        agents_data: Dict[str, Dict[str, Any]],
        global_lessons: List[str] = None
    ) -> Dict[str, EvolutionResult]:
        """
        Ewoluuj wiele agentów jednocześnie
        
        Args:
            agents_data: {agent_type: {prompt, performance_data, lessons}}
            global_lessons: Globalne lekcje dla wszystkich agentów
        
        Returns:
            Dict {agent_type: EvolutionResult}
        """
        results = {}
        
        for agent_type, data in agents_data.items():
            if self.should_evolve(agent_type, data.get('performance_data', {})):
                result = self.evolve_agent(
                    agent_type=agent_type,
                    current_prompt=data.get('prompt', ''),
                    performance_data=data.get('performance_data', {}),
                    lessons_learned=(data.get('lessons', []) + (global_lessons or [])),
                    success_patterns=data.get('success_patterns'),
                    failure_patterns=data.get('failure_patterns')
                )
                results[agent_type] = result
        
        logger.info(f"Bulk evolution: {len(results)} agentów ewoluowało")
        return results
    
    def rollback(self, agent_type: str, versions_back: int = 1) -> Optional[str]:
        """
        Rollback agenta do poprzedniej wersji
        
        Args:
            agent_type: Typ agenta
            versions_back: Ile wersji wstecz
        
        Returns:
            Wersja do której rollback
        """
        history = self.evolution_history.get(agent_type, [])
        
        if len(history) < versions_back + 1:
            logger.warning(f"Cannot rollback {agent_type} {versions_back} versions")
            return None
        
        target_version = history[-(versions_back + 1)].version_to
        
        logger.info(f"Rolled back {agent_type} to version {target_version}")
        return target_version


def create_evolution_engine() -> AgentEvolutionEngine:
    """Utwórz AgentEvolutionEngine"""
    return AgentEvolutionEngine()
