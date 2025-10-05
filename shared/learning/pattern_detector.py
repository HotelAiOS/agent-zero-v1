"""
Pattern Detector
Wykrywanie wzorców sukcesu i porażki w projektach (Neo4j integration)
"""

from enum import Enum
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Typy wzorców"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    TEAM_PATTERN = "team_pattern"
    TECH_PATTERN = "tech_pattern"
    PROCESS_PATTERN = "process_pattern"
    TIMING_PATTERN = "timing_pattern"


@dataclass
class Pattern:
    """Wzorzec wykryty w projektach"""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str
    confidence: float  # 0.0 - 1.0
    
    # Statystyki
    occurrences: int
    success_rate: float
    projects: List[str] = field(default_factory=list)
    
    # Charakterystyka
    conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Korelacje
    correlated_patterns: List[str] = field(default_factory=list)
    
    discovered_at: datetime = field(default_factory=datetime.now)


class PatternDetector:
    """
    Pattern Detector
    Wykrywa wzorce w zakończonych projektach
    Używa Neo4j do graph-based pattern matching
    """
    
    def __init__(self, neo4j_enabled: bool = False):
        self.neo4j_enabled = neo4j_enabled
        self.patterns: Dict[str, Pattern] = {}
        self.project_graphs: Dict[str, Dict] = {}
        logger.info(f"PatternDetector zainicjalizowany (Neo4j: {neo4j_enabled})")
    
    def detect_patterns(
        self,
        projects_data: List[Dict[str, Any]],
        min_confidence: float = 0.7
    ) -> List[Pattern]:
        """
        Wykryj wzorce w danych projektów
        
        Args:
            projects_data: Lista danych projektów
            min_confidence: Minimalny confidence level
        
        Returns:
            Lista wykrytych wzorców
        """
        logger.info(f"Wykrywanie wzorców w {len(projects_data)} projektach...")
        
        detected_patterns = []
        
        # Wzorce technologiczne
        tech_patterns = self._detect_tech_patterns(projects_data)
        detected_patterns.extend(tech_patterns)
        
        # Wzorce zespołowe
        team_patterns = self._detect_team_patterns(projects_data)
        detected_patterns.extend(team_patterns)
        
        # Wzorce procesowe
        process_patterns = self._detect_process_patterns(projects_data)
        detected_patterns.extend(process_patterns)
        
        # Wzorce czasowe
        timing_patterns = self._detect_timing_patterns(projects_data)
        detected_patterns.extend(timing_patterns)
        
        # Filtruj po confidence
        filtered = [p for p in detected_patterns if p.confidence >= min_confidence]
        
        # Znajdź korelacje
        self._find_correlations(filtered)
        
        # Zapisz wzorce
        for pattern in filtered:
            self.patterns[pattern.pattern_id] = pattern
        
        logger.info(f"Wykryto {len(filtered)} wzorców (min confidence: {min_confidence})")
        
        return filtered
    
    def _detect_tech_patterns(self, projects: List[Dict[str, Any]]) -> List[Pattern]:
        """Wykryj wzorce technologiczne"""
        patterns = []
        
        # Grupuj po tech stack
        tech_groups: Dict[str, List[Dict]] = {}
        for project in projects:
            tech_stack = tuple(sorted(project.get('tech_stack', [])))
            key = str(tech_stack)
            if key not in tech_groups:
                tech_groups[key] = []
            tech_groups[key].append(project)
        
        # Analizuj każdą grupę
        for tech_key, group in tech_groups.items():
            if len(group) < 2:  # Minimum 2 projekty
                continue
            
            successful = sum(1 for p in group if p.get('success', False))
            success_rate = successful / len(group)
            
            if success_rate >= 0.8 or success_rate <= 0.2:
                # Silny wzorzec sukcesu lub porażki
                pattern_type = (PatternType.SUCCESS_PATTERN 
                               if success_rate >= 0.8 
                               else PatternType.FAILURE_PATTERN)
                
                tech_list = eval(tech_key) if tech_key else []
                
                pattern = Pattern(
                    pattern_id=f"tech_{len(patterns)}",
                    pattern_type=pattern_type,
                    name=f"Tech Stack: {', '.join(tech_list[:3])}...",
                    description=f"Projects using this tech stack have {success_rate:.0%} success rate",
                    confidence=min(0.95, 0.5 + (abs(success_rate - 0.5) * 2)),
                    occurrences=len(group),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in group],
                    conditions={'tech_stack': tech_list},
                    outcomes={'success_rate': success_rate}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_team_patterns(self, projects: List[Dict[str, Any]]) -> List[Pattern]:
        """Wykryj wzorce zespołowe"""
        patterns = []
        
        # Analizuj rozmiar zespołu
        small_teams = [p for p in projects if p.get('team_size', 0) <= 3]
        large_teams = [p for p in projects if p.get('team_size', 0) >= 7]
        
        if small_teams:
            success_rate = sum(1 for p in small_teams if p.get('success', False)) / len(small_teams)
            if success_rate >= 0.75:
                patterns.append(Pattern(
                    pattern_id=f"team_size_small",
                    pattern_type=PatternType.TEAM_PATTERN,
                    name="Small Team Success",
                    description=f"Small teams (≤3) show {success_rate:.0%} success rate",
                    confidence=0.7 + (success_rate - 0.75) * 1.2,
                    occurrences=len(small_teams),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in small_teams],
                    conditions={'team_size': '<=3'},
                    outcomes={'success_rate': success_rate}
                ))
        
        if large_teams:
            success_rate = sum(1 for p in large_teams if p.get('success', False)) / len(large_teams)
            if success_rate <= 0.4:
                patterns.append(Pattern(
                    pattern_id=f"team_size_large",
                    pattern_type=PatternType.FAILURE_PATTERN,
                    name="Large Team Challenges",
                    description=f"Large teams (≥7) show only {success_rate:.0%} success rate",
                    confidence=0.6 + (0.5 - success_rate) * 1.0,
                    occurrences=len(large_teams),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in large_teams],
                    conditions={'team_size': '>=7'},
                    outcomes={'success_rate': success_rate}
                ))
        
        return patterns
    
    def _detect_process_patterns(self, projects: List[Dict[str, Any]]) -> List[Pattern]:
        """Wykryj wzorce procesowe"""
        patterns = []
        
        # Analizuj test coverage
        high_coverage = [p for p in projects if p.get('test_coverage', 0) >= 0.8]
        if high_coverage:
            success_rate = sum(1 for p in high_coverage if p.get('success', False)) / len(high_coverage)
            if success_rate >= 0.85:
                patterns.append(Pattern(
                    pattern_id=f"process_testing",
                    pattern_type=PatternType.PROCESS_PATTERN,
                    name="High Test Coverage Success",
                    description=f"Projects with ≥80% test coverage have {success_rate:.0%} success rate",
                    confidence=0.8 + (success_rate - 0.85) * 1.5,
                    occurrences=len(high_coverage),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in high_coverage],
                    conditions={'test_coverage': '>=0.8'},
                    outcomes={'success_rate': success_rate}
                ))
        
        # Analizuj code review
        with_reviews = [p for p in projects if p.get('code_reviews_count', 0) > 0]
        if with_reviews:
            success_rate = sum(1 for p in with_reviews if p.get('success', False)) / len(with_reviews)
            if success_rate >= 0.8:
                patterns.append(Pattern(
                    pattern_id=f"process_code_review",
                    pattern_type=PatternType.PROCESS_PATTERN,
                    name="Code Review Impact",
                    description=f"Projects with code reviews have {success_rate:.0%} success rate",
                    confidence=0.75 + (success_rate - 0.8) * 1.0,
                    occurrences=len(with_reviews),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in with_reviews],
                    conditions={'code_reviews': '>0'},
                    outcomes={'success_rate': success_rate}
                ))
        
        return patterns
    
    def _detect_timing_patterns(self, projects: List[Dict[str, Any]]) -> List[Pattern]:
        """Wykryj wzorce czasowe"""
        patterns = []
        
        # Analizuj overrun
        on_time = [p for p in projects if p.get('duration_variance', 0) <= 1.1]
        delayed = [p for p in projects if p.get('duration_variance', 0) > 1.3]
        
        if on_time:
            success_rate = sum(1 for p in on_time if p.get('success', False)) / len(on_time)
            if success_rate >= 0.85:
                patterns.append(Pattern(
                    pattern_id=f"timing_on_schedule",
                    pattern_type=PatternType.TIMING_PATTERN,
                    name="On-Time Delivery Success",
                    description=f"Projects delivered on time have {success_rate:.0%} success rate",
                    confidence=0.8,
                    occurrences=len(on_time),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in on_time],
                    conditions={'duration_variance': '<=1.1'},
                    outcomes={'success_rate': success_rate}
                ))
        
        if delayed:
            success_rate = sum(1 for p in delayed if p.get('success', False)) / len(delayed)
            if success_rate <= 0.4:
                patterns.append(Pattern(
                    pattern_id=f"timing_delayed",
                    pattern_type=PatternType.FAILURE_PATTERN,
                    name="Delay Impact on Success",
                    description=f"Projects delayed >30% have only {success_rate:.0%} success rate",
                    confidence=0.75,
                    occurrences=len(delayed),
                    success_rate=success_rate,
                    projects=[p.get('project_id', '') for p in delayed],
                    conditions={'duration_variance': '>1.3'},
                    outcomes={'success_rate': success_rate}
                ))
        
        return patterns
    
    def _find_correlations(self, patterns: List[Pattern]):
        """Znajdź korelacje między wzorcami"""
        # Sprawdź które wzorce występują w tych samych projektach
        for i, pattern1 in enumerate(patterns):
            projects1 = set(pattern1.projects)
            
            for pattern2 in patterns[i+1:]:
                projects2 = set(pattern2.projects)
                
                # Oblicz overlap
                overlap = len(projects1 & projects2)
                min_projects = min(len(projects1), len(projects2))
                
                if overlap / min_projects >= 0.5:  # 50% overlap
                    pattern1.correlated_patterns.append(pattern2.pattern_id)
                    pattern2.correlated_patterns.append(pattern1.pattern_id)
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Pobierz wzorzec"""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Pattern]:
        """Pobierz wzorce po typie"""
        return [p for p in self.patterns.values() if p.pattern_type == pattern_type]
    
    def get_success_patterns(self, min_confidence: float = 0.7) -> List[Pattern]:
        """Pobierz wzorce sukcesu"""
        return [
            p for p in self.patterns.values()
            if p.pattern_type == PatternType.SUCCESS_PATTERN
            and p.confidence >= min_confidence
        ]
    
    def get_failure_patterns(self, min_confidence: float = 0.7) -> List[Pattern]:
        """Pobierz wzorce porażki"""
        return [
            p for p in self.patterns.values()
            if p.pattern_type == PatternType.FAILURE_PATTERN
            and p.confidence >= min_confidence
        ]
    
    def recommend_for_project(
        self,
        project_characteristics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rekomendacje dla nowego projektu na podstawie wzorców
        
        Args:
            project_characteristics: Charakterystyka nowego projektu
        
        Returns:
            Lista rekomendacji
        """
        recommendations = []
        
        # Sprawdź które wzorce pasują
        for pattern in self.patterns.values():
            match = True
            for condition_key, condition_value in pattern.conditions.items():
                project_value = project_characteristics.get(condition_key)
                
                # Prosta heurystyka matching
                if isinstance(condition_value, str) and condition_value.startswith('>='):
                    threshold = float(condition_value[2:])
                    if project_value is None or project_value < threshold:
                        match = False
                        break
                elif isinstance(condition_value, str) and condition_value.startswith('<='):
                    threshold = float(condition_value[2:])
                    if project_value is None or project_value > threshold:
                        match = False
                        break
                elif isinstance(condition_value, list):
                    if project_value not in condition_value:
                        match = False
                        break
            
            if match:
                recommendation = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_name': pattern.name,
                    'pattern_type': pattern.pattern_type.value,
                    'success_rate': pattern.success_rate,
                    'confidence': pattern.confidence,
                    'description': pattern.description,
                    'recommendation': (
                        f"Continue this approach (success rate: {pattern.success_rate:.0%})"
                        if pattern.pattern_type == PatternType.SUCCESS_PATTERN
                        else f"Avoid this approach (failure rate: {1-pattern.success_rate:.0%})"
                    )
                }
                recommendations.append(recommendation)
        
        # Sortuj po confidence
        recommendations.sort(key=lambda r: r['confidence'], reverse=True)
        
        return recommendations


def create_pattern_detector(neo4j_enabled: bool = False) -> PatternDetector:
    """Utwórz PatternDetector"""
    return PatternDetector(neo4j_enabled)
