"""
Anti-pattern Detector
Wykrywanie anty-wzorców (złych praktyk) w projektach
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AntiPatternSeverity(Enum):
    """Poziom ważności anty-wzorca"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AntiPatternCategory(Enum):
    """Kategorie anty-wzorców"""
    ARCHITECTURE = "architecture"
    CODE_QUALITY = "code_quality"
    PROCESS = "process"
    TEAM = "team"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class AntiPattern:
    """Anty-wzorzec (zła praktyka)"""
    antipattern_id: str
    name: str
    category: AntiPatternCategory
    severity: AntiPatternSeverity
    description: str
    
    # Wykrywanie
    detected_in: List[str] = field(default_factory=list)  # project_ids
    detection_count: int = 0
    
    # Impact
    impact_description: str = ""
    avg_negative_impact: float = 0.0  # 0.0 - 1.0
    
    # Remediation
    remediation_steps: List[str] = field(default_factory=list)
    estimated_fix_hours: float = 0.0
    
    first_detected: datetime = field(default_factory=datetime.now)
    last_detected: Optional[datetime] = None


class AntiPatternDetector:
    """
    Anti-pattern Detector
    Wykrywa złe praktyki w projektach
    """
    
    def __init__(self):
        self.known_antipatterns: Dict[str, AntiPattern] = {}
        self.detected_antipatterns: Dict[str, AntiPattern] = {}
        self._load_known_antipatterns()
        logger.info("AntiPatternDetector zainicjalizowany")
    
    def _load_known_antipatterns(self):
        """Załaduj znane anty-wzorce"""
        # God Object / Monolith
        self.known_antipatterns['god_object'] = AntiPattern(
            antipattern_id='god_object',
            name='God Object / Monolith',
            category=AntiPatternCategory.ARCHITECTURE,
            severity=AntiPatternSeverity.HIGH,
            description='Single file/module doing too much (>1000 lines)',
            impact_description='Difficult to maintain, test, and scale',
            avg_negative_impact=0.7,
            remediation_steps=[
                'Break into smaller modules',
                'Apply Single Responsibility Principle',
                'Refactor into microservices if appropriate'
            ],
            estimated_fix_hours=16.0
        )
        
        # No Tests
        self.known_antipatterns['no_tests'] = AntiPattern(
            antipattern_id='no_tests',
            name='Missing Test Coverage',
            category=AntiPatternCategory.CODE_QUALITY,
            severity=AntiPatternSeverity.CRITICAL,
            description='Test coverage below 50%',
            impact_description='High bug rate, regression risks',
            avg_negative_impact=0.85,
            remediation_steps=[
                'Write unit tests for critical paths',
                'Add integration tests',
                'Implement CI/CD with test gates'
            ],
            estimated_fix_hours=24.0
        )
        
        # No Code Review
        self.known_antipatterns['no_code_review'] = AntiPattern(
            antipattern_id='no_code_review',
            name='No Code Review Process',
            category=AntiPatternCategory.PROCESS,
            severity=AntiPatternSeverity.HIGH,
            description='Code merged without review',
            impact_description='Quality issues, knowledge silos',
            avg_negative_impact=0.6,
            remediation_steps=[
                'Implement mandatory code review',
                'Set up branch protection rules',
                'Train team on review best practices'
            ],
            estimated_fix_hours=4.0
        )
        
        # Hardcoded Secrets
        self.known_antipatterns['hardcoded_secrets'] = AntiPattern(
            antipattern_id='hardcoded_secrets',
            name='Hardcoded Secrets',
            category=AntiPatternCategory.SECURITY,
            severity=AntiPatternSeverity.CRITICAL,
            description='Secrets/credentials in code',
            impact_description='Security breach risk',
            avg_negative_impact=0.95,
            remediation_steps=[
                'Move secrets to environment variables',
                'Use secret management tool (Vault, AWS Secrets)',
                'Rotate compromised credentials',
                'Scan codebase for secrets'
            ],
            estimated_fix_hours=8.0
        )
        
        # Copy-Paste Programming
        self.known_antipatterns['copy_paste'] = AntiPattern(
            antipattern_id='copy_paste',
            name='Copy-Paste Programming',
            category=AntiPatternCategory.CODE_QUALITY,
            severity=AntiPatternSeverity.MEDIUM,
            description='High code duplication (>20%)',
            impact_description='Difficult maintenance, inconsistent behavior',
            avg_negative_impact=0.5,
            remediation_steps=[
                'Extract common code to functions/classes',
                'Apply DRY principle',
                'Refactor duplicated logic'
            ],
            estimated_fix_hours=12.0
        )
        
        # Big Ball of Mud
        self.known_antipatterns['big_ball_mud'] = AntiPattern(
            antipattern_id='big_ball_mud',
            name='Big Ball of Mud',
            category=AntiPatternCategory.ARCHITECTURE,
            severity=AntiPatternSeverity.HIGH,
            description='No clear architecture, spaghetti dependencies',
            impact_description='Unmaintainable, hard to understand',
            avg_negative_impact=0.8,
            remediation_steps=[
                'Document current architecture',
                'Define clean architecture layers',
                'Gradual refactoring with dependency injection'
            ],
            estimated_fix_hours=40.0
        )
        
        # Golden Hammer
        self.known_antipatterns['golden_hammer'] = AntiPattern(
            antipattern_id='golden_hammer',
            name='Golden Hammer',
            category=AntiPatternCategory.ARCHITECTURE,
            severity=AntiPatternSeverity.MEDIUM,
            description='Using same tool/tech for every problem',
            impact_description='Suboptimal solutions, tech debt',
            avg_negative_impact=0.4,
            remediation_steps=[
                'Evaluate alternatives for each use case',
                'Use right tool for the job',
                'Conduct POCs before committing'
            ],
            estimated_fix_hours=8.0
        )
        
        logger.info(f"Załadowano {len(self.known_antipatterns)} znanych anty-wzorców")
    
    def detect_in_project(
        self,
        project_id: str,
        project_data: Dict[str, Any]
    ) -> List[AntiPattern]:
        """
        Wykryj anty-wzorce w projekcie
        
        Args:
            project_id: ID projektu
            project_data: Dane projektu
        
        Returns:
            Lista wykrytych anty-wzorców
        """
        detected = []
        
        # No Tests
        test_coverage = project_data.get('test_coverage', 1.0)
        if test_coverage < 0.5:
            ap = self._clone_antipattern('no_tests', project_id)
            detected.append(ap)
        
        # No Code Review
        code_reviews = project_data.get('code_reviews_count', 0)
        commits = project_data.get('commits_count', 1)
        review_rate = code_reviews / max(1, commits)
        if review_rate < 0.3:
            ap = self._clone_antipattern('no_code_review', project_id)
            detected.append(ap)
        
        # Hardcoded Secrets
        secrets_found = project_data.get('secrets_in_code', 0)
        if secrets_found > 0:
            ap = self._clone_antipattern('hardcoded_secrets', project_id)
            detected.append(ap)
        
        # Copy-Paste
        code_duplication = project_data.get('code_duplication', 0.0)
        if code_duplication > 0.2:
            ap = self._clone_antipattern('copy_paste', project_id)
            detected.append(ap)
        
        # God Object
        max_file_lines = project_data.get('max_file_lines', 0)
        if max_file_lines > 1000:
            ap = self._clone_antipattern('god_object', project_id)
            detected.append(ap)
        
        # Big Ball of Mud
        cyclomatic_complexity = project_data.get('avg_cyclomatic_complexity', 0)
        if cyclomatic_complexity > 15:
            ap = self._clone_antipattern('big_ball_mud', project_id)
            detected.append(ap)
        
        # Golden Hammer
        tech_diversity = project_data.get('tech_diversity_score', 1.0)
        if tech_diversity < 0.3:
            ap = self._clone_antipattern('golden_hammer', project_id)
            detected.append(ap)
        
        if detected:
            logger.warning(
                f"Wykryto {len(detected)} anty-wzorców w projekcie {project_id}"
            )
            for ap in detected:
                logger.warning(f"  - {ap.name} ({ap.severity.value})")
        
        return detected
    
    def _clone_antipattern(self, antipattern_id: str, project_id: str) -> AntiPattern:
        """Sklonuj znany anty-wzorzec z dodaniem project_id"""
        original = self.known_antipatterns[antipattern_id]
        
        clone = AntiPattern(
            antipattern_id=f"{antipattern_id}_{project_id}",
            name=original.name,
            category=original.category,
            severity=original.severity,
            description=original.description,
            detected_in=[project_id],
            detection_count=1,
            impact_description=original.impact_description,
            avg_negative_impact=original.avg_negative_impact,
            remediation_steps=original.remediation_steps.copy(),
            estimated_fix_hours=original.estimated_fix_hours,
            last_detected=datetime.now()
        )
        
        # Zapisz
        self.detected_antipatterns[clone.antipattern_id] = clone
        
        return clone
    
    def get_critical_antipatterns(self) -> List[AntiPattern]:
        """Pobierz krytyczne anty-wzorce"""
        return [
            ap for ap in self.detected_antipatterns.values()
            if ap.severity == AntiPatternSeverity.CRITICAL
        ]
    
    def get_by_category(self, category: AntiPatternCategory) -> List[AntiPattern]:
        """Pobierz anty-wzorce po kategorii"""
        return [
            ap for ap in self.detected_antipatterns.values()
            if ap.category == category
        ]
    
    def get_remediation_plan(
        self,
        antipattern_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Utwórz plan remediacji anty-wzorców
        
        Args:
            antipattern_ids: Lista IDs anty-wzorców do naprawienia
        
        Returns:
            Plan remediacji
        """
        antipatterns = [
            self.detected_antipatterns[ap_id]
            for ap_id in antipattern_ids
            if ap_id in self.detected_antipatterns
        ]
        
        # Sortuj po severity
        severity_order = {
            AntiPatternSeverity.CRITICAL: 4,
            AntiPatternSeverity.HIGH: 3,
            AntiPatternSeverity.MEDIUM: 2,
            AntiPatternSeverity.LOW: 1
        }
        antipatterns.sort(key=lambda ap: severity_order[ap.severity], reverse=True)
        
        total_hours = sum(ap.estimated_fix_hours for ap in antipatterns)
        
        plan = {
            'total_antipatterns': len(antipatterns),
            'estimated_hours': total_hours,
            'estimated_days': total_hours / 8,
            'priority_order': [
                {
                    'antipattern_id': ap.antipattern_id,
                    'name': ap.name,
                    'severity': ap.severity.value,
                    'estimated_hours': ap.estimated_fix_hours,
                    'steps': ap.remediation_steps
                }
                for ap in antipatterns
            ]
        }
        
        return plan


def create_antipattern_detector() -> AntiPatternDetector:
    """Utwórz AntiPatternDetector"""
    return AntiPatternDetector()
