"""
Project Manager
Zarządzanie wykonaniem projektu przez zespół agentów
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectPhase(Enum):
    """Fazy projektu"""
    PLANNING = "planning"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"


@dataclass
class PhaseResult:
    """Wynik fazy projektu"""
    phase: ProjectPhase
    status: str  # success, failed, partial
    deliverables: List[str]
    issues: List[str]
    duration_hours: float
    quality_score: float


class ProjectManager:
    """
    Project Manager
    Zarządza wykonaniem projektu przez agentów
    """
    
    def __init__(self, core_engine: Any):
        self.core = core_engine
        self.current_phase: Dict[str, ProjectPhase] = {}
        self.phase_results: Dict[str, List[PhaseResult]] = {}
        logger.info("ProjectManager zainicjalizowany")
    
    def execute_project(
        self,
        project_execution: Any,
        auto_advance: bool = True
    ) -> Dict[str, Any]:
        """
        Wykonaj projekt przez wszystkie fazy
        
        Args:
            project_execution: ProjectExecution z engine
            auto_advance: Czy automatycznie przejść przez fazy
        
        Returns:
            Wyniki wykonania
        """
        project_id = project_execution.project_id
        logger.info(f"\n{'='*70}")
        logger.info(f"▶️  Rozpoczęcie wykonania: {project_execution.project_name}")
        logger.info(f"{'='*70}")
        
        self.current_phase[project_id] = ProjectPhase.PLANNING
        self.phase_results[project_id] = []
        
        phases = [
            ProjectPhase.PLANNING,
            ProjectPhase.DESIGN,
            ProjectPhase.IMPLEMENTATION,
            ProjectPhase.TESTING,
            ProjectPhase.REVIEW,
            ProjectPhase.DEPLOYMENT
        ]
        
        for phase in phases:
            logger.info(f"\n📍 Faza: {phase.value.upper()}")
            
            result = self._execute_phase(project_execution, phase)
            self.phase_results[project_id].append(result)
            
            logger.info(f"   Status: {result.status}")
            logger.info(f"   Quality: {result.quality_score:.2f}")
            logger.info(f"   Duration: {result.duration_hours:.1f}h")
            
            if result.status == 'failed' and not auto_advance:
                logger.error(f"   ❌ Faza {phase.value} failed!")
                break
            
            project_execution.progress = (phases.index(phase) + 1) / len(phases)
        
        # Podsumowanie
        total_duration = sum(r.duration_hours for r in self.phase_results[project_id])
        avg_quality = sum(r.quality_score for r in self.phase_results[project_id]) / len(self.phase_results[project_id])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Projekt zakończony!")
        logger.info(f"   Total duration: {total_duration:.1f}h ({total_duration/8:.1f} dni)")
        logger.info(f"   Average quality: {avg_quality:.2f}")
        logger.info(f"{'='*70}\n")
        
        return {
            'project_id': project_id,
            'total_duration_hours': total_duration,
            'average_quality': avg_quality,
            'phases_completed': len(self.phase_results[project_id]),
            'phase_results': self.phase_results[project_id]
        }
    
    def _execute_phase(
        self,
        project_execution: Any,
        phase: ProjectPhase
    ) -> PhaseResult:
        """Wykonaj pojedynczą fazę"""
        start_time = datetime.now()
        
        # Symulacja wykonania fazy
        deliverables = []
        issues = []
        
        if phase == ProjectPhase.PLANNING:
            deliverables = ['Project plan', 'Task breakdown', 'Schedule']
            duration = 8.0
            quality = 0.9
            
        elif phase == ProjectPhase.DESIGN:
            deliverables = ['Architecture diagram', 'API specs', 'Database schema']
            duration = 16.0
            quality = 0.85
            
            # Uruchom consensus protocol dla architektury
            protocol = self.core.start_protocol(
                project_execution.project_id,
                'consensus',
                {
                    'topic': 'Architecture decision',
                    'description': 'Choose architecture pattern',
                    'options': ['Microservices', 'Monolith', 'Hybrid'],
                    'method': 'MAJORITY',
                    'participants': list(project_execution.team.keys())[:4]
                }
            )
            
        elif phase == ProjectPhase.IMPLEMENTATION:
            deliverables = ['Source code', 'Unit tests', 'Integration tests']
            duration = 80.0
            quality = 0.8
            
            # Symulacja code reviews
            for i in range(3):
                protocol = self.core.start_protocol(
                    project_execution.project_id,
                    'code_review',
                    {
                        'code_files': [f'module_{i}.py'],
                        'reviewers': list(project_execution.team.keys())[:2],
                        'required_approvals': 1,
                        'description': f'Review module {i}'
                    }
                )
            
        elif phase == ProjectPhase.TESTING:
            deliverables = ['Test results', 'Bug reports', 'Coverage report']
            duration = 24.0
            quality = 0.82
            issues = ['Minor bugs in edge cases']
            
        elif phase == ProjectPhase.REVIEW:
            deliverables = ['Code review summary', 'Security audit', 'Performance report']
            duration = 16.0
            quality = 0.88
            
        elif phase == ProjectPhase.DEPLOYMENT:
            deliverables = ['Deployment scripts', 'Documentation', 'Monitoring setup']
            duration = 12.0
            quality = 0.9
        
        else:
            duration = 8.0
            quality = 0.8
        
        # Zapisz completed tasks
        project_execution.completed_tasks.extend(deliverables)
        
        return PhaseResult(
            phase=phase,
            status='success' if quality >= 0.7 else 'partial',
            deliverables=deliverables,
            issues=issues,
            duration_hours=duration,
            quality_score=quality
        )
    
    def get_phase_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Pobierz status aktualnej fazy"""
        if project_id not in self.current_phase:
            return None
        
        return {
            'current_phase': self.current_phase[project_id].value,
            'completed_phases': len(self.phase_results.get(project_id, [])),
            'results': [
                {
                    'phase': r.phase.value,
                    'status': r.status,
                    'quality': r.quality_score,
                    'duration': r.duration_hours
                }
                for r in self.phase_results.get(project_id, [])
            ]
        }


def create_project_manager(core_engine: Any) -> ProjectManager:
    """Utwórz ProjectManager"""
    return ProjectManager(core_engine)
