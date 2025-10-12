"""
Post-mortem Analyzer
Automatyczna analiza zakończonych projektów
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """Status projektu"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisCategory(Enum):
    """Kategorie analizy"""
    TIMELINE = "timeline"
    QUALITY = "quality"
    TEAM_PERFORMANCE = "team_performance"
    TECHNICAL_DECISIONS = "technical_decisions"
    COMMUNICATION = "communication"
    RISKS = "risks"


@dataclass
class AnalysisInsight:
    """Pojedynczy insight z analizy"""
    insight_id: str
    category: AnalysisCategory
    title: str
    description: str
    impact: str  # positive, negative, neutral
    severity: int  # 1-5
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ProjectAnalysis:
    """Analiza post-mortem projektu"""
    analysis_id: str
    project_id: str
    project_name: str
    status: ProjectStatus
    analyzed_at: datetime
    
    # Metryki
    planned_duration_days: float
    actual_duration_days: float
    planned_cost: float
    actual_cost: float
    quality_score: float  # 0.0 - 1.0
    
    # Insights
    insights: List[AnalysisInsight] = field(default_factory=list)
    
    # Wnioski
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    
    # Success factors
    success_factors: Dict[str, float] = field(default_factory=dict)
    failure_factors: Dict[str, float] = field(default_factory=dict)


class PostMortemAnalyzer:
    """
    Post-mortem Analyzer
    Automatycznie analizuje zakończone projekty
    """
    
    def __init__(self):
        self.analyses: Dict[str, ProjectAnalysis] = {}
        logger.info("PostMortemAnalyzer zainicjalizowany")
    
    def analyze_project(
        self,
        project_id: str,
        project_name: str,
        project_data: Dict[str, Any]
    ) -> ProjectAnalysis:
        """
        Analizuj zakończony projekt
        
        Args:
            project_id: ID projektu
            project_name: Nazwa projektu
            project_data: Dane projektu (tasks, timeline, metrics, etc.)
        
        Returns:
            ProjectAnalysis
        """
        logger.info(f"Rozpoczęto analizę post-mortem projektu: {project_name}")
        
        # Podstawowe metryki
        planned_duration = project_data.get('planned_duration_days', 0)
        actual_duration = project_data.get('actual_duration_days', 0)
        planned_cost = project_data.get('planned_cost', 0)
        actual_cost = project_data.get('actual_cost', 0)
        
        # Określ status projektu
        status = self._determine_project_status(project_data)
        
        # Oblicz quality score
        quality_score = self._calculate_quality_score(project_data)
        
        # Utwórz analizę
        analysis = ProjectAnalysis(
            analysis_id=f"analysis_{project_id}",
            project_id=project_id,
            project_name=project_name,
            status=status,
            analyzed_at=datetime.now(),
            planned_duration_days=planned_duration,
            actual_duration_days=actual_duration,
            planned_cost=planned_cost,
            actual_cost=actual_cost,
            quality_score=quality_score
        )
        
        # Analiza timeline
        self._analyze_timeline(analysis, project_data)
        
        # Analiza jakości
        self._analyze_quality(analysis, project_data)
        
        # Analiza team performance
        self._analyze_team_performance(analysis, project_data)
        
        # Analiza decyzji technicznych
        self._analyze_technical_decisions(analysis, project_data)
        
        # Analiza komunikacji
        self._analyze_communication(analysis, project_data)
        
        # Analiza ryzyk
        self._analyze_risks(analysis, project_data)
        
        # Wygeneruj wnioski
        self._generate_conclusions(analysis)
        
        # Zapisz analizę
        self.analyses[analysis.analysis_id] = analysis
        
        logger.info(
            f"Analiza zakończona: {len(analysis.insights)} insights, "
            f"status: {status.value}, quality: {quality_score:.2f}"
        )
        
        return analysis
    
    def _determine_project_status(self, project_data: Dict[str, Any]) -> ProjectStatus:
        """Określ status projektu"""
        # Prosta heurystyka
        completion = project_data.get('completion_rate', 0.0)
        cancelled = project_data.get('cancelled', False)
        
        if cancelled:
            return ProjectStatus.CANCELLED
        elif completion >= 0.95:
            return ProjectStatus.SUCCESS
        elif completion >= 0.7:
            return ProjectStatus.PARTIAL_SUCCESS
        else:
            return ProjectStatus.FAILED
    
    def _calculate_quality_score(self, project_data: Dict[str, Any]) -> float:
        """Oblicz quality score"""
        # Weighted average różnych metryk
        test_coverage = project_data.get('test_coverage', 0.0)
        code_quality = project_data.get('code_quality_score', 0.8)
        bug_count = project_data.get('bugs_found', 0)
        security_issues = project_data.get('security_issues', 0)
        
        # Prosta formuła
        score = (
            test_coverage * 0.3 +
            code_quality * 0.3 +
            max(0, 1.0 - bug_count / 100) * 0.2 +
            max(0, 1.0 - security_issues / 10) * 0.2
        )
        
        return min(1.0, max(0.0, score))
    
    def _analyze_timeline(self, analysis: ProjectAnalysis, data: Dict[str, Any]):
        """Analiza timeline"""
        planned = analysis.planned_duration_days
        actual = analysis.actual_duration_days
        
        if actual > planned * 1.2:
            # Znaczące opóźnienie
            analysis.insights.append(AnalysisInsight(
                insight_id=f"timeline_{len(analysis.insights)}",
                category=AnalysisCategory.TIMELINE,
                title="Significant Timeline Overrun",
                description=f"Project took {actual:.1f} days vs planned {planned:.1f} days ({(actual/planned - 1)*100:.0f}% overrun)",
                impact="negative",
                severity=4,
                evidence=[f"Planned: {planned}d", f"Actual: {actual}d"],
                recommendations=[
                    "Improve estimation accuracy",
                    "Add buffer time for unknowns",
                    "Break down tasks into smaller units"
                ]
            ))
        elif actual < planned * 0.8:
            # Szybciej niż planowano
            analysis.insights.append(AnalysisInsight(
                insight_id=f"timeline_{len(analysis.insights)}",
                category=AnalysisCategory.TIMELINE,
                title="Ahead of Schedule",
                description=f"Project completed faster than planned ({(1 - actual/planned)*100:.0f}% faster)",
                impact="positive",
                severity=2,
                evidence=[f"Planned: {planned}d", f"Actual: {actual}d"],
                recommendations=["Document efficient practices"]
            ))
    
    def _analyze_quality(self, analysis: ProjectAnalysis, data: Dict[str, Any]):
        """Analiza jakości"""
        if analysis.quality_score >= 0.8:
            analysis.insights.append(AnalysisInsight(
                insight_id=f"quality_{len(analysis.insights)}",
                category=AnalysisCategory.QUALITY,
                title="High Quality Delivery",
                description=f"Quality score: {analysis.quality_score:.2f}",
                impact="positive",
                severity=3,
                evidence=[f"Test coverage: {data.get('test_coverage', 0):.0%}"],
                recommendations=["Share quality practices with other teams"]
            ))
        elif analysis.quality_score < 0.6:
            analysis.insights.append(AnalysisInsight(
                insight_id=f"quality_{len(analysis.insights)}",
                category=AnalysisCategory.QUALITY,
                title="Quality Issues Detected",
                description=f"Quality score below acceptable: {analysis.quality_score:.2f}",
                impact="negative",
                severity=5,
                evidence=[
                    f"Test coverage: {data.get('test_coverage', 0):.0%}",
                    f"Bugs found: {data.get('bugs_found', 0)}"
                ],
                recommendations=[
                    "Increase test coverage requirement",
                    "Add code review checkpoints",
                    "Implement automated quality gates"
                ]
            ))
    
    def _analyze_team_performance(self, analysis: ProjectAnalysis, data: Dict[str, Any]):
        """Analiza team performance"""
        team_size = data.get('team_size', 1)
        tasks_completed = data.get('tasks_completed', 0)
        velocity = tasks_completed / max(1, team_size)
        
        if velocity > 10:
            analysis.insights.append(AnalysisInsight(
                insight_id=f"team_{len(analysis.insights)}",
                category=AnalysisCategory.TEAM_PERFORMANCE,
                title="High Team Velocity",
                description=f"Team completed {tasks_completed} tasks with {team_size} members",
                impact="positive",
                severity=3,
                evidence=[f"Velocity: {velocity:.1f} tasks/member"],
                recommendations=["Document successful team practices"]
            ))
    
    def _analyze_technical_decisions(self, analysis: ProjectAnalysis, data: Dict[str, Any]):
        """Analiza decyzji technicznych"""
        tech_stack = data.get('tech_stack', [])
        tech_issues = data.get('tech_issues', [])
        
        if tech_issues:
            analysis.insights.append(AnalysisInsight(
                insight_id=f"tech_{len(analysis.insights)}",
                category=AnalysisCategory.TECHNICAL_DECISIONS,
                title="Technical Challenges Encountered",
                description=f"{len(tech_issues)} technical issues during project",
                impact="negative",
                severity=3,
                evidence=tech_issues[:3],
                recommendations=[
                    "Review technology choices",
                    "Conduct POCs before committing to new tech"
                ]
            ))
    
    def _analyze_communication(self, analysis: ProjectAnalysis, data: Dict[str, Any]):
        """Analiza komunikacji"""
        protocols_used = data.get('protocols_used', [])
        escalations = data.get('escalations_count', 0)
        
        if escalations > 5:
            analysis.insights.append(AnalysisInsight(
                insight_id=f"comm_{len(analysis.insights)}",
                category=AnalysisCategory.COMMUNICATION,
                title="High Number of Escalations",
                description=f"{escalations} escalations occurred",
                impact="negative",
                severity=4,
                evidence=[f"Escalations: {escalations}"],
                recommendations=[
                    "Improve initial requirements clarity",
                    "Increase team autonomy",
                    "Better risk assessment upfront"
                ]
            ))
    
    def _analyze_risks(self, analysis: ProjectAnalysis, data: Dict[str, Any]):
        """Analiza ryzyk"""
        risks_identified = data.get('risks_identified', 0)
        risks_materialized = data.get('risks_materialized', 0)
        
        if risks_materialized > 0:
            analysis.insights.append(AnalysisInsight(
                insight_id=f"risk_{len(analysis.insights)}",
                category=AnalysisCategory.RISKS,
                title="Risks Materialized",
                description=f"{risks_materialized} out of {risks_identified} risks occurred",
                impact="negative",
                severity=3,
                evidence=[f"Risk realization rate: {risks_materialized/max(1,risks_identified):.0%}"],
                recommendations=[
                    "Improve risk mitigation strategies",
                    "Allocate contingency resources"
                ]
            ))
    
    def _generate_conclusions(self, analysis: ProjectAnalysis):
        """Wygeneruj wnioski końcowe"""
        # What went well
        positive_insights = [i for i in analysis.insights if i.impact == "positive"]
        analysis.what_went_well = [i.title for i in positive_insights]
        
        # What went wrong
        negative_insights = [i for i in analysis.insights if i.impact == "negative"]
        analysis.what_went_wrong = [i.title for i in negative_insights]
        
        # Lessons learned
        analysis.lessons_learned = list(set(
            rec for insight in analysis.insights 
            for rec in insight.recommendations
        ))[:10]
        
        # Action items
        critical_insights = [i for i in analysis.insights if i.severity >= 4]
        analysis.action_items = [
            f"Address: {i.title}"
            for i in critical_insights
        ]
    
    def get_analysis(self, analysis_id: str) -> Optional[ProjectAnalysis]:
        """Pobierz analizę"""
        return self.analyses.get(analysis_id)
    
    def get_summary(self, analysis_id: str) -> Dict[str, Any]:
        """Pobierz podsumowanie analizy"""
        analysis = self.analyses.get(analysis_id)
        if not analysis:
            return {}
        
        return {
            'project_name': analysis.project_name,
            'status': analysis.status.value,
            'quality_score': analysis.quality_score,
            'timeline_performance': {
                'planned_days': analysis.planned_duration_days,
                'actual_days': analysis.actual_duration_days,
                'variance': analysis.actual_duration_days - analysis.planned_duration_days
            },
            'cost_performance': {
                'planned': analysis.planned_cost,
                'actual': analysis.actual_cost,
                'variance': analysis.actual_cost - analysis.planned_cost
            },
            'insights_count': len(analysis.insights),
            'positive_insights': len([i for i in analysis.insights if i.impact == "positive"]),
            'negative_insights': len([i for i in analysis.insights if i.impact == "negative"]),
            'top_lessons': analysis.lessons_learned[:5],
            'action_items': analysis.action_items
        }


def create_post_mortem_analyzer() -> PostMortemAnalyzer:
    """Utwórz PostMortemAnalyzer"""
    return PostMortemAnalyzer()
