"""
Recommendation Engine
AI-powered rekomendacje dla projektów na podstawie historii
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationCategory(Enum):
    """Kategorie rekomendacji"""
    TECHNOLOGY = "technology"
    PROCESS = "process"
    TEAM = "team"
    ARCHITECTURE = "architecture"
    QUALITY = "quality"
    TIMELINE = "timeline"


@dataclass
class Recommendation:
    """Rekomendacja"""
    recommendation_id: str
    category: RecommendationCategory
    title: str
    description: str
    rationale: str
    
    # Confidence & Impact
    confidence: float  # 0.0 - 1.0
    expected_impact: str  # high, medium, low
    
    # Implementation
    action_items: List[str] = field(default_factory=list)
    estimated_effort_hours: float = 0.0
    
    # Evidence
    based_on_projects: List[str] = field(default_factory=list)
    success_rate_with: float = 0.0
    success_rate_without: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)


class RecommendationEngine:
    """
    Recommendation Engine
    Generuje rekomendacje na podstawie historycznych danych
    """
    
    def __init__(self):
        self.recommendations: Dict[str, Recommendation] = {}
        logger.info("RecommendationEngine zainicjalizowany")
    
    def generate_recommendations(
        self,
        project_plan: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        patterns: List[Any] = None,
        antipatterns: List[Any] = None
    ) -> List[Recommendation]:
        """
        Generuj rekomendacje dla nowego projektu
        
        Args:
            project_plan: Plan nowego projektu
            historical_data: Dane historyczne projektów
            patterns: Wykryte wzorce sukcesu
            antipatterns: Wykryte anty-wzorce
        
        Returns:
            Lista rekomendacji
        """
        recommendations = []
        
        # Rekomendacje technologiczne
        tech_recs = self._recommend_technology(project_plan, historical_data)
        recommendations.extend(tech_recs)
        
        # Rekomendacje procesowe
        process_recs = self._recommend_process(project_plan, historical_data)
        recommendations.extend(process_recs)
        
        # Rekomendacje zespołowe
        team_recs = self._recommend_team_structure(project_plan, historical_data)
        recommendations.extend(team_recs)
        
        # Rekomendacje jakościowe
        quality_recs = self._recommend_quality_practices(project_plan, historical_data)
        recommendations.extend(quality_recs)
        
        # Rekomendacje timeline
        timeline_recs = self._recommend_timeline_adjustments(project_plan, historical_data)
        recommendations.extend(timeline_recs)
        
        # Zapisz
        for rec in recommendations:
            self.recommendations[rec.recommendation_id] = rec
        
        logger.info(f"Wygenerowano {len(recommendations)} rekomendacji")
        
        return recommendations
    
    def _recommend_technology(
        self,
        project_plan: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Rekomendacje technologiczne"""
        recommendations = []
        
        # Sprawdź planowany tech stack
        planned_tech = set(project_plan.get('tech_stack', []))
        
        # Znajdź podobne projekty
        similar_projects = [
            p for p in historical_data
            if len(set(p.get('tech_stack', [])) & planned_tech) >= 2
        ]
        
        if similar_projects:
            successful = [p for p in similar_projects if p.get('success', False)]
            success_rate = len(successful) / len(similar_projects)
            
            if success_rate < 0.6:
                # Niska success rate z tym tech stackiem
                recommendations.append(Recommendation(
                    recommendation_id=f"tech_warning_{len(recommendations)}",
                    category=RecommendationCategory.TECHNOLOGY,
                    title="Consider Alternative Technology Stack",
                    description=f"Similar projects with this tech stack have only {success_rate:.0%} success rate",
                    rationale=f"Based on {len(similar_projects)} historical projects",
                    confidence=0.7,
                    expected_impact="high",
                    action_items=[
                        "Review technology choices",
                        "Consider proven alternatives",
                        "Conduct POC with risky technologies"
                    ],
                    estimated_effort_hours=16.0,
                    based_on_projects=[p.get('project_id', '') for p in similar_projects],
                    success_rate_with=success_rate
                ))
        
        return recommendations
    
    def _recommend_process(
        self,
        project_plan: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Rekomendacje procesowe"""
        recommendations = []
        
        # Test coverage
        planned_coverage = project_plan.get('target_test_coverage', 0.0)
        if planned_coverage < 0.8:
            high_coverage_projects = [
                p for p in historical_data
                if p.get('test_coverage', 0) >= 0.8
            ]
            
            if high_coverage_projects:
                success_rate = sum(
                    1 for p in high_coverage_projects if p.get('success', False)
                ) / len(high_coverage_projects)
                
                if success_rate >= 0.85:
                    recommendations.append(Recommendation(
                        recommendation_id=f"process_testing_{len(recommendations)}",
                        category=RecommendationCategory.PROCESS,
                        title="Increase Test Coverage Target to 80%",
                        description=f"Projects with ≥80% coverage have {success_rate:.0%} success rate",
                        rationale=f"Based on {len(high_coverage_projects)} projects",
                        confidence=0.85,
                        expected_impact="high",
                        action_items=[
                            "Set test coverage requirement to 80%",
                            "Add coverage checks to CI/CD",
                            "Write tests alongside features"
                        ],
                        estimated_effort_hours=0.0,  # Part of development
                        based_on_projects=[p.get('project_id', '') for p in high_coverage_projects],
                        success_rate_with=success_rate
                    ))
        
        return recommendations
    
    def _recommend_team_structure(
        self,
        project_plan: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Rekomendacje zespołowe"""
        recommendations = []
        
        planned_team_size = project_plan.get('team_size', 0)
        
        if planned_team_size > 7:
            large_teams = [p for p in historical_data if p.get('team_size', 0) > 7]
            if large_teams:
                success_rate = sum(1 for p in large_teams if p.get('success', False)) / len(large_teams)
                
                if success_rate < 0.5:
                    recommendations.append(Recommendation(
                        recommendation_id=f"team_size_{len(recommendations)}",
                        category=RecommendationCategory.TEAM,
                        title="Consider Smaller Team or Sub-teams",
                        description=f"Large teams (>7) have only {success_rate:.0%} success rate",
                        rationale=f"Based on {len(large_teams)} large team projects",
                        confidence=0.75,
                        expected_impact="medium",
                        action_items=[
                            "Split into 2-3 person sub-teams",
                            "Define clear interfaces between teams",
                            "Assign team leads for coordination"
                        ],
                        estimated_effort_hours=8.0,
                        based_on_projects=[p.get('project_id', '') for p in large_teams],
                        success_rate_with=success_rate
                    ))
        
        return recommendations
    
    def _recommend_quality_practices(
        self,
        project_plan: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Rekomendacje jakościowe"""
        recommendations = []
        
        # Code review
        if not project_plan.get('code_review_required', False):
            with_reviews = [p for p in historical_data if p.get('code_reviews_count', 0) > 0]
            if with_reviews:
                success_rate = sum(1 for p in with_reviews if p.get('success', False)) / len(with_reviews)
                
                if success_rate >= 0.8:
                    recommendations.append(Recommendation(
                        recommendation_id=f"quality_review_{len(recommendations)}",
                        category=RecommendationCategory.QUALITY,
                        title="Implement Mandatory Code Review",
                        description=f"Projects with code reviews have {success_rate:.0%} success rate",
                        rationale=f"Based on {len(with_reviews)} projects",
                        confidence=0.8,
                        expected_impact="high",
                        action_items=[
                            "Enable branch protection rules",
                            "Require 1+ approvals before merge",
                            "Train team on review best practices"
                        ],
                        estimated_effort_hours=4.0,
                        based_on_projects=[p.get('project_id', '') for p in with_reviews],
                        success_rate_with=success_rate
                    ))
        
        return recommendations
    
    def _recommend_timeline_adjustments(
        self,
        project_plan: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Rekomendacje timeline"""
        recommendations = []
        
        planned_duration = project_plan.get('estimated_duration_days', 0)
        
        # Znajdź podobne projekty po złożoności
        complexity_score = project_plan.get('complexity_score', 5)
        similar = [
            p for p in historical_data
            if abs(p.get('complexity_score', 5) - complexity_score) <= 2
        ]
        
        if similar:
            avg_actual = sum(p.get('actual_duration_days', 0) for p in similar) / len(similar)
            
            if planned_duration < avg_actual * 0.8:
                recommendations.append(Recommendation(
                    recommendation_id=f"timeline_buffer_{len(recommendations)}",
                    category=RecommendationCategory.TIMELINE,
                    title="Add Timeline Buffer",
                    description=f"Similar projects took average {avg_actual:.1f} days, you planned {planned_duration:.1f}",
                    rationale=f"Based on {len(similar)} similar projects",
                    confidence=0.7,
                    expected_impact="medium",
                    action_items=[
                        f"Increase timeline to {avg_actual * 1.1:.1f} days",
                        "Add buffer for unknowns (10-20%)",
                        "Plan for contingencies"
                    ],
                    estimated_effort_hours=0.0,
                    based_on_projects=[p.get('project_id', '') for p in similar]
                ))
        
        return recommendations
    
    def get_top_recommendations(
        self,
        limit: int = 5,
        min_confidence: float = 0.7
    ) -> List[Recommendation]:
        """Pobierz top rekomendacje"""
        filtered = [
            r for r in self.recommendations.values()
            if r.confidence >= min_confidence
        ]
        
        # Sortuj po confidence i expected_impact
        impact_score = {'high': 3, 'medium': 2, 'low': 1}
        filtered.sort(
            key=lambda r: (impact_score.get(r.expected_impact, 0), r.confidence),
            reverse=True
        )
        
        return filtered[:limit]


def create_recommendation_engine() -> RecommendationEngine:
    """Utwórz RecommendationEngine"""
    return RecommendationEngine()
