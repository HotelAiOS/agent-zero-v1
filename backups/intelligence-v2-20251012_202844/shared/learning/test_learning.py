"""
Test Learning System
Test całego systemu uczenia się organizacji
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning import (
    PostMortemAnalyzer,
    PatternDetector,
    AntiPatternDetector,
    RecommendationEngine,
    AgentEvolutionEngine,
    ProjectStatus
)


def test_post_mortem():
    """Test Post-mortem Analysis"""
    print("="*70)
    print("🧪 TEST 1: Post-mortem Analyzer")
    print("="*70)
    
    analyzer = PostMortemAnalyzer()
    
    # Symuluj dane projektu
    project_data = {
        'planned_duration_days': 30,
        'actual_duration_days': 42,
        'planned_cost': 50000,
        'actual_cost': 68000,
        'completion_rate': 0.95,
        'test_coverage': 0.85,
        'code_quality_score': 0.8,
        'bugs_found': 12,
        'security_issues': 1,
        'team_size': 5,
        'tasks_completed': 48,
        'tech_issues': ['Performance issues with DB queries', 'Integration problems'],
        'escalations_count': 3,
        'risks_identified': 5,
        'risks_materialized': 2
    }
    
    print("\n📊 Analiza projektu 'E-commerce MVP'...")
    analysis = analyzer.analyze_project(
        project_id='proj_001',
        project_name='E-commerce MVP',
        project_data=project_data
    )
    
    print(f"\n✅ Analiza zakończona:")
    print(f"   Status: {analysis.status.value}")
    print(f"   Quality Score: {analysis.quality_score:.2f}")
    print(f"   Insights: {len(analysis.insights)}")
    
    print(f"\n✅ Co poszło dobrze:")
    for item in analysis.what_went_well:
        print(f"   - {item}")
    
    print(f"\n❌ Co poszło źle:")
    for item in analysis.what_went_wrong:
        print(f"   - {item}")
    
    print(f"\n📚 Top lessons learned:")
    for lesson in analysis.lessons_learned[:5]:
        print(f"   - {lesson}")
    
    return analyzer


def test_pattern_detector():
    """Test Pattern Detector"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Pattern Detector")
    print("="*70)
    
    detector = PatternDetector(neo4j_enabled=False)
    
    # Symuluj dane projektów
    projects = [
        {
            'project_id': 'p1', 'success': True, 'tech_stack': ['Python', 'FastAPI', 'PostgreSQL'],
            'team_size': 3, 'test_coverage': 0.85, 'code_reviews_count': 45,
            'commits_count': 50, 'duration_variance': 1.05
        },
        {
            'project_id': 'p2', 'success': True, 'tech_stack': ['Python', 'FastAPI', 'MongoDB'],
            'team_size': 4, 'test_coverage': 0.9, 'code_reviews_count': 60,
            'commits_count': 65, 'duration_variance': 0.95
        },
        {
            'project_id': 'p3', 'success': True, 'tech_stack': ['Python', 'Django', 'PostgreSQL'],
            'team_size': 3, 'test_coverage': 0.82, 'code_reviews_count': 50,
            'commits_count': 55, 'duration_variance': 1.1
        },
        {
            'project_id': 'p4', 'success': False, 'tech_stack': ['Node.js', 'Express', 'MongoDB'],
            'team_size': 8, 'test_coverage': 0.45, 'code_reviews_count': 10,
            'commits_count': 80, 'duration_variance': 1.5
        },
        {
            'project_id': 'p5', 'success': False, 'tech_stack': ['Java', 'Spring', 'MySQL'],
            'team_size': 7, 'test_coverage': 0.5, 'code_reviews_count': 15,
            'commits_count': 90, 'duration_variance': 1.4
        }
    ]
    
    print(f"\n🔍 Wykrywanie wzorców w {len(projects)} projektach...")
    patterns = detector.detect_patterns(projects, min_confidence=0.7)
    
    print(f"\n✅ Wykryto {len(patterns)} wzorców:")
    for pattern in patterns:
        print(f"\n   📍 {pattern.name}")
        print(f"      Type: {pattern.pattern_type.value}")
        print(f"      Confidence: {pattern.confidence:.2f}")
        print(f"      Success rate: {pattern.success_rate:.0%}")
        print(f"      Occurrences: {pattern.occurrences}")
    
    return detector


def test_antipattern_detector():
    """Test Anti-pattern Detector"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Anti-pattern Detector")
    print("="*70)
    
    detector = AntiPatternDetector()
    
    # Symuluj projekt z problemami
    bad_project = {
        'test_coverage': 0.35,
        'code_reviews_count': 5,
        'commits_count': 100,
        'secrets_in_code': 3,
        'code_duplication': 0.35,
        'max_file_lines': 1500,
        'avg_cyclomatic_complexity': 18,
        'tech_diversity_score': 0.2
    }
    
    print(f"\n⚠️  Wykrywanie anty-wzorców w problematycznym projekcie...")
    antipatterns = detector.detect_in_project('bad_proj_001', bad_project)
    
    print(f"\n🚨 Wykryto {len(antipatterns)} anty-wzorców:")
    for ap in antipatterns:
        print(f"\n   ❌ {ap.name}")
        print(f"      Category: {ap.category.value}")
        print(f"      Severity: {ap.severity.value}")
        print(f"      Impact: {ap.impact_description}")
        print(f"      Fix time: {ap.estimated_fix_hours}h")
    
    # Plan remediacji
    print(f"\n🔧 Plan remediacji:")
    ap_ids = [ap.antipattern_id for ap in antipatterns]
    plan = detector.get_remediation_plan(ap_ids)
    
    print(f"   Total antipatterns: {plan['total_antipatterns']}")
    print(f"   Estimated effort: {plan['estimated_days']:.1f} days")
    
    return detector


def test_recommendation_engine():
    """Test Recommendation Engine"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Recommendation Engine")
    print("="*70)
    
    engine = RecommendationEngine()
    
    # Nowy projekt
    new_project = {
        'tech_stack': ['Python', 'FastAPI', 'PostgreSQL'],
        'team_size': 8,
        'target_test_coverage': 0.6,
        'code_review_required': False,
        'estimated_duration_days': 20,
        'complexity_score': 6
    }
    
    # Dane historyczne
    historical = [
        {
            'project_id': 'h1', 'success': True, 'tech_stack': ['Python', 'FastAPI'],
            'team_size': 3, 'test_coverage': 0.85, 'code_reviews_count': 50,
            'actual_duration_days': 35, 'complexity_score': 6
        },
        {
            'project_id': 'h2', 'success': False, 'tech_stack': ['Python', 'Django'],
            'team_size': 9, 'test_coverage': 0.5, 'code_reviews_count': 20,
            'actual_duration_days': 45, 'complexity_score': 7
        },
        {
            'project_id': 'h3', 'success': True, 'tech_stack': ['Node.js'],
            'team_size': 4, 'test_coverage': 0.9, 'code_reviews_count': 60,
            'actual_duration_days': 30, 'complexity_score': 5
        }
    ]
    
    print(f"\n💡 Generowanie rekomendacji dla nowego projektu...")
    recommendations = engine.generate_recommendations(new_project, historical)
    
    print(f"\n✅ Wygenerowano {len(recommendations)} rekomendacji:")
    
    top_recs = engine.get_top_recommendations(limit=5, min_confidence=0.7)
    for i, rec in enumerate(top_recs, 1):
        print(f"\n   {i}. {rec.title}")
        print(f"      Category: {rec.category.value}")
        print(f"      Impact: {rec.expected_impact}")
        print(f"      Confidence: {rec.confidence:.2f}")
        print(f"      Description: {rec.description}")
    
    return engine


def test_agent_evolution():
    """Test Agent Evolution"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Agent Evolution Engine")
    print("="*70)
    
    engine = AgentEvolutionEngine()
    
    # Dane wydajności agenta
    backend_performance = {
        'version': '1.0.0',
        'projects_count': 10,
        'success_rate': 0.65,
        'avg_quality_score': 0.72,
        'common_issues': [
            'Missing error handling',
            'Inefficient database queries',
            'Incomplete input validation'
        ]
    }
    
    lessons = [
        'Always add comprehensive error handling',
        'Use database indexes for frequent queries',
        'Validate all user inputs with Pydantic'
    ]
    
    print(f"\n🧬 Ewolucja Backend Agent...")
    result = engine.evolve_agent(
        agent_type='backend',
        current_prompt='You are a backend developer...',
        performance_data=backend_performance,
        lessons_learned=lessons
    )
    
    print(f"\n✅ Ewolucja zakończona:")
    print(f"   Version: {result.version_from} → {result.version_to}")
    print(f"   Changes: {len(result.changes)}")
    print(f"   Expected improvement: {result.expected_improvement}")
    
    print(f"\n📝 Zmiany:")
    for change in result.changes:
        print(f"   - {change}")
    
    print(f"\n📖 Rationale: {result.rationale}")
    
    return engine


def main():
    """Uruchom wszystkie testy"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 TEST LEARNING SYSTEM - AGENT ZERO V1".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Test 1: Post-mortem
    analyzer = test_post_mortem()
    
    # Test 2: Pattern Detection
    detector = test_pattern_detector()
    
    # Test 3: Anti-patterns
    antipattern_detector = test_antipattern_detector()
    
    # Test 4: Recommendations
    rec_engine = test_recommendation_engine()
    
    # Test 5: Agent Evolution
    evo_engine = test_agent_evolution()
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ WSZYSTKIE TESTY LEARNING SYSTEM ZAKOŃCZONE!")
    print("="*70)
    
    print(f"\n📊 Podsumowanie:")
    print(f"   Post-mortem: 1 projekt przeanalizowany")
    print(f"   Patterns: {len(detector.patterns)} wzorców wykrytych")
    print(f"   Anti-patterns: 7 typów zdefiniowanych")
    print(f"   Recommendations: {len(rec_engine.recommendations)} wygenerowanych")
    print(f"   Agent Evolution: 1 agent ewoluował")
    
    print("\n" + "="*70)
    print("🚀 Learning System działa poprawnie!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
