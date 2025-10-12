"""
Prompt Builder
Build specialized prompts for different agent types
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Context for prompt building"""
    agent_type: str
    task_name: str
    task_description: str
    tech_stack: List[str]
    requirements: List[str]
    context: Dict[str, Any] = None


class PromptBuilder:
    """
    Build optimized prompts for specialized agents
    """
    
    # System prompts per agent type
    SYSTEM_PROMPTS = {
        'architect': """Jesteś doświadczonym architektem systemów. 
Twoim zadaniem jest projektowanie skalowalnych, maintainable architektur.
Zawsze myśl o: scalability, maintainability, security, performance.
Używaj proven patterns i best practices.""",
        
        'backend': """Jesteś ekspertem backend developmentu.
Piszesz czysty, testowalny kod zgodny z SOLID principles.
Zawsze: type hints, docstrings, error handling, logging.
Focus na: clean code, performance, security.""",
        
        'frontend': """Jesteś ekspertem frontend/UI.
Projektujesz intuitive, accessible interfaces.
Używasz: React best practices, responsive design, accessibility.
Focus na: UX, performance, accessibility.""",
        
        'database': """Jesteś ekspertem baz danych.
Projektujesz zoptymalizowane schematy i queries.
Myślisz o: normalization, indexing, performance, scaling.
Używasz: PostgreSQL/MySQL best practices.""",
        
        'devops': """Jesteś ekspertem DevOps/Infrastructure.
Automatyzujesz deployment i infrastructure.
Używasz: Docker, Kubernetes, CI/CD, IaC.
Focus na: automation, reliability, monitoring.""",
        
        'security': """Jesteś ekspertem security.
Znajdźwasz vulnerabilities i security issues.
Sprawdzasz: OWASP Top 10, secure coding, auth/authz.
Focus na: zero-trust, defense-in-depth.""",
        
        'tester': """Jesteś ekspertem QA/Testing.
Projektujesz comprehensive test suites.
Używasz: unit tests, integration tests, e2e tests.
Focus na: coverage, edge cases, reliability.""",
        
        'performance': """Jesteś ekspertem performance optimization.
Znajdujesz bottlenecks i optimization opportunities.
Analizujesz: time complexity, memory usage, I/O.
Focus na: profiling, benchmarking, optimization."""
    }
    
    @staticmethod
    def build_task_prompt(context: PromptContext) -> List[Dict[str, str]]:
        """
        Build task execution prompt
        
        Args:
            context: Prompt context
        
        Returns:
            List of messages for chat
        """
        system_prompt = PromptBuilder.SYSTEM_PROMPTS.get(
            context.agent_type,
            "Jesteś doświadczonym software developerem."
        )
        
        user_prompt = f"""# Zadanie: {context.task_name}

## Opis:
{context.task_description}

## Tech Stack:
{', '.join(context.tech_stack)}

## Wymagania:
{chr(10).join(f'- {req}' for req in context.requirements)}

## Twoje zadanie:
Wykonaj to zadanie zgodnie z najlepszymi praktykami.
Wygeneruj kompletny, production-ready kod.
Dołącz komentarze i dokumentację.

## Output format:
Rozpocznij implementację:"""
        
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    
    @staticmethod
    def build_code_review_prompt(code: str, language: str = "python") -> List[Dict[str, str]]:
        """Build code review prompt"""
        return [
            {
                'role': 'system',
                'content': """Jesteś doświadczonym code reviewerem.
Analizujesz kod pod kątem: correctness, performance, security, maintainability.
Zgłaszasz: bugs, code smells, security issues, performance problems."""
            },
            {
                'role': 'user',
                'content': f"""Przeanalizuj poniższy kod {language}:



Znajdź:
1. Błędy logiczne
2. Security issues
3. Performance problems
4. Code quality issues
5. Sugestie ulepszeń

Format:
**Severity**: Critical/High/Medium/Low
**Issue**: Description
**Location**: Line numbers
**Fix**: Suggested fix"""
            }
        ]
    
    @staticmethod
    def build_problem_solving_prompt(problem: str, context: str = "") -> List[Dict[str, str]]:
        """Build problem solving prompt"""
        return [
            {
                'role': 'system',
                'content': """Jesteś ekspertem problem solving.
Używasz structured thinking do analizy problemów.
Rozważasz multiple perspectives i trade-offs."""
            },
            {
                'role': 'user',
                'content': f"""# Problem:
{problem}

{f'# Context:{chr(10)}{context}' if context else ''}

Przeanalizuj problem metodą:
1. Define - dokładnie zdefiniuj problem
2. Brainstorm - wygeneruj 3-5 możliwych rozwiązań
3. Analyze - oceń pros/cons każdego rozwiązania
4. Decide - wybierz najlepsze rozwiązanie i uzasadnij
5. Plan - zaplanuj implementację

Rozpocznij analizę:"""
            }
        ]
    
    @staticmethod
    def build_post_mortem_prompt(project_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build post-mortem analysis prompt"""
        return [
            {
                'role': 'system',
                'content': """Jesteś ekspertem project retrospective analysis.
Analizujesz projekty aby znaleźć lessons learned i improvements."""
            },
            {
                'role': 'user',
                'content': f"""Przeanalizuj zakończony projekt:

Nazwa: {project_data.get('name', 'Unknown')}
Typ: {project_data.get('type', 'Unknown')}
Status: {project_data.get('status', 'Unknown')}
Jakość: {project_data.get('quality', 0):.2f}
Zadania: {project_data.get('tasks_completed', 0)}/{project_data.get('tasks_total', 0)}

Wygeneruj analizę:
1. Co poszło dobrze (3-5 punktów)
2. Co poszło źle (3-5 punktów)
3. Lessons learned (5 kluczowych wniosków)
4. Rekomendacje na przyszłość (3-5 actionable items)

Bądź konkretny i praktyczny."""
            }
        ]
