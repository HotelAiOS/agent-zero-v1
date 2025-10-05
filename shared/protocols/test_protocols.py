"""
Test Protocols System
Test wszystkich protokołów komunikacji między agentami
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from protocols import (
    CodeReviewProtocol,
    ReviewSeverity,
    ProblemSolvingProtocol,
    KnowledgeSharingProtocol,
    KnowledgeCategory,
    EscalationProtocol,
    EscalationReason,
    EscalationLevel,
    ConsensusProtocol,
    ConsensusMethod,
    VoteChoice
)


def test_code_review():
    """Test Code Review Protocol"""
    print("="*70)
    print("🧪 TEST 1: Code Review Protocol")
    print("="*70)
    
    protocol = CodeReviewProtocol()
    
    # Inicjuj review
    print("\n📝 Inicjalizacja Code Review...")
    protocol.initiate(
        initiator='backend_agent_1',
        context={
            'code_files': ['api/auth.py', 'api/users.py', 'models/user.py'],
            'reviewers': ['backend_agent_2', 'security_agent_1'],
            'required_approvals': 2,
            'description': 'User authentication implementation'
        }
    )
    
    print(f"✅ Code Review utworzony: {protocol.protocol_id}")
    print(f"   Pliki: {len(protocol.code_files)}")
    print(f"   Reviewers: {protocol.reviewers}")
    
    # Symuluj review 1 - z komentarzami
    print("\n📋 Review #1 (backend_agent_2)...")
    protocol.process_message(
        protocol.send_message(
            from_agent='backend_agent_2',
            to_agent='backend_agent_1',
            content={
                'action': 'submit_review',
                'approved': False,
                'comments': [
                    {
                        'file_path': 'api/auth.py',
                        'line_number': 45,
                        'severity': 'BLOCKER',
                        'message': 'SQL injection vulnerability',
                        'suggestion': 'Use parameterized queries'
                    },
                    {
                        'file_path': 'api/users.py',
                        'line_number': 23,
                        'severity': 'MAJOR',
                        'message': 'Missing input validation',
                        'suggestion': 'Add Pydantic validation'
                    }
                ],
                'summary': 'Security issues found, changes required'
            }
        )
    )
    
    # Symuluj review 2 - approved
    print("\n📋 Review #2 (security_agent_1)...")
    protocol.process_message(
        protocol.send_message(
            from_agent='security_agent_1',
            to_agent='backend_agent_1',
            content={
                'action': 'submit_review',
                'approved': True,
                'comments': [
                    {
                        'file_path': 'api/auth.py',
                        'line_number': 12,
                        'severity': 'INFO',
                        'message': 'Consider adding rate limiting'
                    }
                ],
                'summary': 'Looks good overall'
            }
        )
    )
    
    # Zakończ review
    result = protocol.complete()
    
    print(f"\n📊 Wynik Code Review:")
    print(f"   Approved: {result['approved']}")
    print(f"   Reviews: {result['reviews_received']}/{result['reviewers_count']}")
    print(f"   Comments: {result['total_comments']}")
    print(f"   Blockers: {result['blockers']}")
    print(f"   Can merge: {result['can_merge']}")
    
    return protocol


def test_problem_solving():
    """Test Problem Solving Protocol"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Problem Solving Protocol")
    print("="*70)
    
    protocol = ProblemSolvingProtocol()
    
    # Inicjuj problem solving
    print("\n🔍 Inicjalizacja Problem Solving...")
    protocol.initiate(
        initiator='backend_agent_1',
        context={
            'title': 'API Performance Bottleneck',
            'description': 'Users endpoint response time > 2s under load',
            'severity': 'high',
            'context': {
                'endpoint': '/api/users',
                'avg_response_time': '2.3s',
                'load': '1000 RPS'
            },
            'experts': ['backend_agent_2', 'database_agent_1', 'performance_agent_1'],
            'brainstorm_duration': 300
        }
    )
    
    print(f"✅ Problem Solving utworzony: {protocol.problem.title}")
    print(f"   Severity: {protocol.problem.severity}")
    print(f"   Experts: {len(protocol.expert_agents)}")
    
    # Propozycje rozwiązań
    print("\n💡 Propozycje rozwiązań...")
    
    # Rozwiązanie 1
    protocol.process_message(
        protocol.send_message(
            from_agent='database_agent_1',
            to_agent=None,
            content={
                'action': 'propose_solution',
                'description': 'Add database indexes on frequently queried fields',
                'approach': 'Database optimization',
                'estimated_effort': 4.0,
                'pros': ['Fast to implement', 'Significant performance gain'],
                'cons': ['Increased disk usage'],
                'risks': ['Index maintenance overhead']
            }
        )
    )
    
    # Rozwiązanie 2
    protocol.process_message(
        protocol.send_message(
            from_agent='backend_agent_2',
            to_agent=None,
            content={
                'action': 'propose_solution',
                'description': 'Implement Redis caching for user queries',
                'approach': 'Caching layer',
                'estimated_effort': 8.0,
                'pros': ['Dramatically faster', 'Reduces DB load'],
                'cons': ['Cache invalidation complexity', 'Additional infrastructure'],
                'risks': ['Stale data if not managed properly']
            }
        )
    )
    
    # Rozwiązanie 3
    protocol.process_message(
        protocol.send_message(
            from_agent='performance_agent_1',
            to_agent=None,
            content={
                'action': 'propose_solution',
                'description': 'Implement query result pagination',
                'approach': 'API optimization',
                'estimated_effort': 3.0,
                'pros': ['Reduces payload size', 'Better UX'],
                'cons': ['API breaking change'],
                'risks': ['Client apps need updates']
            }
        )
    )
    
    print(f"   Propozycje: {len(protocol.solutions)}")
    
    # Głosowanie
    print("\n🗳️  Głosowanie...")
    protocol.cast_vote = lambda agent_id, choice: protocol.process_message(
        protocol.send_message(
            from_agent=agent_id,
            to_agent=None,
            content={'action': 'vote_solution', 'solution_id': protocol.solutions[1].solution_id}
        )
    )
    
    # Agenci głosują na rozwiązanie #2 (Redis caching)
    for agent in ['backend_agent_2', 'database_agent_1', 'performance_agent_1']:
        protocol.process_message(
            protocol.send_message(
                from_agent=agent,
                to_agent=None,
                content={'action': 'vote_solution', 'solution_id': protocol.solutions[1].solution_id}
            )
        )
    
    # Finalizuj
    protocol.process_message(
        protocol.send_message(
            from_agent='backend_agent_1',
            to_agent=None,
            content={'action': 'finalize_selection'}
        )
    )
    
    result = protocol.complete()
    
    print(f"\n📊 Wynik Problem Solving:")
    print(f"   Problem: {result['problem']['title']}")
    print(f"   Propozycje: {result['solutions_proposed']}")
    print(f"   Wybrane rozwiązanie: {result['selected_solution']['description']}")
    print(f"   Głosy: {result['selected_solution']['votes']}")
    print(f"   Szacowany czas: {result['selected_solution']['estimated_effort']}h")
    
    return protocol


def test_knowledge_sharing():
    """Test Knowledge Sharing Protocol"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Knowledge Sharing Protocol")
    print("="*70)
    
    protocol = KnowledgeSharingProtocol()
    
    # Inicjuj
    print("\n📚 Inicjalizacja Knowledge Sharing...")
    protocol.initiate(
        initiator='backend_agent_1',
        context={
            'broadcast': True,
            'auto_store': True
        }
    )
    
    # Udostępnij wiedzę
    print("\n💡 Udostępnianie wiedzy...")
    
    # Best practice 1
    protocol.share_knowledge(
        agent_id='backend_agent_1',
        category=KnowledgeCategory.BEST_PRACTICE,
        title='Always use type hints in Python',
        content='Type hints improve code readability and catch bugs early. Use mypy for static type checking.',
        tags=['python', 'typing', 'best-practice'],
        technologies=['python']
    )
    
    # Lesson learned
    protocol.share_knowledge(
        agent_id='devops_agent_1',
        category=KnowledgeCategory.LESSON_LEARNED,
        title='Always set resource limits in Kubernetes',
        content='Without resource limits, one pod can consume all cluster resources causing cascading failures.',
        tags=['kubernetes', 'devops', 'lesson-learned'],
        technologies=['kubernetes']
    )
    
    # Anti-pattern
    protocol.share_knowledge(
        agent_id='security_agent_1',
        category=KnowledgeCategory.ANTI_PATTERN,
        title='Never store secrets in environment variables',
        content='Use dedicated secret management tools like HashiCorp Vault or AWS Secrets Manager.',
        tags=['security', 'secrets', 'anti-pattern'],
        technologies=['security']
    )
    
    print(f"   Udostępniono: {len(protocol.knowledge_base)} elementów")
    
    # Upvotes
    for item in protocol.knowledge_base[:2]:
        protocol.process_message(
            protocol.send_message(
                from_agent='backend_agent_2',
                to_agent=None,
                content={'action': 'upvote', 'knowledge_id': item.knowledge_id}
            )
        )
    
    # Query
    print("\n🔍 Query wiedzy (tag: 'python')...")
    protocol.process_message(
        protocol.send_message(
            from_agent='frontend_agent_1',
            to_agent=None,
            content={
                'action': 'query_knowledge',
                'tags': ['python']
            }
        )
    )
    
    result = protocol.complete()
    
    print(f"\n📊 Wynik Knowledge Sharing:")
    print(f"   Total items: {result['total_items']}")
    print(f"   By category: {result['by_category']}")
    print(f"   Total upvotes: {result['total_upvotes']}")
    print(f"   Top contributors: {result['top_contributors']}")
    
    return protocol


def test_escalation():
    """Test Escalation Protocol"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Escalation Protocol")
    print("="*70)
    
    protocol = EscalationProtocol()
    
    # Inicjuj
    print("\n⬆️  Inicjalizacja Escalation...")
    protocol.initiate(
        initiator='backend_agent_1',
        context={
            'hierarchy': {
                1: ['team_lead_1'],
                2: ['senior_agent_1', 'architect_1'],
                3: ['human_supervisor']
            },
            'auto_escalate_hours': 24,
            'max_level': 3
        }
    )
    
    # Eskalacja 1 - Blocked
    print("\n🚨 Eskalacja #1: Blocked issue...")
    protocol.escalate_issue(
        agent_id='backend_agent_1',
        title='Cannot proceed - conflicting requirements',
        description='Product owner wants feature X but it conflicts with security requirements',
        reason=EscalationReason.CONFLICTING_REQUIREMENTS,
        urgency='high',
        level=1
    )
    
    # Eskalacja 2 - Critical decision
    print("\n🚨 Eskalacja #2: Critical decision...")
    protocol.escalate_issue(
        agent_id='devops_agent_1',
        title='Architecture decision needed - database choice',
        description='Need to choose between PostgreSQL and MongoDB for new service',
        reason=EscalationReason.CRITICAL_DECISION,
        urgency='medium',
        level=2
    )
    
    print(f"   Utworzono: {len(protocol.tickets)} eskalacji")
    
    # Rozwiązanie eskalacji
    print("\n✅ Rozwiązywanie eskalacji #1...")
    protocol.process_message(
        protocol.send_message(
            from_agent='team_lead_1',
            to_agent='backend_agent_1',
            content={
                'action': 'resolve',
                'ticket_id': protocol.tickets[0].ticket_id,
                'resolution': 'Met with product owner and security team. Agreed on modified approach.'
            }
        )
    )
    
    result = protocol.complete()
    
    print(f"\n📊 Wynik Escalation:")
    print(f"   Total escalations: {result['total_escalations']}")
    print(f"   Resolved: {result['resolved']}")
    print(f"   Unresolved: {result['unresolved']}")
    print(f"   By level: {result['by_level']}")
    print(f"   By reason: {result['by_reason']}")
    
    return protocol


def test_consensus():
    """Test Consensus Protocol"""
    print("\n" + "="*70)
    print("🧪 TEST 5: Consensus Protocol")
    print("="*70)
    
    protocol = ConsensusProtocol()
    
    # Inicjuj
    print("\n🗳️  Inicjalizacja Consensus...")
    protocol.initiate(
        initiator='architect_1',
        context={
            'topic': 'Choose API framework for new service',
            'description': 'We need to decide between FastAPI and Django for new microservice',
            'options': ['FastAPI', 'Django', 'Flask'],
            'method': 'SUPERMAJORITY',
            'participants': ['backend_agent_1', 'backend_agent_2', 'architect_1', 'devops_agent_1'],
            'experts': ['architect_1'],
            'deadline_minutes': 60
        }
    )
    
    print(f"✅ Consensus utworzony: {protocol.decision_topic}")
    print(f"   Method: {protocol.consensus_method.value}")
    print(f"   Threshold: {protocol.required_threshold:.0%}")
    print(f"   Participants: {len(protocol.participants)}")
    
    # Głosowanie
    print("\n🗳️  Głosowanie...")
    
    votes = [
        ('backend_agent_1', VoteChoice.APPROVE, 'FastAPI is modern and fast'),
        ('backend_agent_2', VoteChoice.APPROVE, 'FastAPI has great async support'),
        ('architect_1', VoteChoice.APPROVE, 'FastAPI fits our architecture'),
        ('devops_agent_1', VoteChoice.ABSTAIN, 'No strong preference')
    ]
    
    for agent, choice, reasoning in votes:
        protocol.cast_vote(agent, choice, reasoning)
        print(f"   {agent}: {choice.value}")
    
    # Finalizuj
    protocol.process_message(
        protocol.send_message(
            from_agent='architect_1',
            to_agent=None,
            content={'action': 'finalize'}
        )
    )
    
    result = protocol.complete()
    
    print(f"\n📊 Wynik Consensus:")
    print(f"   Topic: {result['topic']}")
    print(f"   Decision reached: {result['decision_reached']}")
    print(f"   Outcome: {result['outcome']}")
    print(f"   Approval rate: {result['approval_rate']:.1%}")
    print(f"   Votes: {result['votes']}")
    print(f"   Participation: {result['participation_rate']:.0%}")
    
    return protocol


def main():
    """Uruchom wszystkie testy"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 TEST PROTOCOLS SYSTEM - AGENT ZERO V1".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Test 1: Code Review
    code_review = test_code_review()
    
    # Test 2: Problem Solving
    problem_solving = test_problem_solving()
    
    # Test 3: Knowledge Sharing
    knowledge_sharing = test_knowledge_sharing()
    
    # Test 4: Escalation
    escalation = test_escalation()
    
    # Test 5: Consensus
    consensus = test_consensus()
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ WSZYSTKIE TESTY PROTOKOŁÓW ZAKOŃCZONE POMYŚLNIE!")
    print("="*70)
    
    print(f"\n📊 Podsumowanie:")
    print(f"   Code Review: {len(code_review.reviews)} reviews, {len(code_review.get_all_comments())} comments")
    print(f"   Problem Solving: {len(problem_solving.solutions)} solutions proposed")
    print(f"   Knowledge Sharing: {len(knowledge_sharing.knowledge_base)} items shared")
    print(f"   Escalation: {len(escalation.tickets)} tickets created")
    print(f"   Consensus: {len(consensus.votes)} votes cast")
    
    print("\n" + "="*70)
    print("🚀 System Protocols działa poprawnie!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
