"""
Tests for IntelligentAgentSelector
Week 43 - Priority 1 Component
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from shared.ai.intelligent_agent_selector import (
    IntelligentAgentSelector,
    AgentCapability,
    TaskComplexity,
    AgentSelection,
    create_intelligent_agent_selector
)


@pytest.fixture
def mock_ai_system():
    """Mock ProductionAISystem"""
    ai_system = Mock()
    ai_system.generate = AsyncMock()
    return ai_system


@pytest.fixture
def agent_selector(mock_ai_system):
    """Create IntelligentAgentSelector instance"""
    return IntelligentAgentSelector(
        ai_system=mock_ai_system,
        neo4j_client=None,
        experience_tracker=None,
        enable_learning=False
    )


@pytest.fixture
def sample_agents():
    """Sample agent capabilities"""
    return [
        AgentCapability(
            agent_id="agent_backend_1",
            name="Backend Developer",
            capabilities=["python", "fastapi", "database"],
            performance_score=0.85,
            current_load=2,
            max_load=10,
            cost_per_task=0.01,
            success_rate=0.90,
            avg_latency_ms=500.0,
            specializations=["api_development", "database_design"]
        ),
        AgentCapability(
            agent_id="agent_frontend_1",
            name="Frontend Developer",
            capabilities=["javascript", "react", "css"],
            performance_score=0.80,
            current_load=5,
            max_load=10,
            cost_per_task=0.008,
            success_rate=0.88,
            avg_latency_ms=450.0,
            specializations=["ui_design", "responsive_web"]
        ),
        AgentCapability(
            agent_id="agent_devops_1",
            name="DevOps Engineer",
            capabilities=["docker", "kubernetes", "ci_cd"],
            performance_score=0.92,
            current_load=1,
            max_load=8,
            cost_per_task=0.015,
            success_rate=0.95,
            avg_latency_ms=600.0,
            specializations=["container_orchestration", "deployment_automation"]
        )
    ]


@pytest.mark.asyncio
async def test_register_agent(agent_selector, sample_agents):
    """Test agent registration"""
    agent = sample_agents[0]
    agent_selector.register_agent(agent)
    
    assert agent.agent_id in agent_selector.agents
    assert agent_selector.agents[agent.agent_id] == agent


@pytest.mark.asyncio
async def test_analyze_task_complexity(agent_selector, mock_ai_system):
    """Test task complexity analysis with AI"""
    # Mock AI response
    mock_ai_system.generate.return_value = '''
    {
        "complexity_score": 0.7,
        "required_capabilities": ["python", "fastapi"],
        "estimated_duration_ms": 5000,
        "priority": "high",
        "reasoning": "Complex API development task"
    }
    '''
    
    task_description = "Create RESTful API endpoint for user authentication"
    context = {"project_id": "test_project", "task_id": "task_001"}
    
    complexity = await agent_selector.analyze_task_complexity(task_description, context)
    
    assert complexity.complexity_score == 0.7
    assert "python" in complexity.required_capabilities
    assert "fastapi" in complexity.required_capabilities
    assert complexity.priority == "high"
    assert complexity.estimated_duration_ms == 5000
    
    # Verify AI was called
    mock_ai_system.generate.assert_called_once()


@pytest.mark.asyncio
async def test_select_agent_basic(agent_selector, mock_ai_system, sample_agents):
    """Test basic agent selection"""
    # Register agents
    for agent in sample_agents:
        agent_selector.register_agent(agent)
    
    # Mock AI responses
    mock_ai_system.generate.side_effect = [
        # Task analysis response
        '''
        {
            "complexity_score": 0.6,
            "required_capabilities": ["python", "fastapi"],
            "estimated_duration_ms": 3000,
            "priority": "medium",
            "reasoning": "Standard API development"
        }
        ''',
        # Agent selection response
        '''
        {
            "selected_agent_id": "agent_backend_1",
            "confidence": 0.85,
            "reasoning": "Best match for Python/FastAPI skills with good performance history",
            "alternatives": [
                {"agent_id": "agent_devops_1", "confidence": 0.6}
            ]
        }
        '''
    ]
    
    task_description = "Create API endpoint for user management"
    task = await agent_selector.analyze_task_complexity(task_description)
    
    selection = await agent_selector.select_agent(task)
    
    assert selection.agent_id == "agent_backend_1"
    assert selection.confidence == 0.85
    assert selection.agent_name == "Backend Developer"
    assert len(selection.alternatives) == 1


@pytest.mark.asyncio
async def test_select_agent_with_constraints(agent_selector, sample_agents):
    """Test agent selection with cost constraints"""
    for agent in sample_agents:
        agent_selector.register_agent(agent)
    
    task = TaskComplexity(
        task_id="task_002",
        description="Deploy application to production",
        complexity_score=0.8,
        required_capabilities=["docker", "kubernetes"],
        estimated_duration_ms=10000,
        priority="high"
    )
    
    constraints = {"max_cost": 0.012}  # Exclude DevOps (0.015)
    
    # Should filter out DevOps agent due to cost constraint
    # But we need at least one matching agent
    # This will fallback to heuristic selection
    
    selection = await agent_selector.select_agent(task, constraints)
    
    assert selection.estimated_cost <= constraints["max_cost"]


@pytest.mark.asyncio
async def test_update_agent_load(agent_selector, sample_agents):
    """Test agent load management"""
    agent = sample_agents[0]
    agent_selector.register_agent(agent)
    
    initial_load = agent.current_load
    
    await agent_selector.update_agent_load(agent.agent_id, 3)
    assert agent_selector.agents[agent.agent_id].current_load == initial_load + 3
    
    await agent_selector.update_agent_load(agent.agent_id, -2)
    assert agent_selector.agents[agent.agent_id].current_load == initial_load + 1
    
    # Should not go below 0
    await agent_selector.update_agent_load(agent.agent_id, -100)
    assert agent_selector.agents[agent.agent_id].current_load == 0


@pytest.mark.asyncio
async def test_record_task_outcome(agent_selector, sample_agents):
    """Test learning from task outcomes"""
    agent = sample_agents[0]
    agent_selector.register_agent(agent)
    
    initial_success_rate = agent.success_rate
    initial_latency = agent.avg_latency_ms
    
    # Record successful task with faster execution
    await agent_selector.record_task_outcome(
        task_id="task_003",
        agent_id=agent.agent_id,
        success=True,
        actual_duration_ms=400.0,  # Faster than average
        actual_cost=0.009
    )
    
    updated_agent = agent_selector.agents[agent.agent_id]
    
    # Success rate should increase slightly
    assert updated_agent.success_rate > initial_success_rate
    
    # Average latency should decrease slightly
    assert updated_agent.avg_latency_ms < initial_latency


@pytest.mark.asyncio
async def test_fallback_selection(agent_selector, sample_agents):
    """Test fallback selection when AI fails"""
    for agent in sample_agents:
        agent_selector.register_agent(agent)
    
    task = TaskComplexity(
        task_id="task_004",
        description="Generic development task",
        complexity_score=0.5,
        required_capabilities=["general"],
        estimated_duration_ms=2000,
        priority="low"
    )
    
    # Use fallback directly
    selection = agent_selector._fallback_selection(sample_agents, task)
    
    assert selection.agent_id in [a.agent_id for a in sample_agents]
    assert selection.confidence == 0.6  # Fallback confidence
    assert "fallback" in selection.reasoning.lower()


@pytest.mark.asyncio
async def test_no_available_agents(agent_selector, sample_agents):
    """Test handling when no agents are available"""
    # Register agents but set all to max load
    for agent in sample_agents:
        agent.current_load = agent.max_load
        agent_selector.register_agent(agent)
    
    task = TaskComplexity(
        task_id="task_005",
        description="Urgent task",
        complexity_score=0.9,
        required_capabilities=["python"],
        estimated_duration_ms=5000,
        priority="critical"
    )
    
    with pytest.raises(ValueError, match="No available agents"):
        await agent_selector.select_agent(task)


@pytest.mark.asyncio
async def test_factory_function():
    """Test factory function for creating selector"""
    mock_ai = Mock()
    
    selector = await create_intelligent_agent_selector(
        ai_system=mock_ai,
        enable_learning=False
    )
    
    assert isinstance(selector, IntelligentAgentSelector)
    assert selector.ai_system == mock_ai
    assert selector.neo4j_client is None
    assert not selector.enable_learning


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
