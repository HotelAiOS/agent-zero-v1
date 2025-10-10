# tests/test_business_requirements_parser.py
"""
Unit tests for Business Requirements Parser - Final validation
"""

import pytest
from business.requirements_parser import BusinessRequirementsParser, BusinessRequest, ValidationResponse
import json

class TestBusinessRequirementsParser:
    
    @pytest.fixture
    def parser(self):
        return BusinessRequirementsParser()
    
    def test_parse_intent_create(self, parser):
        """Test intent parsing for CREATE operations"""
        request = "Create a user registration API with JWT authentication"
        intent = parser.parse_intent(request)
        assert intent == "CREATE"
    
    def test_parse_intent_update(self, parser):
        """Test intent parsing for UPDATE operations"""  
        request = "Update the existing database schema to include user preferences"
        intent = parser.parse_intent(request)
        assert intent == "UPDATE"
    
    def test_extract_entities_multiple(self, parser):
        """Test entity extraction with multiple entities"""
        request = "Create a user management API with database integration and file upload"
        entities = parser.extract_entities(request)
        expected_entities = ['user', 'api', 'database', 'file']
        assert all(entity in entities for entity in expected_entities)
    
    def test_complexity_assessment_simple(self, parser):
        """Test complexity assessment for simple tasks"""
        request = "Create a simple user login form"
        entities = ['user']
        complexity = parser.assess_complexity(request, entities)
        assert complexity in ["Simple", "Moderate"]
    
    def test_complexity_assessment_enterprise(self, parser):
        """Test complexity assessment for enterprise tasks"""
        request = "Build a scalable distributed microservices architecture with advanced security, authentication, authorization, and complex data processing pipelines"
        entities = ['api', 'database', 'user', 'file']
        complexity = parser.assess_complexity(request, entities)
        assert complexity == "Enterprise"
    
    def test_agent_selection(self, parser):
        """Test agent selection logic"""
        agents = parser.select_agents("CREATE", ["api", "database"], "Complex")
        expected_agents = ['orchestrator', 'code_generator', 'api_specialist', 'data_specialist', 'architect']
        assert all(agent in agents for agent in expected_agents)
    
    def test_cost_estimation(self, parser):
        """Test cost and time estimation"""
        cost, time = parser.estimate_cost_and_time("Moderate", ["orchestrator", "api_specialist"], ["api", "user"])
        assert cost > 0
        assert time > 0
        assert isinstance(cost, float)
        assert isinstance(time, int)
    
    def test_technical_spec_generation(self, parser):
        """Test complete technical specification generation"""
        spec = parser.generate_technical_spec(
            intent="CREATE",
            entities=["api", "database"],
            complexity="Moderate",
            business_request="Create a user management API with database"
        )
        
        assert spec['intent'] == "CREATE"
        assert 'api' in spec['entities']
        assert 'database' in spec['entities']
        assert spec['complexity'] == "Moderate"
        assert len(spec['agents_needed']) > 0
        assert spec['estimated_cost'] > 0
        assert spec['estimated_time_minutes'] > 0
        assert 'api' in spec['technical_requirements']
        assert 'database' in spec['technical_requirements']
        assert spec['confidence_score'] > 0.7
    
    def test_validation_valid_request(self, parser):
        """Test validation of a valid request"""
        validation = parser.validate_request("Create a comprehensive user management system with authentication")
        assert validation.is_valid == True
        assert len(validation.errors) == 0
    
    def test_validation_invalid_request_too_short(self, parser):
        """Test validation of invalid request (too short)"""
        validation = parser.validate_request("Create")
        assert validation.is_valid == False
        assert len(validation.errors) > 0
        assert "too short" in validation.errors[0].lower()
    
    def test_validation_security_suggestion(self, parser):
        """Test validation provides security suggestions"""
        validation = parser.validate_request("Create an authentication system with password handling")
        assert len(validation.suggestions) > 0
        assert any("security" in suggestion.lower() for suggestion in validation.suggestions)

# === INTEGRATION TEST ===
def test_api_endpoint_integration():
    """Test API endpoint integration"""
    from fastapi.testclient import TestClient
    from business.requirements_parser import router
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Test parse endpoint
    response = client.post("/api/business/parse", json={
        "request": "Create a user registration API with email verification",
        "priority": "high",
        "context": {"project": "user_management"}
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data['intent'] == "CREATE"
    assert 'user' in data['entities']
    assert data['estimated_cost'] > 0
    
    # Test validation endpoint
    response = client.post("/api/business/validate", json={
        "request": "Create a user registration API with email verification"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data['is_valid'] == True
    
    # Test health endpoint
    response = client.get("/api/business/health")
    assert response.status_code == 200
    assert response.json()['status'] == "healthy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
