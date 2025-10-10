import pytest
import asyncio
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json

# Add shared module to path for Agent Zero V1 microservices architecture
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "services" / "agent-orchestrator" / "src"))

# Import from agent-orchestrator service
from business.requirements_parser import (
    BusinessRequirementsParser, 
    router,
    BusinessRequest,
    TechnicalSpec,
    ValidationResponse
)

class TestBusinessRequirementsParser:
    """Unit tests for Business Requirements Parser core functionality"""

    @pytest.fixture
    def parser(self):
        return BusinessRequirementsParser()

    def test_intent_parsing_create(self, parser):
        """Test CREATE intent detection"""
        requests = [
            "Create a user registration API",
            "Build a new dashboard",
            "Develop authentication system"
        ]

        for request in requests:
            intent = parser.parse_intent(request)
            assert intent == "CREATE", f"Failed for: {request}"

    def test_intent_parsing_update(self, parser):
        """Test UPDATE intent detection"""
        requests = [
            "Update the existing database schema",
            "Modify user preferences",
            "Enhance the current API"
        ]

        for request in requests:
            intent = parser.parse_intent(request)
            assert intent == "UPDATE", f"Failed for: {request}"

    def test_entity_extraction_comprehensive(self, parser):
        """Test comprehensive entity extraction"""
        request = "Create a user management API with database integration and reporting dashboard"
        entities = parser.extract_entities(request)

        expected_entities = ['user', 'api', 'database', 'report']
        for entity in expected_entities:
            assert entity in entities, f"Missing entity: {entity}"

    def test_complexity_assessment_simple(self, parser):
        """Test simple complexity assessment"""
        request = "Create a simple contact form"
        entities = ['user', 'ui']
        complexity = parser.assess_complexity(request, entities)

        assert complexity in ["Simple", "Moderate"], f"Expected Simple/Moderate, got {complexity}"

    def test_complexity_assessment_enterprise(self, parser):
        """Test enterprise complexity assessment"""
        request = """Build a scalable, distributed microservices architecture 
                    with advanced security, real-time analytics, and complex 
                    data processing pipelines for enterprise customers"""
        entities = ['api', 'database', 'user', 'report', 'auth']
        complexity = parser.assess_complexity(request, entities)

        assert complexity == "Enterprise", f"Expected Enterprise, got {complexity}"

    def test_agent_selection_logic(self, parser):
        """Test agent selection based on requirements"""
        # Test API + Database task
        agents = parser.select_agents("CREATE", ["api", "database"], "Complex")

        expected_agents = ['orchestrator', 'code_generator', 'api_specialist', 
                          'database_specialist', 'solution_architect']

        for agent in expected_agents:
            assert agent in agents, f"Missing agent: {agent}"

    def test_cost_time_estimation(self, parser):
        """Test cost and time estimation accuracy"""
        # Simple task
        cost_simple, time_simple = parser.estimate_cost_and_time("Simple", ["orchestrator"], ["api"])
        assert 0.01 <= cost_simple <= 0.10, f"Simple cost out of range: {cost_simple}"
        assert 5 <= time_simple <= 30, f"Simple time out of range: {time_simple}"

        # Enterprise task
        cost_enterprise, time_enterprise = parser.estimate_cost_and_time(
            "Enterprise", 
            ["orchestrator", "solution_architect", "security_specialist"], 
            ["api", "database", "auth", "ui"]
        )
        assert cost_enterprise > cost_simple, "Enterprise should cost more than Simple"
        assert time_enterprise > time_simple, "Enterprise should take longer than Simple"

    def test_technical_spec_generation_comprehensive(self, parser):
        """Test complete technical specification generation"""
        spec = parser.generate_technical_spec(
            intent="CREATE",
            entities=["api", "database", "auth"],
            complexity="Complex",
            business_request="Create a secure user management API with database"
        )

        # Validate required fields
        assert spec['intent'] == "CREATE"
        assert 'api' in spec['entities']
        assert 'database' in spec['entities']
        assert 'auth' in spec['entities']
        assert spec['complexity'] == "Complex"
        assert len(spec['agents_needed']) >= 3
        assert spec['estimated_cost'] > 0
        assert spec['estimated_time_minutes'] > 0
        assert spec['confidence_score'] > 0.7

        # Validate technical requirements
        assert 'api' in spec['technical_requirements']
        assert 'database' in spec['technical_requirements']
        assert 'security' in spec['technical_requirements']

        # Check API requirements
        api_req = spec['technical_requirements']['api']
        assert api_req['type'] == 'REST'
        assert api_req['authentication'] == 'JWT'
        assert api_req['documentation'] == 'OpenAPI/Swagger'

    def test_validation_valid_request(self, parser):
        """Test validation of valid business request"""
        request = "Create a comprehensive user management system with authentication and reporting"
        validation = parser.validate_request(request)

        assert validation['is_valid'] == True
        assert len(validation['errors']) == 0
        assert validation['confidence'] > 0.7

    def test_validation_invalid_request_too_short(self, parser):
        """Test validation of too short request"""
        request = "Create"
        validation = parser.validate_request(request)

        assert validation['is_valid'] == False
        assert len(validation['errors']) > 0
        assert any("too short" in error.lower() for error in validation['errors'])

    def test_validation_security_suggestions(self, parser):
        """Test security suggestions in validation"""
        request = "Create user authentication system with password handling"
        validation = parser.validate_request(request)

        # Should suggest security considerations
        assert len(validation['suggestions']) > 0
        security_suggested = any("security" in suggestion.lower() 
                                for suggestion in validation['suggestions'])
        assert security_suggested, "Should suggest security requirements"

    def test_sanitization(self, parser):
        """Test input sanitization"""
        dangerous_input = '<script>alert("xss")</script>Create an API with "quotes" and <tags>'
        sanitized = parser.sanitize_input(dangerous_input)

        # Should remove dangerous content
        assert '<script>' not in sanitized
        assert 'alert(' not in sanitized
        assert '<tags>' not in sanitized
        assert '"quotes"' not in sanitized

        # Should preserve safe content
        assert 'Create an API' in sanitized


class TestBusinessParserAPI:
    """Integration tests for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_parse_endpoint_success(self, client):
        """Test successful parsing via API endpoint"""
        response = client.post("/api/business/parse", json={
            "request": "Create a user registration API with JWT authentication and email verification",
            "priority": "high",
            "context": {"project": "user_management", "team": "backend"}
        })

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert data['intent'] == "CREATE"
        assert 'user' in data['entities']
        assert 'api' in data['entities']
        assert 'auth' in data['entities']
        assert data['complexity'] in ["Moderate", "Complex"]
        assert data['estimated_cost'] > 0
        assert data['estimated_time_minutes'] > 0
        assert data['confidence_score'] > 0.7
        assert len(data['agents_needed']) >= 3

    def test_parse_endpoint_invalid_request(self, client):
        """Test parsing endpoint with invalid input"""
        response = client.post("/api/business/parse", json={
            "request": "API",  # Too short
            "priority": "high"
        })

        assert response.status_code == 400
        error_detail = response.json()['detail']
        assert error_detail['message'] == "Invalid business request"
        assert len(error_detail['errors']) > 0

    def test_validate_endpoint_success(self, client):
        """Test validation endpoint with valid input"""
        response = client.post("/api/business/validate", json={
            "request": "Create a comprehensive user management system with authentication"
        })

        assert response.status_code == 200
        data = response.json()
        assert data['is_valid'] == True
        assert data['confidence'] > 0.7

    def test_validate_endpoint_with_suggestions(self, client):
        """Test validation endpoint provides helpful suggestions"""
        response = client.post("/api/business/validate", json={
            "request": "Create authentication with passwords"
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data['suggestions']) > 0

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/business/health")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == "healthy"
        assert data['service'] == "business_requirements_parser"
        assert data['version'] == "1.0.0"
        assert 'components' in data

    def test_capabilities_endpoint(self, client):
        """Test capabilities endpoint"""
        response = client.get("/api/business/capabilities")

        assert response.status_code == 200
        data = response.json()
        assert 'supported_intents' in data
        assert 'supported_entities' in data
        assert 'complexity_levels' in data
        assert 'features' in data

        # Validate capabilities content
        assert "CREATE" in data['supported_intents']
        assert "api" in data['supported_entities']
        assert "Simple" in data['complexity_levels']
        assert data['features']['validation'] == True


class TestBusinessParserIntegration:
    """End-to-end integration tests"""

    @pytest.fixture
    def parser(self):
        return BusinessRequirementsParser()

    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_complete_workflow_simple_task(self, client):
        """Test complete workflow for simple task"""
        # Step 1: Validate request
        validate_response = client.post("/api/business/validate", json={
            "request": "Create a simple contact form with email sending"
        })
        assert validate_response.status_code == 200
        assert validate_response.json()['is_valid'] == True

        # Step 2: Parse requirements
        parse_response = client.post("/api/business/parse", json={
            "request": "Create a simple contact form with email sending",
            "priority": "medium"
        })
        assert parse_response.status_code == 200

        spec = parse_response.json()
        assert spec['intent'] == "CREATE"
        assert spec['complexity'] in ["Simple", "Moderate"]
        assert spec['estimated_cost'] < 0.20  # Should be relatively cheap
        assert spec['estimated_time_minutes'] < 60  # Should be quick

    def test_complete_workflow_complex_task(self, client):
        """Test complete workflow for complex enterprise task"""
        complex_request = """Build a scalable e-commerce platform with user authentication, 
                           payment processing, inventory management, real-time analytics, 
                           mobile API, and admin dashboard with role-based access control"""

        # Step 1: Validate
        validate_response = client.post("/api/business/validate", json={
            "request": complex_request
        })
        assert validate_response.status_code == 200

        # Step 2: Parse
        parse_response = client.post("/api/business/parse", json={
            "request": complex_request,
            "priority": "critical",
            "context": {"budget": "enterprise", "timeline": "6_months"}
        })
        assert parse_response.status_code == 200

        spec = parse_response.json()
        assert spec['intent'] == "CREATE"  # or "BUILD"
        assert spec['complexity'] in ["Complex", "Enterprise"]
        assert len(spec['entities']) >= 4  # Multiple components
        assert len(spec['agents_needed']) >= 5  # Multiple specialists
        assert spec['estimated_cost'] > 0.30  # Enterprise pricing
        assert spec['estimated_time_minutes'] > 120  # Significant time

        # Validate technical requirements
        tech_req = spec['technical_requirements']
        assert 'api' in tech_req
        assert 'database' in tech_req
        assert 'security' in tech_req

    def test_error_handling_robustness(self, client):
        """Test system robustness with edge cases"""
        edge_cases = [
            "",  # Empty request
            "   ",  # Whitespace only
            "a" * 6000,  # Too long request
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
        ]

        for edge_case in edge_cases:
            response = client.post("/api/business/validate", json={
                "request": edge_case
            })

            # Should handle gracefully (either 200 with errors or 400)
            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                data = response.json()
                if edge_case.strip() == "" or len(edge_case.strip()) < 10:
                    assert data['is_valid'] == False

    def test_performance_benchmark(self, parser):
        """Test performance benchmarks"""
        import time

        test_request = "Create a user management API with database integration and authentication"

        # Benchmark parsing speed
        start_time = time.time()
        for _ in range(10):
            spec = parser.generate_technical_spec(
                intent=parser.parse_intent(test_request),
                entities=parser.extract_entities(test_request),
                complexity=parser.assess_complexity(test_request, []),
                business_request=test_request
            )
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.1, f"Parsing too slow: {avg_time:.3f}s average"

        # Benchmark validation speed
        start_time = time.time()
        for _ in range(10):
            validation = parser.validate_request(test_request)
        end_time = time.time()

        avg_validation_time = (end_time - start_time) / 10
        assert avg_validation_time < 0.05, f"Validation too slow: {avg_validation_time:.3f}s average"


# === PYTEST CONFIGURATION ===
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
