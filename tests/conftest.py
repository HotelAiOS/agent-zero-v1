"""
Pytest configuration for Agent Zero V1 tests
"""
import pytest
import os


@pytest.fixture(scope="session")
def docker_services():
    """Ensure Docker services are running"""
    return True


@pytest.fixture
def neo4j_credentials():
    """Neo4j connection credentials"""
    return {
        "uri": "bolt://localhost:7687",
        "username": "neo4j", 
        "password": "agent-pass"
    }


@pytest.fixture
def redis_config():
    """Redis connection configuration"""
    return {
        "host": "localhost",
        "port": 6379,
        "db": 0
    }


@pytest.fixture  
def rabbitmq_config():
    """RabbitMQ connection configuration"""
    return {
        "host": "localhost",
        "username": "admin",
        "password": "SecureRabbitPass123"
    }
