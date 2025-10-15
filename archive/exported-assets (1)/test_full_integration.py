"""
Full Integration Test Suite for Agent Zero V1
Tests complete system functionality including Neo4j, RabbitMQ, Redis
"""

import pytest
import asyncio
import time
import requests
from neo4j import GraphDatabase
import pika
import redis


class TestFullIntegration:
    """Test complete system integration"""

    @pytest.fixture(autouse=True)
    def setup_services(self):
        """Verify all services are running"""
        self.neo4j_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "agent-pass")
        )
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Test RabbitMQ connection
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host='localhost',
                credentials=pika.PlainCredentials('admin', 'SecureRabbitPass123')
            )
        )
        self.rabbitmq_channel = connection.channel()

        yield

        # Cleanup
        self.neo4j_driver.close()
        connection.close()

    def test_neo4j_connection(self):
        """Test Neo4j database connection and basic operations"""
        with self.neo4j_driver.session() as session:
            # Test basic query
            result = session.run("RETURN 'Hello Neo4j' as message")
            record = result.single()
            assert record["message"] == "Hello Neo4j"

            # Test APOC availability
            result = session.run("CALL apoc.version()")
            version = result.single()
            assert version is not None
            print(f"✅ Neo4j APOC version: {version['value']}")

    def test_redis_connection(self):
        """Test Redis connection and basic operations"""
        # Test basic set/get
        self.redis_client.set("test_key", "test_value")
        value = self.redis_client.get("test_key")
        assert value.decode() == "test_value"

        # Test expiration
        self.redis_client.setex("temp_key", 1, "temp_value")
        time.sleep(2)
        assert self.redis_client.get("temp_key") is None
        print("✅ Redis operations working")

    def test_rabbitmq_connection(self):
        """Test RabbitMQ connection and message passing"""
        # Declare queue
        queue_name = "test_queue"
        self.rabbitmq_channel.queue_declare(queue=queue_name, durable=False)

        # Send message
        message = "Hello RabbitMQ"
        self.rabbitmq_channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message
        )

        # Receive message
        method_frame, header_frame, body = self.rabbitmq_channel.basic_get(
            queue=queue_name, auto_ack=True
        )

        assert body.decode() == message
        print("✅ RabbitMQ message passing working")

    def test_docker_services_health(self):
        """Test all Docker services health endpoints"""
        # Neo4j HTTP endpoint
        response = requests.get("http://localhost:7474")
        assert response.status_code == 200

        # RabbitMQ management interface
        response = requests.get(
            "http://localhost:15672/api/overview",
            auth=("admin", "SecureRabbitPass123")
        )
        assert response.status_code == 200

        print("✅ All HTTP endpoints accessible")

    def test_system_integration_flow(self):
        """Test complete system workflow"""
        # 1. Store data in Neo4j
        with self.neo4j_driver.session() as session:
            session.run(
                "CREATE (a:Agent {name: $name, status: $status})",
                name="TestAgent", status="active"
            )

        # 2. Cache in Redis
        self.redis_client.hset("agent:TestAgent", "status", "active")

        # 3. Send notification via RabbitMQ
        self.rabbitmq_channel.queue_declare(queue="agent_notifications")
        self.rabbitmq_channel.basic_publish(
            exchange='',
            routing_key="agent_notifications",
            body="Agent TestAgent created"
        )

        # 4. Verify workflow
        with self.neo4j_driver.session() as session:
            result = session.run(
                "MATCH (a:Agent {name: $name}) RETURN a.status as status",
                name="TestAgent"
            )
            record = result.single()
            assert record["status"] == "active"

        cached_status = self.redis_client.hget("agent:TestAgent", "status")
        assert cached_status.decode() == "active"

        # Clean up
        with self.neo4j_driver.session() as session:
            session.run("MATCH (a:Agent {name: $name}) DELETE a", name="TestAgent")

        print("✅ Complete integration workflow successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
