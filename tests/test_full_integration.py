"""
Full Integration Test Suite for Agent Zero V1
Tests complete system functionality including Neo4j, RabbitMQ, Redis
"""

import pytest
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
        # Wait for services to be ready
        time.sleep(2)

        self.neo4j_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "agent-pass")
        )
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Test RabbitMQ connection
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host='localhost',
                    credentials=pika.PlainCredentials('admin', 'SecureRabbitPass123')
                )
            )
            self.rabbitmq_channel = connection.channel()
            self.rabbitmq_connection = connection
        except Exception as e:
            print(f"RabbitMQ connection failed: {e}")
            self.rabbitmq_channel = None
            self.rabbitmq_connection = None

        yield

        # Cleanup
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
        if hasattr(self, 'rabbitmq_connection') and self.rabbitmq_connection:
            self.rabbitmq_connection.close()

    def test_neo4j_connection(self):
        """Test Neo4j database connection and basic operations"""
        try:
            with self.neo4j_driver.session() as session:
                # Test basic query
                result = session.run("RETURN 'Hello Neo4j' as message")
                record = result.single()
                assert record["message"] == "Hello Neo4j"

                # Test APOC availability if installed
                try:
                    result = session.run("CALL apoc.version()")
                    version = result.single()
                    if version:
                        print(f"✅ Neo4j APOC version: {version['value']}")
                except Exception:
                    print("⚠️  APOC not available (optional)")

                print("✅ Neo4j connection and queries working")
        except Exception as e:
            pytest.fail(f"Neo4j test failed: {e}")

    def test_redis_connection(self):
        """Test Redis connection and basic operations"""
        try:
            # Test basic set/get
            self.redis_client.set("test_key", "test_value")
            value = self.redis_client.get("test_key")
            assert value.decode() == "test_value"

            # Test expiration
            self.redis_client.setex("temp_key", 1, "temp_value")
            time.sleep(2)
            assert self.redis_client.get("temp_key") is None

            # Cleanup
            self.redis_client.delete("test_key")

            print("✅ Redis operations working")
        except Exception as e:
            pytest.fail(f"Redis test failed: {e}")

    def test_rabbitmq_connection(self):
        """Test RabbitMQ connection and message passing"""
        if not self.rabbitmq_channel:
            pytest.skip("RabbitMQ connection not available")

        try:
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

            if method_frame:
                assert body.decode() == message
                print("✅ RabbitMQ message passing working")
            else:
                pytest.fail("No message received from RabbitMQ")

            # Cleanup
            self.rabbitmq_channel.queue_delete(queue=queue_name)

        except Exception as e:
            pytest.fail(f"RabbitMQ test failed: {e}")

    def test_docker_services_health(self):
        """Test all Docker services health endpoints"""
        endpoints = [
            ("Neo4j HTTP", "http://localhost:7474", 200),
            ("RabbitMQ Management", "http://localhost:15672", 200),
        ]

        for name, url, expected_status in endpoints:
            try:
                response = requests.get(url, timeout=10)
                assert response.status_code == expected_status
                print(f"✅ {name}: HTTP {response.status_code}")
            except Exception as e:
                pytest.fail(f"{name} endpoint failed: {e}")

    def test_system_integration_flow(self):
        """Test complete system workflow"""
        try:
            # 1. Store data in Neo4j
            with self.neo4j_driver.session() as session:
                session.run(
                    "CREATE (a:Agent {name: $name, status: $status})",
                    name="TestAgent", status="active"
                )

            # 2. Cache in Redis
            self.redis_client.hset("agent:TestAgent", "status", "active")

            # 3. Send notification via RabbitMQ (if available)
            if self.rabbitmq_channel:
                self.rabbitmq_channel.queue_declare(queue="agent_notifications", durable=False)
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

            self.redis_client.delete("agent:TestAgent")

            if self.rabbitmq_channel:
                self.rabbitmq_channel.queue_delete(queue="agent_notifications")

            print("✅ Complete integration workflow successful")

        except Exception as e:
            pytest.fail(f"Integration workflow failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
