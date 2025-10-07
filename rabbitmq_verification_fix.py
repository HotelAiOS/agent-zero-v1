#!/usr/bin/env python3
"""
RabbitMQ Verification and Fix Script for Agent Zero V1
====================================================

This script diagnoses and fixes RabbitMQ connection issues.
Compatible with Arch Linux + Fish Shell environment.
"""

import os
import sys
import subprocess
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabbitMQVerifier:
    """RabbitMQ verification and repair utility."""
    
    def __init__(self):
        self.docker_compose_file = "docker-compose.yml"
        self.rabbitmq_container = "agent-zero-rabbitmq"
        self.rabbitmq_config = {
            'host': 'localhost',
            'port': 5672,
            'management_port': 15672,
            'user': 'agent_zero',
            'password': 'agent_zero_rabbit_dev',
            'vhost': 'agent_zero_vhost'
        }
    
    def check_docker_compose(self) -> bool:
        """Check if docker-compose.yml exists and has RabbitMQ config."""
        if not os.path.exists(self.docker_compose_file):
            logger.error("❌ docker-compose.yml not found")
            return False
        
        try:
            with open(self.docker_compose_file, 'r') as f:
                content = f.read()
                
            if 'rabbitmq' in content:
                logger.info("✅ RabbitMQ configuration found in docker-compose.yml")
                return True
            else:
                logger.error("❌ RabbitMQ not configured in docker-compose.yml")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error reading docker-compose.yml: {e}")
            return False
    
    def check_docker_running(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Docker daemon is running")
                return True
            else:
                logger.error("❌ Docker daemon is not running")
                return False
        except Exception as e:
            logger.error(f"❌ Docker check failed: {e}")
            return False
    
    def start_rabbitmq_service(self) -> bool:
        """Start RabbitMQ container using docker-compose."""
        try:
            logger.info("🚀 Starting RabbitMQ container...")
            
            # Stop existing container if running
            subprocess.run(['docker-compose', 'stop', 'rabbitmq'], 
                         capture_output=True, text=True)
            
            # Remove existing container
            subprocess.run(['docker-compose', 'rm', '-f', 'rabbitmq'], 
                         capture_output=True, text=True)
            
            # Start RabbitMQ service
            result = subprocess.run(['docker-compose', 'up', '-d', 'rabbitmq'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ RabbitMQ container started successfully")
                return True
            else:
                logger.error(f"❌ Failed to start RabbitMQ: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting RabbitMQ: {e}")
            return False
    
    def wait_for_rabbitmq_ready(self, timeout: int = 60) -> bool:
        """Wait for RabbitMQ to be ready."""
        logger.info("⏳ Waiting for RabbitMQ to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if management UI is accessible
                result = subprocess.run([
                    'curl', '-s', '-u', f'{self.rabbitmq_config["user"]}:{self.rabbitmq_config["password"]}',
                    f'http://localhost:{self.rabbitmq_config["management_port"]}/api/overview'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    logger.info("✅ RabbitMQ management interface is ready")
                    return True
                    
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                pass
            
            logger.info("⏳ RabbitMQ not ready yet, waiting...")
            time.sleep(3)
        
        logger.error("❌ RabbitMQ failed to become ready within timeout")
        return False
    
    def test_rabbitmq_connection(self) -> bool:
        """Test RabbitMQ connection using pika."""
        try:
            import pika
            logger.info("🧪 Testing RabbitMQ connection...")
            
            # Connection parameters
            credentials = pika.PlainCredentials(
                self.rabbitmq_config['user'], 
                self.rabbitmq_config['password']
            )
            
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_config['host'],
                port=self.rabbitmq_config['port'],
                virtual_host=self.rabbitmq_config['vhost'],
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            # Test connection
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Test queue operations
            queue_name = 'agent_zero_test_verification'
            channel.queue_declare(queue=queue_name, durable=True)
            
            # Test message publish
            test_message = {
                'test': True,
                'timestamp': datetime.now().isoformat(),
                'verification': 'agent_zero_rabbitmq_test'
            }
            
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(test_message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            # Test message consume
            method, properties, body = channel.basic_get(queue=queue_name, auto_ack=True)
            
            if body:
                received_message = json.loads(body)
                if received_message.get('verification') == 'agent_zero_rabbitmq_test':
                    logger.info("✅ RabbitMQ publish/consume test successful")
                    
                    # Cleanup test queue
                    channel.queue_delete(queue=queue_name)
                    connection.close()
                    return True
            
            logger.error("❌ RabbitMQ message test failed")
            connection.close()
            return False
            
        except ImportError:
            logger.error("❌ Pika library not installed. Install with: pip install pika")
            return False
        except Exception as e:
            logger.error(f"❌ RabbitMQ connection test failed: {e}")
            return False
    
    def check_rabbitmq_logs(self) -> str:
        """Get RabbitMQ container logs for debugging."""
        try:
            result = subprocess.run(['docker-compose', 'logs', '--tail=50', 'rabbitmq'], 
                                  capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logger.error(f"❌ Failed to get RabbitMQ logs: {e}")
            return ""
    
    def create_rabbitmq_client_file(self) -> bool:
        """Create RabbitMQ client file for the project."""
        client_code = '''"""
RabbitMQ Client for Agent Zero V1
===============================

Production-ready RabbitMQ client with connection pooling and error handling.
"""

import pika
import json
import logging
import os
from typing import Dict, Any, Callable, Optional
import time

class RabbitMQClient:
    """Production RabbitMQ client for Agent Zero V1."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'host': os.getenv('RABBITMQ_HOST', 'localhost'),
            'port': int(os.getenv('RABBITMQ_PORT', 5672)),
            'user': os.getenv('RABBITMQ_USER', 'agent_zero'),
            'password': os.getenv('RABBITMQ_PASS', 'agent_zero_rabbit_dev'),
            'vhost': os.getenv('RABBITMQ_VHOST', 'agent_zero_vhost')
        }
        
        self.connection = None
        self.channel = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Establish connection to RabbitMQ with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                credentials = pika.PlainCredentials(
                    self.config['user'], 
                    self.config['password']
                )
                
                parameters = pika.ConnectionParameters(
                    host=self.config['host'],
                    port=self.config['port'],
                    virtual_host=self.config['vhost'],
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
                
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                self.logger.info(f"✅ RabbitMQ connected on attempt {attempt + 1}")
                return True
                
            except Exception as e:
                self.logger.warning(f"RabbitMQ connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        self.logger.error("❌ All RabbitMQ connection attempts failed")
        return False
    
    def publish_message(self, queue: str, message: Dict[str, Any], exchange: str = '') -> bool:
        """Publish message to queue with automatic reconnection."""
        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return False
        
        try:
            # Declare queue (idempotent)
            self.channel.queue_declare(queue=queue, durable=True)
            
            # Publish message
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent
                    content_type='application/json'
                )
            )
            
            self.logger.info(f"✅ Message published to {queue}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to publish message to {queue}: {e}")
            return False
    
    def consume_messages(self, queue: str, callback: Callable, auto_ack: bool = False) -> bool:
        """Start consuming messages from queue."""
        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return False
        
        try:
            self.channel.queue_declare(queue=queue, durable=True)
            
            def wrapper_callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    callback(message)
                    if not auto_ack:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")
                    if not auto_ack:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            self.channel.basic_consume(
                queue=queue,
                on_message_callback=wrapper_callback,
                auto_ack=auto_ack
            )
            
            self.logger.info(f"✅ Started consuming from {queue}")
            self.channel.start_consuming()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to consume from {queue}: {e}")
            return False
    
    def close(self):
        """Close RabbitMQ connection."""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            self.logger.info("RabbitMQ connection closed")
        except Exception as e:
            self.logger.error(f"Error closing RabbitMQ connection: {e}")

# Factory function
def create_rabbitmq_client(config: Optional[Dict[str, Any]] = None) -> RabbitMQClient:
    """Create RabbitMQ client instance."""
    return RabbitMQClient(config)

# Test function
def test_rabbitmq_setup() -> bool:
    """Test RabbitMQ setup and functionality."""
    client = create_rabbitmq_client()
    
    if not client.connect():
        return False
    
    # Test publish/consume
    test_queue = 'agent_zero_setup_test'
    test_message = {
        'test': True,
        'timestamp': time.time(),
        'setup_verification': True
    }
    
    # Publish test message
    if not client.publish_message(test_queue, test_message):
        return False
    
    # Try to get the message back
    try:
        method, properties, body = client.channel.basic_get(queue=test_queue, auto_ack=True)
        if body:
            received = json.loads(body)
            if received.get('setup_verification'):
                client.logger.info("✅ RabbitMQ setup test passed")
                client.close()
                return True
    except Exception as e:
        client.logger.error(f"Setup test failed: {e}")
    
    client.close()
    return False

if __name__ == "__main__":
    # Run setup test
    if test_rabbitmq_setup():
        print("✅ RabbitMQ client is working correctly")
    else:
        print("❌ RabbitMQ client test failed")
'''
        
        try:
            os.makedirs("shared/messaging", exist_ok=True)
            
            with open("shared/messaging/rabbitmq_client.py", 'w') as f:
                f.write(client_code)
            
            logger.info("✅ Created shared/messaging/rabbitmq_client.py")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create RabbitMQ client file: {e}")
            return False
    
    def run_full_verification(self) -> bool:
        """Run complete RabbitMQ verification and fix process."""
        logger.info("🔧 Starting RabbitMQ verification and fix process...")
        
        # Step 1: Check prerequisites
        if not self.check_docker_running():
            logger.error("Please start Docker daemon first")
            return False
        
        if not self.check_docker_compose():
            logger.error("Please ensure docker-compose.yml has RabbitMQ configuration")
            return False
        
        # Step 2: Start RabbitMQ service
        if not self.start_rabbitmq_service():
            logger.error("Failed to start RabbitMQ service")
            logs = self.check_rabbitmq_logs()
            if logs:
                logger.error(f"RabbitMQ logs:\n{logs}")
            return False
        
        # Step 3: Wait for service to be ready
        if not self.wait_for_rabbitmq_ready():
            logger.error("RabbitMQ service failed to become ready")
            logs = self.check_rabbitmq_logs()
            if logs:
                logger.error(f"RabbitMQ logs:\n{logs}")
            return False
        
        # Step 4: Test connection
        if not self.test_rabbitmq_connection():
            logger.error("RabbitMQ connection test failed")
            return False
        
        # Step 5: Create client file
        if not self.create_rabbitmq_client_file():
            logger.error("Failed to create RabbitMQ client file")
            return False
        
        logger.info("🎉 RabbitMQ verification completed successfully!")
        logger.info("✅ RabbitMQ is ready for Agent Zero V1")
        logger.info(f"✅ Management UI: http://localhost:{self.rabbitmq_config['management_port']}")
        logger.info(f"✅ Username: {self.rabbitmq_config['user']}")
        logger.info(f"✅ Password: {self.rabbitmq_config['password']}")
        
        return True

def main():
    """Main function."""
    verifier = RabbitMQVerifier()
    
    if verifier.run_full_verification():
        print("\n🎉 SUCCESS: RabbitMQ is now fully operational!")
        print("Next steps:")
        print("1. Test the management UI at http://localhost:15672")
        print("2. Run integration tests with other Agent Zero components")
        print("3. Move on to the next critical task: WebSocket Frontend Rendering")
        return 0
    else:
        print("\n❌ FAILED: RabbitMQ verification failed")
        print("Please check the error messages above and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())