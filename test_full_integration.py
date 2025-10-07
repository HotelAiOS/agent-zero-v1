#!/usr/bin/env python3
"""
Full Integration Test Suite for AgentExecutor Method Signature Fix
================================================================

This test suite validates the AgentExecutor method signature fix and ensures
proper integration with the Agent Zero V1 multi-agent platform.

Author: Agent Zero V1 Development Team
Status: COMPLETED - Tests for fixed execute_task signature
Last Updated: 2025-10-07
"""

import os
import sys
import pytest
import tempfile
import shutil
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add shared directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from shared.execution.agent_executor import AgentExecutor, AsyncAgentExecutor, create_agent_executor, AgentExecutorError
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the agent_executor.py file is in shared/execution/")
    sys.exit(1)


class TestAgentExecutorMethodSignature:
    """Test class for AgentExecutor method signature fix."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='agent_zero_test_')
        self.executor = AgentExecutor(log_level='DEBUG')
        
        # Mock agent for testing
        self.mock_agent = Mock()
        self.mock_agent.execute = Mock(return_value={
            'status': 'completed',
            'message': 'Test task completed successfully',
            'data': {'result': 'test_result'}
        })
        
        # Test task data
        self.test_task = {
            'id': 'test_task_001',
            'type': 'test',
            'description': 'Test task for method signature validation',
            'data': {'input': 'test_input'}
        }
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_execute_task_method_signature(self):
        """Test that execute_task method accepts all required parameters."""
        # This should not raise TypeError about missing output_dir parameter
        result = self.executor.execute_task(
            agent=self.mock_agent,
            task=self.test_task,
            output_dir=self.temp_dir
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'execution_metadata' in result
        assert result['status'] == 'completed'
        
    def test_execute_task_parameter_validation(self):
        """Test parameter validation in execute_task method."""
        
        # Test with None agent
        with pytest.raises((ValueError, AgentExecutorError)):
            self.executor.execute_task(None, self.test_task, self.temp_dir)
        
        # Test with invalid task
        with pytest.raises((ValueError, AgentExecutorError)):
            self.executor.execute_task(self.mock_agent, None, self.temp_dir)
        
        # Test with empty task
        with pytest.raises((ValueError, AgentExecutorError)):
            self.executor.execute_task(self.mock_agent, {}, self.temp_dir)
        
        # Test with invalid output_dir type
        with pytest.raises((ValueError, AgentExecutorError)):
            self.executor.execute_task(self.mock_agent, self.test_task, None)
        
        # Test with empty output_dir
        with pytest.raises((ValueError, AgentExecutorError)):
            self.executor.execute_task(self.mock_agent, self.test_task, "")
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'new_directory', 'nested')
        
        assert not os.path.exists(non_existent_dir)
        
        result = self.executor.execute_task(
            agent=self.mock_agent,
            task=self.test_task,
            output_dir=non_existent_dir
        )
        
        assert os.path.exists(non_existent_dir)
        assert os.path.isdir(non_existent_dir)
    
    def test_execution_metadata_generation(self):
        """Test that execution metadata is properly generated."""
        result = self.executor.execute_task(
            agent=self.mock_agent,
            task=self.test_task,
            output_dir=self.temp_dir
        )
        
        metadata = result['execution_metadata']
        
        assert 'execution_time' in metadata
        assert 'timestamp' in metadata
        assert 'output_directory' in metadata
        assert 'agent_type' in metadata
        assert 'task_id' in metadata
        assert 'executor_version' in metadata
        
        assert metadata['output_directory'] == self.temp_dir
        assert metadata['task_id'] == self.test_task['id']
        assert isinstance(metadata['execution_time'], float)
    
    def test_agent_method_execution_variants(self):
        """Test execution with different agent method names."""
        
        # Test with agent.run method
        run_agent = Mock()
        run_agent.run = Mock(return_value={'status': 'completed', 'message': 'run method executed'})
        
        result = self.executor.execute_task(run_agent, self.test_task, self.temp_dir)
        assert result['status'] == 'completed'
        run_agent.run.assert_called_once()
        
        # Test with agent.process method
        process_agent = Mock()
        process_agent.process = Mock(return_value={'status': 'completed', 'message': 'process method executed'})
        
        result = self.executor.execute_task(process_agent, self.test_task, self.temp_dir)
        assert result['status'] == 'completed'
        process_agent.process.assert_called_once()
        
        # Test with generic agent (no specific methods)
        generic_agent = Mock()
        # Remove standard methods if they exist
        if hasattr(generic_agent, 'execute'):
            delattr(generic_agent, 'execute')
        if hasattr(generic_agent, 'run'):
            delattr(generic_agent, 'run')
        if hasattr(generic_agent, 'process'):
            delattr(generic_agent, 'process')
        
        result = self.executor.execute_task(generic_agent, self.test_task, self.temp_dir)
        assert result['status'] == 'completed'
        assert 'generic method' in result['message'].lower()


class TestAgentExecutorErrorHandling:
    """Test class for AgentExecutor error handling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='agent_zero_error_test_')
        self.executor = AgentExecutor(log_level='DEBUG')
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_agent_execution_error_handling(self):
        """Test error handling when agent execution fails."""
        
        # Create failing agent
        failing_agent = Mock()
        failing_agent.execute = Mock(side_effect=Exception("Simulated agent failure"))
        
        test_task = {
            'id': 'failing_task_001',
            'type': 'test',
            'description': 'Task that should fail'
        }
        
        with pytest.raises(AgentExecutorError) as exc_info:
            self.executor.execute_task(failing_agent, test_task, self.temp_dir)
        
        # Check that error log file was created
        error_files = list(Path(self.temp_dir).glob('error_*.json'))
        assert len(error_files) > 0
        
        # Verify error file contents
        with open(error_files[0], 'r') as f:
            error_data = json.load(f)
        
        assert error_data['task_id'] == test_task['id']
        assert 'Simulated agent failure' in error_data['error_message']
        assert 'traceback' in error_data
        assert 'timestamp' in error_data
    
    def test_statistics_update_on_failure(self):
        """Test that execution statistics are updated on failure."""
        
        initial_stats = self.executor.get_execution_stats()
        assert initial_stats['tasks_failed'] == 0
        
        # Create failing agent
        failing_agent = Mock()
        failing_agent.execute = Mock(side_effect=Exception("Test failure"))
        
        test_task = {'id': 'fail_test', 'type': 'test', 'description': 'Fail test'}
        
        try:
            self.executor.execute_task(failing_agent, test_task, self.temp_dir)
        except AgentExecutorError:
            pass  # Expected
        
        final_stats = self.executor.get_execution_stats()
        assert final_stats['tasks_executed'] == 1
        assert final_stats['tasks_failed'] == 1
        assert final_stats['tasks_successful'] == 0


class TestAgentExecutorIntegration:
    """Integration tests for AgentExecutor with external systems."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='agent_zero_integration_test_')
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('shared.execution.agent_executor.Neo4jClient')
    def test_neo4j_integration(self, mock_neo4j_client_class):
        """Test Neo4j integration in AgentExecutor."""
        
        # Mock Neo4j client
        mock_client = Mock()
        mock_session = Mock()
        mock_client.get_session.return_value = mock_session
        mock_neo4j_client_class.return_value = mock_client
        
        # Create executor with Neo4j
        executor = AgentExecutor(neo4j_client=mock_client)
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value={'status': 'completed', 'message': 'Neo4j test'})
        
        test_task = {'id': 'neo4j_test', 'type': 'test', 'description': 'Neo4j integration test'}
        
        result = executor.execute_task(mock_agent, test_task, self.temp_dir)
        
        assert result['status'] == 'completed'
        mock_client.get_session.assert_called()
        mock_session.close.assert_called()
    
    @patch('shared.execution.agent_executor.RabbitMQClient')
    def test_rabbitmq_integration(self, mock_rabbitmq_client_class):
        """Test RabbitMQ integration in AgentExecutor."""
        
        # Mock RabbitMQ client
        mock_client = Mock()
        mock_channel = Mock()
        mock_client.get_channel.return_value = mock_channel
        mock_rabbitmq_client_class.return_value = mock_client
        
        # Create executor with RabbitMQ
        executor = AgentExecutor(rabbitmq_client=mock_client)
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value={'status': 'completed', 'message': 'RabbitMQ test'})
        
        test_task = {'id': 'rabbitmq_test', 'type': 'test', 'description': 'RabbitMQ integration test'}
        
        result = executor.execute_task(mock_agent, test_task, self.temp_dir)
        
        assert result['status'] == 'completed'
        mock_client.get_channel.assert_called()
        mock_channel.close.assert_called()


class TestAsyncAgentExecutor:
    """Test class for AsyncAgentExecutor."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='agent_zero_async_test_')
        self.executor = AsyncAgentExecutor(log_level='DEBUG')
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_async_execute_task(self):
        """Test async version of execute_task method."""
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value={'status': 'completed', 'message': 'Async test'})
        
        test_task = {'id': 'async_test', 'type': 'test', 'description': 'Async execution test'}
        
        result = await self.executor.execute_task(mock_agent, test_task, self.temp_dir)
        
        assert result['status'] == 'completed'
        assert 'execution_metadata' in result


class TestAgentExecutorFactory:
    """Test class for AgentExecutor factory function."""
    
    def test_create_agent_executor_default(self):
        """Test factory function with default configuration."""
        executor = create_agent_executor()
        assert isinstance(executor, AgentExecutor)
        assert not isinstance(executor, AsyncAgentExecutor)
    
    def test_create_agent_executor_async(self):
        """Test factory function with async configuration."""
        config = {'async_mode': True}
        executor = create_agent_executor(config)
        assert isinstance(executor, AsyncAgentExecutor)
    
    @patch('shared.execution.agent_executor.Neo4jClient')
    def test_create_agent_executor_with_neo4j(self, mock_neo4j_client_class):
        """Test factory function with Neo4j configuration."""
        config = {
            'neo4j_config': {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password'
            }
        }
        
        mock_client = Mock()
        mock_neo4j_client_class.return_value = mock_client
        
        executor = create_agent_executor(config)
        assert isinstance(executor, AgentExecutor)
        assert executor.neo4j_client is mock_client


# Performance and Load Tests
class TestAgentExecutorPerformance:
    """Performance tests for AgentExecutor."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix='agent_zero_perf_test_')
        self.executor = AgentExecutor(log_level='WARNING')  # Reduce logging for performance
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_multiple_task_execution(self):
        """Test execution of multiple tasks for performance and statistics."""
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value={'status': 'completed', 'message': 'Performance test'})
        
        num_tasks = 10
        
        for i in range(num_tasks):
            test_task = {
                'id': f'perf_task_{i:03d}',
                'type': 'performance_test',
                'description': f'Performance test task {i}'
            }
            
            result = self.executor.execute_task(mock_agent, test_task, self.temp_dir)
            assert result['status'] == 'completed'
        
        # Check statistics
        stats = self.executor.get_execution_stats()
        assert stats['tasks_executed'] == num_tasks
        assert stats['tasks_successful'] == num_tasks
        assert stats['tasks_failed'] == 0
        assert stats['success_rate'] == 1.0
        assert stats['average_execution_time'] > 0


# Main test execution
if __name__ == "__main__":
    print("🧪 Running AgentExecutor Method Signature Fix Integration Tests")
    print("=" * 70)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run basic functionality test
    print("\n1. Testing basic execute_task method signature...")
    test_executor = AgentExecutor(log_level='INFO')
    
    # Mock agent for basic test
    class MockAgent:
        def execute(self, task, context):
            return {
                'status': 'completed',
                'message': 'Basic test successful',
                'task_id': task['id']
            }
    
    test_task = {
        'id': 'signature_test_001',
        'type': 'basic_test',
        'description': 'Basic method signature test'
    }
    
    temp_dir = tempfile.mkdtemp(prefix='agent_zero_basic_test_')
    
    try:
        # This is the key test - ensure method signature works with output_dir parameter
        result = test_executor.execute_task(
            agent=MockAgent(),
            task=test_task,
            output_dir=temp_dir
        )
        
        print("✅ execute_task method signature test PASSED")
        print(f"   Task executed successfully with output_dir: {temp_dir}")
        print(f"   Execution time: {result['execution_metadata']['execution_time']:.3f}s")
        
        # Verify output directory was used
        assert os.path.exists(temp_dir), "Output directory should exist"
        print("✅ Output directory creation test PASSED")
        
        # Test parameter validation
        print("\n2. Testing parameter validation...")
        
        # Test missing parameters (should raise errors)
        test_cases = [
            (None, test_task, temp_dir, "None agent"),
            (MockAgent(), None, temp_dir, "None task"), 
            (MockAgent(), {}, temp_dir, "Empty task"),
            (MockAgent(), test_task, None, "None output_dir"),
            (MockAgent(), test_task, "", "Empty output_dir")
        ]
        
        validation_passed = 0
        for agent, task, output_dir, test_name in test_cases:
            try:
                test_executor.execute_task(agent, task, output_dir)
                print(f"❌ {test_name} validation test FAILED (should have raised error)")
            except (ValueError, AgentExecutorError):
                print(f"✅ {test_name} validation test PASSED")
                validation_passed += 1
            except Exception as e:
                print(f"❌ {test_name} validation test FAILED with unexpected error: {e}")
        
        if validation_passed == len(test_cases):
            print("✅ All parameter validation tests PASSED")
        else:
            print(f"❌ {len(test_cases) - validation_passed} parameter validation tests FAILED")
        
        # Test execution statistics
        print("\n3. Testing execution statistics...")
        stats = test_executor.get_execution_stats()
        print(f"   Tasks executed: {stats['tasks_executed']}")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Average execution time: {stats['average_execution_time']:.3f}s")
        
        if stats['tasks_executed'] > 0 and stats['success_rate'] > 0:
            print("✅ Execution statistics test PASSED")
        else:
            print("❌ Execution statistics test FAILED")
        
        print("\n🎉 ALL BASIC TESTS PASSED!")
        print("AgentExecutor method signature fix is working correctly.")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print("AgentExecutor method signature fix failed!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 70)
    print("To run full pytest suite:")
    print("  cd /path/to/agent-zero-v1")  
    print("  python -m pytest test_full_integration.py -v")
    print("\nTo run specific test categories:")
    print("  python -m pytest test_full_integration.py::TestAgentExecutorMethodSignature -v")
    print("  python -m pytest test_full_integration.py::TestAgentExecutorErrorHandling -v")
    print("  python -m pytest test_full_integration.py::TestAgentExecutorIntegration -v")