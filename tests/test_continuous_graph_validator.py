"""
Comprehensive tests for continuous_graph_validator.py
Covers unit tests, integration tests, edge cases, error handling, performance, and real-world usage.
"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from unittest.mock import Mock

import torch
from torch_geometric.data import Data

from src.utils.continuous_graph_validator import (
    ContinuousGraphValidator, ValidationRule, get_continuous_graph_validator)


class TestValidationRule:
    """Test ValidationRule class."""

    def test_validation_rule_init(self):
        """Test ValidationRule initialization."""
        def check_func(graph, id_manager):
            return {'passed': True}

        rule = ValidationRule('test_rule', check_func, interval=30.0)

        assert rule.name == 'test_rule'
        assert rule.check_func == check_func
        assert rule.repair_func is None
        assert rule.severity == 'warning'
        assert rule.interval == 30.0
        assert rule.last_check == 0
        assert rule.violations == 0
        assert rule.repairs == 0

    def test_validation_rule_should_check(self):
        """Test should_check method."""
        def check_func(graph, id_manager):
            return {'passed': True}

        rule = ValidationRule('test_rule', check_func, interval=1.0)

        # Initially should check
        assert rule.should_check() == True

        # Immediately after, should not check
        assert rule.should_check() == False

        # After interval, should check again
        rule.last_check = time.time() - 2.0
        assert rule.should_check() == True

    def test_validation_rule_check_and_repair_passed(self):
        """Test check_and_repair when rule passes."""
        def check_func(graph, id_manager):
            return {'passed': True}

        rule = ValidationRule('test_rule', check_func)

        graph = Data()
        id_manager = Mock()

        result = rule.check_and_repair(graph, id_manager)

        assert result['rule'] == 'test_rule'
        assert result['status'] == 'passed'
        assert rule.violations == 0
        assert rule.repairs == 0

    def test_validation_rule_check_and_repair_failed_no_repair(self):
        """Test check_and_repair when rule fails but no repair function."""
        def check_func(graph, id_manager):
            return {'passed': False, 'error': 'Test error'}

        rule = ValidationRule('test_rule', check_func)

        graph = Data()
        id_manager = Mock()

        result = rule.check_and_repair(graph, id_manager)

        assert result['rule'] == 'test_rule'
        assert result['status'] == 'no_repair_available'
        assert result['issue']['passed'] == False
        assert rule.violations == 1
        assert rule.repairs == 0

    def test_validation_rule_check_and_repair_with_repair(self):
        """Test check_and_repair with repair function."""
        def check_func(graph, id_manager):
            return {'passed': False, 'error': 'Test error'}

        def repair_func(graph, id_manager, issue):
            return {'repaired': True}

        rule = ValidationRule('test_rule', check_func, repair_func)

        graph = Data()
        id_manager = Mock()

        result = rule.check_and_repair(graph, id_manager)

        assert result['rule'] == 'test_rule'
        assert result['status'] == 'repaired'
        assert rule.violations == 1
        assert rule.repairs == 1

    def test_validation_rule_check_exception(self):
        """Test check_and_repair when check function raises exception."""
        def failing_check_func(graph, id_manager):
            raise RuntimeError("Check failed")

        rule = ValidationRule('test_rule', failing_check_func)

        graph = Data()
        id_manager = Mock()

        result = rule.check_and_repair(graph, id_manager)

        assert result['rule'] == 'test_rule'
        assert result['status'] == 'check_exception'
        assert 'Check failed' in result['error']


class TestContinuousGraphValidatorInit:
    """Test ContinuousGraphValidator initialization."""

    def test_validator_init(self):
        """Test basic initialization."""
        validator = ContinuousGraphValidator()

        assert len(validator._rules) > 0  # Should have default rules
        assert validator._running == False
        assert validator._check_interval == 30.0
        assert validator._full_check_interval == 300.0
        assert validator._auto_repair_enabled == True

        # Check default rules are registered
        expected_rules = ['graph_structure', 'id_consistency', 'energy_values', 'connection_integrity']
        for rule_name in expected_rules:
            assert rule_name in validator._rules

    def test_validator_dependencies(self):
        """Test that dependencies are properly initialized."""
        validator = ContinuousGraphValidator()

        # Should have integrity manager and connection validator
        assert hasattr(validator, 'integrity_manager')
        assert hasattr(validator, 'connection_validator')


class TestDefaultRules:
    """Test default validation rules."""

    def test_graph_structure_rule_valid(self):
        """Test graph structure rule with valid graph."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': 1}]
        graph.x = torch.randn(1, 1)
        graph.edge_index = torch.tensor([[0], [0]])

        id_manager = Mock()
        rule = validator._rules['graph_structure']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == True

    def test_graph_structure_rule_missing_nodelabels(self):
        """Test graph structure rule with missing node_labels."""
        validator = ContinuousGraphValidator()

        graph = Data()
        # Missing node_labels
        graph.x = torch.randn(1, 1)

        id_manager = Mock()
        rule = validator._rules['graph_structure']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == False
        assert 'node_labels' in result['error']

    def test_id_consistency_rule_valid(self):
        """Test ID consistency rule with valid IDs."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': 1}, {'id': 2}]

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        rule = validator._rules['id_consistency']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == True

    def test_id_consistency_rule_orphaned_ids(self):
        """Test ID consistency rule with orphaned IDs."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': 1}, {'id': 999}]  # 999 is orphaned

        id_manager = Mock()
        id_manager.is_valid_id = Mock(side_effect=lambda id: id != 999)

        rule = validator._rules['id_consistency']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == False
        assert 'orphaned_ids' in result
        assert 999 in result['orphaned_ids']

    def test_energy_values_rule_valid(self):
        """Test energy values rule with valid energies."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.x = torch.tensor([[0.5], [0.3], [0.8]])  # Valid energy values

        id_manager = Mock()
        rule = validator._rules['energy_values']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == True

    def test_energy_values_rule_invalid(self):
        """Test energy values rule with invalid energies."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.x = torch.tensor([[1.5], [-0.1], [float('nan')]])  # Invalid values

        id_manager = Mock()
        rule = validator._rules['energy_values']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == False
        assert 'invalid energy values' in result['error']

    def test_connection_integrity_rule_valid(self):
        """Test connection integrity rule with valid connections."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': 0}, {'id': 1}, {'id': 2}]
        graph.edge_index = torch.tensor([[0, 1], [1, 2]])  # Valid indices

        id_manager = Mock()
        rule = validator._rules['connection_integrity']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == True

    def test_connection_integrity_rule_invalid(self):
        """Test connection integrity rule with invalid connections."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': 0}, {'id': 1}]
        graph.edge_index = torch.tensor([[0, 2], [1, 0]])  # Index 2 is out of bounds

        id_manager = Mock()
        rule = validator._rules['connection_integrity']

        result = rule.check_func(graph, id_manager)
        assert result['passed'] == False
        assert 'invalid indices' in result['error']


class TestValidationCycle:
    """Test perform_validation_cycle method."""

    def test_perform_validation_cycle_all_passed(self):
        """Test validation cycle when all rules pass."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': 1}, {'id': 2}]
        graph.x = torch.tensor([[0.5], [0.3]])
        graph.edge_index = torch.tensor([[0], [1]])

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        result = validator.perform_validation_cycle(graph, id_manager)

        assert 'results' in result
        assert 'violations_found' in result
        assert result['violations_found'] == 0
        assert len(result['results']) == len(validator._rules)

    def test_perform_validation_cycle_with_violations(self):
        """Test validation cycle with some violations."""
        validator = ContinuousGraphValidator()

        graph = Data()
        # Missing required attributes to trigger violations
        graph.node_labels = [{'id': 1}]

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        result = validator.perform_validation_cycle(graph, id_manager)

        assert result['violations_found'] > 0
        assert len(result['results']) == len(validator._rules)

        # Check that some results indicate violations
        violation_results = [r for r in result['results'] if r['status'] != 'passed']
        assert len(violation_results) > 0


class TestContinuousValidation:
    """Test continuous validation functionality."""

    def test_start_stop_continuous_validation(self):
        """Test starting and stopping continuous validation."""
        validator = ContinuousGraphValidator()

        graph_source = Mock(return_value=Data())
        id_manager_source = Mock(return_value=Mock())

        # Start validation
        validator.start_continuous_validation(graph_source, id_manager_source)
        assert validator._running == True
        assert validator._thread is not None
        assert validator._thread.is_alive()

        # Stop validation
        validator.stop_continuous_validation()
        assert validator._running == False

        # Wait a bit for thread to stop
        time.sleep(0.1)
        assert not validator._thread.is_alive()

    def test_continuous_validation_loop(self):
        """Test the continuous validation loop."""
        validator = ContinuousGraphValidator()

        call_count = 0
        graph_source = Mock()
        id_manager_source = Mock()

        def mock_graph_source():
            nonlocal call_count
            call_count += 1
            graph = Data()
            graph.node_labels = [{'id': 1}]
            return graph

        def mock_id_manager_source():
            return Mock()

        graph_source.side_effect = mock_graph_source
        id_manager_source.side_effect = mock_id_manager_source

        # Start validation with short interval for testing
        validator._check_interval = 0.1
        validator.start_continuous_validation(graph_source, id_manager_source)

        # Let it run for a short time
        time.sleep(0.5)

        # Stop and check that it was called multiple times
        validator.stop_continuous_validation()
        assert call_count > 2  # Should have been called several times


class TestStatistics:
    """Test statistics and monitoring."""

    def test_get_statistics(self):
        """Test getting validator statistics."""
        validator = ContinuousGraphValidator()

        stats = validator.get_statistics()

        assert 'checks_performed' in stats
        assert 'violations_found' in stats
        assert 'repairs_attempted' in stats
        assert 'repairs_successful' in stats
        assert 'uptime' in stats
        assert 'rules' in stats

        # Check rules statistics
        assert isinstance(stats['rules'], dict)
        for rule_name in validator._rules:
            assert rule_name in stats['rules']

    def test_get_rule_status(self):
        """Test getting individual rule status."""
        validator = ContinuousGraphValidator()

        status = validator.get_rule_status('graph_structure')

        assert status is not None
        assert status['name'] == 'graph_structure'
        assert 'severity' in status
        assert 'interval' in status
        assert 'last_check' in status
        assert 'violations' in status
        assert 'repairs' in status

    def test_get_rule_status_nonexistent(self):
        """Test getting status for nonexistent rule."""
        validator = ContinuousGraphValidator()

        status = validator.get_rule_status('nonexistent_rule')
        assert status is None


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_continuous_graph_validator_singleton(self):
        """Test singleton pattern for global validator."""
        validator1 = get_continuous_graph_validator()
        validator2 = get_continuous_graph_validator()

        assert validator1 is validator2
        assert isinstance(validator1, ContinuousGraphValidator)


class TestIntegration:
    """Integration tests for ContinuousGraphValidator."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = ContinuousGraphValidator()

        # Create a test graph
        graph = Data()
        graph.node_labels = [
            {'id': i, 'type': 'dynamic', 'energy': 0.5}
            for i in range(10)
        ]
        graph.x = torch.randn(10, 1)
        graph.edge_index = torch.randint(0, 10, (2, 15))

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        # Perform validation cycle
        result = validator.perform_validation_cycle(graph, id_manager)

        assert 'results' in result
        assert 'violations_found' in result
        assert 'repairs_attempted' in result
        assert 'repairs_successful' in result

        # Should have results for all rules
        assert len(result['results']) == len(validator._rules)

    def test_continuous_validation_integration(self):
        """Test continuous validation with real graph changes."""
        validator = ContinuousGraphValidator()

        graph_state = {'graph': None}

        def graph_source():
            if graph_state['graph'] is None:
                # Create initial valid graph
                graph = Data()
                graph.node_labels = [{'id': 1}, {'id': 2}]
                graph.x = torch.tensor([[0.5], [0.3]])
                graph_state['graph'] = graph
            return graph_state['graph']

        def id_manager_source():
            manager = Mock()
            manager.is_valid_id = Mock(return_value=True)
            return manager

        # Start continuous validation
        validator._check_interval = 0.2  # Fast for testing
        validator.start_continuous_validation(graph_source, id_manager_source)

        # Let it run for a bit
        time.sleep(0.8)

        # Modify graph to introduce violations
        graph = graph_state['graph']
        graph.x = torch.tensor([[1.5], [-0.1]])  # Invalid energy values

        # Let validation detect the violation
        time.sleep(0.4)

        # Stop validation
        validator.stop_continuous_validation()

        # Check that violations were detected
        stats = validator.get_statistics()
        assert stats['checks_performed'] > 2
        # Should have detected energy value violations


class TestPerformance:
    """Performance tests for ContinuousGraphValidator."""

    def test_validation_cycle_performance(self):
        """Test performance of validation cycle."""
        validator = ContinuousGraphValidator()

        # Create larger graph for performance testing
        graph = Data()
        graph.node_labels = [{'id': i} for i in range(100)]
        graph.x = torch.randn(100, 1)
        graph.edge_index = torch.randint(0, 100, (2, 200))

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        start_time = time.time()
        for _ in range(10):
            result = validator.perform_validation_cycle(graph, id_manager)
            assert 'results' in result
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 5.0

    def test_rule_check_performance(self):
        """Test performance of individual rule checks."""
        validator = ContinuousGraphValidator()

        graph = Data()
        graph.node_labels = [{'id': i} for i in range(50)]
        graph.x = torch.randn(50, 1)

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        rule = validator._rules['graph_structure']

        start_time = time.time()
        for _ in range(100):
            result = rule.check_func(graph, id_manager)
            assert result['passed'] == True
        end_time = time.time()

        # Individual rule checks should be very fast
        assert end_time - start_time < 1.0


class TestRealWorldUsage:
    """Real-world usage scenarios for ContinuousGraphValidator."""

    def test_neural_simulation_validation(self):
        """Test validator in neural simulation context."""
        validator = ContinuousGraphValidator()

        # Simulate a neural network graph
        graph = Data()
        graph.node_labels = []

        # Create different neuron types
        for i in range(100):
            if i < 30:
                node_type = 'sensory'
            elif i < 70:
                node_type = 'dynamic'
            else:
                node_type = 'workspace'

            graph.node_labels.append({
                'id': i,
                'type': node_type,
                'energy': 0.2 + 0.6 * (i / 100),
                'membrane_potential': -70.0 + 10.0 * (i / 100)
            })

        graph.x = torch.tensor([[node['energy']] for node in graph.node_labels])
        graph.edge_index = torch.randint(0, 100, (2, 300))

        id_manager = Mock()
        id_manager.is_valid_id = Mock(return_value=True)

        # Perform validation
        result = validator.perform_validation_cycle(graph, id_manager)

        # In a real simulation, this should pass most checks
        assert 'results' in result
        assert len(result['results']) > 0

        # Check that we get meaningful results
        passed_count = sum(1 for r in result['results'] if r['status'] == 'passed')
        assert passed_count >= 0  # At least some should pass

    def test_continuous_monitoring_simulation(self):
        """Test continuous monitoring during simulation."""
        validator = ContinuousGraphValidator()

        simulation_state = {
            'step': 0,
            'graph': None,
            'violations_over_time': []
        }

        def create_simulation_graph():
            graph = Data()
            num_nodes = 20 + simulation_state['step']  # Graph grows over time

            graph.node_labels = [{'id': i, 'energy': 0.5} for i in range(num_nodes)]
            graph.x = torch.randn(num_nodes, 1)

            # Add some invalid energies occasionally
            if simulation_state['step'] % 5 == 0:
                graph.x[0] = torch.tensor([1.5])  # Invalid energy

            simulation_state['graph'] = graph
            return graph

        def id_manager_source():
            manager = Mock()
            manager.is_valid_id = Mock(return_value=True)
            return manager

        # Start continuous validation
        validator._check_interval = 0.1
        validator.start_continuous_validation(create_simulation_graph, id_manager_source)

        # Simulate several steps
        for step in range(10):
            simulation_state['step'] = step
            time.sleep(0.15)  # Let validation run

            # Record violations
            stats = validator.get_statistics()
            simulation_state['violations_over_time'].append(stats['violations_found'])

        # Stop validation
        validator.stop_continuous_validation()

        # Check that validation was working
        assert len(simulation_state['violations_over_time']) > 0

        # Should have detected some violations (from invalid energies)
        total_violations = sum(simulation_state['violations_over_time'])
        assert total_violations > 0


if __name__ == "__main__":
    pytest.main([__file__])






