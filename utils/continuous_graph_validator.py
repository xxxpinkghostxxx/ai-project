"""
Continuous Graph Validator
Provides ongoing validation and automatic repair of neural simulation graphs.
"""

import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from utils.logging_utils import log_step
from utils.graph_integrity_manager import get_graph_integrity_manager
from utils.connection_validator import get_connection_validator
from utils.reader_writer_lock import get_graph_lock


class ValidationRule:
    """Represents a validation rule with repair capability."""

    def __init__(self, name: str, check_func: Callable, repair_func: Optional[Callable] = None,
                 severity: str = 'warning', interval: float = 60.0):
        self.name = name
        self.check_func = check_func
        self.repair_func = repair_func
        self.severity = severity  # 'info', 'warning', 'error', 'critical'
        self.interval = interval
        self.last_check = 0
        self.violations = 0
        self.repairs = 0

    def should_check(self) -> bool:
        """Check if this rule should be evaluated."""
        return time.time() - self.last_check >= self.interval

    def check_and_repair(self, graph, id_manager) -> Dict[str, Any]:
        """Check the rule and attempt repair if needed."""
        self.last_check = time.time()

        try:
            result = self.check_func(graph, id_manager)

            if not result['passed']:
                self.violations += 1
                log_step(f"Validation rule '{self.name}' failed",
                        severity=self.severity, violations=self.violations)

                # Attempt repair if repair function is available
                if self.repair_func:
                    try:
                        repair_result = self.repair_func(graph, id_manager, result)
                        if repair_result['repaired']:
                            self.repairs += 1
                            log_step(f"Validation rule '{self.name}' repaired",
                                    repairs=self.repairs)
                            return {
                                'rule': self.name,
                                'status': 'repaired',
                                'original_issue': result,
                                'repair_result': repair_result
                            }
                        else:
                            return {
                                'rule': self.name,
                                'status': 'repair_failed',
                                'issue': result,
                                'repair_error': repair_result.get('error', 'Unknown repair error')
                            }
                    except Exception as e:
                        log_step(f"Repair failed for rule '{self.name}'", error=str(e))
                        return {
                            'rule': self.name,
                            'status': 'repair_exception',
                            'issue': result,
                            'repair_error': str(e)
                        }
                else:
                    return {
                        'rule': self.name,
                        'status': 'no_repair_available',
                        'issue': result
                    }
            else:
                return {
                    'rule': self.name,
                    'status': 'passed'
                }

        except Exception as e:
            log_step(f"Validation rule '{self.name}' check failed", error=str(e))
            return {
                'rule': self.name,
                'status': 'check_exception',
                'error': str(e)
            }


class ContinuousGraphValidator:
    """Provides continuous validation and repair of neural graphs."""

    def __init__(self):
        self._lock = get_graph_lock()
        self._rules: Dict[str, ValidationRule] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._check_interval = 30.0  # Check every 30 seconds
        self._last_full_check = 0
        self._full_check_interval = 300.0  # Full check every 5 minutes

        self._stats = {
            'checks_performed': 0,
            'violations_found': 0,
            'repairs_attempted': 0,
            'repairs_successful': 0,
            'uptime': 0,
            'start_time': time.time()
        }

        # Get dependencies
        self.integrity_manager = get_graph_integrity_manager()
        self.connection_validator = get_connection_validator()

        # Register default validation rules
        self._register_default_rules()

        log_step("ContinuousGraphValidator initialized")

    def _register_default_rules(self):
        """Register the default set of validation rules."""

        # Rule 1: Graph structure integrity
        def check_graph_structure(graph, id_manager):
            if not hasattr(graph, 'node_labels') or not graph.node_labels:
                return {'passed': False, 'error': 'Graph missing node_labels'}
            if not hasattr(graph, 'x') or graph.x is None:
                return {'passed': False, 'error': 'Graph missing energy tensor'}
            return {'passed': True}

        def repair_graph_structure(graph, id_manager, issue):
            # Attempt to create missing attributes
            if not hasattr(graph, 'node_labels'):
                graph.node_labels = []
            if not hasattr(graph, 'x'):
                graph.x = torch.empty((0, 1), dtype=torch.float32)
            if not hasattr(graph, 'edge_index'):
                graph.edge_index = torch.empty((2, 0), dtype=torch.long)
            return {'repaired': True}

        self.register_rule(ValidationRule(
            'graph_structure',
            check_graph_structure,
            repair_graph_structure,
            severity='critical',
            interval=10.0
        ))

        # Rule 2: ID consistency
        def check_id_consistency(graph, id_manager):
            if not hasattr(graph, 'node_labels'):
                return {'passed': True}  # Skip if no nodes

            orphaned_ids = []
            for node in graph.node_labels:
                node_id = node.get('id')
                if node_id and not id_manager.is_valid_id(node_id):
                    orphaned_ids.append(node_id)

            if orphaned_ids:
                return {
                    'passed': False,
                    'error': f'Found {len(orphaned_ids)} orphaned IDs',
                    'orphaned_ids': orphaned_ids
                }
            return {'passed': True}

        def repair_id_consistency(graph, id_manager, issue):
            # Clean up orphaned IDs
            if 'orphaned_ids' in issue:
                for node_id in issue['orphaned_ids']:
                    # Remove from graph
                    graph.node_labels = [
                        node for node in graph.node_labels
                        if node.get('id') != node_id
                    ]
            return {'repaired': True}

        self.register_rule(ValidationRule(
            'id_consistency',
            check_id_consistency,
            repair_id_consistency,
            severity='error',
            interval=60.0
        ))

        # Rule 3: Energy value validation
        def check_energy_values(graph, id_manager):
            if not hasattr(graph, 'x') or graph.x is None or graph.x.numel() == 0:
                return {'passed': True}  # Skip if no energy values

            invalid_count = 0
            for i in range(graph.x.shape[0]):
                energy = graph.x[i, 0].item()
                if not (0.0 <= energy <= 1.0) or str(energy).lower() in ['nan', 'inf']:
                    invalid_count += 1

            if invalid_count > 0:
                return {
                    'passed': False,
                    'error': f'Found {invalid_count} invalid energy values',
                    'invalid_count': invalid_count
                }
            return {'passed': True}

        def repair_energy_values(graph, id_manager, issue):
            # Clamp invalid energy values to valid range
            if hasattr(graph, 'x') and graph.x is not None:
                with torch.no_grad():
                    graph.x.clamp_(0.0, 1.0)
                    # Replace NaN and Inf values
                    graph.x[torch.isnan(graph.x)] = 0.5
                    graph.x[torch.isinf(graph.x)] = 0.5
            return {'repaired': True}

        self.register_rule(ValidationRule(
            'energy_values',
            check_energy_values,
            repair_energy_values,
            severity='warning',
            interval=120.0
        ))

        # Rule 4: Connection integrity
        def check_connection_integrity(graph, id_manager):
            if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                return {'passed': True}

            invalid_connections = []
            for i in range(graph.edge_index.shape[1]):
                source_idx = graph.edge_index[0, i].item()
                target_idx = graph.edge_index[1, i].item()

                # Check if indices are within bounds
                if (source_idx >= len(graph.node_labels) or
                    target_idx >= len(graph.node_labels) or
                    source_idx < 0 or target_idx < 0):
                    invalid_connections.append((source_idx, target_idx))

            if invalid_connections:
                return {
                    'passed': False,
                    'error': f'Found {len(invalid_connections)} connections with invalid indices',
                    'invalid_connections': invalid_connections
                }
            return {'passed': True}

        def repair_connection_integrity(graph, id_manager, issue):
            # Remove invalid connections
            if 'invalid_connections' in issue and hasattr(graph, 'edge_index'):
                valid_mask = torch.ones(graph.edge_index.shape[1], dtype=torch.bool)

                for i, (source_idx, target_idx) in enumerate(issue['invalid_connections']):
                    # Find the connection index in the edge tensor
                    for j in range(graph.edge_index.shape[1]):
                        if (graph.edge_index[0, j].item() == source_idx and
                            graph.edge_index[1, j].item() == target_idx):
                            valid_mask[j] = False
                            break

                # Keep only valid connections
                graph.edge_index = graph.edge_index[:, valid_mask]

            return {'repaired': True}

        self.register_rule(ValidationRule(
            'connection_integrity',
            check_connection_integrity,
            repair_connection_integrity,
            severity='error',
            interval=90.0
        ))

    def register_rule(self, rule: ValidationRule):
        """Register a new validation rule."""
        with self._lock.write_lock():
            self._rules[rule.name] = rule
            log_step(f"Registered validation rule: {rule.name}")

    def unregister_rule(self, rule_name: str):
        """Unregister a validation rule."""
        with self._lock.write_lock():
            if rule_name in self._rules:
                del self._rules[rule_name]
                log_step(f"Unregistered validation rule: {rule_name}")

    def start_continuous_validation(self, graph_source: Callable, id_manager_source: Callable):
        """Start continuous validation in a background thread."""
        if self._running:
            log_step("Continuous validation already running")
            return

        self._running = True
        self._graph_source = graph_source
        self._id_manager_source = id_manager_source

        self._thread = threading.Thread(
            target=self._validation_loop,
            name="ContinuousGraphValidator",
            daemon=True
        )
        self._thread.start()

        log_step("Continuous graph validation started")

    def stop_continuous_validation(self):
        """Stop continuous validation."""
        if not self._running:
            return

        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        log_step("Continuous graph validation stopped")

    def _validation_loop(self):
        """Main validation loop."""
        while self._running:
            try:
                # Get current graph and ID manager
                graph = self._graph_source()
                id_manager = self._id_manager_source()

                if graph is not None and id_manager is not None:
                    self.perform_validation_cycle(graph, id_manager)

                # Sleep before next cycle
                time.sleep(self._check_interval)

            except Exception as e:
                log_step("Error in validation loop", error=str(e))
                time.sleep(self._check_interval)

    def perform_validation_cycle(self, graph, id_manager) -> Dict[str, Any]:
        """Perform one complete validation cycle."""
        with self._lock.read_lock():
            self._stats['checks_performed'] += 1

            results = []
            violations_found = 0
            repairs_attempted = 0
            repairs_successful = 0

            # Check each rule
            for rule_name, rule in self._rules.items():
                if rule.should_check():
                    result = rule.check_and_repair(graph, id_manager)
                    results.append(result)

                    if result['status'] != 'passed':
                        violations_found += 1

                    if 'repair' in result['status']:
                        repairs_attempted += 1
                        if result['status'] == 'repaired':
                            repairs_successful += 1

            # Perform full integrity check periodically
            current_time = time.time()
            if current_time - self._last_full_check >= self._full_check_interval:
                integrity_result = self.integrity_manager.check_integrity(graph, id_manager)
                if not integrity_result['is_integrity_intact']:
                    log_step("Full integrity check found violations",
                            violations=len(integrity_result['violations']))
                self._last_full_check = current_time

            # Update statistics
            self._stats['violations_found'] += violations_found
            self._stats['repairs_attempted'] += repairs_attempted
            self._stats['repairs_successful'] += repairs_successful
            self._stats['uptime'] = current_time - self._stats['start_time']

            cycle_result = {
                'cycle_time': current_time,
                'results': results,
                'violations_found': violations_found,
                'repairs_attempted': repairs_attempted,
                'repairs_successful': repairs_successful
            }

            if violations_found > 0:
                log_step("Validation cycle completed with violations",
                        violations=violations_found, repairs=repairs_successful)

            return cycle_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        with self._lock.read_lock():
            stats = self._stats.copy()

            # Add rule-specific statistics
            rule_stats = {}
            for rule_name, rule in self._rules.items():
                rule_stats[rule_name] = {
                    'violations': rule.violations,
                    'repairs': rule.repairs,
                    'last_check': rule.last_check,
                    'severity': rule.severity
                }

            stats['rules'] = rule_stats
            return stats

    def get_rule_status(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific rule."""
        with self._lock.read_lock():
            if rule_name in self._rules:
                rule = self._rules[rule_name]
                return {
                    'name': rule.name,
                    'severity': rule.severity,
                    'interval': rule.interval,
                    'last_check': rule.last_check,
                    'violations': rule.violations,
                    'repairs': rule.repairs,
                    'time_since_last_check': time.time() - rule.last_check
                }
            return None

    def force_validation_cycle(self, graph, id_manager) -> Dict[str, Any]:
        """Force a validation cycle to run immediately."""
        # Reset last check times for all rules
        with self._lock.write_lock():
            for rule in self._rules.values():
                rule.last_check = 0

        return self.perform_validation_cycle(graph, id_manager)


# Global instance
_continuous_validator_instance = None
_continuous_validator_lock = threading.Lock()


def get_continuous_graph_validator() -> ContinuousGraphValidator:
    """Get the global continuous graph validator instance."""
    global _continuous_validator_instance
    if _continuous_validator_instance is None:
        with _continuous_validator_lock:
            if _continuous_validator_instance is None:
                _continuous_validator_instance = ContinuousGraphValidator()
    return _continuous_validator_instance