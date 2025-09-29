
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any

import logging
from src.utils.logging_utils import log_step


@dataclass
class IDTransaction:
    """Represents an atomic ID management transaction."""
    operations: List[Dict[str, Any]]
    timestamp: float
    transaction_id: str

    def __init__(self, transaction_id: str):
        self.operations = []
        self.timestamp = time.time()
        self.transaction_id = transaction_id

    def add_operation(self, op_type: str, **kwargs):
        """Add an operation to this transaction."""
        self.operations.append({
            'type': op_type,
            'timestamp': time.time(),
            **kwargs
        })


class NodeIDManager:

    def __init__(self):
        # Validate and set reasonable defaults
        self._max_graph_size = 1000000  # 1M nodes max

        # Use reentrant lock for thread safety
        self._lock = threading.RLock()
        self._next_id = 1
        self._active_ids: Set[int] = set()
        self._recycled_ids: List[int] = []
        self._id_to_index: Dict[int, int] = {}
        self._index_to_id: Dict[int, int] = {}
        self._node_type_map: Dict[int, str] = {}
        self._id_metadata: Dict[int, Dict[str, Any]] = {}
        self._metadata_size_limit = 1024  # 1KB per node metadata limit

        # Transaction management with memory limits
        self._pending_transactions: Dict[str, IDTransaction] = {}
        self._completed_transactions: List[IDTransaction] = []
        self._max_transaction_history = 1000  # Keep only last 1000 transactions
        self._transaction_lock = threading.RLock()

        # Integrity checking
        self._integrity_check_enabled = True
        self._last_integrity_check = time.time()
        self._integrity_check_interval = 60.0  # Check every 60 seconds

        # Thread-safe statistics
        self._stats = {
            'total_ids_generated': 0,
            'active_ids': 0,
            'recycled_ids': 0,
            'lookup_operations': 0,
            'transactions_completed': 0,
            'integrity_checks_passed': 0,
            'creation_time': time.time(),
            'memory_usage_mb': 0.0
        }
        self._stats_lock = threading.RLock()

        log_step("NodeIDManager initialized with memory limits and thread safety")
    @contextmanager
    def transaction(self, transaction_id: Optional[str] = None):
        """Context manager for atomic ID operations."""
        if transaction_id is None:
            transaction_id = f"txn_{int(time.time() * 1000000)}"

        txn = IDTransaction(transaction_id)

        with self._transaction_lock:
            self._pending_transactions[transaction_id] = txn

        try:
            yield txn
            self._commit_transaction(transaction_id)
        except Exception as e:
            self._rollback_transaction(transaction_id)
            raise e
        finally:
            with self._transaction_lock:
                if transaction_id in self._pending_transactions:
                    del self._pending_transactions[transaction_id]

    def _commit_transaction(self, transaction_id: str):
        """Atomically commit a transaction."""
        with self._transaction_lock:
            if transaction_id not in self._pending_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            txn = self._pending_transactions[transaction_id]

        # Execute all operations atomically
        with self._lock:
            try:
                for operation in txn.operations:
                    self._execute_operation(operation)

                # Move to completed transactions and cleanup old ones
                with self._transaction_lock:
                    self._completed_transactions.append(txn)
                    self._stats['transactions_completed'] += 1

                    # Cleanup old transactions to prevent memory leaks
                    if len(self._completed_transactions) > self._max_transaction_history:
                        removed_count = len(self._completed_transactions) - self._max_transaction_history
                        self._completed_transactions = self._completed_transactions[-self._max_transaction_history:]
                        if removed_count > 0:
                            log_step(f"Cleaned up {removed_count} old transactions")

                log_step(f"Transaction {transaction_id} committed successfully",
                        operations=len(txn.operations))

            except Exception as e:
                log_step(f"Transaction {transaction_id} commit failed, rolling back", error=str(e))
                raise e

    def _rollback_transaction(self, transaction_id: str):
        """Rollback a failed transaction."""
        with self._transaction_lock:
            if transaction_id in self._pending_transactions:
                del self._pending_transactions[transaction_id]
        log_step(f"Transaction {transaction_id} rolled back")

    def _execute_operation(self, operation: Dict[str, Any]):
        """Execute a single operation within a transaction."""
        op_type = operation['type']

        if op_type == 'generate_id':
            self._execute_generate_id(operation)
        elif op_type == 'register_index':
            self._execute_register_index(operation)
        elif op_type == 'recycle_id':
            self._execute_recycle_id(operation)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    def _execute_generate_id(self, operation: Dict[str, Any]):
        """Execute ID generation within transaction (write operation) with validation."""
        with self._lock:
            node_type = operation.get('node_type', 'unknown')
            metadata = operation.get('metadata')

            # Validate inputs
            if not isinstance(node_type, str) or len(node_type) > 50:
                raise ValueError(f"Invalid node_type: must be string <= 50 chars, got {node_type}")

            if metadata is not None:
                if not isinstance(metadata, dict):
                    raise ValueError(f"Invalid metadata: must be dict or None, got {type(metadata)}")

                # Check metadata size limit
                import json
                try:
                    metadata_size = len(json.dumps(metadata).encode('utf-8'))
                    if metadata_size > self._metadata_size_limit:
                        logging.warning("Metadata size %s exceeds limit %s, truncating", metadata_size, self._metadata_size_limit)
                        # Keep only essential metadata
                        metadata = {'size_exceeded': True, 'original_size': metadata_size}
                except Exception as e:
                    logging.warning("Could not calculate metadata size: %s", e)
                    metadata = {'size_error': str(e)}

            # Check if we can generate more IDs
            if len(self._active_ids) >= self._max_graph_size:
                raise RuntimeError(f"Graph expansion limit reached: {self._max_graph_size} nodes.")

            # Generate ID
            if self._recycled_ids:
                node_id = self._recycled_ids.pop(0)
            else:
                node_id = self._next_id
                self._next_id += 1

                # Safety check to prevent infinite loops
                if self._next_id >= self._max_graph_size * 2:
                    node_id = self._find_next_available_id_efficient()
                    if node_id is None:
                        raise RuntimeError(f"Cannot find available ID within limit: {self._max_graph_size}")

            # Validate generated ID
            if node_id < 0 or node_id >= self._max_graph_size:
                raise RuntimeError(f"Generated invalid node ID: {node_id}")

            self._active_ids.add(node_id)
            self._node_type_map[node_id] = node_type
            if metadata:
                self._id_metadata[node_id] = metadata.copy()

            # Update stats thread-safely
            with self._stats_lock:
                self._stats['total_ids_generated'] += 1
                self._stats['active_ids'] = len(self._active_ids)

            # Store the generated ID in the operation for reference
            operation['generated_id'] = node_id

    def _execute_register_index(self, operation: Dict[str, Any]):
        """Execute index registration within transaction (write operation)."""
        with self._lock:
            node_id = operation['node_id']
            index = operation['index']

            if node_id not in self._active_ids:
                raise ValueError(f"Cannot register index for inactive ID: {node_id}")

            self._id_to_index[node_id] = index
            self._index_to_id[index] = node_id

    def _execute_recycle_id(self, operation: Dict[str, Any]):
        """Execute ID recycling within transaction (write operation)."""
        with self._lock:
            node_id = operation['node_id']

            if node_id not in self._active_ids:
                raise ValueError(f"Cannot recycle inactive ID: {node_id}")

            self._active_ids.remove(node_id)

            # Clean up mappings
            if node_id in self._id_to_index:
                index = self._id_to_index[node_id]
                del self._id_to_index[node_id]
                if index in self._index_to_id:
                    del self._index_to_id[index]

            if node_id in self._node_type_map:
                del self._node_type_map[node_id]

            if node_id in self._id_metadata:
                del self._id_metadata[node_id]

            self._recycled_ids.append(node_id)
            self._stats['active_ids'] = len(self._active_ids)
            self._stats['recycled_ids'] = len(self._recycled_ids)

    def set_max_graph_size(self, max_size: int):
        """Set maximum graph size (write operation)."""
        with self._lock:
            self._max_graph_size = max_size
            log_step("Max graph size set", max_size=max_size)
    def generate_unique_id(self, node_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> int:
        """Generate a unique ID using atomic transaction with validation."""
        # Input validation
        if not isinstance(node_type, str):
            node_type = str(node_type)
        if len(node_type) > 50:
            node_type = node_type[:50]

        if metadata is not None and not isinstance(metadata, dict):
            logging.warning(f"Invalid metadata type {type(metadata)}, ignoring")
            metadata = None

        try:
            with self.transaction() as txn:
                txn.add_operation('generate_id', node_type=node_type, metadata=metadata)

            # The transaction will have stored the generated ID
            generated_id = None
            for op in txn.operations:
                if op['type'] == 'generate_id':
                    generated_id = op.get('generated_id')
                    break

            if generated_id is None:
                raise RuntimeError("Failed to generate ID in transaction")

            if generated_id % 10000 == 0:
                log_step("ID generation batch", node_id=generated_id, node_type=node_type, batch_size=10000)

            return generated_id

        except Exception as e:
            logging.error(f"Failed to generate unique ID for type '{node_type}': {e}")
            raise
    def _find_next_available_id_efficient(self) -> Optional[int]:
        """Find next available ID efficiently (read operation)."""
        with self._lock:
            # Start from a reasonable point to avoid conflicts with low IDs
            start_id = max(1000, self._next_id - 10000)

            # Try a range around the current next_id first
            for i in range(start_id, min(start_id + 20000, self._max_graph_size)):
                if i not in self._active_ids:
                    return i

            # If that fails, search from the beginning in larger steps
            for i in range(1, self._max_graph_size, 100):  # Check every 100th ID
                if i not in self._active_ids:
                    return i

            # Last resort: linear search in a limited range
            search_limit = min(10000, self._max_graph_size)
            for i in range(1, search_limit):
                if i not in self._active_ids:
                    return i

            return None

    def _find_next_available_id(self) -> Optional[int]:
        """Legacy method - redirects to efficient implementation."""
        return self._find_next_available_id_efficient()
    def register_node_index(self, node_id: int, index: int) -> bool:
        """Register node index using atomic transaction with validation."""
        # Input validation
        if not isinstance(node_id, int) or node_id < 0:
            logging.error(f"Invalid node_id: {node_id}")
            return False

        if not isinstance(index, int) or index < 0:
            logging.error(f"Invalid index: {index}")
            return False

        if index >= self._max_graph_size:
            logging.error(f"Index {index} exceeds max graph size {self._max_graph_size}")
            return False

        try:
            with self.transaction() as txn:
                txn.add_operation('register_index', node_id=node_id, index=index)

            if index % 5000 == 0:
                log_step("Node index registered batch", node_id=node_id, index=index, batch_size=5000)
            return True
        except ValueError as e:
            logging.warning(f"Failed to register index for node {node_id}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error registering index for node {node_id}: {e}")
            return False
    def get_node_index(self, node_id: int) -> Optional[int]:
        """Get node index (read operation)."""
        with self._lock:
            self._stats['lookup_operations'] += 1
            return self._id_to_index.get(node_id)

    def get_node_id(self, index: int) -> Optional[int]:
        """Get node ID by index (read operation)."""
        with self._lock:
            self._stats['lookup_operations'] += 1
            return self._index_to_id.get(index)

    def is_valid_id(self, node_id: int) -> bool:
        """Check if ID is valid (read operation)."""
        with self._lock:
            return node_id in self._active_ids

    def get_node_type(self, node_id: int) -> Optional[str]:
        """Get node type (read operation)."""
        with self._lock:
            return self._node_type_map.get(node_id)

    def get_node_metadata(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get node metadata (read operation)."""
        with self._lock:
            return self._id_metadata.get(node_id)
    def recycle_node_id(self, node_id: int) -> bool:
        """Recycle node ID using atomic transaction with validation."""
        # Input validation
        if not isinstance(node_id, int) or node_id <= 0:
            logging.error(f"Invalid node_id for recycling: {node_id}")
            return False

        try:
            with self.transaction() as txn:
                txn.add_operation('recycle_id', node_id=node_id)

            log_step("Node ID recycled", node_id=node_id)
            return True
        except ValueError as e:
            logging.warning(f"Failed to recycle node ID {node_id}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error recycling node ID {node_id}: {e}")
            return False
    def get_all_active_ids(self) -> List[int]:
        """Get all active IDs (read operation)."""
        with self._lock:
            return list(self._active_ids)

    def get_ids_by_type(self, node_type: str) -> List[int]:
        """Get IDs by type (read operation)."""
        with self._lock:
            return [node_id for node_id, ntype in self._node_type_map.items()
                   if ntype == node_type and node_id in self._active_ids]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics with thread safety and memory usage tracking."""
        with self._lock:
            with self._stats_lock:
                stats = self._stats.copy()

            stats['uptime'] = time.time() - stats['creation_time']
            stats['recycled_ids_count'] = len(self._recycled_ids)

            # Calculate memory usage
            try:
                import sys
                # Rough memory estimation
                memory_usage = (
                    sys.getsizeof(self._active_ids) +
                    sys.getsizeof(self._recycled_ids) +
                    sys.getsizeof(self._id_to_index) +
                    sys.getsizeof(self._index_to_id) +
                    sys.getsizeof(self._node_type_map) +
                    sys.getsizeof(self._id_metadata) +
                    sum(sys.getsizeof(txn) for txn in self._completed_transactions)
                )
                stats['memory_usage_mb'] = memory_usage / (1024 * 1024)
            except Exception:
                stats['memory_usage_mb'] = 0.0

            # Add transaction statistics
            with self._transaction_lock:
                stats['pending_transactions'] = len(self._pending_transactions)
                stats['completed_transactions'] = len(self._completed_transactions)

            # Add integrity check statistics
            stats['integrity_checks_enabled'] = self._integrity_check_enabled
            stats['time_since_last_integrity_check'] = time.time() - self._last_integrity_check

            # Add capacity information
            stats['expansion_capacity'] = self.get_expansion_capacity()
            stats['utilization_percent'] = (len(self._active_ids) / self._max_graph_size * 100) if self._max_graph_size > 0 else 0.0

            return stats

    def get_transaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transaction history (read operation)."""
        with self._lock:
            with self._transaction_lock:
                recent_transactions = self._completed_transactions[-limit:]
                return [
                    {
                        'transaction_id': txn.transaction_id,
                        'timestamp': txn.timestamp,
                        'operations_count': len(txn.operations),
                        'operation_types': [op['type'] for op in txn.operations]
                    }
                    for txn in recent_transactions
                ]
    def validate_graph_consistency(self, graph) -> Dict[str, Any]:
        """Enhanced graph consistency validation with integrity checking (read operation)."""

        with self._lock:
            validation_results = {
                'is_consistent': True,
                'errors': [],
                'warnings': [],
                'node_count_mismatch': False,
                'missing_ids': [],
                'orphaned_ids': [],
                'integrity_check_passed': False,
                'last_integrity_check': self._last_integrity_check
            }

            if not hasattr(graph, 'node_labels'):
                validation_results['errors'].append("Graph missing node_labels")
                validation_results['is_consistent'] = False
                return validation_results

            graph_node_count = len(graph.node_labels)
            active_id_count = len(self._active_ids)

            if graph_node_count != active_id_count:
                validation_results['node_count_mismatch'] = True
                validation_results['warnings'].append(
                    f"Node count mismatch: graph has {graph_node_count} nodes, "
                    f"ID manager has {active_id_count} active IDs"
                )

            # Enhanced validation with integrity checks
            for i, node_label in enumerate(graph.node_labels):
                node_id = node_label.get('id')
                if node_id is None:
                    validation_results['missing_ids'].append(i)
                    validation_results['errors'].append(f"Node at index {i} missing ID")
                elif not self.is_valid_id(node_id):
                    validation_results['errors'].append(f"Node at index {i} has invalid ID: {node_id}")
                elif self.get_node_index(node_id) != i:
                    validation_results['errors'].append(
                        f"Node ID {node_id} index mismatch: expected {i}, got {self.get_node_index(node_id)}"
                    )
                else:
                    # Additional integrity checks
                    expected_type = node_label.get('type', 'unknown')
                    actual_type = self.get_node_type(node_id)
                    if expected_type != actual_type:
                        validation_results['warnings'].append(
                            f"Node {node_id} type mismatch: expected {expected_type}, got {actual_type}"
                        )

            for node_id in self._active_ids:
                if node_id not in self._id_to_index:
                    validation_results['orphaned_ids'].append(node_id)
                    validation_results['warnings'].append(f"Orphaned ID: {node_id}")

            # Perform periodic integrity check
            current_time = time.time()
            if self._integrity_check_enabled and (current_time - self._last_integrity_check) > self._integrity_check_interval:
                integrity_result = self._perform_integrity_check()
                validation_results['integrity_check_passed'] = integrity_result['passed']
                if not integrity_result['passed']:
                    validation_results['errors'].extend(integrity_result['errors'])
                self._last_integrity_check = current_time
                self._stats['integrity_checks_passed'] += 1 if integrity_result['passed'] else 0

            if validation_results['errors']:
                validation_results['is_consistent'] = False
            return validation_results

    def _perform_integrity_check(self) -> Dict[str, Any]:
        """Perform comprehensive integrity check of ID mappings (read operation)."""
        with self._lock:
            result = {'passed': True, 'errors': [], 'warnings': []}

            try:
                # Check bidirectional consistency
                for node_id, index in self._id_to_index.items():
                    if node_id not in self._active_ids:
                        result['errors'].append(f"ID {node_id} in index map but not active")
                        result['passed'] = False

                    reverse_id = self._index_to_id.get(index)
                    if reverse_id != node_id:
                        result['errors'].append(f"Index {index} maps to {reverse_id}, expected {node_id}")
                        result['passed'] = False

                # Check for duplicate indices
                indices = list(self._id_to_index.values())
                if len(indices) != len(set(indices)):
                    result['errors'].append("Duplicate indices found in ID-to-index mapping")
                    result['passed'] = False

                # Check for duplicate IDs
                if len(self._active_ids) != len(self._id_to_index):
                    result['errors'].append("Mismatch between active IDs and index mappings")
                    result['passed'] = False

                # Check recycled IDs don't conflict
                for recycled_id in self._recycled_ids:
                    if recycled_id in self._active_ids:
                        result['errors'].append(f"Recycled ID {recycled_id} still marked as active")
                        result['passed'] = False

            except Exception as e:
                result['passed'] = False
                result['errors'].append(f"Integrity check failed with exception: {e}")

            return result

    def enable_integrity_checks(self, enabled: bool = True, interval: float = 60.0):
        """Enable or disable periodic integrity checks (write operation)."""
        with self._lock:
            self._integrity_check_enabled = enabled
            self._integrity_check_interval = interval
            log_step(f"Integrity checks {'enabled' if enabled else 'disabled'}",
                    interval=interval)
    def can_expand_graph(self, additional_nodes: int = 1) -> bool:
        """Check if the graph can expand by the given number of nodes (read operation)."""
        with self._lock:
            return len(self._active_ids) + additional_nodes <= self._max_graph_size

    def get_expansion_capacity(self) -> int:
        """Get the number of additional nodes that can be added before hitting the limit (read operation)."""
        with self._lock:
            return max(0, self._max_graph_size - len(self._active_ids))

    def cleanup_orphaned_ids(self, graph) -> int:
        """Remove IDs that are no longer in the graph and return count of cleaned IDs (write operation)."""
        with self._lock:
            if not hasattr(graph, 'node_labels'):
                return 0

            graph_node_ids = {node.get('id') for node in graph.node_labels if node.get('id') is not None}
            orphaned_ids = self._active_ids - graph_node_ids

            for node_id in orphaned_ids:
                self.recycle_node_id(node_id)

            log_step(f"Cleaned up {len(orphaned_ids)} orphaned IDs")
            return len(orphaned_ids)

    def reset(self):
        """Reset the ID manager (write operation) with memory cleanup."""
        with self._lock:
            self._next_id = 1
            self._active_ids.clear()
            self._recycled_ids.clear()
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._node_type_map.clear()
            self._id_metadata.clear()

            # Clear transaction history
            with self._transaction_lock:
                self._pending_transactions.clear()
                self._completed_transactions.clear()

            # Reset stats
            with self._stats_lock:
                self._stats = {
                    'total_ids_generated': 0,
                    'active_ids': 0,
                    'recycled_ids': 0,
                    'lookup_operations': 0,
                    'transactions_completed': 0,
                    'integrity_checks_passed': 0,
                    'creation_time': time.time(),
                    'memory_usage_mb': 0.0
                }

            # Reset max graph size to default
            self._max_graph_size = 1000000  # 1 million nodes default
            log_step("NodeIDManager reset", max_size=self._max_graph_size)

    def cleanup_memory(self):
        """Clean up memory by removing old transaction history and optimizing data structures."""
        with self._lock:
            # Clean up old transactions
            with self._transaction_lock:
                if len(self._completed_transactions) > self._max_transaction_history // 2:
                    self._completed_transactions = self._completed_transactions[-self._max_transaction_history // 2:]

            # Clean up orphaned metadata (IDs that exist in metadata but not in active_ids)
            orphaned_metadata = []
            for node_id in self._id_metadata:
                if node_id not in self._active_ids:
                    orphaned_metadata.append(node_id)

            for node_id in orphaned_metadata:
                del self._id_metadata[node_id]

            # Optimize recycled IDs list if it gets too large
            if len(self._recycled_ids) > 10000:
                # Keep only the most recently recycled IDs
                self._recycled_ids = self._recycled_ids[-5000:]

            log_step("Memory cleanup completed",
                    transactions_cleaned=len(self._completed_transactions),
                    metadata_cleaned=len(orphaned_metadata),
                    recycled_trimmed=len(self._recycled_ids))
_id_manager_instance = None
_id_manager_lock = threading.Lock()


def get_id_manager() -> NodeIDManager:

    global _id_manager_instance
    if _id_manager_instance is None:
        with _id_manager_lock:
            if _id_manager_instance is None:
                _id_manager_instance = NodeIDManager()
    return _id_manager_instance


def reset_id_manager():
    global _id_manager_instance
    with _id_manager_lock:
        if _id_manager_instance is not None:
            _id_manager_instance.reset()
        else:
            _id_manager_instance = NodeIDManager()

def force_reset_id_manager():
    """Force reset the ID manager, creating a new instance."""
    global _id_manager_instance
    with _id_manager_lock:
        _id_manager_instance = None
        log_step("ID manager force reset - new instance will be created")


def generate_node_id(node_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> int:
    return get_id_manager().generate_unique_id(node_type, metadata)


def get_node_index_by_id(node_id: int) -> Optional[int]:
    return get_id_manager().get_node_index(node_id)


def get_node_id_by_index(index: int) -> Optional[int]:
    return get_id_manager().get_node_id(index)


def is_valid_node_id(node_id: int) -> bool:
    return get_id_manager().is_valid_id(node_id)


def recycle_node_id(node_id: int) -> bool:
    return get_id_manager().recycle_node_id(node_id)







