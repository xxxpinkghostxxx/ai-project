"""
Graph Integrity Manager
Provides versioning and integrity checking for neural simulation graphs.
"""

import hashlib
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from src.utils.logging_utils import log_step


@dataclass
class GraphVersion:
    """Represents a version of the graph with integrity information."""
    version_id: str
    timestamp: float
    node_count: int
    edge_count: int
    content_hash: str
    id_manager_hash: str
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None


@dataclass
class IntegrityViolation:
    """Represents an integrity violation detected in the graph."""
    violation_type: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    timestamp: float
    affected_nodes: List[int]
    suggested_fix: Optional[str] = None


class GraphIntegrityManager:
    """Manages graph versioning and integrity checking."""

    def __init__(self):
        self._lock = threading.RLock()
        self._versions: Dict[str, GraphVersion] = {}
        self._current_version: Optional[str] = None
        self._violations: List[IntegrityViolation] = []
        self._integrity_check_interval = 30.0  # Check every 30 seconds
        self._last_integrity_check = time.time()
        self._auto_repair_enabled = True

        # Version management
        self._max_versions = 100  # Keep last 100 versions
        self._version_counter = 0

        log_step("GraphIntegrityManager initialized")

    def create_version(self, graph, id_manager, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new version of the graph with integrity information."""
        with self._lock:
            version_id = f"v{self._version_counter}_{int(time.time())}"
            self._version_counter += 1

            # Calculate content hashes
            content_hash = self._calculate_graph_hash(graph)
            id_manager_hash = self._calculate_id_manager_hash(id_manager)

            # Get graph statistics
            node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
            edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0

            version = GraphVersion(
                version_id=version_id,
                timestamp=time.time(),
                node_count=node_count,
                edge_count=edge_count,
                content_hash=content_hash,
                id_manager_hash=id_manager_hash,
                metadata=metadata or {},
                parent_version=self._current_version
            )

            self._versions[version_id] = version
            self._current_version = version_id

            # Clean up old versions if needed
            self._cleanup_old_versions()

            log_step(f"Graph version created: {version_id}",
                    node_count=node_count, edge_count=edge_count)
            return version_id

    def check_integrity(self, graph, id_manager) -> Dict[str, Any]:
        """Check graph integrity against current version."""
        with self._lock:
            result = {
                'is_integrity_intact': True,
                'violations': [],
                'current_version': self._current_version,
                'last_check': self._last_integrity_check
            }

            # Check for None graph
            if graph is None:
                result['violations'].append(IntegrityViolation(
                    'invalid_graph',
                    'Graph is None - cannot perform integrity check',
                    'critical',
                    time.time(),
                    []
                ))
                result['is_integrity_intact'] = False
                return result

            if not self._current_version:
                result['violations'].append(IntegrityViolation(
                    'no_baseline_version',
                    'No baseline version exists for integrity checking',
                    'warning',
                    time.time(),
                    []
                ))
                return result

            current_version = self._versions[self._current_version]

            # Check content hash
            current_hash = self._calculate_graph_hash(graph)
            if current_hash != current_version.content_hash:
                violation = IntegrityViolation(
                    'content_hash_mismatch',
                    f'Graph content hash changed from {current_version.content_hash} to {current_hash}',
                    'critical',
                    time.time(),
                    [],
                    'Consider reverting to previous version or repairing graph consistency'
                )
                result['violations'].append(violation)
                result['is_integrity_intact'] = False

            # Check ID manager hash
            id_hash = self._calculate_id_manager_hash(id_manager)
            if id_hash != current_version.id_manager_hash:
                violation = IntegrityViolation(
                    'id_manager_hash_mismatch',
                    f'ID manager state changed from {current_version.id_manager_hash} to {id_hash}',
                    'critical',
                    time.time(),
                    [],
                    'Check ID manager consistency and repair mappings'
                )
                result['violations'].append(violation)
                result['is_integrity_intact'] = False

            # Check node/edge counts
            current_node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
            current_edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0

            if current_node_count != current_version.node_count:
                violation = IntegrityViolation(
                    'node_count_mismatch',
                    f'Node count changed from {current_version.node_count} to {current_node_count}',
                    'warning',
                    time.time(),
                    []
                )
                result['violations'].append(violation)

            if current_edge_count != current_version.edge_count:
                violation = IntegrityViolation(
                    'edge_count_mismatch',
                    f'Edge count changed from {current_version.edge_count} to {current_edge_count}',
                    'warning',
                    time.time(),
                    []
                )
                result['violations'].append(violation)

            # Store violations for later analysis
            self._violations.extend(result['violations'])
            self._last_integrity_check = time.time()

            return result

    def _calculate_graph_hash(self, graph) -> str:
        """Calculate a hash of the graph's content."""
        try:
            # Include node labels and edge indices in hash
            hash_components = []

            if hasattr(graph, 'node_labels'):
                # Sort node labels by ID for consistent hashing
                sorted_nodes = sorted(graph.node_labels, key=lambda x: x.get('id', 0))
                hash_components.append(str(sorted_nodes))

            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                hash_components.append(str(graph.edge_index.tolist()))

            if hasattr(graph, 'x') and graph.x is not None:
                hash_components.append(str(graph.x.tolist()))

            content = '|'.join(hash_components)
            return hashlib.sha256(content.encode()).hexdigest()[:16]  # Short hash for efficiency

        except Exception as e:
            log_step(f"Error calculating graph hash: {e}")
            return "error_hash"

    def _calculate_id_manager_hash(self, id_manager) -> str:
        """Calculate a hash of the ID manager's state."""
        try:
            # Include key ID manager state in hash
            state_components = [
                str(sorted(id_manager._active_ids)),
                str(sorted(id_manager._id_to_index.items())),
                str(sorted(id_manager._node_type_map.items()))
            ]
            content = '|'.join(state_components)
            return hashlib.sha256(content.encode()).hexdigest()[:16]

        except Exception as e:
            log_step(f"Error calculating ID manager hash: {e}")
            return "error_hash"

    def _cleanup_old_versions(self):
        """Clean up old versions to prevent memory bloat."""
        if len(self._versions) > self._max_versions:
            # Remove oldest versions, keeping the most recent
            version_items = sorted(self._versions.items(),
                                 key=lambda x: x[1].timestamp,
                                 reverse=True)
            versions_to_keep = version_items[:self._max_versions]
            self._versions = dict(versions_to_keep)

    def get_version_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent version history."""
        with self._lock:
            recent_versions = sorted(self._versions.values(),
                                   key=lambda v: v.timestamp,
                                   reverse=True)[:limit]
            return [
                {
                    'version_id': v.version_id,
                    'timestamp': v.timestamp,
                    'node_count': v.node_count,
                    'edge_count': v.edge_count,
                    'parent_version': v.parent_version
                }
                for v in recent_versions
            ]

    def get_violation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent integrity violations."""
        with self._lock:
            recent_violations = self._violations[-limit:]
            return [
                {
                    'violation_type': v.violation_type,
                    'description': v.description,
                    'severity': v.severity,
                    'timestamp': v.timestamp,
                    'affected_nodes_count': len(v.affected_nodes),
                    'suggested_fix': v.suggested_fix
                }
                for v in recent_violations
            ]

    def enable_auto_repair(self, enabled: bool = True):
        """Enable or disable automatic integrity repair."""
        self._auto_repair_enabled = enabled
        log_step(f"Auto repair {'enabled' if enabled else 'disabled'}")

    def set_integrity_check_interval(self, interval: float):
        """Set the interval for automatic integrity checks."""
        self._integrity_check_interval = interval
        log_step(f"Integrity check interval set to {interval}s")

    def get_statistics(self) -> Dict[str, Any]:
        """Get integrity manager statistics."""
        with self._lock:
            return {
                'total_versions': len(self._versions),
                'current_version': self._current_version,
                'total_violations': len(self._violations),
                'auto_repair_enabled': self._auto_repair_enabled,
                'integrity_check_interval': self._integrity_check_interval,
                'time_since_last_check': time.time() - self._last_integrity_check,
                'versions_kept': self._max_versions
            }


# Global instance
_integrity_manager_instance = None
_integrity_manager_lock = threading.Lock()


def get_graph_integrity_manager() -> GraphIntegrityManager:
    """Get the global graph integrity manager instance."""
    global _integrity_manager_instance
    if _integrity_manager_instance is None:
        with _integrity_manager_lock:
            if _integrity_manager_instance is None:
                _integrity_manager_instance = GraphIntegrityManager()
    return _integrity_manager_instance






