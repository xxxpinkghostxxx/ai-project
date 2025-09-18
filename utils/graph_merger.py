"""
Graph Merger
Handles merging of neural simulation graphs with proper ID conflict resolution.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from utils.logging_utils import log_step
from utils.reader_writer_lock import get_graph_lock


class GraphMergeConflictError(Exception):
    """Exception raised when graph merging encounters unresolvable conflicts."""
    pass


class GraphMerger:
    """Handles merging of neural simulation graphs with ID conflict resolution."""

    def __init__(self):
        self._lock = get_graph_lock()
        self._merge_history: List[Dict[str, Any]] = []
        self._stats = {
            'total_merges': 0,
            'successful_merges': 0,
            'failed_merges': 0,
            'id_conflicts_resolved': 0,
            'nodes_merged': 0,
            'edges_merged': 0
        }

    def merge_graphs(self, primary_graph, secondary_graph,
                    conflict_resolution: str = 'auto',
                    preserve_primary_ids: bool = True) -> Dict[str, Any]:
        """
        Merge two graphs with intelligent ID conflict resolution.

        Args:
            primary_graph: The primary graph (keeps its IDs when possible)
            secondary_graph: The secondary graph (IDs may be remapped)
            conflict_resolution: Strategy for resolving conflicts ('auto', 'rename', 'overwrite')
            preserve_primary_ids: Whether to preserve primary graph IDs

        Returns:
            Dict containing merged graph and merge statistics
        """
        with self._lock.write_lock():
            start_time = time.time()

            try:
                # Validate input graphs
                validation_result = self._validate_graphs_for_merge(primary_graph, secondary_graph)
                if not validation_result['valid']:
                    raise GraphMergeConflictError(f"Graph validation failed: {validation_result['errors']}")

                # Analyze ID conflicts
                conflict_analysis = self._analyze_id_conflicts(primary_graph, secondary_graph)

                # Create ID mapping for conflict resolution
                id_mapping = self._create_id_mapping(
                    primary_graph, secondary_graph, conflict_analysis,
                    conflict_resolution, preserve_primary_ids
                )

                # Perform the merge
                merged_graph = self._perform_merge(
                    primary_graph, secondary_graph, id_mapping
                )

                # Update statistics
                merge_time = time.time() - start_time
                merge_stats = {
                    'merge_time': merge_time,
                    'conflicts_resolved': len(conflict_analysis['conflicts']),
                    'nodes_in_result': len(merged_graph.node_labels) if hasattr(merged_graph, 'node_labels') else 0,
                    'edges_in_result': merged_graph.edge_index.shape[1] if hasattr(merged_graph, 'edge_index') else 0,
                    'id_mapping_size': len(id_mapping)
                }

                # Record merge in history
                self._record_merge_history(primary_graph, secondary_graph, merged_graph, merge_stats)

                # Update global statistics
                self._stats['total_merges'] += 1
                self._stats['successful_merges'] += 1
                self._stats['id_conflicts_resolved'] += len(conflict_analysis['conflicts'])
                self._stats['nodes_merged'] += merge_stats['nodes_in_result']
                self._stats['edges_merged'] += merge_stats['edges_in_result']

                log_step("Graph merge completed successfully",
                        conflicts_resolved=len(conflict_analysis['conflicts']),
                        nodes_merged=merge_stats['nodes_in_result'],
                        merge_time=f"{merge_time:.3f}s")

                return {
                    'merged_graph': merged_graph,
                    'id_mapping': id_mapping,
                    'statistics': merge_stats,
                    'conflicts': conflict_analysis['conflicts']
                }

            except Exception as e:
                self._stats['failed_merges'] += 1
                log_step("Graph merge failed", error=str(e))
                raise GraphMergeConflictError(f"Graph merge failed: {e}")

    def _validate_graphs_for_merge(self, graph1, graph2) -> Dict[str, Any]:
        """Validate that graphs can be merged."""
        result = {'valid': True, 'errors': [], 'warnings': []}

        # Check for required attributes
        required_attrs = ['node_labels', 'x']
        for attr in required_attrs:
            if not hasattr(graph1, attr):
                result['errors'].append(f"Primary graph missing required attribute: {attr}")
                result['valid'] = False
            if not hasattr(graph2, attr):
                result['errors'].append(f"Secondary graph missing required attribute: {attr}")
                result['valid'] = False

        # Check graph sizes
        if hasattr(graph1, 'node_labels') and hasattr(graph2, 'node_labels'):
            size1 = len(graph1.node_labels)
            size2 = len(graph2.node_labels)
            total_size = size1 + size2

            if total_size > 1000000:  # 1M node limit
                result['warnings'].append(f"Large merge operation: {total_size} total nodes")

        return result

    def _analyze_id_conflicts(self, graph1, graph2) -> Dict[str, Any]:
        """Analyze ID conflicts between the two graphs."""
        conflicts = []
        graph1_ids = set()
        graph2_ids = set()

        # Extract IDs from both graphs
        if hasattr(graph1, 'node_labels'):
            graph1_ids = {node.get('id') for node in graph1.node_labels if node.get('id') is not None}

        if hasattr(graph2, 'node_labels'):
            graph2_ids = {node.get('id') for node in graph2.node_labels if node.get('id') is not None}

        # Find conflicts
        conflicting_ids = graph1_ids & graph2_ids
        for conflict_id in conflicting_ids:
            # Get node details for conflict analysis
            graph1_node = next((node for node in graph1.node_labels if node.get('id') == conflict_id), None)
            graph2_node = next((node for node in graph2.node_labels if node.get('id') == conflict_id), None)

            conflicts.append({
                'id': conflict_id,
                'graph1_node': graph1_node,
                'graph2_node': graph2_node,
                'severity': self._assess_conflict_severity(graph1_node, graph2_node)
            })

        return {
            'conflicts': conflicts,
            'graph1_unique_ids': len(graph1_ids - graph2_ids),
            'graph2_unique_ids': len(graph2_ids - graph1_ids),
            'total_conflicts': len(conflicts)
        }

    def _assess_conflict_severity(self, node1, node2) -> str:
        """Assess the severity of an ID conflict."""
        if not node1 or not node2:
            return 'unknown'

        # Check if nodes are identical
        keys_to_check = ['type', 'energy', 'x', 'y', 'behavior']
        identical = True
        for key in keys_to_check:
            val1 = node1.get(key)
            val2 = node2.get(key)
            if val1 != val2:
                identical = False
                break

        if identical:
            return 'identical'  # Can safely keep one
        else:
            return 'different'  # Need to resolve conflict

    def _create_id_mapping(self, graph1, graph2, conflict_analysis,
                          resolution_strategy: str, preserve_primary: bool) -> Dict[int, int]:
        """Create mapping for resolving ID conflicts."""
        id_mapping = {}
        next_available_id = self._find_next_available_id(graph1, graph2)

        for conflict in conflict_analysis['conflicts']:
            conflict_id = conflict['id']

            if resolution_strategy == 'auto':
                if conflict['severity'] == 'identical':
                    # Keep primary, map secondary to primary
                    id_mapping[conflict_id] = conflict_id
                else:
                    # Generate new ID for secondary
                    new_id = next_available_id
                    next_available_id += 1
                    id_mapping[conflict_id] = new_id

            elif resolution_strategy == 'rename':
                # Always rename conflicting IDs
                new_id = next_available_id
                next_available_id += 1
                id_mapping[conflict_id] = new_id

            elif resolution_strategy == 'overwrite':
                # Keep primary IDs, overwrite secondary
                id_mapping[conflict_id] = conflict_id

        return id_mapping

    def _find_next_available_id(self, graph1, graph2) -> int:
        """Find the next available ID that doesn't conflict with either graph."""
        used_ids = set()

        # Collect all used IDs
        for graph in [graph1, graph2]:
            if hasattr(graph, 'node_labels'):
                for node in graph.node_labels:
                    node_id = node.get('id')
                    if node_id is not None:
                        used_ids.add(node_id)

        # Find next available ID
        candidate_id = 1
        while candidate_id in used_ids:
            candidate_id += 1

        return candidate_id

    def _perform_merge(self, graph1, graph2, id_mapping: Dict[int, int]):
        """Perform the actual graph merge."""
        import torch

        # Start with a copy of the primary graph
        merged_graph = self._copy_graph_structure(graph1)

        # Add nodes from secondary graph with ID remapping
        if hasattr(graph2, 'node_labels'):
            for node in graph2.node_labels:
                original_id = node.get('id')
                if original_id is not None:
                    # Remap ID if needed
                    new_id = id_mapping.get(original_id, original_id)

                    # Create new node with remapped ID
                    new_node = node.copy()
                    new_node['id'] = new_id

                    merged_graph.node_labels.append(new_node)

                    # Add energy value if available
                    if hasattr(graph2, 'x') and hasattr(merged_graph, 'x'):
                        # Find the index of this node in the secondary graph
                        node_idx = graph2.node_labels.index(node)
                        if node_idx < graph2.x.shape[0]:
                            energy_value = graph2.x[node_idx].clone()
                            if merged_graph.x is None:
                                merged_graph.x = energy_value.unsqueeze(0)
                            else:
                                merged_graph.x = torch.cat([merged_graph.x, energy_value.unsqueeze(0)], dim=0)

        # Merge edge indices with ID remapping
        if hasattr(graph2, 'edge_index') and graph2.edge_index is not None:
            remapped_edges = graph2.edge_index.clone()

            # Remap source and target IDs in edges
            for i in range(remapped_edges.shape[1]):
                source_id = remapped_edges[0, i].item()
                target_id = remapped_edges[1, i].item()

                # Find the node indices in the merged graph
                source_idx = self._find_node_index_by_id(merged_graph, id_mapping.get(source_id, source_id))
                target_idx = self._find_node_index_by_id(merged_graph, id_mapping.get(target_id, target_id))

                if source_idx is not None and target_idx is not None:
                    remapped_edges[0, i] = source_idx
                    remapped_edges[1, i] = target_idx

            # Add remapped edges to merged graph
            if merged_graph.edge_index is None:
                merged_graph.edge_index = remapped_edges
            else:
                merged_graph.edge_index = torch.cat([merged_graph.edge_index, remapped_edges], dim=1)

        return merged_graph

    def _copy_graph_structure(self, graph):
        """Create a copy of the graph structure."""
        import torch

        # Create a basic copy
        copied_graph = type(graph)()

        # Copy attributes
        if hasattr(graph, 'node_labels'):
            copied_graph.node_labels = graph.node_labels.copy()

        if hasattr(graph, 'x') and graph.x is not None:
            copied_graph.x = graph.x.clone()

        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            copied_graph.edge_index = graph.edge_index.clone()

        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            copied_graph.edge_attr = graph.edge_attr.clone()

        # Copy other attributes
        for attr in dir(graph):
            if not attr.startswith('_') and attr not in ['node_labels', 'x', 'edge_index', 'edge_attr']:
                try:
                    value = getattr(graph, attr)
                    if not callable(value):
                        setattr(copied_graph, attr, value)
                except:
                    pass

        return copied_graph

    def _find_node_index_by_id(self, graph, node_id: int) -> Optional[int]:
        """Find the index of a node by its ID in the graph."""
        if hasattr(graph, 'node_labels'):
            for idx, node in enumerate(graph.node_labels):
                if node.get('id') == node_id:
                    return idx
        return None

    def _record_merge_history(self, graph1, graph2, merged_graph, stats: Dict[str, Any]):
        """Record merge operation in history."""
        history_entry = {
            'timestamp': time.time(),
            'graph1_size': len(graph1.node_labels) if hasattr(graph1, 'node_labels') else 0,
            'graph2_size': len(graph2.node_labels) if hasattr(graph2, 'node_labels') else 0,
            'merged_size': len(merged_graph.node_labels) if hasattr(merged_graph, 'node_labels') else 0,
            'statistics': stats
        }

        self._merge_history.append(history_entry)

        # Keep only recent history
        if len(self._merge_history) > 100:
            self._merge_history = self._merge_history[-100:]

    def get_merge_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent merge history."""
        with self._lock.read_lock():
            return self._merge_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get merger statistics."""
        with self._lock.read_lock():
            return self._stats.copy()


# Global instance
_graph_merger_instance = None
_graph_merger_lock = threading.Lock()


def get_graph_merger() -> GraphMerger:
    """Get the global graph merger instance."""
    global _graph_merger_instance
    if _graph_merger_instance is None:
        with _graph_merger_lock:
            if _graph_merger_instance is None:
                _graph_merger_instance = GraphMerger()
    return _graph_merger_instance