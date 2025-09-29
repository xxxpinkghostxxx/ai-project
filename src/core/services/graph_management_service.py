"""
GraphManagementService implementation - Neural graph management service.

This module provides the concrete implementation of IGraphManager,
handling neural graph operations including persistence, validation,
and structural integrity management.
"""

import json
import os
import pickle
import time
from typing import Any, Dict, Optional

import torch
import torch_geometric
from torch_geometric.data import Data

from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.graph_manager import IGraphManager


class GraphManagementService(IGraphManager):
    """
    Concrete implementation of IGraphManager.

    This service provides comprehensive neural graph management including
    initialization, persistence, validation, and structural integrity checks.
    """

    def __init__(self,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator):
        """
        Initialize the GraphManagementService.

        Args:
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
        """
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator

        # Graph management settings
        self._default_graph_size = 1000
        self._max_graph_size = 10000
        self._persistence_format = "pytorch"  # "pytorch", "json", "pickle"

        # Graph statistics cache
        self._graph_stats_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp = 0
        self._cache_ttl = 60  # seconds

    def get_default_graph_size(self) -> int:
        """Get the default graph size."""
        return self._default_graph_size

    def get_max_graph_size(self) -> int:
        """Get the maximum graph size."""
        return self._max_graph_size

    def get_persistence_format(self) -> str:
        """Get the persistence format."""
        return self._persistence_format

    def get_cache_ttl(self) -> int:
        """Get the cache time-to-live value."""
        return self._cache_ttl

    def get_graph_stats_cache(self) -> Optional[Dict[str, Any]]:
        """Get the graph statistics cache."""
        return self._graph_stats_cache

    def set_graph_stats_cache(self, cache: Optional[Dict[str, Any]]) -> None:
        """Set the graph statistics cache."""
        self._graph_stats_cache = cache
        self._cache_timestamp = time.time() if cache is not None else 0

    def is_cache_expired(self) -> bool:
        """Check if the cache is expired."""
        current_time = time.time()
        return (current_time - self._cache_timestamp) > self._cache_ttl

    def convert_graph_to_json(self, graph: Data) -> Dict[str, Any]:
        """Convert graph to JSON-serializable format."""
        return self._graph_to_json(graph)

    def convert_json_to_graph(self, data: Dict[str, Any]) -> Data:
        """Convert JSON data back to PyTorch Geometric Data object."""
        return self._json_to_graph(data)

    def initialize_graph(self, config: Optional[Dict[str, Any]] = None) -> Data:
        """
        Initialize a new neural graph.

        Args:
            config: Optional configuration for graph initialization

        Returns:
            Data: Initialized neural graph
        """
        try:
            # Get configuration
            if config is None:
                config = {}

            graph_size = config.get('size', self.get_default_graph_size())
            graph_size = min(graph_size, self.get_max_graph_size())

            # Create node labels
            node_labels = []
            node_features = []
            edge_indices = [[], []]
            edge_attributes = []

            # Create different types of nodes
            node_types = ['sensory', 'dynamic', 'oscillator', 'integrator', 'relay', 'highway']
            nodes_per_type = graph_size // len(node_types)

            node_id = 0
            for node_type in node_types:
                for i in range(nodes_per_type):
                    # Create node
                    node_label = {
                        'id': f"{node_type}_{node_id}",
                        'type': node_type,
                        'energy': 0.5 + 0.3 * (i / nodes_per_type),  # Varied energy levels
                        'x': (node_id % 50) * 10,
                        'y': (node_id // 50) * 10,
                        'membrane_potential': 0.0,
                        'threshold': 0.5,
                        'behavior': node_type,
                        'state': 'active'
                    }
                    node_labels.append(node_label)
                    node_features.append([node_label['energy']])
                    node_id += 1

            # Create connections
            for i, node_label in enumerate(node_labels):
                # Connect to nearby nodes
                for j in range(max(0, i-5), min(len(node_labels), i+6)):
                    if i != j:
                        edge_indices[0].append(i)
                        edge_indices[1].append(j)

                        # Create edge attributes
                        weight = 0.5 + 0.3 * torch.rand(1).item()
                        edge_attr = {
                            'source': node_label['id'],
                            'target': node_labels[j]['id'],
                            'weight': weight,
                            'type': 'excitatory' if weight > 0.5 else 'inhibitory'
                        }
                        edge_attributes.append(edge_attr)

            # Create PyTorch Geometric Data object
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            x = torch.tensor(node_features, dtype=torch.float)

            graph = Data(
                x=x,
                edge_index=edge_index,
                node_labels=node_labels,
                edge_attributes=edge_attributes
            )

            # Publish graph initialization event
            self.event_coordinator.publish("graph_initialized", {
                "nodes": len(node_labels),
                "edges": edge_index.shape[1],
                "node_types": node_types,
                "timestamp": time.time()
            })

            return graph

        except (ValueError, RuntimeError, AttributeError, TypeError) as e:
            print(f"Error initializing graph: {e}")
            # Return minimal graph on error
            return Data(
                x=torch.empty(0, 1),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                node_labels=[],
                edge_attributes=[]
            )

    def load_graph(self, filepath: str) -> Optional[Data]:
        """
        Load graph from file.

        Args:
            filepath: Path to the saved graph file

        Returns:
            Optional[Data]: Loaded graph or None if loading failed
        """
        try:
            if not os.path.exists(filepath):
                print(f"Graph file not found: {filepath}")
                return None

            # Determine file format
            if filepath.endswith('.pt') or filepath.endswith('.pth'):
                # PyTorch format - handle PyTorch Geometric Data objects
                try:
                    # First try with weights_only=False for PyTorch Geometric compatibility
                    graph = torch.load(filepath, weights_only=False)
                except (RuntimeError, FileNotFoundError, pickle.UnpicklingError) as e:
                    print(f"Failed to load with weights_only=False: {e}")
                    # Fallback to safe loading
                    try:
                        torch.serialization.add_safe_globals([torch_geometric.data.data.Data])
                        graph = torch.load(filepath, weights_only=True)
                    except (RuntimeError, AttributeError) as e2:
                        print(f"Failed to load with safe globals: {e2}")
                        return None
            elif filepath.endswith('.pkl'):
                # Pickle format
                with open(filepath, 'rb') as f:
                    graph = pickle.load(f)
            elif filepath.endswith('.json'):
                # JSON format (limited support)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Convert back to Data object (simplified)
                graph = self.convert_json_to_graph(data)
            else:
                print(f"Unsupported file format: {filepath}")
                return None

            # Validate loaded graph
            validation = self.validate_graph_integrity(graph)
            if not validation.get('valid', False):
                print(f"Loaded graph validation failed: {validation.get('issues', [])}")
                return None

            # Publish graph loading event
            self.event_coordinator.publish("graph_loaded", {
                "filepath": filepath,
                "nodes": len(graph.node_labels) if hasattr(graph, 'node_labels') else 0,
                "edges": graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0,
                "timestamp": time.time()
            })

            return graph

        except (FileNotFoundError, IOError, json.JSONDecodeError, pickle.UnpicklingError) as e:
            print(f"Error loading graph: {e}")
            return None

    def save_graph(self, graph: Data, filepath: str) -> bool:
        """
        Save graph to file.

        Args:
            graph: Graph to save
            filepath: Path to save the graph

        Returns:
            bool: True if saving successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Determine save format
            if filepath.endswith('.pt') or filepath.endswith('.pth'):
                # PyTorch format
                torch.save(graph, filepath)
            elif filepath.endswith('.pkl'):
                # Pickle format
                with open(filepath, 'wb') as f:
                    pickle.dump(graph, f)
            elif filepath.endswith('.json'):
                # JSON format (limited support)
                data = self.convert_graph_to_json(graph)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                # Default to PyTorch format
                torch.save(graph, filepath)

            # Publish graph saving event
            self.event_coordinator.publish("graph_saved", {
                "filepath": filepath,
                "nodes": len(graph.node_labels) if hasattr(graph, 'node_labels') else 0,
                "edges": graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0,
                "timestamp": time.time()
            })

            return True

        except (OSError, IOError, TypeError) as e:
            print(f"Error saving graph: {e}")
            return False

    def validate_graph_integrity(self, graph: Data) -> Dict[str, Any]:
        """
        Validate graph structure and data integrity.

        Args:
            graph: Graph to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        issues = []

        try:
            # Check basic structure
            if graph is None:
                issues.append("Graph is None")
                return {"valid": False, "issues": issues}

            # Check node features
            if not hasattr(graph, 'x') or graph.x is None:
                issues.append("Missing node features tensor")
            elif graph.x.numel() == 0:
                issues.append("Empty node features tensor")

            # Check edge index
            if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                issues.append("Missing edge index tensor")
            elif graph.edge_index.numel() == 0:
                issues.append("Empty edge index tensor")
            elif graph.edge_index.shape[0] != 2:
                issues.append(f"Invalid edge index shape: {graph.edge_index.shape}")

            # Check node labels
            if hasattr(graph, 'node_labels'):
                if not graph.node_labels:
                    issues.append("Empty node labels")
                elif len(graph.node_labels) != graph.x.shape[0]:
                    issues.append(
                        f"Node labels count ({len(graph.node_labels)}) doesn't match "
                        f"node features ({graph.x.shape[0]})"
                    )

                # Check node label structure
                for i, node in enumerate(graph.node_labels):
                    if not isinstance(node, dict):
                        issues.append(f"Node {i} label is not a dictionary")
                    elif 'id' not in node:
                        issues.append(f"Node {i} missing 'id' field")

            # Check edge attributes
            if hasattr(graph, 'edge_attributes'):
                if (graph.edge_attributes and
                        len(graph.edge_attributes) != graph.edge_index.shape[1]):
                    issues.append(
                        f"Edge attributes count ({len(graph.edge_attributes)}) doesn't "
                        f"match edges ({graph.edge_index.shape[1]})"
                    )

            # Check for NaN or infinite values
            if hasattr(graph, 'x') and graph.x is not None:
                if torch.isnan(graph.x).any():
                    issues.append("Node features contain NaN values")
                if torch.isinf(graph.x).any():
                    issues.append("Node features contain infinite values")

        except (AttributeError, TypeError, ValueError) as e:
            issues.append(f"Validation error: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "nodes": len(graph.node_labels) if hasattr(graph, 'node_labels') else 0,
            "edges": graph.edge_index.shape[1] if hasattr(graph, 'edge_index') else 0,
            "node_features_shape": graph.x.shape if hasattr(graph, 'x') else None
        }

    def get_graph_statistics(self, graph: Data) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics.

        Args:
            graph: Graph to analyze

        Returns:
            Dict[str, Any]: Graph statistics
        """
        try:
            stats = {
                "nodes": 0,
                "edges": 0,
                "node_types": {},
                "edge_types": {},
                "energy_distribution": {},
                "connectivity_stats": {},
                "graph_density": 0.0
            }

            if graph is None:
                return stats

            # Basic counts
            if hasattr(graph, 'node_labels') and graph.node_labels:
                stats["nodes"] = len(graph.node_labels)

                # Node type distribution
                for node in graph.node_labels:
                    node_type = node.get('type', 'unknown')
                    stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

                    # Energy distribution
                    energy = node.get('energy', 0.0)
                    energy_bin = int(energy * 10) / 10.0  # Round to nearest 0.1
                    stats["energy_distribution"][energy_bin] = (
                        stats["energy_distribution"].get(energy_bin, 0) + 1
                    )

            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                stats["edges"] = graph.edge_index.shape[1]

                # Graph density
                if stats["nodes"] > 0:
                    max_edges = stats["nodes"] * (stats["nodes"] - 1)
                    stats["graph_density"] = stats["edges"] / max_edges if max_edges > 0 else 0

            # Edge type distribution
            if hasattr(graph, 'edge_attributes') and graph.edge_attributes:
                for edge in graph.edge_attributes:
                    edge_type = edge.get('type', 'unknown')
                    stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

            # Connectivity statistics
            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                # Degree distribution
                degrees = torch.bincount(graph.edge_index[0], minlength=stats["nodes"])
                stats["connectivity_stats"] = {
                    "average_degree": degrees.float().mean().item(),
                    "max_degree": degrees.max().item(),
                    "min_degree": degrees.min().item()
                }

            return stats

        except (AttributeError, TypeError, RuntimeError, ZeroDivisionError) as e:
            print(f"Error getting graph statistics: {e}")
            return {"error": str(e)}

    def _graph_to_json(self, graph: Data) -> Dict[str, Any]:
        """
        Convert graph to JSON-serializable format.

        Args:
            graph: PyTorch Geometric Data object

        Returns:
            Dict[str, Any]: JSON-serializable representation
        """
        data = {
            "node_labels": graph.node_labels if hasattr(graph, 'node_labels') else [],
            "edge_attributes": graph.edge_attributes if hasattr(graph, 'edge_attributes') else [],
            "x": graph.x.tolist() if hasattr(graph, 'x') and graph.x is not None else [],
            "edge_index": (
                graph.edge_index.tolist()
                if hasattr(graph, 'edge_index') and graph.edge_index is not None
                else []
            )
        }
        return data

    def _json_to_graph(self, data: Dict[str, Any]) -> Data:
        """
        Convert JSON data back to PyTorch Geometric Data object.

        Args:
            data: JSON data

        Returns:
            Data: PyTorch Geometric Data object
        """
        graph = Data()

        if "x" in data and data["x"]:
            graph.x = torch.tensor(data["x"], dtype=torch.float)

        if "edge_index" in data and data["edge_index"]:
            graph.edge_index = torch.tensor(data["edge_index"], dtype=torch.long)

        if "node_labels" in data:
            graph.node_labels = data["node_labels"]

        if "edge_attributes" in data:
            graph.edge_attributes = data["edge_attributes"]

        return graph

    def update_node_lifecycle(self, graph: Any) -> Any:
        # This is a placeholder implementation.
        # In a real scenario, this would involve complex logic for node birth/death.
        return graph

    def cleanup(self) -> None:
        """Clean up resources."""
        self.set_graph_stats_cache(None)

