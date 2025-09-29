"""
IGraphManager interface - Neural graph management service.

This interface defines the contract for managing the neural graph structure,
including node and edge operations, persistence, and graph integrity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch_geometric.data import Data


class IGraphManager(ABC):
    """
    Abstract interface for neural graph management operations.

    This interface defines the contract for managing the neural network
    graph structure, including CRUD operations, persistence, and integrity checks.
    """

    @abstractmethod
    def initialize_graph(self, config: Optional[Dict[str, Any]] = None) -> Data:
        """Initialize a new neural graph."""

    @abstractmethod
    def load_graph(self, filepath: str) -> Optional[Data]:
        """Load graph from file."""

    @abstractmethod
    def save_graph(self, graph: Data, filepath: str) -> bool:
        """Save graph to file."""

    @abstractmethod
    def validate_graph_integrity(self, graph: Data) -> Dict[str, Any]:
        """Validate graph structure and data integrity."""

    @abstractmethod
    def get_graph_statistics(self, graph: Data) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""

    @abstractmethod
    def update_node_lifecycle(self, graph: Data) -> Data:
        """
        Update the lifecycle of nodes in the graph (e.g., birth/death).

        Args:
            graph: The current neural graph.

        Returns:
            The updated neural graph.
        """
