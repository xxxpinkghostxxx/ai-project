"""
Module for node access layer interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IAccessLayer(ABC):  # pylint: disable=too-few-public-methods
    """
    Interface for accessing nodes in the graph.
    """
    @abstractmethod
    def get_node_by_id(self, node_id: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieves a node from the graph by its unique ID.
        """
        raise NotImplementedError()
