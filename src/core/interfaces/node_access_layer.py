from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class IAccessLayer(ABC):
    @abstractmethod
    def get_node_by_id(self, node_id: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieves a node from the graph by its unique ID.
        """
        pass






