"""
Simulation rule validator.

LEGACY: Not yet migrated to TaichiNeuralEngine. The original validator tested
PyGNeuralSystem which no longer exists. Kept as a stub — migrate or remove.
"""

import logging
from typing import Any, Dict, List

from project.config import (
    NODE_ENERGY_CAP,
    NODE_DEATH_THRESHOLD,
    NODE_SPAWN_THRESHOLD,
    NODE_ENERGY_SPAWN_COST,
    MAX_NODE_BIRTHS_PER_STEP,
    NODE_TYPE_SENSORY,
    NODE_TYPE_DYNAMIC,
    NODE_TYPE_WORKSPACE,
    CONN_TYPE_EXCITATORY,
    CONN_TYPE_INHIBITORY,
    CONN_TYPE_GATED,
    CONN_TYPE_PLASTIC,
)

logger = logging.getLogger(__name__)


class SimulationValidator:
    """Validates simulation rules. Currently returns NOT_AVAILABLE pending migration."""

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.test_system: Any = None

    def run_full_test(self, device: str = "cpu") -> Dict[str, Any]:
        """Run comprehensive rule validation test.

        Not yet migrated to TaichiNeuralEngine — returns NOT_AVAILABLE.
        """
        return {
            "status": "NOT_AVAILABLE",
            "message": (
                "SimulationValidator has not been migrated to the Taichi engine. "
                "The old PyGNeuralSystem it tested no longer exists."
            ),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
        }
