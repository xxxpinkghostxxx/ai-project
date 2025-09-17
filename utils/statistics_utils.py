"""
Consolidated statistics utilities to reduce code duplication.
"""

from typing import Dict, Any, Optional, List, Union


class StatisticsManager:
    """Manages and consolidates statistics for different modules."""
    
    def __init__(self):
        self._stats: Dict[str, Dict[str, Any]] = {}
    
    def register_module_stats(self, module_name: str, initial_stats: Dict[str, Any]):
        """Registers initial statistics for a given module."""
        if module_name not in self._stats:
            self._stats[module_name] = initial_stats
        else:
            self._stats[module_name].update(initial_stats)
    
    def update_module_stats(self, module_name: str, key: str, value: Any):
        """Updates a specific statistic for a module."""
        if module_name in self._stats:
            self._stats[module_name][key] = value
        else:
            # Optionally log a warning if trying to update unregistered module
            pass
    
    def increment_module_stat(self, module_name: str, key: str, amount: int = 1):
        """Increments a numeric statistic for a module."""
        if module_name in self._stats and key in self._stats[module_name]:
            self._stats[module_name][key] += amount
        else:
            # Optionally log a warning
            pass
    
    def get_module_stats(self, module_name: str) -> Dict[str, Any]:
        """Retrieves all statistics for a given module."""
        return self._stats.get(module_name, {}).copy()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves statistics for all registered modules."""
        return {k: v.copy() for k, v in self._stats.items()}
    
    def reset_module_stats(self, module_name: str, initial_stats: Optional[Dict[str, Any]] = None):
        """Resets statistics for a module, optionally to new initial values."""
        if module_name in self._stats:
            if initial_stats is not None:
                self._stats[module_name] = initial_stats
            else:
                # Clear existing stats if no new initial stats are provided
                self._stats[module_name] = {}
        else:
            # Optionally log a warning
            pass
    
    def reset_all_stats(self):
        """Resets all registered statistics."""
        self._stats.clear()


_global_stats_manager = StatisticsManager()


def get_global_statistics_manager() -> StatisticsManager:
    """Returns the global statistics manager instance."""
    return _global_stats_manager


def create_standard_stats(module_name: str) -> Dict[str, Any]:
    """
    Creates a standard dictionary for module statistics.
    Can be extended with module-specific metrics.
    """
    return {
        'total_operations': 0,
        'successful_operations': 0,
        'failed_operations': 0,
        'last_reset_time': 0.0,
        'module_name': module_name
    }