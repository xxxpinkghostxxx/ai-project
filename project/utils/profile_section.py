"""Performance profiling utilities for the DGL Neural System."""
import time
import functools
import logging
from typing import Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dgl_system.log'
)
logger = logging.getLogger('profile_section')

class ProfileSection:
    """Context manager for profiling code sections."""

    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            logger.log(self.log_level, f"{self.name}: {duration:.4f}s")

def profile_section(name: Optional[str] = None, log_level: int = logging.INFO):
    """
    Decorator for profiling function execution time.

    Args:
        name: Optional name for the section. If None, uses the function name.
        log_level: Logging level for the profile message.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            section_name = name or func.__name__
            with ProfileSection(section_name, log_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class Profiler:
    """Class for tracking multiple profile sections."""

    def __init__(self):
        self.sections: dict[str, list[float]] = {}

    def add_section(self, name: str, duration: float):
        """Add a duration to a section's history."""
        if name not in self.sections:
            self.sections[name] = []
        self.sections[name].append(duration)

        # Keep only last 100 measurements
        if len(self.sections[name]) > 100:
            self.sections[name].pop(0)

    def get_section_stats(self, name: str) -> dict:
        """Get statistics for a section."""
        if name not in self.sections or not self.sections[name]:
            return {
                'min': 0.0,
                'max': 0.0,
                'avg': 0.0,
                'total': 0.0,
                'count': 0
            }

        durations = self.sections[name]
        return {
            'min': min(durations),
            'max': max(durations),
            'avg': sum(durations) / len(durations),
            'total': sum(durations),
            'count': len(durations)
        }

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all sections."""
        return {name: self.get_section_stats(name) for name in self.sections}

    def clear(self):
        """Clear all section history."""
        self.sections.clear()

# Global profiler instance
profiler = Profiler() 