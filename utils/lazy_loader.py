"""
Lazy loading system for optimized startup performance.
Provides deferred initialization of heavy components.
"""

import threading
import time
from typing import Any, Callable, Optional, Dict
from functools import wraps
import logging

class LazyLoader:
    """Thread-safe lazy loading system for expensive components."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._loaded_components: Dict[str, Any] = {}
        self._loading_flags: Dict[str, bool] = {}
        self._load_times: Dict[str, float] = {}
        self._load_callbacks: Dict[str, list] = {}
        self._factory_functions: Dict[str, Callable[[], Any]] = {}
        self._priorities: Dict[str, int] = {}
        self._dependencies: Dict[str, list] = {}
    
    def lazy_load(self, component_name: str, factory_func: Callable[[], Any],
                    priority: int = 0, dependencies: list = None) -> Any:
        """
        Get a component, loading it lazily if not already loaded.

        Args:
            component_name: Unique name for the component
            factory_func: Function to create the component
            priority: Load priority (higher = load earlier)
            dependencies: List of component names this depends on

        Returns:
            The loaded component
        """
        with self._lock:
            if component_name in self._loaded_components:
                return self._loaded_components[component_name]

            # Store factory function and metadata for potential preloading
            self._factory_functions[component_name] = factory_func
            self._priorities[component_name] = priority
            self._dependencies[component_name] = dependencies or []

            # Load dependencies first
            deps = dependencies or []
            for dep in deps:
                if dep not in self._loaded_components:
                    if dep in self._factory_functions:
                        self.lazy_load(dep, self._factory_functions[dep],
                                     self._priorities.get(dep, 0),
                                     self._dependencies.get(dep, []))
                    else:
                        raise ValueError(f"Dependency {dep} for {component_name} has no factory function")

            # Check if already loading (prevent recursive loading)
            if self._loading_flags.get(component_name, False):
                # Wait for loading to complete
                start_time = time.time()
                while self._loading_flags.get(component_name, False):
                    if time.time() - start_time > 30:  # 30 second timeout
                        raise TimeoutError(f"Timeout waiting for {component_name} to load")
                    time.sleep(0.01)
                return self._loaded_components[component_name]

            # Mark as loading
            self._loading_flags[component_name] = True

            try:
                start_time = time.time()
                component = factory_func()
                load_time = time.time() - start_time

                self._loaded_components[component_name] = component
                self._load_times[component_name] = load_time
                self._loading_flags[component_name] = False

                logging.info(f"Lazy loaded {component_name} in {load_time:.3f}s")

                # Trigger callbacks
                for callback in self._load_callbacks.get(component_name, []):
                    try:
                        callback(component)
                    except Exception as e:
                        logging.error(f"Load callback error for {component_name}: {e}")

                return component

            except Exception as e:
                self._loading_flags[component_name] = False
                logging.error(f"Failed to lazy load {component_name}: {e}")
                raise
    
    def preload_components(self, component_names: list, max_concurrent: int = 3):
        """
        Preload multiple components in parallel with controlled concurrency.

        Args:
            component_names: List of component names to preload
            max_concurrent: Maximum number of concurrent loading operations
        """
        import concurrent.futures

        def load_component(name):
            with self._lock:
                if name in self._loaded_components:
                    return name, True

                # Check if factory function exists
                if name not in self._factory_functions:
                    logging.warning(f"No factory function registered for component: {name}")
                    return name, False

                try:
                    # Load the component using stored factory function
                    component = self.lazy_load(name, self._factory_functions[name],
                                             self._priorities.get(name, 0),
                                             self._dependencies.get(name, []))
                    return name, True
                except Exception as e:
                    logging.error(f"Failed to preload component {name}: {e}")
                    return name, False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(load_component, name) for name in component_names]
            for future in concurrent.futures.as_completed(futures):
                name, success = future.result()
                if success:
                    logging.info(f"Preloaded component: {name}")
                else:
                    logging.warning(f"Failed to preload component: {name}")
    
    def get_load_time(self, component_name: str) -> Optional[float]:
        """Get the load time for a component."""
        with self._lock:
            return self._load_times.get(component_name)
    
    def add_load_callback(self, component_name: str, callback: Callable[[Any], None]):
        """Add a callback to be called when a component is loaded."""
        with self._lock:
            if component_name not in self._load_callbacks:
                self._load_callbacks[component_name] = []
            self._load_callbacks[component_name].append(callback)
    
    def is_loaded(self, component_name: str) -> bool:
        """Check if a component is loaded."""
        with self._lock:
            return component_name in self._loaded_components
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        with self._lock:
            return {
                'loaded_components': list(self._loaded_components.keys()),
                'load_times': self._load_times.copy(),
                'total_components': len(self._loaded_components)
            }
    
    def cleanup(self):
        """Clean up all loaded components."""
        with self._lock:
            for component_name, component in self._loaded_components.items():
                if hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                    except Exception as e:
                        logging.error(f"Error cleaning up {component_name}: {e}")

            self._loaded_components.clear()
            self._loading_flags.clear()
            self._load_times.clear()
            self._load_callbacks.clear()
            self._factory_functions.clear()
            self._priorities.clear()
            self._dependencies.clear()

# Global lazy loader instance
_global_lazy_loader = LazyLoader()

def get_lazy_loader() -> LazyLoader:
    """Get the global lazy loader instance."""
    return _global_lazy_loader

def lazy_component(component_name: str, priority: int = 0, dependencies: list = None):
    """
    Decorator for lazy loading class methods.

    Args:
        component_name: Name of the component to lazy load
        priority: Load priority
        dependencies: List of dependencies
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            loader = get_lazy_loader()
            component = loader.lazy_load(component_name, lambda: func(self),  # Load without args
                                         priority, dependencies)
            return component
        return wrapper
    return decorator

# Component factory functions for common lazy loads
def create_lazy_simulation_manager():
    """Lazy factory for simulation manager."""
    from core.simulation_manager import SimulationManager
    return SimulationManager()

def create_lazy_ui_engine():
    """Lazy factory for UI engine."""
    from ui.ui_engine import create_ui
    return create_ui()

def create_lazy_neural_dynamics():
    """Lazy factory for neural dynamics."""
    from neural.enhanced_neural_dynamics import create_enhanced_neural_dynamics
    return create_enhanced_neural_dynamics()

def create_lazy_performance_monitor():
    """Lazy factory for performance monitor."""
    from utils.performance_monitor import PerformanceMonitor as PerfMonitor
    monitor = PerfMonitor()
    # Don't start monitoring here - let the SimulationManager handle it
    return monitor