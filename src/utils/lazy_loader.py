"""
Lazy loading system for optimized startup performance.
Provides deferred initialization of heavy components.
"""

import logging
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional


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

    def get_lock_internal(self) -> threading.RLock():
        """Get the internal lock."""
        return self._lock

    def get_loaded_components_internal(self) -> Dict[str, Any]:
        """Get the loaded components dictionary."""
        return self._loaded_components

    def set_loaded_component_internal(self, name: str, component: Any) -> None:
        """Set a loaded component."""
        self._loaded_components[name] = component

    def get_loading_flags_internal(self) -> Dict[str, bool]:
        """Get the loading flags dictionary."""
        return self._loading_flags

    def set_loading_flag_internal(self, name: str, flag: bool) -> None:
        """Set a loading flag."""
        self._loading_flags[name] = flag

    def get_load_times_internal(self) -> Dict[str, float]:
        """Get the load times dictionary."""
        return self._load_times

    def set_load_time_internal(self, name: str, time: float) -> None:
        """Set a load time."""
        self._load_times[name] = time

    def get_load_callbacks_internal(self) -> Dict[str, list]:
        """Get the load callbacks dictionary."""
        return self._load_callbacks

    def add_load_callback_internal(self, name: str, callback) -> None:
        """Add a load callback."""
        if name not in self._load_callbacks:
            self._load_callbacks[name] = []
        self._load_callbacks[name].append(callback)

    def get_factory_functions_internal(self) -> Dict[str, Callable[[], Any]]:
        """Get the factory functions dictionary."""
        return self._factory_functions

    def set_factory_function_internal(self, name: str, func: Callable[[], Any]) -> None:
        """Set a factory function."""
        self._factory_functions[name] = func

    def get_priorities_internal(self) -> Dict[str, int]:
        """Get the priorities dictionary."""
        return self._priorities

    def set_priority_internal(self, name: str, priority: int) -> None:
        """Set a priority."""
        self._priorities[name] = priority

    def get_dependencies_internal(self) -> Dict[str, list]:
        """Get the dependencies dictionary."""
        return self._dependencies

    def set_dependencies_internal(self, name: str, deps: list) -> None:
        """Set dependencies."""
        self._dependencies[name] = deps
    
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
        with self.get_lock_internal():
            if component_name in self.get_loaded_components_internal():
                return self.get_loaded_components_internal()[component_name]

            # Store factory function and metadata for potential preloading
            self.set_factory_function_internal(component_name, factory_func)
            self.set_priority_internal(component_name, priority)
            self.set_dependencies_internal(component_name, dependencies or [])

            # Load dependencies first
            deps = dependencies or []
            for dep in deps:
                if dep not in self.get_loaded_components_internal():
                    if dep in self.get_factory_functions_internal():
                        self.lazy_load(dep, self.get_factory_functions_internal()[dep],
                                     self.get_priorities_internal().get(dep, 0),
                                     self.get_dependencies_internal().get(dep, []))
                    else:
                        raise ValueError(f"Dependency {dep} for {component_name} has no factory function")

            # Check if already loading (prevent recursive loading)
            if self.get_loading_flags_internal().get(component_name, False):
                # Wait for loading to complete
                start_time = time.time()
                while self.get_loading_flags_internal().get(component_name, False):
                    if time.time() - start_time > 30:  # 30 second timeout
                        raise TimeoutError(f"Timeout waiting for {component_name} to load")
                    time.sleep(0.01)
                return self.get_loaded_components_internal()[component_name]

            # Mark as loading
            self.set_loading_flag_internal(component_name, True)

            try:
                start_time = time.time()
                component = factory_func()
                load_time = time.time() - start_time

                self.set_loaded_component_internal(component_name, component)
                self.set_load_time_internal(component_name, load_time)
                self.set_loading_flag_internal(component_name, False)

                logging.info(f"Lazy loaded {component_name} in {load_time:.3f}s")

                # Trigger callbacks
                callbacks = self.get_load_callbacks_internal().get(component_name, [])
                for callback in callbacks:
                    try:
                        callback(component)
                    except Exception as e:
                        logging.error(f"Load callback error for {component_name}: {e}")

                return component

            except Exception as e:
                self.set_loading_flag_internal(component_name, False)
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
            with self.get_lock_internal():
                if name in self.get_loaded_components_internal():
                    return name, True

                # Check if factory function exists
                if name not in self.get_factory_functions_internal():
                    logging.warning(f"No factory function registered for component: {name}")
                    return name, False

                try:
                    # Load the component using stored factory function
                    self.lazy_load(name, self.get_factory_functions_internal()[name],
                                   self.get_priorities_internal().get(name, 0),
                                   self.get_dependencies_internal().get(name, []))
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
        with self.get_lock_internal():
            return self.get_load_times_internal().get(component_name)
    
    def add_load_callback(self, component_name: str, callback: Callable[[Any], None]):
        """Add a callback to be called when a component is loaded."""
        with self.get_lock_internal():
            self.add_load_callback_internal(component_name, callback)
    
    def is_loaded(self, component_name: str) -> bool:
        """Check if a component is loaded."""
        with self.get_lock_internal():
            return component_name in self.get_loaded_components_internal()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        with self.get_lock_internal():
            return {
                'loaded_components': list(self.get_loaded_components_internal().keys()),
                'load_times': self.get_load_times_internal().copy(),
                'total_components': len(self.get_loaded_components_internal())
            }
    
    def cleanup(self):
        """Clean up all loaded components."""
        with self.get_lock_internal():
            # Iterate over a copy of keys to avoid "dictionary changed size during iteration" error
            loaded_components = self.get_loaded_components_internal()
            for component_name in list(loaded_components.keys()):
                component = loaded_components.get(component_name)
                if component and hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                    except Exception as e:
                        logging.error(f"Error cleaning up {component_name}: {e}")

            loaded_components.clear()
            self.get_loading_flags_internal().clear()
            self.get_load_times_internal().clear()
            self.get_load_callbacks_internal().clear()
            self.get_factory_functions_internal().clear()
            self.get_priorities_internal().clear()
            self.get_dependencies_internal().clear()

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
def create_lazy_simulation_coordinator():
    """Lazy factory for simulation coordinator."""
    from src.core.services.simulation_coordinator import SimulationCoordinator
    return SimulationCoordinator(None, None, None, None, None, None, None, None, None)

def create_lazy_ui_engine():
    """Lazy factory for UI engine."""
    from src.ui.ui_engine import create_ui
    return create_ui()

def create_lazy_neural_dynamics():
    """Lazy factory for neural dynamics."""
    from src.neural.enhanced_neural_dynamics import \
        create_enhanced_neural_dynamics
    return create_enhanced_neural_dynamics()

def create_lazy_performance_monitor():
    """Lazy factory for performance monitor."""
    from src.utils.unified_performance_system import get_performance_monitor
    return get_performance_monitor()






