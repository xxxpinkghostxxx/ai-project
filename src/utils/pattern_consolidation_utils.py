"""
Pattern consolidation utilities to reduce code duplication.
"""

from typing import Any, Callable, Dict


def _get_constants():
    """Lazy import of consolidated constants to avoid circular imports."""
    # pylint: disable=import-outside-toplevel
    from config.consolidated_constants import (CONNECTION_PROPERTIES,
                                                CONNECTION_TYPES,
                                                DEFAULT_VALUES, ERROR_MESSAGES,
                                                LOG_MESSAGES, NODE_PROPERTIES,
                                                NODE_STATES, NODE_TYPES,
                                                PERFORMANCE_METRICS,
                                                SYSTEM_STATES, THRESHOLDS,
                                                UI_CONSTANTS)
    return {
        'UI_CONSTANTS': UI_CONSTANTS,
        'ERROR_MESSAGES': ERROR_MESSAGES,
        'LOG_MESSAGES': LOG_MESSAGES,
        'NODE_PROPERTIES': NODE_PROPERTIES,
        'CONNECTION_PROPERTIES': CONNECTION_PROPERTIES,
        'SYSTEM_STATES': SYSTEM_STATES,
        'NODE_STATES': NODE_STATES,
        'NODE_TYPES': NODE_TYPES,
        'CONNECTION_TYPES': CONNECTION_TYPES,
        'PERFORMANCE_METRICS': PERFORMANCE_METRICS,
        'THRESHOLDS': THRESHOLDS,
        'DEFAULT_VALUES': DEFAULT_VALUES
    }


# Cache for constants
_CONSTANTS_CACHE = None


def _get_cached_constants():
    """Get cached constants with lazy loading."""
    global _CONSTANTS_CACHE  # pylint: disable=global-statement
    if _CONSTANTS_CACHE is None:
        _CONSTANTS_CACHE = _get_constants()
    return _CONSTANTS_CACHE


def create_standard_node(node_id: int, node_type: str = 'dynamic',
                        subtype: str = 'standard', **kwargs) -> Dict[str, Any]:
    """
    Create a standard node with common properties.
    Replaces repeated node creation patterns.
    """
    consts = _get_cached_constants()
    node_properties = consts['NODE_PROPERTIES']
    node_states = consts['NODE_STATES']
    default_values = consts['DEFAULT_VALUES']
    return {
        node_properties['ID']: node_id,
        node_properties['TYPE']: node_type,
        node_properties['SUBTYPE']: subtype,
        node_properties['ENERGY']: kwargs.get('energy', 0.0),
        node_properties['STATE']: kwargs.get('state', node_states['INACTIVE']),
        node_properties['THRESHOLD']: kwargs.get('threshold', default_values['THRESHOLD_DEFAULT']),
        node_properties['MEMBRANE_POTENTIAL']: kwargs.get('membrane_potential', 0.0),
        node_properties['REFRACTORY_TIMER']: kwargs.get('refractory_timer', 0.0),
        node_properties['LAST_UPDATE']: kwargs.get('last_update', 0),
        node_properties['PLASTICITY_ENABLED']: kwargs.get('plasticity_enabled', True),
        node_properties['ELIGIBILITY_TRACE']: kwargs.get('eligibility_trace', default_values['ELIGIBILITY_TRACE_DEFAULT']),
        node_properties['ENHANCED_BEHAVIOR']: kwargs.get('enhanced_behavior', False),
        **kwargs
    }


def create_workspace_node(node_id: int, x: int, y: int, **kwargs) -> Dict[str, Any]:
    """
    Create a workspace node with spatial coordinates.
    Replaces repeated workspace node creation patterns.
    """
    return create_standard_node(
        node_id=node_id,
        node_type='workspace',
        subtype='workspace',
        x=x,
        y=y,
        behavior='workspace',
        **kwargs
    )


def create_sensory_node(node_id: int, x: float, y: float, energy: float = 0.0, **kwargs) -> Dict[str, Any]:
    """
    Create a sensory node with spatial coordinates and energy.
    Replaces repeated sensory node creation patterns.
    """
    consts = _get_cached_constants()
    node_states = consts['NODE_STATES']
    return create_standard_node(
        node_id=node_id,
        node_type='sensory',
        subtype='sensory',
        x=x,
        y=y,
        behavior='sensory',
        energy=energy,
        state=node_states['ACTIVE'],
        **kwargs
    )


def create_dynamic_node(node_id: int, x: float, y: float, **kwargs) -> Dict[str, Any]:
    """
    Create a dynamic node with spatial coordinates.
    Replaces repeated dynamic node creation patterns.
    """
    return create_standard_node(
        node_id=node_id,
        node_type='dynamic',
        subtype='dynamic',
        x=x,
        y=y,
        behavior='dynamic',
        **kwargs
    )


def create_standard_connection(source_id: int, target_id: int,
                              connection_type: str = 'excitatory',
                              weight: float = 1.0, **kwargs) -> Dict[str, Any]:
    """
    Create a standard connection with common properties.
    Replaces repeated connection creation patterns.
    """
    consts = _get_cached_constants()
    connection_properties = consts['CONNECTION_PROPERTIES']
    default_values = consts['DEFAULT_VALUES']
    return {
        connection_properties['SOURCE']: source_id,
        connection_properties['TARGET']: target_id,
        connection_properties['TYPE']: connection_type,
        connection_properties['WEIGHT']: weight,
        connection_properties['DELAY']: kwargs.get('delay', default_values['DELAY_DEFAULT']),
        connection_properties['ACTIVE']: kwargs.get('active', True),
        connection_properties['ELIGIBILITY_TRACE']: kwargs.get('eligibility_trace', default_values['ELIGIBILITY_TRACE_DEFAULT']),
        connection_properties['LAST_ACTIVITY']: kwargs.get('last_activity', 0.0),
        connection_properties['ACTIVATION_COUNT']: kwargs.get('activation_count', 0),
        **kwargs
    }


def create_standard_statistics() -> Dict[str, Any]:
    """
    Create standard statistics dictionary.
    Replaces repeated statistics initialization patterns.
    """
    return {
        'total_operations': 0,
        'successful_operations': 0,
        'failed_operations': 0,
        'error_count': 0,
        'warning_count': 0,
        'last_update': 0.0,
        'uptime': 0.0,
        'performance_score': 100.0
    }


def create_standard_performance_metrics() -> Dict[str, Any]:
    """
    Create standard performance metrics dictionary.
    Replaces repeated performance metrics initialization patterns.
    """
    consts = _get_cached_constants()
    performance_metrics = consts['PERFORMANCE_METRICS']
    return {
        performance_metrics['STEP_TIME']: 0.0,
        performance_metrics['TOTAL_RUNTIME']: 0.0,
        performance_metrics['FPS']: 0.0,
        performance_metrics['MEMORY_USAGE']: 0.0,
        performance_metrics['CPU_PERCENT']: 0.0,
        performance_metrics['GPU_USAGE']: 0.0,
        performance_metrics['ERROR_RATE']: 0.0,
        performance_metrics['NODE_COUNT']: 0,
        performance_metrics['EDGE_COUNT']: 0,
        performance_metrics['THROUGHPUT']: 0.0
    }


def create_standard_ui_elements() -> Dict[str, str]:
    """
    Create standard UI element tags.
    Replaces repeated UI element creation patterns.
    """
    consts = _get_cached_constants()
    ui_constants = consts['UI_CONSTANTS']
    return {
        'main_window': ui_constants['MAIN_WINDOW_TAG'],
        'status_text': ui_constants['STATUS_TEXT_TAG'],
        'nodes_text': ui_constants['NODES_TEXT_TAG'],
        'edges_text': ui_constants['EDGES_TEXT_TAG'],
        'energy_text': ui_constants['ENERGY_TEXT_TAG'],
        'connections_text': ui_constants['CONNECTIONS_TEXT_TAG'],
        'legend_window': ui_constants['LEGEND_WINDOW_TAG'],
        'help_window': ui_constants['HELP_WINDOW_TAG']
    }


def create_standard_error_handler() -> Dict[str, Callable]:
    """
    Create standard error handling patterns.
    Replaces repeated error handling patterns.
    """
    def handle_value_error(_error: Exception, _context: str = "") -> bool:
        """Handle ValueError with standard pattern."""
        return False

    def handle_type_error(_error: Exception, _context: str = "") -> bool:
        """Handle TypeError with standard pattern."""
        return False

    def handle_attribute_error(_error: Exception, _context: str = "") -> bool:
        """Handle AttributeError with standard pattern."""
        return False

    def handle_key_error(_error: Exception, _context: str = "") -> bool:
        """Handle KeyError with standard pattern."""
        return False

    def handle_runtime_error(_error: Exception, _context: str = "") -> bool:
        """Handle RuntimeError with standard pattern."""
        return False

    def handle_memory_error(_error: Exception, _context: str = "") -> bool:
        """Handle MemoryError with standard pattern."""
        return False

    def handle_os_error(_error: Exception, _context: str = "") -> bool:
        """Handle OSError with standard pattern."""
        return False

    return {
        'ValueError': handle_value_error,
        'TypeError': handle_type_error,
        'AttributeError': handle_attribute_error,
        'KeyError': handle_key_error,
        'RuntimeError': handle_runtime_error,
        'MemoryError': handle_memory_error,
        'OSError': handle_os_error
    }


def create_standard_validation_patterns() -> Dict[str, Callable]:
    """
    Create standard validation patterns.
    Replaces repeated validation patterns.
    """
    consts = _get_cached_constants()
    default_values = consts['DEFAULT_VALUES']
    node_types = consts['NODE_TYPES']
    connection_types = consts['CONNECTION_TYPES']

    def validate_node_id(node_id: Any) -> bool:
        """Validate node ID with standard pattern."""
        return isinstance(node_id, int) and node_id >= 0

    def validate_energy_value(energy: Any) -> bool:
        """Validate energy value with standard pattern."""
        return isinstance(energy, (int, float)) and 0 <= energy <= default_values['ENERGY_CAP']

    def validate_threshold_value(threshold: Any) -> bool:
        """Validate threshold value with standard pattern."""
        return isinstance(threshold, (int, float)) and 0 <= threshold <= 1.0

    def validate_weight_value(weight: Any) -> bool:
        """Validate weight value with standard pattern."""
        return isinstance(weight, (int, float)) and 0 <= weight <= 5.0

    def validate_node_type(node_type: Any) -> bool:
        """Validate node type with standard pattern."""
        return isinstance(node_type, str) and node_type in node_types.values()

    def validate_connection_type(connection_type: Any) -> bool:
        """Validate connection type with standard pattern."""
        return isinstance(connection_type, str) and connection_type in connection_types.values()

    return {
        'node_id': validate_node_id,
        'energy': validate_energy_value,
        'threshold': validate_threshold_value,
        'weight': validate_weight_value,
        'node_type': validate_node_type,
        'connection_type': validate_connection_type
    }


def create_standard_initialization_patterns() -> Dict[str, Callable]:
    """
    Create standard initialization patterns.
    Replaces repeated initialization patterns.
    """
    def initialize_basic_properties(self, **kwargs) -> None:
        """Initialize basic properties with standard pattern."""
        self.id = kwargs.get('id', 0)
        self.type = kwargs.get('type', 'unknown')
        self.active = kwargs.get('active', True)
        self.created_at = kwargs.get('created_at', 0.0)
        self.last_updated = kwargs.get('last_updated', 0.0)

    def initialize_statistics(self) -> None:
        """Initialize statistics with standard pattern."""
        self.stats = create_standard_statistics()
        self.performance_metrics = create_standard_performance_metrics()

    def initialize_callbacks(self) -> None:
        """Initialize callbacks with standard pattern."""
        self.callbacks = []
        self.error_callbacks = []
        self.success_callbacks = []

    def initialize_locks(self) -> None:
        """Initialize locks with standard pattern."""
        # pylint: disable=import-outside-toplevel
        import threading
        self._lock = threading.RLock()
        self._stats_lock = threading.RLock()

    return {
        'basic_properties': initialize_basic_properties,
        'statistics': initialize_statistics,
        'callbacks': initialize_callbacks,
        'locks': initialize_locks
    }


def create_standard_cleanup_patterns() -> Dict[str, Callable]:
    """
    Create standard cleanup patterns.
    Replaces repeated cleanup patterns.
    """
    def cleanup_basic_properties(self) -> None:
        """Cleanup basic properties with standard pattern."""
        self.id = None
        self.type = None
        self.active = False
        self.created_at = 0.0
        self.last_updated = 0.0

    def cleanup_statistics(self) -> None:
        """Cleanup statistics with standard pattern."""
        if hasattr(self, 'stats'):
            self.stats.clear()
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics.clear()

    def cleanup_callbacks(self) -> None:
        """Cleanup callbacks with standard pattern."""
        if hasattr(self, 'callbacks'):
            self.callbacks.clear()
        if hasattr(self, 'error_callbacks'):
            self.error_callbacks.clear()
        if hasattr(self, 'success_callbacks'):
            self.success_callbacks.clear()

    def cleanup_locks(self) -> None:
        """Cleanup locks with standard pattern."""
        if hasattr(self, '_lock'):
            del self._lock
        if hasattr(self, '_stats_lock'):
            del self._stats_lock

    return {
        'basic_properties': cleanup_basic_properties,
        'statistics': cleanup_statistics,
        'callbacks': cleanup_callbacks,
        'locks': cleanup_locks
    }
