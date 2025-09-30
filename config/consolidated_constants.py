"""
Consolidated constants to reduce string duplication across the codebase.
"""

import logging
from config.unified_config_manager import get_config_manager

# Setup logging
logger = logging.getLogger(__name__)

# Common UI Constants
UI_CONSTANTS = {
    'SIMULATION_STATUS_RUNNING': 'Running',
    'SIMULATION_STATUS_STOPPED': 'Stopped',
    'NEURAL_SIMULATION_TITLE': 'Neural Simulation',
    'MAIN_WINDOW_TAG': 'main_window',
    'STATUS_TEXT_TAG': 'status_text',
    'NODES_TEXT_TAG': 'nodes_text',
    'EDGES_TEXT_TAG': 'edges_text',
    'ENERGY_TEXT_TAG': 'energy_text',
    'CONNECTIONS_TEXT_TAG': 'connections_text',
    'LEGEND_WINDOW_TAG': 'legend_window',
    'HELP_WINDOW_TAG': 'help_window'
}

# Common Error Messages
ERROR_MESSAGES = {
    'GRAPH_NONE': 'Graph is None',
    'INVALID_NODE_ID': 'Invalid node ID',
    'MISSING_ATTRIBUTE': 'Missing required attribute',
    'CALLBACK_ERROR': 'Callback execution failed',
    'UI_UPDATE_ERROR': 'UI update failed',
    'INVALID_SLOT': 'Invalid slot number',
    'FILE_OPERATION_FAILED': 'File operation failed',
    'GRAPH_OPERATION_FAILED': 'Graph operation failed',
    'CONFIG_LOAD_FAILED': 'Configuration loading failed',
    'MEMORY_ALLOCATION_FAILED': 'Memory allocation failed',
    'NETWORK_OPERATION_FAILED': 'Network operation failed',
    'SIMULATION_STEP_FAILED': 'Simulation step failed',
    'NODE_OPERATION_FAILED': 'Node operation failed',
    'EXCEPTION_OCCURRED': 'Exception occurred',
    'ERROR_OCCURRED': 'Error occurred',
    'WARNING_OCCURRED': 'Warning occurred',
    'DEBUG_INFO': 'Debug info',
    'SUCCESS_MESSAGE': 'Success',
    'FAILURE_MESSAGE': 'Failure',
    'UI_ERROR': 'UI Error',
    'PROCESSING_ERROR': 'Error processing',
    'SIMULATION_ERROR': 'Error in simulation'
}

# Common Log Messages
LOG_MESSAGES = {
    'SYSTEM_STARTED': 'System started',
    'SYSTEM_STOPPED': 'System stopped',
    'SIMULATION_STARTED': 'Simulation started',
    'SIMULATION_STOPPED': 'Simulation stopped',
    'GRAPH_UPDATED': 'Graph updated',
    'NODE_CREATED': 'Node created',
    'NODE_DELETED': 'Node deleted',
    'CONNECTION_CREATED': 'Connection created',
    'CONNECTION_DELETED': 'Connection deleted',
    'ENERGY_UPDATED': 'Energy updated',
    'MEMORY_CONSOLIDATED': 'Memory consolidated',
    'LEARNING_APPLIED': 'Learning applied'
}

# Common File Paths
FILE_PATHS = {
    'NEURAL_MAPS_DIR': 'data/neural_maps',
    'SLOT_METADATA': 'slot_metadata.json',
    'CONFIG_FILE': 'config/config.ini',
    'LOG_DIR': 'logs',
    'PROFILE_DIR': 'profiles'
}

# Common Node Properties
NODE_PROPERTIES = {
    'ID': 'id',
    'TYPE': 'type',
    'SUBTYPE': 'subtype',
    'ENERGY': 'energy',
    'STATE': 'state',
    'THRESHOLD': 'threshold',
    'MEMBRANE_POTENTIAL': 'membrane_potential',
    'REFRACTORY_TIMER': 'refractory_timer',
    'LAST_UPDATE': 'last_update',
    'PLASTICITY_ENABLED': 'plasticity_enabled',
    'ELIGIBILITY_TRACE': 'eligibility_trace',
    'ENHANCED_BEHAVIOR': 'enhanced_behavior'
}

# Common Connection Properties
CONNECTION_PROPERTIES = {
    'SOURCE': 'source',
    'TARGET': 'target',
    'WEIGHT': 'weight',
    'DELAY': 'delay',
    'LAST_ACTIVITY': 'last_activity',
    'ACTIVATION_COUNT': 'activation_count'
}

# Common System States
SYSTEM_STATES = {
    'INITIALIZING': 'initializing',
    'RUNNING': 'running',
    'STOPPED': 'stopped',
    'PAUSED': 'paused',
    'ERROR': 'error',
    'CLEANUP': 'cleanup'
}

# Common Node States
NODE_STATES = {
    'ACTIVE': 'active',
    'INACTIVE': 'inactive',
    'PENDING': 'pending',
    'CONSOLIDATING': 'consolidating',
    'SYNTHESIZING': 'synthesizing',
    'PLANNING': 'planning',
    'IMAGINING': 'imagining',
    'REGULATING': 'regulating'
}

# Common Node Types
NODE_TYPES = {
    'SENSORY': 'sensory',
    'DYNAMIC': 'dynamic',
    'OSCILLATOR': 'oscillator',
    'INTEGRATOR': 'integrator',
    'RELAY': 'relay',
    'HIGHWAY': 'highway',
    'WORKSPACE': 'workspace',
    'TRANSMITTER': 'transmitter',
    'RESONATOR': 'resonator',
    'DAMPENER': 'dampener'
}

# Common Connection Types
CONNECTION_TYPES = {
    'EXCITATORY': 'excitatory',
    'INHIBITORY': 'inhibitory',
    'MODULATORY': 'modulatory',
    'PLASTIC': 'plastic',
    'BURST': 'burst',
    'GATED': 'gated'
}

# Common Performance Metrics
PERFORMANCE_METRICS = {
    'STEP_TIME': 'step_time',
    'TOTAL_RUNTIME': 'total_runtime',
    'FPS': 'fps',
    'MEMORY_USAGE': 'memory_usage_mb',
    'CPU_PERCENT': 'cpu_percent',
    'GPU_USAGE': 'gpu_usage_percent',
    'ERROR_RATE': 'error_rate',
    'NODE_COUNT': 'node_count',
    'EDGE_COUNT': 'edge_count',
    'THROUGHPUT': 'throughput'
}

# Common Thresholds
THRESHOLDS = {
    'MEMORY_WARNING_MB': 2000.0,
    'MEMORY_CRITICAL_MB': 4000.0,
    'CPU_WARNING_PERCENT': 80.0,
    'CPU_CRITICAL_PERCENT': 95.0,
    'STEP_TIME_WARNING_MS': 100.0,
    'STEP_TIME_CRITICAL_MS': 500.0,
    'FPS_WARNING': 20.0,
    'FPS_CRITICAL': 10.0,
    'ERROR_RATE_WARNING': 0.05,
    'ERROR_RATE_CRITICAL': 0.1
}

# Common Default Values
DEFAULT_VALUES = {
    'ENERGY_CAP': 5.0,  # Reduced from 255.0 to make modulation effects more pronounced
    'THRESHOLD_DEFAULT': 0.5,
    'LEARNING_RATE': 0.01,
    'REFRACTORY_PERIOD': 0.1,
    'INTEGRATION_RATE': 0.5,
    'RELAY_AMPLIFICATION': 1.5,
    'HIGHWAY_ENERGY_BOOST': 2.0,
    'WEIGHT_DEFAULT': 1.0,
    'DELAY_DEFAULT': 0.0,
    'ELIGIBILITY_TRACE_DEFAULT': 0.0
}


def load_constants_from_config():
    """Load constants from configuration file."""
    try:
        config_manager = get_config_manager()
        logger.info("Loading constants from config...")

        # Load energy cap
        energy_cap = config_manager.get('SystemConstants.node_energy_cap', 5.0)
        DEFAULT_VALUES['ENERGY_CAP'] = energy_cap
        logger.info("Loaded ENERGY_CAP from config: %s", energy_cap)

        # Load other constants if they exist in config
        threshold = config_manager.get('SystemConstants.threshold_default', 0.5)
        DEFAULT_VALUES['THRESHOLD_DEFAULT'] = threshold
        logger.info("Loaded THRESHOLD_DEFAULT from config: %s", threshold)

        learning_rate = config_manager.get('Learning.plasticity_rate', 0.01)
        DEFAULT_VALUES['LEARNING_RATE'] = learning_rate
        logger.info("Loaded LEARNING_RATE from config: %s", learning_rate)

        refractory = config_manager.get('SystemConstants.refractory_period', 0.1)
        DEFAULT_VALUES['REFRACTORY_PERIOD'] = refractory
        logger.info("Loaded REFRACTORY_PERIOD from config: %s", refractory)

        integration = config_manager.get('EnhancedNodes.integrator_threshold', 0.5)
        DEFAULT_VALUES['INTEGRATION_RATE'] = integration
        logger.info("Loaded INTEGRATION_RATE from config: %s", integration)

        relay_amp = config_manager.get('EnhancedNodes.relay_amplification', 1.5)
        DEFAULT_VALUES['RELAY_AMPLIFICATION'] = relay_amp
        logger.info("Loaded RELAY_AMPLIFICATION from config: %s", relay_amp)

        highway_boost = config_manager.get('EnhancedNodes.highway_energy_boost', 2.0)
        DEFAULT_VALUES['HIGHWAY_ENERGY_BOOST'] = highway_boost
        logger.info("Loaded HIGHWAY_ENERGY_BOOST from config: %s", highway_boost)

        logger.info("Constants loaded successfully from config")

    except ImportError as e:
        # Circular import - will load later when needed
        logger.debug("Deferred loading constants due to circular import: %s", e)
    except (KeyError, ValueError, TypeError) as e:
        logger.error("Failed to load constants from config: %s", e)
        logger.info("Using default hardcoded values")


# Load constants on import (lazy to avoid circular imports)
try:
    load_constants_from_config()
except ImportError:
    # If there's a circular import, constants will be loaded when first accessed
    pass

# Common Print Patterns
PRINT_PATTERNS = {
    'ERROR_PREFIX': 'Error:',
    'WARNING_PREFIX': 'Warning:',
    'INFO_PREFIX': 'Info:',
    'DEBUG_PREFIX': 'Debug:',
    'SUCCESS_PREFIX': 'Success:',
    'FAILURE_PREFIX': 'Failure:',
    'EXCEPTION_PATTERN': 'except Exception as e:\n        print(f',
    'PRINT_F_PATTERN': 'print(f',
    'PRINT_PATTERN': 'print(',
    'ERROR_PRINT_PATTERN': 'print(f"Error: {e}")',
    'WARNING_PRINT_PATTERN': 'print(f"Warning: {message}")',
    'INFO_PRINT_PATTERN': 'print(f"Info: {message}")',
    'UI_ERROR_PATTERN': 'print(f"UI Error: {e}")',
    'PROCESSING_ERROR_PATTERN': 'print(f"Error processing {file_path}: {e}")',
    'SIMULATION_ERROR_PATTERN': 'print(f"Error in simulation: {e}")',
    'INVALID_SLOT_PATTERN': 'print(f"Invalid slot number: {slot_number}")'
}

# Common Exception Types
EXCEPTION_TYPES = {
    'VALUE_ERROR': ValueError,
    'TYPE_ERROR': TypeError,
    'ATTRIBUTE_ERROR': AttributeError,
    'KEY_ERROR': KeyError,
    'RUNTIME_ERROR': RuntimeError,
    'MEMORY_ERROR': MemoryError,
    'OS_ERROR': OSError,
    'IO_ERROR': IOError
}

# Common Function Names
FUNCTION_NAMES = {
    'INIT': '__init__',
    'RESET': 'reset',
    'UPDATE': 'update',
    'START': 'start',
    'STOP': 'stop',
    'GET_STATISTICS': 'get_statistics',
    'RESET_STATISTICS': 'reset_statistics',
    'MAIN': 'main'
}

# Common Class Names
CLASS_NAMES = {
    'SIMULATION_COORDINATOR': 'SimulationCoordinator',
    'BEHAVIOR_ENGINE': 'BehaviorEngine',
    'LEARNING_ENGINE': 'LearningEngine',
    'MEMORY_SYSTEM': 'MemorySystem',
    'ERROR_HANDLER': 'ErrorHandler',
    'PERFORMANCE_MONITOR': 'PerformanceMonitor',
    'NETWORK_METRICS': 'NetworkMetrics',
    'UI_ENGINE': 'UIEngine',
    'UI_STATE_MANAGER': 'UIStateManager'
}
