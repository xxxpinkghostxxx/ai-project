



"""Energy constants module for neural system configuration and defaults."""

import time

from config.unified_config_manager import (get_enhanced_nodes_config,
                                           get_learning_config,
                                           get_system_constants)


class EnergyConstants:
    """Constants and configuration methods for energy-related parameters in the neural system."""

    TIME_STEP = 0.01
    REFRACTORY_PERIOD_SHORT = 0.1
    REFRACTORY_PERIOD_MEDIUM = 0.5
    REFRACTORY_PERIOD_LONG = 1.0
    ACTIVATION_THRESHOLD_DEFAULT = 0.5
    MEMBRANE_POTENTIAL_CAP = 1.0
    ENERGY_TRANSFER_FRACTION = 0.2
    PULSE_ENERGY_FRACTION = 0.1
    PULSE_ENERGY_FRACTION_LARGE = 0.15
    INTEGRATION_RATE_DEFAULT = 0.5
    RELAY_AMPLIFICATION_DEFAULT = 1.5
    HIGHWAY_ENERGY_BOOST_DEFAULT = 2.0
    ENERGY_THRESHOLD_LOW = 100.0
    ENERGY_BOOST_AMOUNT = 50.0
    DISTRIBUTION_ENERGY_BASE = 10.0
    DECAY_RATE_DEFAULT = 0.02
    PLASTICITY_THRESHOLD_DEFAULT = 0.3
    ELIGIBILITY_TRACE_DECAY = 0.95
    CONNECTION_WEIGHT_DEFAULT = 1.0
    CONNECTION_WEIGHT_LOW = 0.1
    CONNECTION_WEIGHT_MEDIUM = 0.6
    CONNECTION_WEIGHT_HIGH = 0.8
    CONNECTION_WEIGHT_MODULATORY = 0.5
    WEIGHT_CAP_MAX = 5.0
    WEIGHT_MIN = 0.1
    EDGE_DELAY_DEFAULT = 0.0
    ELIGIBILITY_TRACE_DEFAULT = 0.0
    LAST_ACTIVITY_DEFAULT = 0.0
    MEMBRANE_POTENTIAL_RESET = 0.0
    DYNAMIC_ENERGY_THRESHOLD_FRACTION = 0.8
    @classmethod
    def get_activation_threshold(cls) -> float:
        """Get the activation threshold from configuration or return default."""
        try:
            config = get_learning_config()
            if config is None:
                return cls.ACTIVATION_THRESHOLD_DEFAULT
            value = config.get('activation_threshold', cls.ACTIVATION_THRESHOLD_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.ACTIVATION_THRESHOLD_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.ACTIVATION_THRESHOLD_DEFAULT
    @classmethod
    def get_refractory_period(cls) -> float:
        """Get the refractory period from configuration or return default."""
        try:
            config = get_learning_config()
            if config is None:
                return cls.REFRACTORY_PERIOD_LONG
            value = config.get('refractory_period', cls.REFRACTORY_PERIOD_LONG)
            if not isinstance(value, (int, float)) or value <= 0:
                return cls.REFRACTORY_PERIOD_LONG
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.REFRACTORY_PERIOD_LONG
    @classmethod
    def get_integration_rate(cls) -> float:
        """Get the integration rate from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.INTEGRATION_RATE_DEFAULT
            value = config.get('integration_rate', cls.INTEGRATION_RATE_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.INTEGRATION_RATE_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.INTEGRATION_RATE_DEFAULT
    @classmethod
    def get_relay_amplification(cls) -> float:
        """Get the relay amplification from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            value = config.get('relay_amplification', cls.RELAY_AMPLIFICATION_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.RELAY_AMPLIFICATION_DEFAULT
    @classmethod
    def get_highway_energy_boost(cls) -> float:
        """Get the highway energy boost from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            value = config.get('highway_energy_boost', cls.HIGHWAY_ENERGY_BOOST_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
    @classmethod
    def get_decay_rate(cls) -> float:
        """Get the decay rate from configuration or return default."""
        try:
            config = get_learning_config()
            if config is None:
                return cls.DECAY_RATE_DEFAULT
            value = config.get('energy_leak_rate', cls.DECAY_RATE_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.DECAY_RATE_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.DECAY_RATE_DEFAULT
    @classmethod
    def get_plasticity_threshold(cls) -> float:
        """Get the plasticity threshold from configuration or return default."""
        try:
            config = get_learning_config()
            if config is None:
                return cls.PLASTICITY_THRESHOLD_DEFAULT
            value = config.get('plasticity_gate_threshold', cls.PLASTICITY_THRESHOLD_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.PLASTICITY_THRESHOLD_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.PLASTICITY_THRESHOLD_DEFAULT
    @classmethod
    def get_dynamic_energy_threshold(cls) -> float:
        """Get the dynamic energy threshold based on energy cap."""
        return cls.DYNAMIC_ENERGY_THRESHOLD_FRACTION * get_node_energy_cap()


# pylint: disable=too-few-public-methods
class ConnectionConstants:
    """Constants for connection types and weights in the neural network."""
    EDGE_TYPES = ['excitatory', 'inhibitory', 'modulatory']
    DEFAULT_EDGE_WEIGHT = 1.0
    DEFAULT_EDGE_DELAY = 0.0
    EXCITATORY_WEIGHT = 1.0
    INHIBITORY_WEIGHT = -1.0
    MODULATORY_WEIGHT = 0.5
    DYNAMIC_CONNECTION_WEIGHT = 0.6
    HIGHWAY_CONNECTION_WEIGHT = 0.8
    SENSORY_CONNECTION_WEIGHT = 0.1
    LEARNING_RATE_DEFAULT = 0.01
    WEIGHT_CHANGE_FACTOR = 0.1
    WEIGHT_CAP_MAX = 5.0
    WEIGHT_MIN = 0.1
    ELIGIBILITY_TRACE_DECAY = 0.95
    ELIGIBILITY_TRACE_UPDATE = 0.1
    DYNAMIC_ENERGY_THRESHOLD_FRACTION = 0.8


# pylint: disable=too-few-public-methods
class OscillatorConstants:
    """Constants for oscillator behavior in neural nodes."""
    OSCILLATION_FREQUENCY_DEFAULT = 1.0
    PULSE_ENERGY_FRACTION = 0.1
    REFRACTORY_PERIOD_SHORT = 0.1
    @classmethod
    def get_oscillation_frequency(cls) -> float:
        """Get the oscillation frequency from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.OSCILLATION_FREQUENCY_DEFAULT
            value = config.get('oscillator_frequency', cls.OSCILLATION_FREQUENCY_DEFAULT)
            if not isinstance(value, (int, float)) or value <= 0:
                return cls.OSCILLATION_FREQUENCY_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.OSCILLATION_FREQUENCY_DEFAULT


class IntegratorConstants:
    """Constants for integrator node behavior."""
    INTEGRATION_RATE_DEFAULT = 0.5
    INTEGRATOR_THRESHOLD_DEFAULT = 0.8
    @classmethod
    def get_integration_rate(cls) -> float:
        """Get the integration rate from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.INTEGRATION_RATE_DEFAULT
            value = config.get('integration_rate', cls.INTEGRATION_RATE_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.INTEGRATION_RATE_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.INTEGRATION_RATE_DEFAULT
    @classmethod
    def get_integrator_threshold(cls) -> float:
        """Get the integrator threshold from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.INTEGRATOR_THRESHOLD_DEFAULT
            value = config.get('integrator_threshold', cls.INTEGRATOR_THRESHOLD_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.INTEGRATOR_THRESHOLD_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.INTEGRATOR_THRESHOLD_DEFAULT


# pylint: disable=too-few-public-methods
class RelayConstants:
    """Constants for relay node amplification."""
    RELAY_AMPLIFICATION_DEFAULT = 1.5
    ENERGY_TRANSFER_FRACTION = 0.2
    @classmethod
    def get_relay_amplification(cls) -> float:
        """Get the relay amplification from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            value = config.get('relay_amplification', cls.RELAY_AMPLIFICATION_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.RELAY_AMPLIFICATION_DEFAULT


# pylint: disable=too-few-public-methods
class HighwayConstants:
    """Constants for highway node energy management."""
    HIGHWAY_ENERGY_BOOST_DEFAULT = 2.0
    ENERGY_THRESHOLD_LOW = 100.0
    ENERGY_BOOST_AMOUNT = 50.0
    DISTRIBUTION_ENERGY_BASE = 10.0
    @classmethod
    def get_highway_energy_boost(cls) -> float:
        """Get the highway energy boost from configuration or return default."""
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            value = config.get('highway_energy_boost', cls.HIGHWAY_ENERGY_BOOST_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            return value
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls.HIGHWAY_ENERGY_BOOST_DEFAULT


class _EnergyCapCache:
    """Simple cache for energy cap values to avoid repeated calculations."""

    def __init__(self):
        self._cache = None
        self._cache_time = 0
        self._cache_ttl = 300

    def get(self):
        """Retrieve cached value if valid."""
        current_time = time.time()
        if (self._cache is not None and
            current_time - self._cache_time < self._cache_ttl):
            return self._cache
        return None

    def set(self, value):
        """Set the cached value with current timestamp."""
        self._cache = value
        self._cache_time = time.time()

_energy_cap_cache = _EnergyCapCache()


def get_node_energy_cap():
    """Get node energy cap with assertions for safety."""
    current_time = time.time()
    assert current_time > 0, "Current time must be positive"

    # Check cache first
    cached_value = _energy_cap_cache.get()
    if cached_value is not None:
        assert cached_value > 0, "Cached energy cap must be positive"
        return cached_value

    # Calculate new value
    constants = get_system_constants()
    energy_cap = constants.get('node_energy_cap', 5.0)  # Updated default to match new config
    assert energy_cap > 0, "Energy cap must be positive"
    _energy_cap_cache.set(energy_cap)
    return energy_cap


_ENERGY_CAP_PRECACHED = None


def _precache_energy_cap():
    # pylint: disable=global-statement
    global _ENERGY_CAP_PRECACHED
    if _ENERGY_CAP_PRECACHED is None:
        _ENERGY_CAP_PRECACHED = get_node_energy_cap()
        # Ensure we have a reasonable fallback if config is not loaded
        if _ENERGY_CAP_PRECACHED <= 0 or _ENERGY_CAP_PRECACHED > 100:
            _ENERGY_CAP_PRECACHED = 5.0  # Updated default to match new config
    return _ENERGY_CAP_PRECACHED


_precache_energy_cap()







