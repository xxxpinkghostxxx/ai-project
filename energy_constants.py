"""
energy_constants.py

Constants for energy behavior calculations to replace magic numbers.
All energy-related constants are centralized here for maintainability.
"""

from config_manager import get_system_constants, get_learning_config, get_enhanced_nodes_config


class EnergyConstants:
    """Centralized energy-related constants."""
    
    # Time constants
    TIME_STEP = 0.01
    REFRACTORY_PERIOD_SHORT = 0.1
    REFRACTORY_PERIOD_MEDIUM = 0.5
    REFRACTORY_PERIOD_LONG = 1.0
    
    # Energy thresholds and limits
    ACTIVATION_THRESHOLD_DEFAULT = 0.5
    MEMBRANE_POTENTIAL_CAP = 1.0
    ENERGY_TRANSFER_FRACTION = 0.2
    PULSE_ENERGY_FRACTION = 0.1
    PULSE_ENERGY_FRACTION_LARGE = 0.15
    
    # Integration and dynamics
    INTEGRATION_RATE_DEFAULT = 0.5
    RELAY_AMPLIFICATION_DEFAULT = 1.5
    HIGHWAY_ENERGY_BOOST_DEFAULT = 2.0
    
    # Energy levels
    ENERGY_THRESHOLD_LOW = 100.0
    ENERGY_BOOST_AMOUNT = 50.0
    DISTRIBUTION_ENERGY_BASE = 10.0
    
    # Decay and plasticity
    DECAY_RATE_DEFAULT = 0.02
    PLASTICITY_THRESHOLD_DEFAULT = 0.3
    ELIGIBILITY_TRACE_DECAY = 0.95
    
    # Connection weights
    CONNECTION_WEIGHT_DEFAULT = 1.0
    CONNECTION_WEIGHT_LOW = 0.1
    CONNECTION_WEIGHT_MEDIUM = 0.6
    CONNECTION_WEIGHT_HIGH = 0.8
    CONNECTION_WEIGHT_MODULATORY = 0.5
    
    # Weight limits
    WEIGHT_CAP_MAX = 5.0
    WEIGHT_MIN = 0.1
    
    # Edge properties
    EDGE_DELAY_DEFAULT = 0.0
    ELIGIBILITY_TRACE_DEFAULT = 0.0
    LAST_ACTIVITY_DEFAULT = 0.0
    
    # Membrane potential
    MEMBRANE_POTENTIAL_RESET = 0.0
    
    # Dynamic energy
    DYNAMIC_ENERGY_THRESHOLD_FRACTION = 0.8
    
    @classmethod
    def get_activation_threshold(cls) -> float:
        """Get activation threshold from config or default."""
        config = get_learning_config()
        return config.get('activation_threshold', cls.ACTIVATION_THRESHOLD_DEFAULT)
    
    @classmethod
    def get_refractory_period(cls) -> float:
        """Get refractory period from config or default."""
        config = get_learning_config()
        return config.get('refractory_period', cls.REFRACTORY_PERIOD_LONG)
    
    @classmethod
    def get_integration_rate(cls) -> float:
        """Get integration rate from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('integration_rate', cls.INTEGRATION_RATE_DEFAULT)
    
    @classmethod
    def get_relay_amplification(cls) -> float:
        """Get relay amplification from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('relay_amplification', cls.RELAY_AMPLIFICATION_DEFAULT)
    
    @classmethod
    def get_highway_energy_boost(cls) -> float:
        """Get highway energy boost from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('highway_energy_boost', cls.HIGHWAY_ENERGY_BOOST_DEFAULT)
    
    @classmethod
    def get_decay_rate(cls) -> float:
        """Get decay rate from config or default."""
        config = get_learning_config()
        return config.get('energy_leak_rate', cls.DECAY_RATE_DEFAULT)
    
    @classmethod
    def get_plasticity_threshold(cls) -> float:
        """Get plasticity threshold from config or default."""
        config = get_learning_config()
        return config.get('plasticity_gate_threshold', cls.PLASTICITY_THRESHOLD_DEFAULT)
    
    @classmethod
    def get_dynamic_energy_threshold(cls) -> float:
        """Get dynamic energy threshold."""
        from connection_logic import get_max_dynamic_energy
        return cls.DYNAMIC_ENERGY_THRESHOLD_FRACTION * get_max_dynamic_energy()


class ConnectionConstants:
    """Constants for connection logic."""
    
    # Edge types
    EDGE_TYPES = ['excitatory', 'inhibitory', 'modulatory']
    
    # Default values
    DEFAULT_EDGE_WEIGHT = 1.0
    DEFAULT_EDGE_DELAY = 0.0
    
    # Connection weights by type
    EXCITATORY_WEIGHT = 1.0
    INHIBITORY_WEIGHT = -1.0
    MODULATORY_WEIGHT = 0.5
    
    # Dynamic connection weights
    DYNAMIC_CONNECTION_WEIGHT = 0.6
    HIGHWAY_CONNECTION_WEIGHT = 0.8
    SENSORY_CONNECTION_WEIGHT = 0.1
    
    # Learning parameters
    LEARNING_RATE_DEFAULT = 0.01
    WEIGHT_CHANGE_FACTOR = 0.1
    
    # Weight limits
    WEIGHT_CAP_MAX = 5.0
    WEIGHT_MIN = 0.1
    
    # Eligibility trace
    ELIGIBILITY_TRACE_DECAY = 0.95
    ELIGIBILITY_TRACE_UPDATE = 0.1


class OscillatorConstants:
    """Constants for oscillator behavior."""
    
    # Frequency and timing
    OSCILLATION_FREQUENCY_DEFAULT = 1.0
    PULSE_ENERGY_FRACTION = 0.1
    REFRACTORY_PERIOD_SHORT = 0.1
    
    @classmethod
    def get_oscillation_frequency(cls) -> float:
        """Get oscillation frequency from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('oscillator_frequency', cls.OSCILLATION_FREQUENCY_DEFAULT)


class IntegratorConstants:
    """Constants for integrator behavior."""
    
    # Integration parameters
    INTEGRATION_RATE_DEFAULT = 0.5
    INTEGRATOR_THRESHOLD_DEFAULT = 0.8
    
    @classmethod
    def get_integration_rate(cls) -> float:
        """Get integration rate from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('integration_rate', cls.INTEGRATION_RATE_DEFAULT)
    
    @classmethod
    def get_integrator_threshold(cls) -> float:
        """Get integrator threshold from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('integrator_threshold', cls.INTEGRATOR_THRESHOLD_DEFAULT)


class RelayConstants:
    """Constants for relay behavior."""
    
    # Relay parameters
    RELAY_AMPLIFICATION_DEFAULT = 1.5
    ENERGY_TRANSFER_FRACTION = 0.2
    
    @classmethod
    def get_relay_amplification(cls) -> float:
        """Get relay amplification from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('relay_amplification', cls.RELAY_AMPLIFICATION_DEFAULT)


class HighwayConstants:
    """Constants for highway behavior."""
    
    # Highway parameters
    HIGHWAY_ENERGY_BOOST_DEFAULT = 2.0
    ENERGY_THRESHOLD_LOW = 100.0
    ENERGY_BOOST_AMOUNT = 50.0
    DISTRIBUTION_ENERGY_BASE = 10.0
    
    @classmethod
    def get_highway_energy_boost(cls) -> float:
        """Get highway energy boost from config or default."""
        config = get_enhanced_nodes_config()
        return config.get('highway_energy_boost', cls.HIGHWAY_ENERGY_BOOST_DEFAULT)
