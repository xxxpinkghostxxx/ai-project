



from config.unified_config_manager import (get_enhanced_nodes_config,
                                           get_learning_config)


class EnergyConstants:
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
        try:
            config = get_learning_config()
            if config is None:
                return cls.ACTIVATION_THRESHOLD_DEFAULT
            value = config.get('activation_threshold', cls.ACTIVATION_THRESHOLD_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.ACTIVATION_THRESHOLD_DEFAULT
            return value
        except Exception:
            return cls.ACTIVATION_THRESHOLD_DEFAULT
    @classmethod
    def get_refractory_period(cls) -> float:
        try:
            config = get_learning_config()
            if config is None:
                return cls.REFRACTORY_PERIOD_LONG
            value = config.get('refractory_period', cls.REFRACTORY_PERIOD_LONG)
            if not isinstance(value, (int, float)) or value <= 0:
                return cls.REFRACTORY_PERIOD_LONG
            return value
        except Exception:
            return cls.REFRACTORY_PERIOD_LONG
    @classmethod
    def get_integration_rate(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.INTEGRATION_RATE_DEFAULT
            value = config.get('integration_rate', cls.INTEGRATION_RATE_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.INTEGRATION_RATE_DEFAULT
            return value
        except Exception:
            return cls.INTEGRATION_RATE_DEFAULT
    @classmethod
    def get_relay_amplification(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            value = config.get('relay_amplification', cls.RELAY_AMPLIFICATION_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            return value
        except Exception:
            return cls.RELAY_AMPLIFICATION_DEFAULT
    @classmethod
    def get_highway_energy_boost(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            value = config.get('highway_energy_boost', cls.HIGHWAY_ENERGY_BOOST_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            return value
        except Exception:
            return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
    @classmethod
    def get_decay_rate(cls) -> float:
        try:
            config = get_learning_config()
            if config is None:
                return cls.DECAY_RATE_DEFAULT
            value = config.get('energy_leak_rate', cls.DECAY_RATE_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.DECAY_RATE_DEFAULT
            return value
        except Exception:
            return cls.DECAY_RATE_DEFAULT
    @classmethod
    def get_plasticity_threshold(cls) -> float:
        try:
            config = get_learning_config()
            if config is None:
                return cls.PLASTICITY_THRESHOLD_DEFAULT
            value = config.get('plasticity_gate_threshold', cls.PLASTICITY_THRESHOLD_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.PLASTICITY_THRESHOLD_DEFAULT
            return value
        except Exception:
            return cls.PLASTICITY_THRESHOLD_DEFAULT
    @classmethod
    def get_dynamic_energy_threshold(cls) -> float:
        from src.energy.energy_behavior import get_node_energy_cap
        return cls.DYNAMIC_ENERGY_THRESHOLD_FRACTION * get_node_energy_cap()


class ConnectionConstants:
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


class OscillatorConstants:
    OSCILLATION_FREQUENCY_DEFAULT = 1.0
    PULSE_ENERGY_FRACTION = 0.1
    REFRACTORY_PERIOD_SHORT = 0.1
    @classmethod
    def get_oscillation_frequency(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.OSCILLATION_FREQUENCY_DEFAULT
            value = config.get('oscillator_frequency', cls.OSCILLATION_FREQUENCY_DEFAULT)
            if not isinstance(value, (int, float)) or value <= 0:
                return cls.OSCILLATION_FREQUENCY_DEFAULT
            return value
        except Exception:
            return cls.OSCILLATION_FREQUENCY_DEFAULT


class IntegratorConstants:
    INTEGRATION_RATE_DEFAULT = 0.5
    INTEGRATOR_THRESHOLD_DEFAULT = 0.8
    @classmethod
    def get_integration_rate(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.INTEGRATION_RATE_DEFAULT
            value = config.get('integration_rate', cls.INTEGRATION_RATE_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.INTEGRATION_RATE_DEFAULT
            return value
        except Exception:
            return cls.INTEGRATION_RATE_DEFAULT
    @classmethod
    def get_integrator_threshold(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.INTEGRATOR_THRESHOLD_DEFAULT
            value = config.get('integrator_threshold', cls.INTEGRATOR_THRESHOLD_DEFAULT)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return cls.INTEGRATOR_THRESHOLD_DEFAULT
            return value
        except Exception:
            return cls.INTEGRATOR_THRESHOLD_DEFAULT


class RelayConstants:
    RELAY_AMPLIFICATION_DEFAULT = 1.5
    ENERGY_TRANSFER_FRACTION = 0.2
    @classmethod
    def get_relay_amplification(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            value = config.get('relay_amplification', cls.RELAY_AMPLIFICATION_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.RELAY_AMPLIFICATION_DEFAULT
            return value
        except Exception:
            return cls.RELAY_AMPLIFICATION_DEFAULT


class HighwayConstants:
    HIGHWAY_ENERGY_BOOST_DEFAULT = 2.0
    ENERGY_THRESHOLD_LOW = 100.0
    ENERGY_BOOST_AMOUNT = 50.0
    DISTRIBUTION_ENERGY_BASE = 10.0
    @classmethod
    def get_highway_energy_boost(cls) -> float:
        try:
            config = get_enhanced_nodes_config()
            if config is None:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            value = config.get('highway_energy_boost', cls.HIGHWAY_ENERGY_BOOST_DEFAULT)
            if not isinstance(value, (int, float)) or value < 1:
                return cls.HIGHWAY_ENERGY_BOOST_DEFAULT
            return value
        except Exception:
            return cls.HIGHWAY_ENERGY_BOOST_DEFAULT







