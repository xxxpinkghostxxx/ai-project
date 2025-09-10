"""
energy_parameters.py

Data classes for energy behavior parameters to reduce feature envy and long parameter lists.
These classes encapsulate related parameters and provide a clean interface for energy operations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torch_geometric.data import Data


@dataclass
class EnergyUpdateParams:
    """Parameters for energy update operations."""
    node_id: int
    delta_energy: float
    current_energy: float
    membrane_potential: Optional[float] = None
    threshold: Optional[float] = None
    refractory_timer: Optional[float] = None
    last_activation: Optional[float] = None
    behavior: Optional[str] = None


@dataclass
class OscillatorParams:
    """Parameters for oscillator behavior."""
    node_id: int
    current_time: float
    oscillation_frequency: float = 1.0
    pulse_energy_fraction: float = 0.1
    refractory_period: float = 0.1
    last_activation: Optional[float] = None


@dataclass
class IntegratorParams:
    """Parameters for integrator behavior."""
    node_id: int
    integration_rate: float = 0.5
    integrator_threshold: float = 0.8
    accumulated_energy: float = 0.0
    membrane_potential: Optional[float] = None


@dataclass
class RelayParams:
    """Parameters for relay behavior."""
    node_id: int
    relay_amplification: float = 1.5
    energy_transfer_fraction: float = 0.2
    current_energy: Optional[float] = None


@dataclass
class HighwayParams:
    """Parameters for highway behavior."""
    node_id: int
    highway_energy_boost: float = 2.0
    energy_threshold_low: float = 100.0
    energy_boost_amount: float = 50.0
    distribution_energy_base: float = 10.0
    current_energy: Optional[float] = None


@dataclass
class ConnectionParams:
    """Parameters for connection operations."""
    source_id: int
    target_id: int
    weight: float = 1.0
    edge_type: str = 'excitatory'
    delay: float = 0.0
    plasticity_tag: bool = False


@dataclass
class EnergyPulseParams:
    """Parameters for energy pulse operations."""
    source_node_idx: int
    pulse_energy: float
    pulse_targets: List[int]
    energy_fraction: float = 0.15


@dataclass
class MembranePotentialParams:
    """Parameters for membrane potential calculations."""
    node_id: int
    current_energy: float
    energy_cap: float
    membrane_potential_cap: float = 1.0


@dataclass
class RefractoryParams:
    """Parameters for refractory period management."""
    node_id: int
    refractory_timer: float
    time_step: float = 0.01
    membrane_potential_reset: float = 0.0


class EnergyParameterFactory:
    """Factory for creating energy parameter objects from graph data."""
    
    @staticmethod
    def create_energy_update_params(graph: Data, node_id: int, delta_energy: float) -> EnergyUpdateParams:
        """Create energy update parameters from graph data."""
        if node_id >= len(graph.node_labels):
            raise ValueError(f"Invalid node_id: {node_id}")
        
        node = graph.node_labels[node_id]
        current_energy = graph.x[node_id, 0].item()
        
        return EnergyUpdateParams(
            node_id=node_id,
            delta_energy=delta_energy,
            current_energy=current_energy,
            membrane_potential=node.get('membrane_potential'),
            threshold=node.get('threshold'),
            refractory_timer=node.get('refractory_timer'),
            last_activation=node.get('last_activation'),
            behavior=node.get('behavior')
        )
    
    @staticmethod
    def create_oscillator_params(graph: Data, node_id: int, current_time: float) -> OscillatorParams:
        """Create oscillator parameters from graph data."""
        if node_id >= len(graph.node_labels):
            raise ValueError(f"Invalid node_id: {node_id}")
        
        node = graph.node_labels[node_id]
        
        return OscillatorParams(
            node_id=node_id,
            current_time=current_time,
            oscillation_frequency=node.get('oscillation_freq', 1.0),
            last_activation=node.get('last_activation')
        )
    
    @staticmethod
    def create_integrator_params(graph: Data, node_id: int) -> IntegratorParams:
        """Create integrator parameters from graph data."""
        if node_id >= len(graph.node_labels):
            raise ValueError(f"Invalid node_id: {node_id}")
        
        node = graph.node_labels[node_id]
        
        return IntegratorParams(
            node_id=node_id,
            integration_rate=node.get('integration_rate', 0.5),
            integrator_threshold=node.get('integrator_threshold', 0.8),
            membrane_potential=node.get('membrane_potential')
        )
    
    @staticmethod
    def create_relay_params(graph: Data, node_id: int) -> RelayParams:
        """Create relay parameters from graph data."""
        if node_id >= len(graph.node_labels):
            raise ValueError(f"Invalid node_id: {node_id}")
        
        node = graph.node_labels[node_id]
        
        return RelayParams(
            node_id=node_id,
            relay_amplification=node.get('relay_amplification', 1.5)
        )
    
    @staticmethod
    def create_highway_params(graph: Data, node_id: int) -> HighwayParams:
        """Create highway parameters from graph data."""
        if node_id >= len(graph.node_labels):
            raise ValueError(f"Invalid node_id: {node_id}")
        
        node = graph.node_labels[node_id]
        
        return HighwayParams(
            node_id=node_id,
            highway_energy_boost=node.get('highway_energy_boost', 2.0)
        )
    
    @staticmethod
    def create_connection_params(source_id: int, target_id: int, weight: float = 1.0, 
                               edge_type: str = 'excitatory') -> ConnectionParams:
        """Create connection parameters."""
        return ConnectionParams(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            edge_type=edge_type
        )
    
    @staticmethod
    def create_energy_pulse_params(source_node_idx: int, source_energy: float, 
                                 energy_fraction: float = 0.15) -> EnergyPulseParams:
        """Create energy pulse parameters."""
        return EnergyPulseParams(
            source_node_idx=source_node_idx,
            pulse_energy=source_energy * energy_fraction,
            pulse_targets=[],
            energy_fraction=energy_fraction
        )
    
    @staticmethod
    def create_membrane_potential_params(node_id: int, current_energy: float, 
                                       energy_cap: float) -> MembranePotentialParams:
        """Create membrane potential parameters."""
        return MembranePotentialParams(
            node_id=node_id,
            current_energy=current_energy,
            energy_cap=energy_cap
        )
    
    @staticmethod
    def create_refractory_params(node_id: int, refractory_timer: float) -> RefractoryParams:
        """Create refractory parameters."""
        return RefractoryParams(
            node_id=node_id,
            refractory_timer=refractory_timer
        )


class EnergyParameterValidator:
    """Validator for energy parameter objects."""
    
    @staticmethod
    def validate_energy_update_params(params: EnergyUpdateParams) -> bool:
        """Validate energy update parameters."""
        if params.node_id < 0:
            return False
        if params.delta_energy is None:
            return False
        if params.current_energy < 0:
            return False
        return True
    
    @staticmethod
    def validate_oscillator_params(params: OscillatorParams) -> bool:
        """Validate oscillator parameters."""
        if params.node_id < 0:
            return False
        if params.oscillation_frequency <= 0:
            return False
        if params.current_time < 0:
            return False
        return True
    
    @staticmethod
    def validate_integrator_params(params: IntegratorParams) -> bool:
        """Validate integrator parameters."""
        if params.node_id < 0:
            return False
        if not 0 <= params.integration_rate <= 1:
            return False
        if not 0 <= params.integrator_threshold <= 1:
            return False
        return True
    
    @staticmethod
    def validate_relay_params(params: RelayParams) -> bool:
        """Validate relay parameters."""
        if params.node_id < 0:
            return False
        if params.relay_amplification <= 0:
            return False
        if not 0 <= params.energy_transfer_fraction <= 1:
            return False
        return True
    
    @staticmethod
    def validate_highway_params(params: HighwayParams) -> bool:
        """Validate highway parameters."""
        if params.node_id < 0:
            return False
        if params.highway_energy_boost <= 0:
            return False
        if params.energy_threshold_low < 0:
            return False
        if params.energy_boost_amount < 0:
            return False
        return True
    
    @staticmethod
    def validate_connection_params(params: ConnectionParams) -> bool:
        """Validate connection parameters."""
        if params.source_id < 0 or params.target_id < 0:
            return False
        if params.source_id == params.target_id:
            return False
        if params.weight < 0:
            return False
        if params.edge_type not in ['excitatory', 'inhibitory', 'modulatory']:
            return False
        return True
