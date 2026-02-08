"""
Energy Prediction Calculator for Simulation Validation

This module provides a comprehensive energy calculator that predicts expected
energy values for all node types, connection types, and combinations.
"""

from typing import Dict, List, Tuple, Optional
import torch

from project.pyg_neural_system import (
    NODE_TYPE_SENSORY,
    NODE_TYPE_DYNAMIC,
    NODE_TYPE_WORKSPACE,
    SUBTYPE_TRANSMITTER,
    SUBTYPE_RESONATOR,
    SUBTYPE_DAMPENER,
    CONN_TYPE_EXCITATORY,
    CONN_TYPE_INHIBITORY,
    CONN_TYPE_GATED,
    CONN_TYPE_PLASTIC,
    CONN_SUBTYPE3_ONE_WAY_OUT,
    CONN_SUBTYPE3_ONE_WAY_IN,
    CONN_SUBTYPE3_FREE_FLOW,
    NODE_ENERGY_CAP,
    NODE_DEATH_THRESHOLD,
    GATE_THRESHOLD,
    TRANSMISSION_LOSS,
    CONNECTION_MAINTENANCE_COST,
    NODE_ENERGY_DECAY,
)

from project.config import (
    CONN_ENERGY_TRANSFER_CAPACITY,
    CONN_MAINTENANCE_COST,
)

# Match runtime config behavior (transfer_capacity is read from ConfigManager in pyg_neural_system)
try:
    from project.utils.config_manager import ConfigManager  # type: ignore[import-not-found]
except Exception:  # pylint: disable=broad-exception-caught
    ConfigManager = None  # type: ignore[assignment]


class EnergyCalculator:
    """Predicts expected energy values based on node types, connections, and rules."""
    
    def __init__(self):
        # NOTE: In the current PyG implementation, dynamic "decay" is a per-outgoing-edge maintenance cost,
        # not an energy-proportional decay. (See: out_deg * CONNECTION_MAINTENANCE_COST)
        self.conn_maintenance_cost = CONN_MAINTENANCE_COST  # 0.122

        # Connection transfer capacity:
        # Runtime uses ConfigManager('system.conn_transfer_capacity'); fall back to CONN_ENERGY_TRANSFER_CAPACITY.
        self.transfer_capacity = self._get_runtime_transfer_capacity(default=CONN_ENERGY_TRANSFER_CAPACITY)
        
        # Transmission loss
        self.transmission_loss = TRANSMISSION_LOSS  # 0.9
        
        # Workspace energy adjustment target (70% of cap)
        self.workspace_target_energy = NODE_ENERGY_CAP * 0.7  # 170.8
        self.workspace_adjustment_rate = 0.02  # 2% per step

    def _get_runtime_transfer_capacity(self, default: float) -> float:
        if ConfigManager is None:
            return float(default)
        try:
            cfg = ConfigManager()
            cap = cfg.get_config('system', 'conn_transfer_capacity')
            if isinstance(cap, (int, float)):
                return float(cap)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return float(default)
    
    def calculate_energy_transfer(
        self,
        src_energy: float,
        dst_energy: float,
        weight: float,
        conn_type: int,
        src_subtype: Optional[int] = None,
        dst_subtype: Optional[int] = None,
        gate_threshold: float = GATE_THRESHOLD,
    ) -> Tuple[float, float]:
        """
        Calculate energy transfer between two nodes.
        
        Returns:
            (energy_loss_from_src, energy_gain_to_dst)
        
        Note: In the actual simulation, TRANSMISSION_LOSS is applied to the transfer
        amount BEFORE calculating both energy_loss and energy_gain. This means:
        - energy_loss = base_transfer * subtype_scales * TRANSMISSION_LOSS
        - energy_gain = base_transfer * subtype_scales * TRANSMISSION_LOSS (or negative for inhibitory)
        """
        # Check if connection is gated and source energy is below threshold
        if conn_type == CONN_TYPE_GATED and src_energy < gate_threshold:
            return (0.0, 0.0)
        
        # Base transfer amount
        base_transfer = src_energy * weight * self.transfer_capacity
        
        # Apply dynamic subtype modulation (source) - matches actual implementation
        if src_subtype == SUBTYPE_TRANSMITTER:
            base_transfer *= 1.2  # Boost outgoing (matches actual: 1.2)
        elif src_subtype == SUBTYPE_DAMPENER:
            base_transfer *= 0.6  # Reduce outgoing (matches actual: 0.6)
        # SUBTYPE_RESONATOR: 1.0 (no change)
        
        # Apply dynamic subtype modulation (destination) - matches actual implementation
        if dst_subtype == SUBTYPE_RESONATOR:
            base_transfer *= 1.2  # Boost incoming (matches actual: 1.2)
        elif dst_subtype == SUBTYPE_DAMPENER:
            base_transfer *= 0.5  # Reduce incoming (matches actual: 0.5)
        
        # Apply transmission loss to base_transfer FIRST (matches actual simulation)
        # In the actual code: transfer_amounts.mul_(TRANSMISSION_LOSS) is applied before
        # calculating both energy_loss and energy_gain
        transfer_after_loss = base_transfer * self.transmission_loss
        
        # Apply connection type effects
        if conn_type == CONN_TYPE_EXCITATORY:
            # Source loses energy, destination gains (both use transmission-loss-adjusted amount)
            energy_loss = transfer_after_loss
            energy_gain = transfer_after_loss
        elif conn_type == CONN_TYPE_INHIBITORY:
            # Source doesn't lose energy, destination loses
            energy_loss = 0.0
            energy_gain = -transfer_after_loss
        elif conn_type == CONN_TYPE_GATED:
            # Same as excitatory (gate already checked)
            energy_loss = transfer_after_loss
            energy_gain = transfer_after_loss
        elif conn_type == CONN_TYPE_PLASTIC:
            # Same as excitatory (weight may change over time, but we predict with current weight)
            energy_loss = transfer_after_loss
            energy_gain = transfer_after_loss
        else:
            # Unknown type, no transfer
            energy_loss = 0.0
            energy_gain = 0.0
        
        return (energy_loss, energy_gain)
    
    def calculate_node_energy_change(
        self,
        node_type: int,
        current_energy: float,
        incoming_transfers: List[float],
        outgoing_transfers: List[float],
        num_connections: int = 0,
        is_workspace: bool = False,
        sensory_true_value: Optional[float] = None,
    ) -> float:
        """
        Calculate expected energy change for a node in one step.
        
        Args:
            node_type: Type of node
            current_energy: Current energy value
            incoming_transfers: List of energy gains from incoming connections
            outgoing_transfers: List of energy losses from outgoing connections
            num_connections: Number of connections (for maintenance cost)
            is_workspace: Whether this is a workspace node (for adjustment)
            sensory_true_value: True value for sensory nodes (for restoration)
        
        Returns:
            Expected energy change (delta)
        """
        delta = 0.0
        
        # Sensory nodes: restored to true value, no decay, no connection effects
        if node_type == NODE_TYPE_SENSORY:
            if sensory_true_value is not None:
                # Will be restored to true value
                delta = sensory_true_value - current_energy
            else:
                # No change if no true value set
                delta = 0.0
            return delta
        
        # Dynamic nodes: per-edge maintenance + transfers
        if node_type == NODE_TYPE_DYNAMIC:
            # Connection maintenance cost (matches runtime: out_deg * CONNECTION_MAINTENANCE_COST)
            delta -= float(num_connections) * float(self.conn_maintenance_cost)
            
            # Incoming transfers (gain)
            delta += sum(incoming_transfers)
            
            # Outgoing transfers (loss)
            delta -= sum(outgoing_transfers)
            
            return delta
        
        # Workspace nodes: no decay, receive transfers, energy adjustment
        if node_type == NODE_TYPE_WORKSPACE:
            # Incoming transfers (gain)
            delta += sum(incoming_transfers)
            
            # Energy adjustment towards target (only if energy > 0.1)
            if current_energy > 0.1:
                adjustment = (self.workspace_target_energy - current_energy) * self.workspace_adjustment_rate
                delta += adjustment
            
            # Clamp to prevent negative (minimum 0.0)
            if current_energy + delta < 0.0:
                delta = -current_energy
            
            return delta
        
        # Unknown type: no change
        return 0.0
    
    def predict_energy_after_steps(
        self,
        initial_energies: Dict[int, float],
        node_types: Dict[int, int],
        node_subtypes: Dict[int, Optional[int]],
        connections: List[Tuple[int, int, float, int, int]],  # (src, dst, weight, conn_type, conn_subtype3)
        num_steps: int,
        sensory_true_values: Optional[Dict[int, float]] = None,
    ) -> Dict[int, List[float]]:
        """
        Predict energy values for each node over multiple steps.
        
        Args:
            initial_energies: {node_idx: initial_energy}
            node_types: {node_idx: node_type}
            node_subtypes: {node_idx: dynamic_subtype or None}
            connections: List of (src_idx, dst_idx, weight, conn_type, conn_subtype3)
            num_steps: Number of steps to predict
            sensory_true_values: {sensory_node_idx: true_value} for sensory nodes
        
        Returns:
            {node_idx: [energy_at_step_0, energy_at_step_1, ..., energy_at_step_N]}
        """
        # Initialize energy history
        energy_history: Dict[int, List[float]] = {
            idx: [initial_energies[idx]] for idx in initial_energies.keys()
        }
        
        # Build connection maps for efficient lookup
        incoming_connections: Dict[int, List[Tuple[int, int, float, int, int]]] = {}
        outgoing_connections: Dict[int, List[Tuple[int, int, float, int, int]]] = {}
        
        for src, dst, weight, conn_type, conn_subtype3 in connections:
            if dst not in incoming_connections:
                incoming_connections[dst] = []
            incoming_connections[dst].append((src, dst, weight, conn_type, conn_subtype3))
            
            if src not in outgoing_connections:
                outgoing_connections[src] = []
            outgoing_connections[src].append((src, dst, weight, conn_type, conn_subtype3))
        
        # Predict each step
        current_energies = initial_energies.copy()
        
        for step in range(num_steps):
            # Calculate energy changes for each node
            energy_changes: Dict[int, float] = {}
            
            for node_idx in current_energies.keys():
                node_type = node_types.get(node_idx, NODE_TYPE_DYNAMIC)
                node_subtype = node_subtypes.get(node_idx)
                current_energy = current_energies[node_idx]
                
                # Sensory nodes: overwritten to true value each step (both before/after transfers in runtime)
                if node_type == NODE_TYPE_SENSORY and sensory_true_values is not None:
                    sensory_true = sensory_true_values.get(node_idx)
                    if sensory_true is not None:
                        current_energies[node_idx] = float(sensory_true)
                        energy_history[node_idx][-1] = float(sensory_true)

                # Calculate incoming transfers
                incoming_transfers = []
                for src, dst, weight, conn_type, conn_subtype3 in incoming_connections.get(node_idx, []):
                    src_energy = current_energies.get(src, 0.0)
                    src_subtype = node_subtypes.get(src)
                    dst_subtype = node_subtype
                    
                    # Check connection directionality
                    if conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_OUT:
                        # Only works if src is parent of dst (simplified: always allow for prediction)
                        pass
                    elif conn_subtype3 == CONN_SUBTYPE3_ONE_WAY_IN:
                        # Only works if src is NOT parent of dst (simplified: always allow for prediction)
                        pass
                    # CONN_SUBTYPE3_FREE_FLOW: always works
                    
                    _, energy_gain = self.calculate_energy_transfer(
                        src_energy, current_energy, weight, conn_type,
                        src_subtype, dst_subtype
                    )
                    incoming_transfers.append(energy_gain)
                
                # Calculate outgoing transfers
                outgoing_transfers = []
                for src, dst, weight, conn_type, conn_subtype3 in outgoing_connections.get(node_idx, []):
                    src_energy = current_energy
                    src_subtype = node_subtype
                    dst_energy = current_energies.get(dst, 0.0)
                    dst_subtype = node_subtypes.get(dst)

                    # Transfers to sensory are blocked; attempt burns energy then refunds half => net loss = 50%
                    if node_types.get(dst, NODE_TYPE_DYNAMIC) == NODE_TYPE_SENSORY:
                        # In runtime: net_loss = attempted_base_transfer * TRANSMISSION_LOSS * 0.5
                        # calculate_energy_transfer already applies TRANSMISSION_LOSS, so energy_loss
                        # is already base_transfer * TRANSMISSION_LOSS. We multiply by 0.5 for net loss.
                        energy_loss, _ = self.calculate_energy_transfer(
                            src_energy, dst_energy, weight, conn_type,
                            src_subtype, None
                        )
                        # Net loss is 50% of the would-be transfer (energy_loss already has TRANSMISSION_LOSS applied)
                        outgoing_transfers.append(energy_loss * 0.5)
                    else:
                        energy_loss, _ = self.calculate_energy_transfer(
                            src_energy, dst_energy, weight, conn_type,
                            src_subtype, dst_subtype
                        )
                        outgoing_transfers.append(energy_loss)
                
                # Get number of connections for maintenance cost
                num_conns = len(outgoing_connections.get(node_idx, []))
                
                # Get sensory true value if applicable
                sensory_true = None
                if node_type == NODE_TYPE_SENSORY and sensory_true_values:
                    sensory_true = sensory_true_values.get(node_idx)
                
                # Calculate energy change
                delta = self.calculate_node_energy_change(
                    node_type, current_energy, incoming_transfers, outgoing_transfers,
                    num_conns, node_type == NODE_TYPE_WORKSPACE, sensory_true
                )
                
                energy_changes[node_idx] = delta
            
            # Apply energy changes
            for node_idx, delta in energy_changes.items():
                new_energy = current_energies[node_idx] + delta
                # Clamp to valid range
                new_energy = max(NODE_DEATH_THRESHOLD, min(NODE_ENERGY_CAP, new_energy))
                current_energies[node_idx] = new_energy
                energy_history[node_idx].append(new_energy)
        
        return energy_history
    
    def compare_predicted_vs_actual(
        self,
        predicted: Dict[int, List[float]],
        actual: Dict[int, List[float]],
        tolerance: float = 0.1,
    ) -> Dict[str, any]:
        """
        Compare predicted energy values with actual values.
        
        Returns:
            Dictionary with comparison results, errors, and warnings
        """
        results = {
            "matches": [],
            "errors": [],
            "warnings": [],
            "max_deviation": 0.0,
            "avg_deviation": 0.0,
        }
        
        all_deviations = []
        
        for node_idx in predicted.keys():
            if node_idx not in actual:
                results["errors"].append(f"Node {node_idx}: Predicted but not found in actual")
                continue
            
            pred_values = predicted[node_idx]
            actual_values = actual[node_idx]
            
            if len(pred_values) != len(actual_values):
                results["errors"].append(
                    f"Node {node_idx}: Length mismatch - predicted {len(pred_values)}, actual {len(actual_values)}"
                )
                continue
            
            for step, (pred, act) in enumerate(zip(pred_values, actual_values)):
                deviation = abs(pred - act)
                relative_deviation = deviation / max(abs(pred), abs(act), 1.0)  # Avoid division by zero
                all_deviations.append(deviation)
                
                if relative_deviation > tolerance:
                    results["warnings"].append(
                        f"Node {node_idx} step {step}: Predicted {pred:.3f}, Actual {act:.3f}, "
                        f"Deviation {deviation:.3f} ({relative_deviation*100:.1f}%)"
                    )
                else:
                    results["matches"].append(f"Node {node_idx} step {step}: Match (pred={pred:.3f}, act={act:.3f})")
        
        if all_deviations:
            results["max_deviation"] = max(all_deviations)
            results["avg_deviation"] = sum(all_deviations) / len(all_deviations)
        
        return results
