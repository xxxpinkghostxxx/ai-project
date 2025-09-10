"""
homeostasis_controller.py

This module implements the homeostatic control system for the energy-based neural system.
It includes network regulation, criticality optimization, and energy balance control
mechanisms adapted from research concepts to work with the current energy dynamics framework.
"""

import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state

# Import configuration manager
from config_manager import get_homeostasis_config

# Homeostasis constants with configuration fallbacks
def get_homeostasis_config_values():
    config = get_homeostasis_config()
    return {
        'target_energy_ratio': config.get('target_energy_ratio', 0.6),
        'criticality_threshold': config.get('criticality_threshold', 0.1),
        'regulation_rate': config.get('regulation_rate', 0.001),
        'regulation_interval': config.get('regulation_interval', 100),
        'branching_target': 1.0,  # Default branching target
        'energy_variance_threshold': 0.2  # Default variance threshold
    }


class HomeostasisController:
    """
    Network regulation and criticality optimization system.
    Implements homeostatic control mechanisms to maintain network stability
    and drive the system toward critical dynamics.
    """
    
    def __init__(self):
        """Initialize the homeostasis controller with regulation parameters from configuration."""
        config = get_homeostasis_config_values()
        self.target_energy_ratio = config['target_energy_ratio']
        self.criticality_threshold = config['criticality_threshold']
        self.regulation_rate = config['regulation_rate']
        self.regulation_interval = config['regulation_interval']
        self.branching_target = config['branching_target']
        self.energy_variance_threshold = config['energy_variance_threshold']
        
        # Regulation statistics
        self.regulation_stats = {
            'energy_regulations': 0,
            'criticality_regulations': 0,
            'threshold_adjustments': 0,
            'total_regulation_events': 0,
            'last_regulation_time': time.time()
        }
        
        # Historical data for trend analysis
        self.energy_history = []
        self.branching_history = []
        self.variance_history = []
        self.max_history_length = 1000
    
    def regulate_network_activity(self, graph):
        """
        Monitor total energy and node counts, adjust birth/death thresholds automatically.
        This maintains energy balance and prevents runaway growth or collapse.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            Modified graph with adjusted thresholds
        """
        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
            return graph
        
        # Use network metrics if available for more comprehensive regulation
        if hasattr(graph, 'network_metrics') and graph.network_metrics:
            try:
                metrics = graph.network_metrics.calculate_comprehensive_metrics(graph)
                energy_metrics = metrics.get('energy_balance', {})
                connectivity_metrics = metrics.get('connectivity', {})
                
                # Use metrics-based energy assessment
                total_energy = energy_metrics.get('total_energy', 0.0)
                avg_energy = energy_metrics.get('average_energy', 0.0)
                energy_variance = energy_metrics.get('energy_variance', 0.0)
                energy_ratio = energy_metrics.get('energy_ratio', 0.0)
                
                # Get connectivity information
                connectivity_density = connectivity_metrics.get('density', 0.0)
                num_nodes = len(graph.node_labels)
                
                log_step("Metrics-based energy regulation",
                        total_energy=total_energy,
                        avg_energy=avg_energy,
                        energy_variance=energy_variance,
                        connectivity_density=connectivity_density)
                
            except Exception as e:
                log_step("Metrics calculation failed, using fallback", error=str(e))
                # Fallback to basic energy calculation
                total_energy = float(torch.sum(graph.x[:, 0]).item())
                num_nodes = len(graph.node_labels)
                avg_energy = total_energy / num_nodes if num_nodes > 0 else 0
                
                # Calculate energy ratio relative to maximum possible
                max_possible_energy = num_nodes * 244.0  # NODE_ENERGY_CAP
                energy_ratio = total_energy / max_possible_energy if max_possible_energy > 0 else 0
                
                # Calculate energy variance
                energy_values = graph.x[:, 0].cpu().numpy()
                energy_variance = np.var(energy_values) if len(energy_values) > 1 else 0
        else:
            # Fallback to basic energy calculation
            total_energy = float(torch.sum(graph.x[:, 0]).item())
            num_nodes = len(graph.node_labels)
            avg_energy = total_energy / num_nodes if num_nodes > 0 else 0
            
            # Calculate energy ratio relative to maximum possible
            max_possible_energy = num_nodes * 244.0  # NODE_ENERGY_CAP
            energy_ratio = total_energy / max_possible_energy if max_possible_energy > 0 else 0
            
            # Calculate energy variance
            energy_values = graph.x[:, 0].cpu().numpy()
            energy_variance = np.var(energy_values) if len(energy_values) > 1 else 0
        
        # Store historical data
        self._update_history('energy', energy_ratio)
        self._update_history('variance', energy_variance)
        
        # Determine regulation action
        regulation_needed = False
        regulation_type = None
        
        if energy_ratio > self.target_energy_ratio + self.criticality_threshold:
            # Too much energy - increase death threshold, decrease birth threshold
            regulation_needed = True
            regulation_type = 'reduce_energy'
            log_step("Energy regulation needed", 
                    action="reduce_energy",
                    current_ratio=energy_ratio,
                    target_ratio=self.target_energy_ratio)
            
        elif energy_ratio < self.target_energy_ratio - self.criticality_threshold:
            # Too little energy - decrease death threshold, increase birth threshold
            regulation_needed = True
            regulation_type = 'increase_energy'
            log_step("Energy regulation needed", 
                    action="increase_energy",
                    current_ratio=energy_ratio,
                    target_ratio=self.target_energy_ratio)
        
        # Apply regulation if needed
        if regulation_needed:
            self._apply_energy_regulation(graph, regulation_type, energy_ratio)
            self.regulation_stats['energy_regulations'] += 1
            self.regulation_stats['total_regulation_events'] += 1
        
        return graph
    
    def _apply_energy_regulation(self, graph, regulation_type, current_ratio):
        """
        Apply energy regulation by adjusting system thresholds.
        
        Args:
            graph: PyTorch Geometric graph
            regulation_type: Type of regulation ('reduce_energy' or 'increase_energy')
            current_ratio: Current energy ratio
        """
        # Import thresholds from death_and_birth_logic
        from death_and_birth_logic import (
            NODE_BIRTH_THRESHOLD, NODE_DEATH_THRESHOLD, NODE_BIRTH_COST
        )
        
        if regulation_type == 'reduce_energy':
            # Increase death threshold to remove more nodes
            new_death_threshold = NODE_DEATH_THRESHOLD * (1 + self.regulation_rate)
            new_death_threshold = min(new_death_threshold, 50.0)  # Cap at 50
            
            # Decrease birth threshold to create fewer nodes
            new_birth_threshold = NODE_BIRTH_THRESHOLD * (1 - self.regulation_rate)
            new_birth_threshold = max(new_birth_threshold, 50.0)  # Minimum 50
            
            log_step("Energy reduction regulation applied",
                    old_death_threshold=NODE_DEATH_THRESHOLD,
                    new_death_threshold=new_death_threshold,
                    old_birth_threshold=NODE_BIRTH_THRESHOLD,
                    new_birth_threshold=new_birth_threshold)
            
        elif regulation_type == 'increase_energy':
            # Decrease death threshold to remove fewer nodes
            new_death_threshold = NODE_DEATH_THRESHOLD * (1 - self.regulation_rate)
            new_death_threshold = max(new_death_threshold, 0.0)  # Minimum 0
            
            # Increase birth threshold to create more nodes
            new_birth_threshold = NODE_BIRTH_THRESHOLD * (1 + self.regulation_rate)
            new_birth_threshold = min(new_birth_threshold, 300.0)  # Cap at 300
            
            log_step("Energy increase regulation applied",
                    old_death_threshold=NODE_DEATH_THRESHOLD,
                    new_death_threshold=new_death_threshold,
                    old_birth_threshold=NODE_BIRTH_THRESHOLD,
                    new_birth_threshold=new_birth_threshold)
        
        # Store regulation in graph for access by other systems
        if not hasattr(graph, 'homeostasis_data'):
            graph.homeostasis_data = {}
        
        graph.homeostasis_data['last_regulation'] = {
            'type': regulation_type,
            'timestamp': time.time(),
            'old_energy_ratio': current_ratio,
            'new_death_threshold': new_death_threshold,
            'new_birth_threshold': new_birth_threshold
        }
    
    def optimize_criticality(self, graph):
        """
        Drive network toward critical state by optimizing branching ratio.
        Critical dynamics optimize information processing and learning.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            Modified graph with criticality optimizations
        """
        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'edge_index'):
            return graph
        
        # Use network metrics if available for more accurate criticality assessment
        if hasattr(graph, 'network_metrics') and graph.network_metrics:
            try:
                metrics = graph.network_metrics.calculate_comprehensive_metrics(graph)
                branching_ratio = metrics.get('criticality', 0.0)
                connectivity_metrics = metrics.get('connectivity', {})
                energy_metrics = metrics.get('energy_balance', {})
                
                # Use metrics-based criticality assessment
                log_step("Metrics-based criticality optimization",
                        branching_ratio=branching_ratio,
                        connectivity_density=connectivity_metrics.get('density', 0.0),
                        energy_variance=energy_metrics.get('energy_variance', 0.0))
            except Exception as e:
                log_step("Metrics calculation failed, using fallback", error=str(e))
                branching_ratio = self._calculate_branching_ratio(graph)
        else:
            # Fallback to basic branching ratio calculation
            branching_ratio = self._calculate_branching_ratio(graph)
        
        self._update_history('branching', branching_ratio)
        
        # Determine if criticality regulation is needed
        criticality_deviation = abs(branching_ratio - self.branching_target)
        
        if criticality_deviation > self.criticality_threshold:
            regulation_needed = True
            regulation_type = 'supercritical' if branching_ratio > self.branching_target else 'subcritical'
            
            log_step("Criticality regulation needed",
                    current_ratio=branching_ratio,
                    target_ratio=self.branching_target,
                    deviation=criticality_deviation,
                    regulation_type=regulation_type)
            
            # Apply criticality regulation
            self._apply_criticality_regulation(graph, regulation_type, branching_ratio)
            self.regulation_stats['criticality_regulations'] += 1
            self.regulation_stats['total_regulation_events'] += 1
        
        return graph
    
    def _calculate_branching_ratio(self, graph):
        """
        Calculate the branching ratio: (total new activations) / (total activations).
        A ratio of 1.0 indicates critical dynamics.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            float: Branching ratio
        """
        if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
            return 0.0
        
        # Count total activations (nodes with recent activity)
        current_time = time.time()
        total_activations = 0
        new_activations = 0
        
        for node in graph.node_labels:
            last_activation = node.get('last_activation', 0)
            if last_activation > 0:
                total_activations += 1
                
                # Check if this is a new activation pattern (within last 5 seconds)
                if current_time - last_activation < 5.0:
                    new_activations += 1
        
        # Calculate branching ratio
        if total_activations > 0:
            branching_ratio = new_activations / total_activations
            return branching_ratio
        
        return 0.0
    
    def _apply_criticality_regulation(self, graph, regulation_type, current_ratio):
        """
        Apply criticality regulation by adjusting connection formation parameters.
        
        Args:
            graph: PyTorch Geometric graph
            regulation_type: Type of regulation ('supercritical' or 'subcritical')
            current_ratio: Current branching ratio
        """
        if regulation_type == 'supercritical':
            # Network is too active - reduce excitatory connections
            regulation_action = 'reduce_excitation'
            log_step("Supercritical regulation applied",
                    action="reduce_excitation",
                    current_ratio=current_ratio)
            
        elif regulation_type == 'subcritical':
            # Network is too quiet - increase excitatory connections
            regulation_action = 'increase_excitation'
            log_step("Subcritical regulation applied",
                    action="increase_excitation",
                    current_ratio=current_ratio)
        
        # Store regulation data
        if not hasattr(graph, 'homeostasis_data'):
            graph.homeostasis_data = {}
        
        graph.homeostasis_data['criticality_regulation'] = {
            'type': regulation_type,
            'action': regulation_action,
            'timestamp': time.time(),
            'old_branching_ratio': current_ratio,
            'target_ratio': self.branching_target
        }
    
    def monitor_network_health(self, graph):
        """
        Monitor overall network health and stability.
        Provides early warning of potential issues.
        
        Args:
            graph: PyTorch Geometric graph
        
        Returns:
            dict: Network health metrics
        """
        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
            return {'status': 'unknown', 'warnings': []}
        
        warnings = []
        health_score = 1.0
        
        # Check energy balance
        total_energy = float(torch.sum(graph.x[:, 0]).item())
        num_nodes = len(graph.node_labels)
        avg_energy = total_energy / num_nodes if num_nodes > 0 else 0
        
        if avg_energy < 10.0:
            warnings.append("Low average energy - network may be dying")
            health_score *= 0.5
        elif avg_energy > 200.0:
            warnings.append("High average energy - network may be unstable")
            health_score *= 0.7
        
        # Check node count stability
        if hasattr(graph, 'homeostasis_data'):
            last_regulation = graph.homeostasis_data.get('last_regulation', {})
            if last_regulation:
                regulation_age = time.time() - last_regulation.get('timestamp', 0)
                if regulation_age > 300:  # 5 minutes
                    warnings.append("No recent energy regulation - system may be stuck")
                    health_score *= 0.8
        
        # Check connection density
        if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
            connection_density = graph.edge_index.shape[1] / (num_nodes * num_nodes)
            if connection_density < 0.001:
                warnings.append("Very low connection density - network may be fragmented")
                health_score *= 0.6
            elif connection_density > 0.1:
                warnings.append("Very high connection density - network may be overconnected")
                health_score *= 0.8
        
        # Determine overall health status
        if health_score >= 0.8:
            status = 'healthy'
        elif health_score >= 0.6:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'health_score': health_score,
            'warnings': warnings,
            'metrics': {
                'total_energy': total_energy,
                'num_nodes': num_nodes,
                'avg_energy': avg_energy,
                'connection_density': graph.edge_index.shape[1] / (num_nodes * num_nodes) if num_nodes > 0 else 0
            }
        }
    
    def _update_history(self, metric_type, value):
        """
        Update historical data for trend analysis.
        
        Args:
            metric_type: Type of metric ('energy', 'branching', 'variance')
            value: Metric value
        """
        if metric_type == 'energy':
            self.energy_history.append(value)
            if len(self.energy_history) > self.max_history_length:
                self.energy_history = self.energy_history[-self.max_history_length:]
        
        elif metric_type == 'branching':
            self.branching_history.append(value)
            if len(self.branching_history) > self.max_history_length:
                self.branching_history = self.branching_history[-self.max_history_length:]
        
        elif metric_type == 'variance':
            self.variance_history.append(value)
            if len(self.variance_history) > self.max_history_length:
                self.variance_history = self.variance_history[-self.max_history_length:]
    
    def get_regulation_statistics(self):
        """Get current regulation statistics for monitoring."""
        return self.regulation_stats.copy()
    
    def reset_statistics(self):
        """Reset regulation statistics."""
        self.regulation_stats = {
            'energy_regulations': 0,
            'criticality_regulations': 0,
            'threshold_adjustments': 0,
            'total_regulation_events': 0,
            'last_regulation_time': time.time()
        }
    
    def get_network_trends(self):
        """
        Analyze network trends based on historical data.
        
        Returns:
            dict: Trend analysis results
        """
        trends = {}
        
        # Energy trend analysis
        if len(self.energy_history) > 10:
            recent_energy = self.energy_history[-10:]
            energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
            trends['energy_trend'] = 'increasing' if energy_trend > 0.001 else 'decreasing' if energy_trend < -0.001 else 'stable'
            trends['energy_slope'] = energy_trend
        
        # Branching ratio trend analysis
        if len(self.branching_history) > 10:
            recent_branching = self.branching_history[-10:]
            branching_trend = np.polyfit(range(len(recent_branching)), recent_branching, 1)[0]
            trends['branching_trend'] = 'increasing' if branching_trend > 0.01 else 'decreasing' if branching_trend < -0.01 else 'stable'
            trends['branching_slope'] = branching_trend
        
        # Variance trend analysis
        if len(self.variance_history) > 10:
            recent_variance = self.variance_history[-10:]
            variance_trend = np.polyfit(range(len(recent_variance)), recent_variance, 1)[0]
            trends['variance_trend'] = 'increasing' if variance_trend > 0.1 else 'decreasing' if variance_trend < -0.1 else 'stable'
            trends['variance_slope'] = variance_trend
        
        return trends


# Utility functions for homeostasis analysis
def calculate_network_stability(graph):
    """
    Calculate network stability based on energy distribution and connection patterns.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        float: Stability score (0-1)
    """
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return 0.0
    
    # Energy stability
    energy_values = graph.x[:, 0].cpu().numpy()
    energy_variance = np.var(energy_values) if len(energy_values) > 1 else 0
    energy_stability = 1.0 / (1.0 + energy_variance / 100.0)  # Normalize
    
    # Connection stability
    if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
        num_connections = graph.edge_index.shape[1]
        num_nodes = len(graph.node_labels)
        connection_stability = min(num_connections / (num_nodes * 2), 1.0)  # Normalize
    else:
        connection_stability = 0.0
    
    # Overall stability
    stability = (energy_stability * 0.6 + connection_stability * 0.4)
    return stability


def detect_network_anomalies(graph):
    """
    Detect potential network anomalies that may require intervention.
    
    Args:
        graph: PyTorch Geometric graph
    
    Returns:
        list: List of detected anomalies
    """
    anomalies = []
    
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return anomalies
    
    # Check for energy anomalies
    energy_values = graph.x[:, 0].cpu().numpy()
    if len(energy_values) > 0:
        mean_energy = np.mean(energy_values)
        std_energy = np.std(energy_values)
        
        # Check for extreme energy values
        extreme_high = np.sum(energy_values > mean_energy + 3 * std_energy)
        extreme_low = np.sum(energy_values < mean_energy - 3 * std_energy)
        
        if extreme_high > 0:
            anomalies.append(f"High energy outliers: {extreme_high} nodes")
        if extreme_low > 0:
            anomalies.append(f"Low energy outliers: {extreme_low} nodes")
    
    # Check for connection anomalies
    if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
        num_connections = graph.edge_index.shape[1]
        num_nodes = len(graph.node_labels)
        
        if num_nodes > 0:
            connection_density = num_connections / (num_nodes * num_nodes)
            if connection_density < 0.0001:
                anomalies.append("Extremely low connection density")
            elif connection_density > 0.5:
                anomalies.append("Extremely high connection density")
    
    # Check for behavior anomalies
    behavior_counts = {}
    for node in graph.node_labels:
        behavior = node.get('behavior', 'unknown')
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
    
    if len(behavior_counts) > 0:
        total_nodes = sum(behavior_counts.values())
        for behavior, count in behavior_counts.items():
            if count / total_nodes > 0.8:
                anomalies.append(f"Behavior imbalance: {behavior} dominates ({count}/{total_nodes})")
    
    return anomalies


# Example usage and testing
if __name__ == "__main__":
    # Test homeostasis controller
    controller = HomeostasisController()
    
    print("Homeostasis Controller initialized successfully!")
    print(f"Target energy ratio: {controller.target_energy_ratio}")
    print(f"Criticality threshold: {controller.criticality_threshold}")
    print(f"Regulation rate: {controller.regulation_rate}")
    print(f"Regulation interval: {controller.regulation_interval}")
    
    # Test statistics
    stats = controller.get_regulation_statistics()
    print(f"Initial statistics: {stats}")
    
    print("\nHomeostasis Controller is ready for integration!")
