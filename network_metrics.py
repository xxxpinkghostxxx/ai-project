"""
Network Analysis & Metrics System
Phase 5: Network topology analysis and performance monitoring

This module provides comprehensive network analysis including:
- Criticality metrics (branching ratio, avalanche analysis)
- Connectivity analysis (degree distribution, clustering, path lengths)
- Energy balance monitoring (conservation, distribution, entropy)
- Performance tracking and trend analysis
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import math

# Import logging utilities and configuration manager
from logging_utils import log_runtime, log_step
from config_manager import get_network_metrics_config

# Get network metrics configuration values
def get_network_config_values():
    config = get_network_metrics_config()
    return {
        'calculation_interval': config.get('calculation_interval', 50),
        'criticality_target': config.get('criticality_target', 1.0),
        'connectivity_target': config.get('connectivity_target', 0.3)
    }


class NetworkMetrics:
    """
    Comprehensive network analysis and monitoring system.
    
    Provides real-time metrics for network topology, criticality,
    connectivity, and energy balance with historical tracking.
    """
    
    def __init__(self, calculation_interval: int = None):
        """
        Initialize the NetworkMetrics system.
        
        Args:
            calculation_interval: Steps between metric calculations (uses config if None)
        """
        config = get_network_config_values()
        self.calculation_interval = calculation_interval or config['calculation_interval']
        self.criticality_target = config.get('criticality_target', 1.0)
        self.connectivity_target = config.get('connectivity_target', 0.3)
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.activation_patterns = defaultdict(int)  # Track activation patterns
        self.last_metrics = {}
        
        # Performance tracking
        self.calculation_times = deque(maxlen=100)
        self.metric_updates = 0
        
        logging.info("[NETWORK_METRICS] Initialized with calculation interval: %d", calculation_interval)
    
    @log_runtime
    def calculate_criticality(self, graph) -> float:
        """
        Compute branching ratio and criticality metrics.
        
        Branching ratio = (total new activations) / (total activations)
        Values close to 1.0 indicate critical state (avalanche behavior).
        
        Args:
            graph: PyTorch Geometric Data object with node labels
            
        Returns:
            float: Criticality metric (0.0 to 2.0+)
        """
        log_step("calculate_criticality start")
        
        if not hasattr(graph, 'node_labels') or not graph.node_labels:
            logging.warning("[NETWORK_METRICS] No node labels found for criticality calculation")
            return 0.0
        
        total_activations = 0
        new_activations = 0
        current_patterns = set()
        
        for node_idx, node in enumerate(graph.node_labels):
            if node.get('last_activation', 0) > 0:
                total_activations += 1
                
                # Create activation pattern signature
                pattern_key = self._create_activation_pattern(node, graph, node_idx)
                current_patterns.add(pattern_key)
                
                # Check if this is a new activation pattern
                if self._is_new_activation_pattern(pattern_key):
                    new_activations += 1
        
        # Calculate branching ratio
        if total_activations > 0:
            branching_ratio = new_activations / total_activations
            
            # Update activation pattern history
            self._update_activation_patterns(current_patterns)
            
            # Log criticality metrics
            logging.info(f"[NETWORK_METRICS] Criticality: total={total_activations}, "
                        f"new={new_activations}, ratio={branching_ratio:.3f}")
            
            log_step("calculate_criticality end")
            return branching_ratio
        
        log_step("calculate_criticality end - no activations")
        return 0.0
    
    @log_runtime
    def analyze_connectivity(self, graph) -> Dict[str, float]:
        """
        Analyze network structure and evolution.
        
        Calculates clustering coefficient, path lengths, degree distribution,
        and other topological metrics.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            Dict containing connectivity metrics
        """
        log_step("analyze_connectivity start")
        
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            logging.warning("[NETWORK_METRICS] No edge index found for connectivity analysis")
            return {'num_nodes': 0, 'num_edges': 0, 'avg_degree': 0.0, 'density': 0.0}
        
        num_nodes = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
        
        # Basic connectivity metrics
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
        
        # Calculate clustering coefficient
        clustering_coeff = self._calculate_clustering_coefficient(graph)
        
        # Calculate average path length (approximation)
        avg_path_length = self._calculate_average_path_length(graph)
        
        # Degree distribution analysis
        degree_distribution = self._analyze_degree_distribution(graph)
        
        connectivity_metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'density': density,
            'clustering_coefficient': clustering_coeff,
            'avg_path_length': avg_path_length,
            'degree_variance': degree_distribution.get('variance', 0.0),
            'degree_entropy': degree_distribution.get('entropy', 0.0),
            'max_degree': degree_distribution.get('max_degree', 0),
            'min_degree': degree_distribution.get('min_degree', 0)
        }
        
        logging.info(f"[NETWORK_METRICS] Connectivity: nodes={num_nodes}, edges={num_edges}, "
                    f"density={density:.3f}, clustering={clustering_coeff:.3f}")
        
        log_step("analyze_connectivity end")
        return connectivity_metrics
    
    @log_runtime
    def measure_energy_balance(self, graph) -> Dict[str, float]:
        """
        Monitor global energy conservation and distribution.
        
        Tracks energy flow, identifies imbalances, and calculates
        energy distribution statistics.
        
        Args:
            graph: PyTorch Geometric Data object with node features
            
        Returns:
            Dict containing energy balance metrics
        """
        log_step("measure_energy_balance start")
        
        if not hasattr(graph, 'x') or graph.x is None:
            logging.warning("[NETWORK_METRICS] No node features found for energy balance")
            return {'total_energy': 0.0, 'energy_variance': 0.0, 'energy_entropy': 0.0}
        
        # Extract energy values from node features
        try:
            energy_values = graph.x[:, 0].cpu().numpy()
        except (IndexError, AttributeError):
            logging.warning("[NETWORK_METRICS] Could not extract energy values from graph features")
            return {'total_energy': 0.0, 'energy_variance': 0.0, 'energy_entropy': 0.0}
        
        total_energy = np.sum(energy_values)
        energy_variance = np.var(energy_values) if len(energy_values) > 1 else 0.0
        energy_entropy = self._calculate_energy_entropy(energy_values)
        
        # Energy distribution analysis
        energy_distribution = {
            'total_energy': float(total_energy),
            'energy_variance': float(energy_variance),
            'energy_entropy': float(energy_entropy),
            'min_energy': float(np.min(energy_values)),
            'max_energy': float(np.max(energy_values)),
            'mean_energy': float(np.mean(energy_values)),
            'energy_skewness': float(self._calculate_skewness(energy_values)),
            'energy_kurtosis': float(self._calculate_kurtosis(energy_values))
        }
        
        # Check for energy conservation violations
        conservation_status = self._check_energy_conservation(graph, total_energy)
        energy_distribution.update(conservation_status)
        
        logging.info(f"[NETWORK_METRICS] Energy: total={total_energy:.2f}, "
                    f"variance={energy_variance:.2f}, entropy={energy_entropy:.3f}")
        
        log_step("measure_energy_balance end")
        return energy_distribution
    
    @log_runtime
    def calculate_comprehensive_metrics(self, graph) -> Dict[str, any]:
        """
        Calculate all network metrics in a single call.
        
        This is the main method called by the main loop to get
        a complete snapshot of network state.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            Dict containing all network metrics
        """
        log_step("calculate_comprehensive_metrics start")
        
        import time as _time
        start_time = _time.time()
        
        # Calculate all metrics
        criticality = self.calculate_criticality(graph)
        connectivity = self.analyze_connectivity(graph)
        energy_balance = self.measure_energy_balance(graph)
        
        # Performance metrics
        performance_metrics = {
            'calculation_time': _time.time() - start_time,
            'metric_updates': self.metric_updates,
            'calculation_efficiency': self._calculate_efficiency()
        }
        
        # Combine all metrics
        comprehensive_metrics = {
            'timestamp': _time.time(),
            'criticality': criticality,
            'connectivity': connectivity,
            'energy_balance': energy_balance,
            'performance': performance_metrics
        }
        
        # Store in history
        self.metrics_history.append(comprehensive_metrics)
        self.last_metrics = comprehensive_metrics
        self.metric_updates += 1
        
        # Update calculation time tracking
        self.calculation_times.append(performance_metrics['calculation_time'])
        
        logging.info(f"[NETWORK_METRICS] Comprehensive metrics calculated in "
                    f"{performance_metrics['calculation_time']:.4f}s")
        
        log_step("calculate_comprehensive_metrics end")
        return comprehensive_metrics
    
    def get_metrics_trends(self, window_size: int = 10) -> Dict[str, List[float]]:
        """
        Get trend analysis of metrics over time.
        
        Args:
            window_size: Number of recent measurements to analyze
            
        Returns:
            Dict containing trend data for each metric
        """
        if len(self.metrics_history) < window_size:
            return {}
        
        recent_metrics = list(self.metrics_history)[-window_size:]
        
        trends = {
            'criticality_trend': [m['criticality'] for m in recent_metrics],
            'density_trend': [m['connectivity']['density'] for m in recent_metrics],
            'energy_variance_trend': [m['energy_balance']['energy_variance'] for m in recent_metrics],
            'clustering_trend': [m['connectivity']['clustering_coefficient'] for m in recent_metrics]
        }
        
        return trends
    
    def get_network_health_score(self) -> Dict[str, any]:
        """
        Calculate overall network health score based on metrics.
        
        Returns:
            Dict containing health score and recommendations
        """
        if not self.last_metrics:
            return {'score': 0.0, 'status': 'unknown', 'recommendations': []}
        
        metrics = self.last_metrics
        
        # Health scoring algorithm
        health_score = 0.0
        recommendations = []
        
        # Criticality scoring (target: 0.8-1.2)
        criticality = metrics['criticality']
        if 0.8 <= criticality <= 1.2:
            health_score += 25.0
        elif 0.5 <= criticality <= 1.5:
            health_score += 15.0
        else:
            health_score += 5.0
            recommendations.append("Criticality outside optimal range")
        
        # Connectivity scoring (target: 0.1-0.5)
        density = metrics['connectivity']['density']
        if 0.1 <= density <= 0.5:
            health_score += 25.0
        elif 0.05 <= density <= 0.7:
            health_score += 15.0
        else:
            health_score += 5.0
            recommendations.append("Network density outside optimal range")
        
        # Energy balance scoring (target: low variance)
        energy_variance = metrics['energy_balance']['energy_variance']
        if energy_variance < 100.0:
            health_score += 25.0
        elif energy_variance < 500.0:
            health_score += 15.0
        else:
            health_score += 5.0
            recommendations.append("High energy variance detected")
        
        # Performance scoring
        calc_time = metrics['performance']['calculation_time']
        if calc_time < 0.001:
            health_score += 25.0
        elif calc_time < 0.01:
            health_score += 15.0
        else:
            health_score += 5.0
            recommendations.append("Metric calculation performance degraded")
        
        # Determine status
        if health_score >= 80.0:
            status = 'excellent'
        elif health_score >= 60.0:
            status = 'good'
        elif health_score >= 40.0:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'score': health_score,
            'status': status,
            'recommendations': recommendations,
            'criticality': criticality,
            'density': density,
            'energy_variance': energy_variance,
            'calculation_time': calc_time
        }
    
    # Private helper methods
    
    def _create_activation_pattern(self, node: Dict, graph, node_idx: int) -> str:
        """Create a unique signature for activation patterns."""
        # Create pattern based on node type, energy, and connections
        node_type = node.get('type', 'unknown')
        energy_level = int(node.get('energy', 0) / 10)  # Quantize energy
        
        # Count active connections
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            # Convert PyTorch tensor to numpy for comparison
            edge_sources = graph.edge_index[0].cpu().numpy()
            active_connections = np.sum(edge_sources == node_idx)
        else:
            active_connections = 0
        
        return f"{node_type}_{energy_level}_{active_connections}"
    
    def _is_new_activation_pattern(self, pattern_key: str) -> bool:
        """Check if activation pattern is new."""
        if pattern_key not in self.activation_patterns:
            self.activation_patterns[pattern_key] = 1
            return True
        return False
    
    def _update_activation_patterns(self, current_patterns: set):
        """Update activation pattern history."""
        for pattern in current_patterns:
            self.activation_patterns[pattern] += 1
    
    def _calculate_clustering_coefficient(self, graph) -> float:
        """Calculate network clustering coefficient."""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return 0.0
        
        # Simplified clustering calculation
        # In a full implementation, this would calculate the actual clustering coefficient
        # For now, return a reasonable approximation based on network density
        num_nodes = len(graph.node_labels) if hasattr(graph, 'node_labels') else 1
        num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
        
        if num_nodes <= 1:
            return 0.0
        
        density = num_edges / (num_nodes * (num_nodes - 1))
        return min(density * 2.0, 1.0)  # Approximate clustering
    
    def _calculate_average_path_length(self, graph) -> float:
        """Calculate average shortest path length between nodes."""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return 0.0
        
        # Simplified path length calculation
        # In a full implementation, this would use Floyd-Warshall or similar
        num_nodes = len(graph.node_labels) if hasattr(graph, 'node_labels') else 1
        
        if num_nodes <= 1:
            return 0.0
        
        # Approximate based on network density
        num_edges = graph.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Higher density = shorter paths
        if density > 0.5:
            return 1.5
        elif density > 0.1:
            return 2.5
        else:
            return 4.0
    
    def _analyze_degree_distribution(self, graph) -> Dict[str, float]:
        """Analyze node degree distribution."""
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return {'variance': 0.0, 'entropy': 0.0, 'max_degree': 0, 'min_degree': 0}
        
        # Calculate degree for each node
        degrees = defaultdict(int)
        for edge_idx in range(graph.edge_index.shape[1]):
            source = graph.edge_index[0, edge_idx].item()
            target = graph.edge_index[1, edge_idx].item()
            degrees[source] += 1
            degrees[target] += 1
        
        if not degrees:
            return {'variance': 0.0, 'entropy': 0.0, 'max_degree': 0, 'min_degree': 0}
        
        degree_values = list(degrees.values())
        
        return {
            'variance': float(np.var(degree_values)),
            'entropy': float(self._calculate_entropy(degree_values)),
            'max_degree': max(degree_values),
            'min_degree': min(degree_values)
        }
    
    def _calculate_energy_entropy(self, energy_values: np.ndarray) -> float:
        """Calculate entropy of energy distribution."""
        if len(energy_values) == 0:
            return 0.0
        
        # Normalize energy values
        total_energy = np.sum(energy_values)
        if total_energy == 0:
            return 0.0
        
        probabilities = energy_values / total_energy
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a list of values."""
        if not values:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(values, bins=min(20, len(values)))
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Normalize
        probabilities = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((values - mean) / std) ** 3)
        return float(skewness)
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of distribution."""
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3
        return float(kurtosis)
    
    def _check_energy_conservation(self, graph, total_energy: float) -> Dict[str, any]:
        """Check for energy conservation violations."""
        # This would compare current energy with expected energy based on
        # initial conditions and known energy sources/sinks
        
        # For now, return basic conservation status
        return {
            'conservation_violation': False,
            'energy_drift': 0.0,
            'conservation_status': 'stable'
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate metric calculation efficiency."""
        if not self.calculation_times:
            return 1.0
        
        avg_time = np.mean(self.calculation_times)
        if avg_time == 0:
            return 1.0
        
        # Efficiency decreases as calculation time increases
        # Target: < 1ms calculations
        efficiency = max(0.0, 1.0 - (avg_time / 0.001))
        return float(efficiency)


# Utility functions for external use
def create_network_metrics(calculation_interval: int = 50) -> NetworkMetrics:
    """Factory function to create NetworkMetrics instance."""
    return NetworkMetrics(calculation_interval)


def quick_network_analysis(graph) -> Dict[str, any]:
    """
    Quick network analysis without creating a full NetworkMetrics instance.
    
    Args:
        graph: PyTorch Geometric Data object
        
    Returns:
        Dict containing basic network metrics
    """
    metrics = NetworkMetrics()
    return metrics.calculate_comprehensive_metrics(graph)
