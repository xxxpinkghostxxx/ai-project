"""Module for calculating and analyzing network metrics in neural systems."""

import logging
from collections import defaultdict, deque
from typing import Dict, List

import time
import numpy as np
import torch

from config.unified_config_manager import get_network_metrics_config
from src.utils.logging_utils import log_runtime, log_step


class NetworkMetrics:
    """Class for computing various network metrics including criticality, connectivity, and energy balance."""

    def __init__(self, calculation_interval: int = None):

        config = get_network_metrics_config()
        self.calculation_interval = calculation_interval or config.get('calculation_interval', 50)
        self.criticality_target = config.get('criticality_target', 1.0)
        self.connectivity_target = config.get('connectivity_target', 0.3)
        self.metrics_history = deque(maxlen=1000)
        self.activation_patterns = defaultdict(int)
        self.last_metrics = {}
        self.calculation_times = deque(maxlen=100)
        self.metric_updates = 0
        self._initial_energy = 0.0
        self._energy_sources = 0.0
        self._energy_sinks = 0.0
        logging.info("[NETWORK_METRICS] Initialized with calculation interval: %d", self.calculation_interval)
    @log_runtime
    def calculate_criticality(self, graph) -> float:
        """Calculate the criticality of the network based on activation patterns."""
        log_step("calculate_criticality start")
        if not hasattr(graph, 'node_labels') or not graph.node_labels:
            logging.warning("[NETWORK_METRICS] No node labels found for criticality calculation")
            return 0.0
        total_activations = 0
        new_activations = 0
        current_patterns = set()
        sample_size = min(1000, len(graph.node_labels))
        sample_indices = np.random.choice(len(graph.node_labels), sample_size, replace=False)
        for node_idx in sample_indices:
            node = graph.node_labels[node_idx]
            if node.get('last_activation', 0) > 0:
                total_activations += 1
                pattern_key = self._create_activation_pattern(node, graph, node_idx)
                current_patterns.add(pattern_key)
                if self._is_new_activation_pattern(pattern_key):
                    new_activations += 1
        if total_activations > 0:
            branching_ratio = new_activations / total_activations
            self._update_activation_patterns(current_patterns)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("[NETWORK_METRICS] Criticality: total=%s, new=%s, ratio=%.3f", total_activations, new_activations, branching_ratio)
            log_step("calculate_criticality end")
            return branching_ratio
        log_step("calculate_criticality end - no activations")
        return 0.0
    @log_runtime
    def analyze_connectivity(self, graph) -> Dict[str, float]:
        """Analyze the connectivity metrics of the network graph."""
        log_step("analyze_connectivity start")
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            logging.warning("[NETWORK_METRICS] No edge index found for connectivity analysis")
            return {'num_nodes': 0, 'num_edges': 0, 'avg_degree': 0.0, 'density': 0.0}
        num_nodes = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
        num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
        clustering_coeff = self._calculate_clustering_coefficient(graph)
        avg_path_length = self._calculate_average_path_length(graph)
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
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[NETWORK_METRICS] Connectivity: nodes=%s, edges=%s, density=%.3f, clustering=%.3f", num_nodes, num_edges, density, clustering_coeff)
        log_step("analyze_connectivity end")
        return connectivity_metrics
    @log_runtime
    def measure_energy_balance(self, graph) -> Dict[str, float]:
        """Measure the energy balance and distribution in the network."""
        log_step("measure_energy_balance start")
        if not hasattr(graph, 'x') or graph.x is None:
            logging.warning("[NETWORK_METRICS] No node features found for energy balance")
            return {'total_energy': 0.0, 'energy_variance': 0.0, 'energy_entropy': 0.0}
        total_energy = float(torch.sum(graph.x[:, 0]).item())
        num_nodes = len(graph.node_labels)
        _avg_energy = total_energy / num_nodes if num_nodes > 0 else 0.0
        energy_variance = float(torch.var(graph.x[:, 0]).item()) if len(graph.x) > 1 else 0.0
        energy_entropy = self._calculate_energy_entropy(graph.x[:, 0].cpu().numpy())
        try:
            energy_values = graph.x[:, 0].cpu().numpy()
        except (IndexError, AttributeError):
            logging.warning("[NETWORK_METRICS] Could not extract energy values from graph features")
            return {'total_energy': 0.0, 'energy_variance': 0.0, 'energy_entropy': 0.0}
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
        conservation_status = self._check_energy_conservation(graph, total_energy)
        energy_distribution.update(conservation_status)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[NETWORK_METRICS] Energy: total=%.2f, variance=%.2f, entropy=%.3f", total_energy, energy_variance, energy_entropy)
        log_step("measure_energy_balance end")
        return energy_distribution
    @log_runtime
    def calculate_comprehensive_metrics(self, graph) -> Dict[str, any]:
        """Calculate comprehensive metrics including criticality, connectivity, and energy."""
        log_step("calculate_comprehensive_metrics start")
        start_time = time.time()
        criticality = self.calculate_criticality(graph)
        connectivity = self.analyze_connectivity(graph)
        energy_balance = self.measure_energy_balance(graph)
        performance_metrics = {
            'calculation_time': time.time() - start_time,
            'metric_updates': self.metric_updates,
            'calculation_efficiency': self._calculate_efficiency()
        }
        comprehensive_metrics = {
            'timestamp': time.time(),
            'criticality': criticality,
            'connectivity': connectivity,
            'energy_balance': energy_balance,
            'performance': performance_metrics
        }
        self.metrics_history.append(comprehensive_metrics)
        self.last_metrics = comprehensive_metrics
        self.metric_updates += 1
        self.calculation_times.append(performance_metrics['calculation_time'])
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[NETWORK_METRICS] Comprehensive metrics calculated in %.4fs", performance_metrics['calculation_time'])
        log_step("calculate_comprehensive_metrics end")
        return comprehensive_metrics
    def get_metrics_trends(self, window_size: int = 10) -> Dict[str, List[float]]:
        """Get trends in metrics over a specified window size."""
        if len(self.metrics_history) < window_size:
            return {}
        recent_metrics = list(self.metrics_history)[-window_size:]
        trends = {
            'criticality_trend': [m.get('criticality', 0.0) for m in recent_metrics],
            'density_trend': [m.get('connectivity', {}).get('density', 0.0) for m in recent_metrics],
            'energy_variance_trend': [m.get('energy_balance', {}).get('energy_variance', 0.0) for m in recent_metrics],
            'clustering_trend': [m.get('connectivity', {}).get('clustering_coefficient', 0.0) for m in recent_metrics]
        }
        return trends
    def get_network_health_score(self) -> Dict[str, any]:
        """Calculate and return the network health score based on metrics."""
        if not self.last_metrics:
            return {'score': 0.0, 'status': 'unknown', 'recommendations': []}
        metrics = self.last_metrics
        health_score = 0.0
        recommendations = []
        criticality = metrics.get('criticality', 0.0)
        density = metrics.get('connectivity', {}).get('density', 0.0)
        energy_variance = metrics.get('energy_balance', {}).get('energy_variance', 0.0)
        calc_time = metrics.get('performance', {}).get('calculation_time', 0.0)
        health_checks = [
            ('criticality', criticality, [(0.8, 1.2, 25.0), (0.5, 1.5, 15.0)], "Criticality outside optimal range"),
            ('density', density, [(0.1, 0.5, 25.0), (0.05, 0.7, 15.0)], "Network density outside optimal range"),
            ('energy_variance', energy_variance, [(0.0, 100.0, 25.0), (0.0, 500.0, 15.0)], "High energy variance detected"),
            ('calculation_time', calc_time, [(0.0, 0.001, 25.0), (0.0, 0.01, 15.0)], "Metric calculation performance degraded"),
        ]
        for _, value, ranges, rec in health_checks:
            for low, high, score in ranges:
                if low <= value <= high:
                    health_score += score
                    break
            else:
                health_score += 5.0
                recommendations.append(rec)
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
    def _create_activation_pattern(self, node: Dict, graph, node_idx: int) -> str:
        node_type = node.get('type', 'unknown')
        energy_level = int(node.get('energy', 0) / 10)
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            edge_sources = graph.edge_index[0].cpu().numpy()
            active_connections = np.sum(edge_sources == node_idx)
        else:
            active_connections = 0
        return f"{node_type}_{energy_level}_{active_connections}"
    def _is_new_activation_pattern(self, pattern_key: str) -> bool:
        if pattern_key not in self.activation_patterns:
            self.activation_patterns[pattern_key] = 1
            return True
        return False
    def _update_activation_patterns(self, current_patterns: set):
        for pattern in current_patterns:
            self.activation_patterns[pattern] += 1
    def _calculate_clustering_coefficient(self, graph) -> float:
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return 0.0
        num_nodes = len(graph.node_labels) if hasattr(graph, 'node_labels') else 1
        num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
        if num_nodes <= 1:
            return 0.0
        density = num_edges / (num_nodes * (num_nodes - 1))
        return min(density * 2.0, 1.0)
    def _calculate_average_path_length(self, graph) -> float:
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return 0.0
        num_nodes = len(graph.node_labels) if hasattr(graph, 'node_labels') else 1
        if num_nodes <= 1:
            return 0.0
        num_edges = graph.edge_index.shape[1]
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        if density > 0.5:
            return 1.5
        if density > 0.1:
            return 2.5
        return 4.0
    def _analyze_degree_distribution(self, graph) -> Dict[str, float]:
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return {'variance': 0.0, 'entropy': 0.0, 'max_degree': 0, 'min_degree': 0}
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
        if len(energy_values) == 0:
            return 0.0
        total_energy = np.sum(energy_values)
        if total_energy == 0:
            return 0.0
        probabilities = energy_values / total_energy
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) == 0:
            return 0.0
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    def _calculate_entropy(self, values: List[float]) -> float:
        if not values:
            return 0.0
        hist, _ = np.histogram(values, bins=min(20, len(values)))
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        probabilities = hist / np.sum(hist)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)
    def _calculate_skewness(self, values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        skewness = np.mean(((values - mean) / std) ** 3)
        return float(skewness)
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        if len(values) < 4:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3
        return float(kurtosis)
    def _check_energy_conservation(self, _graph, total_energy: float) -> Dict[str, any]:
        if self._initial_energy == 0.0:
            self._initial_energy = total_energy
        energy_drift = total_energy - self._initial_energy
        drift_percentage = (energy_drift / self._initial_energy) * 100 if self._initial_energy > 0 else 0
        conservation_violation = drift_percentage > 10.0
        return {
            'conservation_violation': conservation_violation,
            'energy_drift': energy_drift,
            'drift_percentage': drift_percentage,
            'conservation_status': 'stable' if not conservation_violation else 'unstable'
        }
    def _calculate_efficiency(self) -> float:
        if not self.calculation_times:
            return 1.0
        avg_time = np.mean(self.calculation_times)
        if avg_time == 0:
            return 1.0
        efficiency = max(0.0, 1.0 - (avg_time / 0.001))
        return float(efficiency)

    def _calculate_ei_ratio(self, graph) -> float:
        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return 0.0
        excitatory_sum = sum(edge.weight for edge in graph.edge_attributes if edge.type == 'excitatory')
        inhibitory_sum = sum(edge.weight for edge in graph.edge_attributes if edge.type == 'inhibitory')
        if inhibitory_sum == 0:
            return 10.0  # High ratio when no inhibitory
        return excitatory_sum / inhibitory_sum

    def _adjust_ei_balance(self, graph, _target_ratio: float):
        if not hasattr(graph, 'edge_attributes') or not graph.edge_attributes:
            return
        for edge in graph.edge_attributes:
            if edge.type == 'excitatory':
                edge.weight *= 0.9  # Decrease
            elif edge.type == 'inhibitory':
                edge.weight *= 1.1  # Increase

    def _adjust_criticality(self, graph, current_criticality: float):
        if current_criticality >= self.criticality_target:
            return
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            graph.edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        # Add a dummy edge
        new_edge = torch.tensor([[0], [1]], dtype=torch.long)
        graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)


def create_network_metrics(calculation_interval: int = 50) -> NetworkMetrics:
    """Create a new NetworkMetrics instance with specified calculation interval."""
    return NetworkMetrics(calculation_interval)


def quick_network_analysis(graph) -> Dict[str, any]:
    """Perform a quick analysis of the network metrics."""
    metrics = NetworkMetrics()
    return metrics.calculate_comprehensive_metrics(graph)







