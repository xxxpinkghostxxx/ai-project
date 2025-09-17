
import time
import numpy as np
import torch
from logging_utils import log_step, log_node_state
from config_manager import get_homeostasis_config


class HomeostasisController:

    def __init__(self):
        config = get_homeostasis_config()
        self.target_energy_ratio = config.get('target_energy_ratio', 0.6)
        self.criticality_threshold = config.get('criticality_threshold', 0.1)
        self.regulation_rate = config.get('regulation_rate', 0.001)
        self.regulation_interval = config.get('regulation_interval', 100)
        self.branching_target = 1.0
        self.energy_variance_threshold = 0.2
        self.regulation_stats = {
            'energy_regulations': 0,
            'criticality_regulations': 0,
            'threshold_adjustments': 0,
            'total_regulation_events': 0,
            'last_regulation_time': time.time()
        }
        self.energy_history = []
        self.branching_history = []
        self.variance_history = []
        self.max_history_length = 1000
    def regulate_network_activity(self, graph):

        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
            return graph
        if hasattr(graph, 'network_metrics') and graph.network_metrics:
            try:
                metrics = graph.network_metrics.calculate_comprehensive_metrics(graph)
                energy_metrics = metrics.get('energy_balance', {})
                connectivity_metrics = metrics.get('connectivity', {})
                total_energy = energy_metrics.get('total_energy', 0.0)
                avg_energy = energy_metrics.get('average_energy', 0.0)
                energy_variance = energy_metrics.get('energy_variance', 0.0)
                energy_ratio = energy_metrics.get('energy_ratio', 0.0)
                connectivity_density = connectivity_metrics.get('density', 0.0)
                num_nodes = len(graph.node_labels)
                log_step("Metrics-based energy regulation",
                        total_energy=total_energy,
                        avg_energy=avg_energy,
                        energy_variance=energy_variance,
                        connectivity_density=connectivity_density)
            except Exception as e:
                log_step("Metrics calculation failed, using fallback", error=str(e))
                total_energy = float(torch.sum(graph.x[:, 0]).item()) if hasattr(graph, 'x') else 0.0
                num_nodes = len(graph.node_labels)
                avg_energy = total_energy / num_nodes if num_nodes > 0 else 0.0
                energy_ratio = avg_energy / 244.0 if total_energy > 0 else 0.0
                energy_variance = float(torch.var(graph.x[:, 0]).item()) if hasattr(graph, 'x') and len(graph.x) > 1 else 0.0
        else:
            total_energy = float(torch.sum(graph.x[:, 0]).item()) if hasattr(graph, 'x') else 0.0
            num_nodes = len(graph.node_labels)
            avg_energy = total_energy / num_nodes if num_nodes > 0 else 0.0
            energy_ratio = avg_energy / 244.0 if total_energy > 0 else 0.0
            energy_variance = float(torch.var(graph.x[:, 0]).item()) if hasattr(graph, 'x') and len(graph.x) > 1 else 0.0
        self._update_history('energy', energy_ratio)
        self._update_history('variance', energy_variance)
        regulation_needed = False
        regulation_type = None
        if energy_ratio > self.target_energy_ratio + self.criticality_threshold:
            regulation_needed = True
            regulation_type = 'reduce_energy'
            log_step("Energy regulation needed",
                    action="reduce_energy",
                    current_ratio=energy_ratio,
                    target_ratio=self.target_energy_ratio)
        elif energy_ratio < self.target_energy_ratio - self.criticality_threshold:
            regulation_needed = True
            regulation_type = 'increase_energy'
            log_step("Energy regulation needed",
                    action="increase_energy",
                    current_ratio=energy_ratio,
                    target_ratio=self.target_energy_ratio)
        if regulation_needed:
            self._apply_energy_regulation(graph, regulation_type, energy_ratio)
            self.regulation_stats['energy_regulations'] += 1
            self.regulation_stats['total_regulation_events'] += 1
        return graph
    def _apply_energy_regulation(self, graph, regulation_type, current_ratio):

        from death_and_birth_logic import (
            get_node_birth_threshold, get_node_death_threshold, get_node_birth_cost
        )
        if regulation_type == 'reduce_energy':
            current_death_threshold = get_node_death_threshold()
            current_birth_threshold = get_node_birth_threshold()
            new_death_threshold = current_death_threshold + (self.regulation_rate * 50.0)
            new_death_threshold = min(new_death_threshold, 50.0)
            new_birth_threshold = current_birth_threshold - (self.regulation_rate * 50.0)
            new_birth_threshold = max(new_birth_threshold, 50.0)
            log_step("Energy reduction regulation applied",
                    old_death_threshold=current_death_threshold,
                    new_death_threshold=new_death_threshold,
                    old_birth_threshold=current_birth_threshold,
                    new_birth_threshold=new_birth_threshold)
        elif regulation_type == 'increase_energy':
            current_death_threshold = get_node_death_threshold()
            current_birth_threshold = get_node_birth_threshold()
            new_death_threshold = current_death_threshold - (self.regulation_rate * 10.0)
            new_death_threshold = max(new_death_threshold, 0.0)
            new_birth_threshold = current_birth_threshold + (self.regulation_rate * 50.0)
            new_birth_threshold = min(new_birth_threshold, 300.0)
            log_step("Energy increase regulation applied",
                    old_death_threshold=current_death_threshold,
                    new_death_threshold=new_death_threshold,
                    old_birth_threshold=current_birth_threshold,
                    new_birth_threshold=new_birth_threshold)
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

        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'edge_index'):
            return graph
        if hasattr(graph, 'network_metrics') and graph.network_metrics:
            try:
                metrics = graph.network_metrics.calculate_comprehensive_metrics(graph)
                branching_ratio = metrics.get('criticality', 0.0)
                connectivity_metrics = metrics.get('connectivity', {})
                energy_metrics = metrics.get('energy_balance', {})
                log_step("Metrics-based criticality optimization",
                        branching_ratio=branching_ratio,
                        connectivity_density=connectivity_metrics.get('density', 0.0),
                        energy_variance=energy_metrics.get('energy_variance', 0.0))
            except Exception as e:
                log_step("Metrics calculation failed, using fallback", error=str(e))
                branching_ratio = self._calculate_branching_ratio(graph)
        else:
            branching_ratio = self._calculate_branching_ratio(graph)
        self._update_history('branching', branching_ratio)
        criticality_deviation = abs(branching_ratio - self.branching_target)
        if criticality_deviation > self.criticality_threshold:
            regulation_needed = True
            regulation_type = 'supercritical' if branching_ratio > self.branching_target else 'subcritical'
            log_step("Criticality regulation needed",
                    current_ratio=branching_ratio,
                    target_ratio=self.branching_target,
                    deviation=criticality_deviation,
                    regulation_type=regulation_type)
            self._apply_criticality_regulation(graph, regulation_type, branching_ratio)
            self.regulation_stats['criticality_regulations'] += 1
            self.regulation_stats['total_regulation_events'] += 1
        return graph
    def _calculate_branching_ratio(self, graph):

        if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
            return 0.0
        current_time = time.time()
        total_activations = 0
        new_activations = 0
        for node in graph.node_labels:
            last_activation = node.get('last_activation', 0)
            if last_activation > 0:
                total_activations += 1
                if current_time - last_activation < 5.0:
                    new_activations += 1
        if total_activations > 0:
            branching_ratio = new_activations / total_activations
            return branching_ratio
        return 0.0
    def _apply_criticality_regulation(self, graph, regulation_type, current_ratio):

        if regulation_type == 'supercritical':
            regulation_action = 'reduce_excitation'
            log_step("Supercritical regulation applied",
                    action="reduce_excitation",
                    current_ratio=current_ratio)
        elif regulation_type == 'subcritical':
            regulation_action = 'increase_excitation'
            log_step("Subcritical regulation applied",
                    action="increase_excitation",
                    current_ratio=current_ratio)
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

        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
            return {'status': 'unknown', 'warnings': []}
        warnings = []
        health_score = 1.0
        total_energy = float(torch.sum(graph.x[:, 0]).item())
        num_nodes = len(graph.node_labels)
        avg_energy = total_energy / num_nodes if num_nodes > 0 else 0
        if avg_energy < 10.0:
            warnings.append("Low average energy - network may be dying")
            health_score *= 0.5
        elif avg_energy > 200.0:
            warnings.append("High average energy - network may be unstable")
            health_score *= 0.7
        if hasattr(graph, 'homeostasis_data'):
            last_regulation = graph.homeostasis_data.get('last_regulation', {})
            if last_regulation:
                regulation_age = time.time() - last_regulation.get('timestamp', 0)
                if regulation_age > 300:
                    warnings.append("No recent energy regulation - system may be stuck")
                    health_score *= 0.8
        if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
            connection_density = graph.edge_index.shape[1] / (num_nodes * num_nodes)
            if connection_density < 0.001:
                warnings.append("Very low connection density - network may be fragmented")
                health_score *= 0.6
            elif connection_density > 0.1:
                warnings.append("Very high connection density - network may be overconnected")
                health_score *= 0.8
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
        return self.regulation_stats.copy()
    def reset_statistics(self):
        self.regulation_stats = {
            'energy_regulations': 0,
            'criticality_regulations': 0,
            'threshold_adjustments': 0,
            'total_regulation_events': 0,
            'last_regulation_time': time.time()
        }
    def get_network_trends(self):

        trends = {}
        if len(self.energy_history) > 10:
            recent_energy = self.energy_history[-10:]
            energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
            trends['energy_trend'] = 'increasing' if energy_trend > 0.001 else 'decreasing' if energy_trend < -0.001 else 'stable'
            trends['energy_slope'] = energy_trend
        if len(self.branching_history) > 10:
            recent_branching = self.branching_history[-10:]
            branching_trend = np.polyfit(range(len(recent_branching)), recent_branching, 1)[0]
            trends['branching_trend'] = 'increasing' if branching_trend > 0.01 else 'decreasing' if branching_trend < -0.01 else 'stable'
            trends['branching_slope'] = branching_trend
        if len(self.variance_history) > 10:
            recent_variance = self.variance_history[-10:]
            variance_trend = np.polyfit(range(len(recent_variance)), recent_variance, 1)[0]
            trends['variance_trend'] = 'increasing' if variance_trend > 0.1 else 'decreasing' if variance_trend < -0.1 else 'stable'
            trends['variance_slope'] = variance_trend
        return trends


def calculate_network_stability(graph):

    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return 0.0
    energy_values = graph.x[:, 0].cpu().numpy()
    energy_variance = np.var(energy_values) if len(energy_values) > 1 else 0
    energy_stability = 1.0 / (1.0 + energy_variance / 100.0)
    if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
        num_connections = graph.edge_index.shape[1]
        num_nodes = len(graph.node_labels)
        connection_stability = min(num_connections / (num_nodes * 2), 1.0)
    else:
        connection_stability = 0.0
    stability = (energy_stability * 0.6 + connection_stability * 0.4)
    return stability


def detect_network_anomalies(graph):

    anomalies = []
    if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
        return anomalies
    energy_values = graph.x[:, 0].cpu().numpy()
    if len(energy_values) > 0:
        mean_energy = np.mean(energy_values)
        std_energy = np.std(energy_values)
        extreme_high = np.sum(energy_values > mean_energy + 3 * std_energy)
        extreme_low = np.sum(energy_values < mean_energy - 3 * std_energy)
        if extreme_high > 0:
            anomalies.append(f"High energy outliers: {extreme_high} nodes")
        if extreme_low > 0:
            anomalies.append(f"Low energy outliers: {extreme_low} nodes")
    if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
        num_connections = graph.edge_index.shape[1]
        num_nodes = len(graph.node_labels)
        if num_nodes > 0:
            connection_density = num_connections / (num_nodes * num_nodes)
            if connection_density < 0.0001:
                anomalies.append("Extremely low connection density")
            elif connection_density > 0.5:
                anomalies.append("Extremely high connection density")
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
