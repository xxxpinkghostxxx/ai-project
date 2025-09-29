"""
LoadBalancingService implementation - Load balancing for distributed neural simulation.

This module provides the concrete implementation of ILoadBalancer,
handling workload distribution and optimization across distributed nodes.
"""

import time
import statistics
from typing import Dict, Any, List
from collections import defaultdict, deque

from ..interfaces.load_balancer import ILoadBalancer, LoadMetrics
from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.event_coordinator import IEventCoordinator
from ..interfaces.distributed_coordinator import IDistributedCoordinator


class LoadBalancingService(ILoadBalancer):
    """
    Concrete implementation of ILoadBalancer.

    This service provides intelligent load balancing across distributed nodes,
    optimizing resource utilization and maintaining performance in multi-node setups.
    """

    def __init__(self,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator,
                 distributed_coordinator: IDistributedCoordinator):
        """
        Initialize the LoadBalancingService.

        Args:
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
            distributed_coordinator: Service for distributed coordination
        """
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator
        self.distributed_coordinator = distributed_coordinator

        # Load balancing state
        self._node_metrics: Dict[str, LoadMetrics] = {}
        self._historical_load: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._load_predictions: Dict[str, List[float]] = defaultdict(list)

        # Configuration
        self._rebalance_interval = 30.0  # seconds
        self._prediction_window = 60  # seconds
        self._load_imbalance_threshold = 0.8
        self._max_tasks_per_node = 10

        # Statistics
        self._rebalance_count = 0
        self._task_migrations = 0
        self._last_rebalance_time = 0

    def assess_node_load(self, node_id: str) -> LoadMetrics:
        """
        Assess the current load of a specific node.

        Args:
            node_id: ID of the node to assess

        Returns:
            LoadMetrics: Current load metrics for the node
        """
        metrics = LoadMetrics()

        try:
            # Get system status from distributed coordinator
            system_status = self.distributed_coordinator.get_system_status()

            # Calculate load metrics based on available data
            if node_id in system_status.get('node_workloads', {}):
                metrics.task_count = system_status['node_workloads'][node_id]
            else:
                # Estimate based on total tasks and nodes
                total_tasks = system_status.get('running_tasks', 0)
                total_nodes = system_status.get('active_nodes', 1)
                metrics.task_count = total_tasks // total_nodes

            # Normalize task count to load factor
            metrics.cpu_usage = min(1.0, metrics.task_count / self._max_tasks_per_node)
            metrics.memory_usage = metrics.cpu_usage * 0.8  # Estimate memory usage
            metrics.energy_level = 1.0 - metrics.cpu_usage * 0.3  # Energy decreases with load
            metrics.network_latency = 0.01 + metrics.cpu_usage * 0.05  # Latency increases with load
            metrics.task_completion_rate = 1.0 - metrics.cpu_usage * 0.2  # Completion rate decreases with load

            # Store historical data
            self._node_metrics[node_id] = metrics
            self._historical_load[node_id].append(metrics.cpu_usage)

            return metrics

        except Exception as e:
            print(f"Error assessing node load: {e}")
            return metrics

    def calculate_optimal_distribution(self, tasks: List[Dict[str, Any]],
                                     nodes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate optimal task distribution across nodes.

        Args:
            tasks: List of tasks to distribute
            nodes: List of available node IDs

        Returns:
            Dict[str, List[Dict[str, Any]]]: Task distribution mapping
        """
        try:
            distribution = {node_id: [] for node_id in nodes}

            if not tasks or not nodes:
                return distribution

            # Sort tasks by priority (highest first)
            sorted_tasks = sorted(tasks, key=lambda t: t.get('priority', 1), reverse=True)

            # Assess current node loads
            node_loads = {}
            for node_id in nodes:
                metrics = self.assess_node_load(node_id)
                node_loads[node_id] = metrics.cpu_usage

            # Distribute tasks using weighted round-robin with load balancing
            for task in sorted_tasks:
                # Find node with lowest current load
                best_node = min(node_loads.keys(), key=lambda n: node_loads[n])

                # Check if node can accept more tasks
                if node_loads[best_node] < self._load_imbalance_threshold:
                    distribution[best_node].append(task)
                    # Update load estimate
                    node_loads[best_node] += 0.1  # Estimate load increase per task
                else:
                    # All nodes are heavily loaded, distribute to least loaded
                    least_loaded = min(node_loads.keys(), key=lambda n: node_loads[n])
                    distribution[least_loaded].append(task)
                    node_loads[least_loaded] += 0.1

            return distribution

        except Exception as e:
            print(f"Error calculating optimal distribution: {e}")
            # Fallback: distribute evenly
            return self._distribute_evenly(tasks, nodes)

    def rebalance_workload(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Rebalance workload across all nodes when imbalance exceeds threshold.

        Args:
            threshold: Load imbalance threshold (0.0-1.0)

        Returns:
            Dict[str, Any]: Rebalancing results
        """
        try:
            current_time = time.time()

            # Check if enough time has passed since last rebalance
            if current_time - self._last_rebalance_time < self._rebalance_interval:
                return {"success": False, "reason": "Too soon since last rebalance"}

            # Get system status
            system_status = self.distributed_coordinator.get_system_status()
            active_nodes = system_status.get('active_nodes', 0)

            if active_nodes < 2:
                return {"success": False, "reason": "Insufficient nodes for rebalancing"}

            # Assess current loads
            node_loads = {}
            for node_id in list(system_status.get('node_workloads', {}).keys()):
                metrics = self.assess_node_load(node_id)
                node_loads[node_id] = metrics.cpu_usage

            if not node_loads:
                return {"success": False, "reason": "No load data available"}

            # Calculate load statistics
            loads = list(node_loads.values())
            avg_load = statistics.mean(loads)
            max_load = max(loads)
            min_load = min(loads)

            # Check if rebalancing is needed
            load_imbalance = max_load - min_load
            if load_imbalance < threshold:
                return {
                    "success": True,
                    "rebalanced": False,
                    "reason": "Load imbalance below threshold",
                    "current_imbalance": load_imbalance,
                    "threshold": threshold
                }

            # Identify overloaded and underloaded nodes
            overloaded = [node for node, load in node_loads.items() if load > avg_load + threshold * 0.5]
            underloaded = [node for node, load in node_loads.items() if load < avg_load - threshold * 0.5]

            # Perform rebalancing using distributed coordinator
            rebalance_result = self.distributed_coordinator.balance_workload()

            self._rebalance_count += 1
            self._last_rebalance_time = current_time

            # Publish rebalance event
            self.event_coordinator.publish("load_rebalanced", {
                "rebalance_count": self._rebalance_count,
                "imbalance_before": load_imbalance,
                "overloaded_nodes": len(overloaded),
                "underloaded_nodes": len(underloaded),
                "timestamp": current_time
            })

            return {
                "success": True,
                "rebalanced": True,
                "imbalance_before": load_imbalance,
                "imbalance_after": rebalance_result.get('average_workload', avg_load),
                "migrations": rebalance_result.get('migrations_performed', 0),
                "timestamp": current_time
            }

        except Exception as e:
            print(f"Error rebalancing workload: {e}")
            return {"success": False, "error": str(e)}

    def predict_load_changes(self, time_window: int = 60) -> Dict[str, Any]:
        """
        Predict future load changes based on historical data.

        Args:
            time_window: Prediction time window in seconds

        Returns:
            Dict[str, Any]: Load prediction results
        """
        try:
            predictions = {}

            for node_id, historical_load in self._historical_load.items():
                if len(historical_load) < 3:
                    # Not enough data for prediction
                    predictions[node_id] = {
                        "predicted_load": 0.5,
                        "confidence": 0.0,
                        "trend": "unknown"
                    }
                    continue

                # Simple linear regression for prediction
                recent_loads = list(historical_load)[-10:]  # Use last 10 measurements
                n = len(recent_loads)

                if n < 2:
                    predictions[node_id] = {
                        "predicted_load": recent_loads[-1] if recent_loads else 0.5,
                        "confidence": 0.5,
                        "trend": "stable"
                    }
                    continue

                # Calculate trend
                x = list(range(n))
                y = recent_loads

                # Simple slope calculation
                slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0

                # Predict next value
                predicted_load = max(0.0, min(1.0, y[-1] + slope))

                # Determine trend
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"

                # Calculate confidence based on data consistency
                load_std = statistics.stdev(y) if len(y) > 1 else 0
                confidence = max(0.0, 1.0 - load_std)

                predictions[node_id] = {
                    "predicted_load": predicted_load,
                    "confidence": confidence,
                    "trend": trend,
                    "historical_points": n
                }

            return {
                "predictions": predictions,
                "time_window": time_window,
                "prediction_method": "linear_regression",
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"Error predicting load changes: {e}")
            return {"error": str(e)}

    def get_load_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive load balancing statistics.

        Returns:
            Dict[str, Any]: Load balancing statistics
        """
        try:
            # Get current system status
            system_status = self.distributed_coordinator.get_system_status()

            # Calculate load statistics
            node_loads = {}
            for node_id in list(system_status.get('node_workloads', {}).keys()):
                metrics = self.assess_node_load(node_id)
                node_loads[node_id] = metrics.cpu_usage

            if node_loads:
                loads = list(node_loads.values())
                stats = {
                    "total_nodes": len(node_loads),
                    "average_load": statistics.mean(loads),
                    "max_load": max(loads),
                    "min_load": min(loads),
                    "load_std_dev": statistics.stdev(loads) if len(loads) > 1 else 0,
                    "load_range": max(loads) - min(loads)
                }
            else:
                stats = {
                    "total_nodes": 0,
                    "average_load": 0.0,
                    "max_load": 0.0,
                    "min_load": 0.0,
                    "load_std_dev": 0.0,
                    "load_range": 0.0
                }

            return {
                "load_statistics": stats,
                "rebalance_count": self._rebalance_count,
                "task_migrations": self._task_migrations,
                "last_rebalance_time": self._last_rebalance_time,
                "node_loads": node_loads,
                "historical_data_points": {node: len(history) for node, history in self._historical_load.items()},
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"Error getting load statistics: {e}")
            return {"error": str(e)}

    def _distribute_evenly(self, tasks: List[Dict[str, Any]], nodes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fallback method to distribute tasks evenly across nodes.

        Args:
            tasks: List of tasks to distribute
            nodes: List of available node IDs

        Returns:
            Dict[str, List[Dict[str, Any]]]: Even task distribution
        """
        distribution = {node_id: [] for node_id in nodes}

        if not tasks or not nodes:
            return distribution

        for i, task in enumerate(tasks):
            node_index = i % len(nodes)
            distribution[nodes[node_index]].append(task)

        return distribution

    def cleanup(self) -> None:
        """Clean up resources."""
        self._node_metrics.clear()
        self._historical_load.clear()
        self._load_predictions.clear()






