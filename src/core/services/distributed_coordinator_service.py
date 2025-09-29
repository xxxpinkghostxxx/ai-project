"""
DistributedCoordinatorService implementation - Distributed coordination
for multi-node neural simulation.

This module provides the concrete implementation of IDistributedCoordinator,
handling coordination of neural simulation across multiple nodes with load balancing,
fault tolerance, and distributed processing capabilities.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import deque
from torch_geometric.data import Data

from ..interfaces.distributed_coordinator import (
    IDistributedCoordinator, NodeInfo, DistributedTask
)
from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.event_coordinator import IEventCoordinator


class DistributedCoordinatorService(IDistributedCoordinator):
    """
    Concrete implementation of IDistributedCoordinator.

    This service coordinates neural simulation activities across multiple nodes,
    providing load balancing, fault tolerance, and distributed processing
    while maintaining energy-based coordination.
    """

    def __init__(self,
                 configuration_service: IConfigurationService,
                 event_coordinator: IEventCoordinator):
        """
        Initialize the DistributedCoordinatorService.

        Args:
            configuration_service: Service for configuration management
            event_coordinator: Service for event publishing
        """
        self.configuration_service = configuration_service
        self.event_coordinator = event_coordinator

        # Distributed system state
        self._nodes: Dict[str, NodeInfo] = {}
        self._tasks: Dict[str, DistributedTask] = {}
        self._task_queue: deque = deque()
        self._completed_tasks: Dict[str, DistributedTask] = {}

        # Coordination settings
        self._heartbeat_interval = 5.0  # seconds
        self._task_timeout = 30.0  # seconds
        self._max_tasks_per_node = 10
        self._load_balance_threshold = 0.8

        # Monitoring
        self._is_coordinating = False
        self._coordination_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Task type routing
        self._task_type_routing = {
            "neural_processing": self._route_neural_task,
            "learning": self._route_learning_task,
            "sensory": self._route_sensory_task,
            "energy": self._route_energy_task
        }

    def initialize_distributed_system(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the distributed neural simulation system.

        Args:
            config: Configuration for distributed system setup

        Returns:
            bool: True if initialization successful
        """
        try:
            with self._lock:
                # Update configuration
                self._heartbeat_interval = config.get('heartbeat_interval', 5.0)
                self._task_timeout = config.get('task_timeout', 30.0)
                self._max_tasks_per_node = config.get('max_tasks_per_node', 10)
                self._load_balance_threshold = config.get('load_balance_threshold', 0.8)

                # Start coordination if not already running
                if not self._is_coordinating:
                    self._is_coordinating = True
                    self._coordination_thread = threading.Thread(
                        target=self._coordination_loop,
                        daemon=True,
                        name="DistributedCoordinator"
                    )
                    self._coordination_thread.start()

                # Publish initialization event
                self.event_coordinator.publish(
                    "distributed_system_initialized",
                    {
                        "node_count": len(self._nodes),
                        "config": config,
                        "timestamp": time.time()
                    }
                )

                return True

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error initializing distributed system: {e}")
            return False

    def register_node(self, node_info: NodeInfo) -> bool:
        """
        Register a new node in the distributed system.

        Args:
            node_info: Information about the node to register

        Returns:
            bool: True if registration successful
        """
        try:
            with self._lock:
                if node_info.node_id in self._nodes:
                    print(f"Node {node_info.node_id} already registered")
                    return False

                node_info.status = "active"
                node_info.last_heartbeat = time.time()
                self._nodes[node_info.node_id] = node_info

                # Publish node registration event
                self.event_coordinator.publish(
                    "node_registered",
                    {
                        "node_id": node_info.node_id,
                        "address": node_info.address,
                        "capabilities": node_info.capabilities,
                        "timestamp": time.time()
                    }
                )

                return True

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error registering node: {e}")
            return False

    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the distributed system.

        Args:
            node_id: ID of the node to unregister

        Returns:
            bool: True if unregistration successful
        """
        try:
            with self._lock:
                if node_id not in self._nodes:
                    return False

                # Handle any tasks assigned to this node
                self._handle_node_removal(node_id)

                # Remove the node
                del self._nodes[node_id]

                # Publish node unregistration event
                self.event_coordinator.publish(
                    "node_unregistered",
                    {
                        "node_id": node_id,
                        "timestamp": time.time()
                    }
                )

                return True

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error unregistering node: {e}")
            return False

    def submit_task(self, task: DistributedTask) -> bool:
        """
        Submit a task for distributed processing.

        Args:
            task: The task to submit

        Returns:
            bool: True if task submitted successfully
        """
        try:
            with self._lock:
                if task.task_id in self._tasks:
                    print(f"Task {task.task_id} already exists")
                    return False

                task.created_time = time.time()
                task.status = "pending"

                self._tasks[task.task_id] = task
                self._task_queue.append(task)

                # Publish task submission event
                self.event_coordinator.publish(
                    "task_submitted",
                    {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "priority": task.priority,
                        "timestamp": task.created_time
                    }
                )

                return True

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error submitting task: {e}")
            return False

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a completed task.

        Args:
            task_id: ID of the task

        Returns:
            Optional[Any]: Task result if available
        """
        with self._lock:
            if task_id in self._completed_tasks:
                task = self._completed_tasks[task_id]
                return task.result
            elif task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == "completed":
                    return task.result

            return None

    def balance_workload(self) -> Dict[str, Any]:
        """
        Balance workload across all active nodes.

        Returns:
            Dict[str, Any]: Workload balancing results
        """
        try:
            with self._lock:
                active_nodes = [node for node in self._nodes.values() if node.status == "active"]

                if not active_nodes:
                    return {"success": False, "error": "No active nodes available"}

                # Calculate current workloads
                total_workload = sum(node.workload for node in active_nodes)
                avg_workload = total_workload / len(active_nodes)

                # Identify overloaded and underloaded nodes
                overloaded = [
                    node for node in active_nodes
                    if node.workload > avg_workload * self._load_balance_threshold
                ]
                underloaded = [
                    node for node in active_nodes
                    if node.workload < avg_workload * (1 - self._load_balance_threshold)
                ]

                # Perform load balancing
                migrations = []
                for overloaded_node in overloaded:
                    for underloaded_node in underloaded:
                        # Migrate tasks from overloaded to underloaded nodes
                        tasks_to_migrate = self._get_tasks_for_node(overloaded_node.node_id)
                        if tasks_to_migrate:
                            task = tasks_to_migrate[0]  # Migrate first task
                            success = self.migrate_task(task.task_id, underloaded_node.node_id)
                            if success:
                                migrations.append({
                                    "task_id": task.task_id,
                                    "from_node": overloaded_node.node_id,
                                    "to_node": underloaded_node.node_id
                                })

                result = {
                    "success": True,
                    "total_nodes": len(active_nodes),
                    "average_workload": avg_workload,
                    "migrations_performed": len(migrations),
                    "migrations": migrations,
                    "timestamp": time.time()
                }

                # Publish workload balancing event
                self.event_coordinator.publish(
                    "workload_balanced",
                    result
                )

                return result

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error balancing workload: {e}")
            return {"success": False, "error": str(e)}

    def handle_node_failure(self, node_id: str) -> bool:
        """
        Handle failure of a distributed node.

        Args:
            node_id: ID of the failed node

        Returns:
            bool: True if failure handled successfully
        """
        try:
            with self._lock:
                if node_id not in self._nodes:
                    return False

                # Mark node as failed
                self._nodes[node_id].status = "failed"

                # Handle tasks assigned to failed node
                self._handle_node_removal(node_id)

                # Publish node failure event
                self.event_coordinator.publish(
                    "node_failed",
                    {
                        "node_id": node_id,
                        "timestamp": time.time()
                    }
                )

                return True

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error handling node failure: {e}")
            return False

    def synchronize_state(self, graph: Data) -> bool:
        """
        Synchronize neural graph state across all nodes.

        Args:
            graph: Current neural graph state

        Returns:
            bool: True if synchronization successful
        """
        try:
            with self._lock:
                active_nodes = [node for node in self._nodes.values() if node.status == "active"]

                if not active_nodes:
                    return False

                # Create synchronization tasks for each node
                sync_tasks = []
                for node in active_nodes:
                    sync_task = DistributedTask(
                        task_id=(
                            f"sync_{node.node_id}_{int(time.time())}"
                        ),
                        task_type="state_sync",
                        data={"graph": graph, "node_id": node.node_id},
                        priority=10  # High priority
                    )
                    sync_tasks.append(sync_task)

                # Submit synchronization tasks
                success_count = 0
                for task in sync_tasks:
                    if self.submit_task(task):
                        success_count += 1

                # Publish synchronization event
                self.event_coordinator.publish(
                    "state_synchronized",
                    {
                        "total_nodes": len(active_nodes),
                        "successful_syncs": success_count,
                        "timestamp": time.time()
                    }
                )

                return success_count == len(active_nodes)

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error synchronizing state: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the distributed system.

        Returns:
            Dict[str, Any]: System status information
        """
        with self._lock:
            active_nodes = [node for node in self._nodes.values() if node.status == "active"]
            failed_nodes = [node for node in self._nodes.values() if node.status == "failed"]

            pending_tasks = [task for task in self._tasks.values() if task.status == "pending"]
            running_tasks = [task for task in self._tasks.values() if task.status == "running"]
            completed_tasks = list(self._completed_tasks.values())

            return {
                "total_nodes": len(self._nodes),
                "active_nodes": len(active_nodes),
                "failed_nodes": len(failed_nodes),
                "pending_tasks": len(pending_tasks),
                "running_tasks": len(running_tasks),
                "completed_tasks": len(completed_tasks),
                "total_tasks": len(self._tasks),
                "is_coordinating": self._is_coordinating,
                "timestamp": time.time()
            }

    def optimize_energy_distribution(self) -> Dict[str, Any]:
        """
        Optimize energy distribution across distributed nodes.

        Returns:
            Dict[str, Any]: Energy optimization results
        """
        try:
            with self._lock:
                active_nodes = [node for node in self._nodes.values() if node.status == "active"]

                if not active_nodes:
                    return {"success": False, "error": "No active nodes available"}

                # Calculate energy distribution
                total_energy = sum(node.energy_level for node in active_nodes)
                avg_energy = total_energy / len(active_nodes)

                # Identify energy imbalances
                high_energy_nodes = [
                    node for node in active_nodes
                    if node.energy_level > avg_energy * 1.2
                ]
                low_energy_nodes = [
                    node for node in active_nodes
                    if node.energy_level < avg_energy * 0.8
                ]

                # Create energy balancing tasks
                energy_tasks = []
                for high_node in high_energy_nodes:
                    for low_node in low_energy_nodes:
                        energy_transfer = min(
                            (high_node.energy_level - avg_energy) * 0.5,
                            (avg_energy - low_node.energy_level) * 0.5
                        )

                        if energy_transfer > 0.01:  # Minimum transfer threshold
                            task = DistributedTask(
                                task_id=(
                                    f"energy_balance_{high_node.node_id}_"
                                    f"{low_node.node_id}_{int(time.time())}"
                                ),
                                task_type="energy_balance",
                                data={
                                    "from_node": high_node.node_id,
                                    "to_node": low_node.node_id,
                                    "energy_amount": energy_transfer
                                },
                                priority=5
                            )
                            energy_tasks.append(task)

                # Submit energy balancing tasks
                success_count = 0
                for task in energy_tasks:
                    if self.submit_task(task):
                        success_count += 1

                result = {
                    "success": True,
                    "total_nodes": len(active_nodes),
                    "average_energy": avg_energy,
                    "energy_tasks_created": len(energy_tasks),
                    "energy_tasks_submitted": success_count,
                    "timestamp": time.time()
                }

                # Publish energy optimization event
                self.event_coordinator.publish(
                    "energy_optimized",
                    result
                )

                return result

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error optimizing energy distribution: {e}")
            return {"success": False, "error": str(e)}

    def migrate_task(self, task_id: str, target_node_id: str) -> bool:
        """
        Migrate a running task to a different node.

        Args:
            task_id: ID of the task to migrate
            target_node_id: ID of the target node

        Returns:
            bool: True if migration successful
        """
        try:
            with self._lock:
                if task_id not in self._tasks:
                    return False

                if target_node_id not in self._nodes:
                    return False

                task = self._tasks[task_id]
                old_node = task.assigned_node

                # Update task assignment
                task.assigned_node = target_node_id

                # Update node workloads
                if old_node and old_node in self._nodes:
                    self._nodes[old_node].workload = max(0, self._nodes[old_node].workload - 1)

                if target_node_id in self._nodes:
                    self._nodes[target_node_id].workload += 1

                # Publish task migration event
                self.event_coordinator.publish(
                    "task_migrated",
                    {
                        "task_id": task_id,
                        "from_node": old_node,
                        "to_node": target_node_id,
                        "timestamp": time.time()
                    }
                )

                return True

        except (ValueError, RuntimeError, OSError) as e:
            print(f"Error migrating task: {e}")
            return False

    def _coordination_loop(self) -> None:
        """Main coordination loop for distributed system management."""
        while self._is_coordinating:
            try:
                # Process pending tasks
                self._process_pending_tasks()

                # Check for task timeouts
                self._check_task_timeouts()

                # Update node heartbeats
                self._update_node_heartbeats()

                # Perform periodic load balancing
                if len(self._nodes) > 1:
                    self.balance_workload()

                # Sleep for coordination interval
                time.sleep(self._heartbeat_interval)

            except (ValueError, RuntimeError, OSError) as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(1.0)

    def _process_pending_tasks(self) -> None:
        """Process pending tasks in the queue."""
        with self._lock:
            # Process up to 10 tasks per iteration
            for _ in range(min(10, len(self._task_queue))):
                if not self._task_queue:
                    break

                task = self._task_queue.popleft()

                # Find suitable node for task
                target_node = self._find_suitable_node(task)
                if target_node:
                    task.assigned_node = target_node.node_id
                    task.status = "running"
                    self._nodes[target_node.node_id].workload += 1

                    # Publish task assignment event
                    self.event_coordinator.publish(
                        "task_assigned",
                        {
                            "task_id": task.task_id,
                            "node_id": target_node.node_id,
                            "task_type": task.task_type,
                            "timestamp": time.time()
                        }
                    )
                else:
                    # No suitable node found, requeue task
                    self._task_queue.append(task)

    def _find_suitable_node(self, task: DistributedTask) -> Optional[NodeInfo]:
        """Find a suitable node for the given task."""
        active_nodes = [node for node in self._nodes.values()
                       if node.status == "active" and node.workload < self._max_tasks_per_node]

        if not active_nodes:
            return None

        # Use task-specific routing if available
        if task.task_type in self._task_type_routing:
            return self._task_type_routing[task.task_type](task, active_nodes)

        # Default routing: lowest workload
        return min(active_nodes, key=lambda node: node.workload)

    def _route_neural_task(
        self, _task: DistributedTask, nodes: List[NodeInfo]
    ) -> Optional[NodeInfo]:
        """Route neural processing tasks to nodes with neural capabilities."""
        neural_nodes = [node for node in nodes if "neural" in node.capabilities]
        if neural_nodes:
            return min(neural_nodes, key=lambda node: node.workload)
        return None

    def _route_learning_task(
        self, _task: DistributedTask, nodes: List[NodeInfo]
    ) -> Optional[NodeInfo]:
        """Route learning tasks to nodes with learning capabilities."""
        learning_nodes = [node for node in nodes if "learning" in node.capabilities]
        if learning_nodes:
            return min(learning_nodes, key=lambda node: node.workload)
        return None

    def _route_sensory_task(
        self, _task: DistributedTask, nodes: List[NodeInfo]
    ) -> Optional[NodeInfo]:
        """Route sensory tasks to nodes with sensory capabilities."""
        sensory_nodes = [node for node in nodes if "sensory" in node.capabilities]
        if sensory_nodes:
            return min(sensory_nodes, key=lambda node: node.workload)
        return None

    def _route_energy_task(
        self, _task: DistributedTask, nodes: List[NodeInfo]
    ) -> Optional[NodeInfo]:
        """Route energy tasks to nodes with highest energy levels."""
        if nodes:
            return max(nodes, key=lambda node: node.energy_level)
        return None

    def _check_task_timeouts(self) -> None:
        """Check for timed out tasks and handle them."""
        current_time = time.time()
        timeout_threshold = current_time - self._task_timeout

        with self._lock:
            timed_out_tasks = []
            for task in self._tasks.values():
                if task.status == "running" and task.created_time < timeout_threshold:
                    timed_out_tasks.append(task)

            for task in timed_out_tasks:
                task.status = "failed"
                if task.assigned_node and task.assigned_node in self._nodes:
                    self._nodes[task.assigned_node].workload = max(
                        0, self._nodes[task.assigned_node].workload - 1
                    )

                # Publish task timeout event
                self.event_coordinator.publish(
                    "task_timed_out",
                    {
                        "task_id": task.task_id,
                        "assigned_node": task.assigned_node,
                        "timeout_duration": self._task_timeout,
                        "timestamp": current_time
                    }
                )

    def _update_node_heartbeats(self) -> None:
        """Update node heartbeat status."""
        current_time = time.time()
        heartbeat_threshold = current_time - (self._heartbeat_interval * 3)

        with self._lock:
            for node in self._nodes.values():
                if node.last_heartbeat < heartbeat_threshold and node.status == "active":
                    # Node missed heartbeat, mark as inactive
                    node.status = "inactive"
                    self._handle_node_removal(node.node_id)

                    # Publish node heartbeat timeout event
                    self.event_coordinator.publish(
                        "node_heartbeat_timeout",
                        {
                            "node_id": node.node_id,
                            "last_heartbeat": node.last_heartbeat,
                            "timestamp": current_time
                        }
                    )

    def _handle_node_removal(self, node_id: str) -> None:
        """Handle removal of a node by reassigning its tasks."""
        # Find tasks assigned to this node
        assigned_tasks = [task for task in self._tasks.values()
                         if task.assigned_node == node_id and task.status == "running"]

        # Requeue tasks
        for task in assigned_tasks:
            task.assigned_node = None
            task.status = "pending"
            self._task_queue.appendleft(task)  # Add to front of queue

    def _get_tasks_for_node(self, node_id: str) -> List[DistributedTask]:
        """Get all tasks assigned to a specific node."""
        return [task for task in self._tasks.values()
                if task.assigned_node == node_id and task.status == "running"]

    def cleanup(self) -> None:
        """Clean up resources."""
        self._is_coordinating = False
        if self._coordination_thread and self._coordination_thread.is_alive():
            self._coordination_thread.join(timeout=2.0)

        with self._lock:
            self._nodes.clear()
            self._tasks.clear()
            self._task_queue.clear()
            self._completed_tasks.clear()
