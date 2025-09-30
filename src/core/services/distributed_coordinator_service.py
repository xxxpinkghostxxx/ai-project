"""
DistributedCoordinatorService implementation - Distributed coordination
for multi-node neural simulation.

This module provides the concrete implementation of IDistributedCoordinator,
handling coordination of neural simulation across multiple nodes with load balancing,
fault tolerance, and distributed processing capabilities.
"""

import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from torch_geometric.data import Data

from ..interfaces.configuration_service import IConfigurationService
from ..interfaces.distributed_coordinator import (DistributedTask,
                                                  IDistributedCoordinator,
                                                  NodeInfo)
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

    # Public accessor methods for private attributes
    def get_nodes(self) -> Dict[str, NodeInfo]:
        """Get all registered nodes."""
        return self._nodes

    def get_tasks(self) -> Dict[str, DistributedTask]:
        """Get all tasks."""
        return self._tasks

    def get_task_queue(self) -> deque:
        """Get the task queue."""
        return self._task_queue

    def get_completed_tasks(self) -> Dict[str, DistributedTask]:
        """Get completed tasks."""
        return self._completed_tasks

    def get_heartbeat_interval(self) -> float:
        """Get heartbeat interval."""
        return self._heartbeat_interval

    def set_heartbeat_interval(self, interval: float) -> None:
        """Set heartbeat interval."""
        self._heartbeat_interval = interval

    def get_task_timeout(self) -> float:
        """Get task timeout."""
        return self._task_timeout

    def set_task_timeout(self, timeout: float) -> None:
        """Set task timeout."""
        self._task_timeout = timeout

    def get_max_tasks_per_node(self) -> int:
        """Get maximum tasks per node."""
        return self._max_tasks_per_node

    def set_max_tasks_per_node(self, max_tasks: int) -> None:
        """Set maximum tasks per node."""
        self._max_tasks_per_node = max_tasks

    def get_load_balance_threshold(self) -> float:
        """Get load balance threshold."""
        return self._load_balance_threshold

    def set_load_balance_threshold(self, threshold: float) -> None:
        """Set load balance threshold."""
        self._load_balance_threshold = threshold

    def is_coordinating(self) -> bool:
        """Check if coordination is active."""
        return self._is_coordinating

    def set_coordinating(self, coordinating: bool) -> None:
        """Set coordination status."""
        self._is_coordinating = coordinating

    def get_coordination_thread(self) -> Optional[threading.Thread]:
        """Get coordination thread."""
        return self._coordination_thread

    def set_coordination_thread(self, thread: Optional[threading.Thread]) -> None:
        """Set coordination thread."""
        self._coordination_thread = thread

    def get_lock(self) -> threading.RLock:
        """Get the coordination lock."""
        return self._lock

    def get_task_type_routing(self) -> Dict[str, callable]:
        """Get task type routing configuration."""
        return self._task_type_routing

    def initialize_distributed_system(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the distributed neural simulation system.

        Args:
            config: Configuration for distributed system setup

        Returns:
            bool: True if initialization successful
        """
        try:
            with self.get_lock():
                # Update configuration
                self.set_heartbeat_interval(config.get('heartbeat_interval', 5.0))
                self.set_task_timeout(config.get('task_timeout', 30.0))
                self.set_max_tasks_per_node(config.get('max_tasks_per_node', 10))
                self.set_load_balance_threshold(config.get('load_balance_threshold', 0.8))

                # Start coordination if not already running
                if not self.is_coordinating():
                    self.set_coordinating(True)
                    coordination_thread = threading.Thread(
                        target=self._coordination_loop,
                        daemon=True,
                        name="DistributedCoordinator"
                    )
                    self.set_coordination_thread(coordination_thread)
                    coordination_thread.start()

                # Publish initialization event
                self.event_coordinator.publish(
                    "distributed_system_initialized",
                    {
                        "node_count": len(self.get_nodes()),
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
            with self.get_lock():
                nodes = self.get_nodes()
                if node_info.node_id in nodes:
                    print(f"Node {node_info.node_id} already registered")
                    return False

                node_info.status = "active"
                node_info.last_heartbeat = time.time()
                nodes[node_info.node_id] = node_info

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
            with self.get_lock():
                nodes = self.get_nodes()
                if node_id not in nodes:
                    return False

                # Handle any tasks assigned to this node
                self._handle_node_removal(node_id)

                # Remove the node
                del nodes[node_id]

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
            with self.get_lock():
                tasks = self.get_tasks()
                task_queue = self.get_task_queue()
                if task.task_id in tasks:
                    print(f"Task {task.task_id} already exists")
                    return False

                task.created_time = time.time()
                task.status = "pending"

                tasks[task.task_id] = task
                task_queue.append(task)

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
        with self.get_lock():
            completed_tasks = self.get_completed_tasks()
            tasks = self.get_tasks()
            if task_id in completed_tasks:
                task = completed_tasks[task_id]
                return task.result
            if task_id in tasks:
                task = tasks[task_id]
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
            with self.get_lock():
                nodes = self.get_nodes()
                load_balance_threshold = self.get_load_balance_threshold()
                active_nodes = [node for node in nodes.values() if node.status == "active"]

                if not active_nodes:
                    return {"success": False, "error": "No active nodes available"}

                # Calculate current workloads
                total_workload = sum(node.workload for node in active_nodes)
                avg_workload = total_workload / len(active_nodes)

                # Identify overloaded and underloaded nodes
                overloaded = [
                    node for node in active_nodes
                    if node.workload > avg_workload * load_balance_threshold
                ]
                underloaded = [
                    node for node in active_nodes
                    if node.workload < avg_workload * (1 - load_balance_threshold)
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
            with self.get_lock():
                nodes = self.get_nodes()
                if node_id not in nodes:
                    return False

                # Mark node as failed
                nodes[node_id].status = "failed"

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
            with self.get_lock():
                nodes = self.get_nodes()
                active_nodes = [node for node in nodes.values() if node.status == "active"]

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
        with self.get_lock():
            nodes = self.get_nodes()
            tasks = self.get_tasks()
            completed_tasks = self.get_completed_tasks()
            active_nodes = [node for node in nodes.values() if node.status == "active"]
            failed_nodes = [node for node in nodes.values() if node.status == "failed"]

            pending_tasks = [task for task in tasks.values() if task.status == "pending"]
            running_tasks = [task for task in tasks.values() if task.status == "running"]
            completed_tasks_list = list(completed_tasks.values())

            return {
                "total_nodes": len(nodes),
                "active_nodes": len(active_nodes),
                "failed_nodes": len(failed_nodes),
                "pending_tasks": len(pending_tasks),
                "running_tasks": len(running_tasks),
                "completed_tasks": len(completed_tasks_list),
                "total_tasks": len(tasks),
                "is_coordinating": self.is_coordinating(),
                "timestamp": time.time()
            }

    def optimize_energy_distribution(self) -> Dict[str, Any]:
        """
        Optimize energy distribution across distributed nodes.

        Returns:
            Dict[str, Any]: Energy optimization results
        """
        try:
            with self.get_lock():
                nodes = self.get_nodes()
                active_nodes = [node for node in nodes.values() if node.status == "active"]

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
            with self.get_lock():
                tasks = self.get_tasks()
                nodes = self.get_nodes()
                if task_id not in tasks:
                    return False

                if target_node_id not in nodes:
                    return False

                task = tasks[task_id]
                old_node = task.assigned_node

                # Update task assignment
                task.assigned_node = target_node_id

                # Update node workloads
                if old_node and old_node in nodes:
                    nodes[old_node].workload = max(0, nodes[old_node].workload - 1)

                if target_node_id in nodes:
                    nodes[target_node_id].workload += 1

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
        while self.is_coordinating():
            try:
                # Process pending tasks
                self._process_pending_tasks()

                # Check for task timeouts
                self._check_task_timeouts()

                # Update node heartbeats
                self._update_node_heartbeats()

                # Perform periodic load balancing
                nodes = self.get_nodes()
                if len(nodes) > 1:
                    self.balance_workload()

                # Sleep for coordination interval
                heartbeat_interval = self.get_heartbeat_interval()
                time.sleep(heartbeat_interval)

            except (ValueError, RuntimeError, OSError) as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(1.0)

    def _process_pending_tasks(self) -> None:
        """Process pending tasks in the queue."""
        with self.get_lock():
            task_queue = self.get_task_queue()
            nodes = self.get_nodes()
            # Process up to 10 tasks per iteration
            for _ in range(min(10, len(task_queue))):
                if not task_queue:
                    break

                task = task_queue.popleft()

                # Find suitable node for task
                target_node = self._find_suitable_node(task)
                if target_node:
                    task.assigned_node = target_node.node_id
                    task.status = "running"
                    nodes[target_node.node_id].workload += 1

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
                    task_queue.append(task)

    def _find_suitable_node(self, task: DistributedTask) -> Optional[NodeInfo]:
        """Find a suitable node for the given task."""
        nodes = self.get_nodes()
        max_tasks_per_node = self.get_max_tasks_per_node()
        task_type_routing = self.get_task_type_routing()
        active_nodes = [node for node in nodes.values()
                       if node.status == "active" and node.workload < max_tasks_per_node]

        if not active_nodes:
            return None

        # Use task-specific routing if available
        if task.task_type in task_type_routing:
            return task_type_routing[task.task_type](task, active_nodes)

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
        task_timeout = self.get_task_timeout()
        timeout_threshold = current_time - task_timeout

        with self.get_lock():
            tasks = self.get_tasks()
            nodes = self.get_nodes()
            timed_out_tasks = []
            for task in tasks.values():
                if task.status == "running" and task.created_time < timeout_threshold:
                    timed_out_tasks.append(task)

            for task in timed_out_tasks:
                task.status = "failed"
                if task.assigned_node and task.assigned_node in nodes:
                    nodes[task.assigned_node].workload = max(
                        0, nodes[task.assigned_node].workload - 1
                    )

                # Publish task timeout event
                self.event_coordinator.publish(
                    "task_timed_out",
                    {
                        "task_id": task.task_id,
                        "assigned_node": task.assigned_node,
                        "timeout_duration": task_timeout,
                        "timestamp": current_time
                    }
                )

    def _update_node_heartbeats(self) -> None:
        """Update node heartbeat status."""
        current_time = time.time()
        heartbeat_interval = self.get_heartbeat_interval()
        heartbeat_threshold = current_time - (heartbeat_interval * 3)

        with self.get_lock():
            nodes = self.get_nodes()
            for node in nodes.values():
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
        tasks = self.get_tasks()
        task_queue = self.get_task_queue()
        # Find tasks assigned to this node
        assigned_tasks = [task for task in tasks.values()
                         if task.assigned_node == node_id and task.status == "running"]

        # Requeue tasks
        for task in assigned_tasks:
            task.assigned_node = None
            task.status = "pending"
            task_queue.appendleft(task)  # Add to front of queue

    def _get_tasks_for_node(self, node_id: str) -> List[DistributedTask]:
        """Get all tasks assigned to a specific node."""
        tasks = self.get_tasks()
        return [task for task in tasks.values()
                if task.assigned_node == node_id and task.status == "running"]

    def cleanup(self) -> None:
        """Clean up resources."""
        self.set_coordinating(False)
        coordination_thread = self.get_coordination_thread()
        if coordination_thread and coordination_thread.is_alive():
            coordination_thread.join(timeout=2.0)

        with self.get_lock():
            self.get_nodes().clear()
            self.get_tasks().clear()
            self.get_task_queue().clear()
            self.get_completed_tasks().clear()
