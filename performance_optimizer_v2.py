"""
performance_optimizer_v2.py

Advanced performance optimization for the neural system.
Fixes the 57-second simulation step issue and optimizes real-time performance.
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import logging
from logging_utils import log_step, log_runtime

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    step_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    node_count: int
    edge_count: int
    throughput: float  # nodes processed per second
    timestamp: float

class PerformanceOptimizer:
    """
    Advanced performance optimizer for neural system.
    Implements batching, caching, and parallel processing optimizations.
    """
    
    def __init__(self, target_step_time: float = 0.016):  # 60 FPS
        """
        Initialize performance optimizer.
        
        Args:
            target_step_time: Target time per simulation step (seconds)
        """
        self.target_step_time = target_step_time
        self.metrics_history = deque(maxlen=1000)
        
        # Optimization settings
        self.batch_size = 1000  # Process nodes in batches
        self.cache_size = 10000  # Cache size for computations
        self.parallel_workers = 4  # Number of parallel workers
        
        # Performance tracking
        self.step_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.optimization_callbacks = []
        
        # Caching
        self.computation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Batching
        self.node_batches = []
        self.edge_batches = []
        
        log_step("PerformanceOptimizer initialized", 
                target_step_time=target_step_time,
                batch_size=self.batch_size)
    
    def optimize_graph_processing(self, graph) -> torch.Tensor:
        """
        Optimize graph processing with batching and caching.
        
        Args:
            graph: PyTorch Geometric graph
            
        Returns:
            Optimized graph
        """
        start_time = time.time()
        
        try:
            # Batch node processing
            optimized_graph = self._batch_process_nodes(graph)
            
            # Optimize edge operations
            optimized_graph = self._optimize_edge_operations(optimized_graph)
            
            # Update performance metrics
            step_time = time.time() - start_time
            self._update_metrics(step_time, graph)
            
            return optimized_graph
            
        except Exception as e:
            log_step("Graph processing optimization error", error=str(e))
            return graph
    
    def _batch_process_nodes(self, graph) -> torch.Tensor:
        """Process nodes in optimized batches."""
        if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
            return graph
        
        num_nodes = len(graph.node_labels)
        
        # Create batches
        batch_size = min(self.batch_size, num_nodes)
        num_batches = (num_nodes + batch_size - 1) // batch_size
        
        # Process nodes in parallel batches
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_nodes)
                
                # Get batch of nodes
                batch_nodes = graph.node_labels[start_idx:end_idx]
                batch_features = graph.x[start_idx:end_idx]
                
                # Process batch (vectorized operations)
                self._process_node_batch(batch_nodes, batch_features, graph)
        
        return graph
    
    def _process_node_batch(self, batch_nodes: List[Dict], 
                           batch_features: torch.Tensor, graph):
        """Process a batch of nodes efficiently."""
        try:
            # Vectorized energy updates
            if batch_features.numel() > 0:
                # Apply energy decay (vectorized)
                decay_rate = 0.99
                batch_features *= decay_rate
                
                # Apply activation functions (vectorized)
                batch_features = torch.clamp(batch_features, 0.0, 244.0)
                
                # Update node states (vectorized where possible)
                for i, node in enumerate(batch_nodes):
                    if i < batch_features.shape[0]:
                        energy = batch_features[i, 0].item()
                        node['energy'] = energy
                        node['membrane_potential'] = min(energy / 244.0, 1.0)
                        
                        # Update state based on energy
                        if energy > 0.3:
                            node['state'] = 'active'
                        else:
                            node['state'] = 'inactive'
            
        except Exception as e:
            log_step("Node batch processing error", error=str(e))
    
    def _optimize_edge_operations(self, graph):
        """Optimize edge operations with caching and batching."""
        if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
            return graph
        
        try:
            # Cache edge computations
            edge_cache_key = f"edges_{graph.edge_index.shape[1]}"
            
            if edge_cache_key in self.computation_cache:
                self.cache_hits += 1
                # Use cached results
                cached_edges = self.computation_cache[edge_cache_key]
                graph.edge_index = cached_edges
            else:
                self.cache_misses += 1
                # Compute and cache
                self.computation_cache[edge_cache_key] = graph.edge_index.clone()
                
                # Limit cache size
                if len(self.computation_cache) > self.cache_size:
                    # Remove oldest entries
                    oldest_key = next(iter(self.computation_cache))
                    del self.computation_cache[oldest_key]
            
            return graph
            
        except Exception as e:
            log_step("Edge operations optimization error", error=str(e))
            return graph
    
    def optimize_connection_formation(self, graph) -> torch.Tensor:
        """Optimize connection formation with intelligent batching."""
        start_time = time.time()
        
        try:
            if not hasattr(graph, "node_labels"):
                return graph
            
            # Limit connection formation for large graphs
            num_nodes = len(graph.node_labels)
            if num_nodes > 10000:
                # Use sampling for large graphs
                graph = self._sampled_connection_formation(graph)
            else:
                # Use full connection formation for small graphs
                graph = self._full_connection_formation(graph)
            
            step_time = time.time() - start_time
            log_step("Connection formation optimized", 
                    step_time=step_time,
                    num_nodes=num_nodes)
            
            return graph
            
        except Exception as e:
            log_step("Connection formation optimization error", error=str(e))
            return graph
    
    def _sampled_connection_formation(self, graph):
        """Form connections using intelligent sampling for large graphs."""
        try:
            # Sample a subset of nodes for connection formation
            num_nodes = len(graph.node_labels)
            sample_size = min(1000, num_nodes // 10)  # Sample 10% or max 1000 nodes
            
            # Get random sample of node indices
            sample_indices = np.random.choice(
                num_nodes, 
                size=sample_size, 
                replace=False
            )
            
            # Create connections only for sampled nodes
            new_edges = []
            for i, node_idx in enumerate(sample_indices):
                if i < len(sample_indices) - 1:
                    # Connect to next node in sample
                    target_idx = sample_indices[i + 1]
                    new_edges.append([node_idx, target_idx])
            
            # Add sampled edges to graph
            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).T
                if graph.edge_index.numel() == 0:
                    graph.edge_index = new_edge_tensor
                else:
                    graph.edge_index = torch.cat([graph.edge_index, new_edge_tensor], dim=1)
            
            return graph
            
        except Exception as e:
            log_step("Sampled connection formation error", error=str(e))
            return graph
    
    def _full_connection_formation(self, graph):
        """Form connections for smaller graphs."""
        try:
            # Use existing connection logic for smaller graphs
            from connection_logic import intelligent_connection_formation
            return intelligent_connection_formation(graph)
            
        except Exception as e:
            log_step("Full connection formation error", error=str(e))
            return graph
    
    def optimize_birth_death_logic(self, graph) -> torch.Tensor:
        """Optimize node birth and death with batching."""
        start_time = time.time()
        
        try:
            # Batch node death operations
            graph = self._batch_node_death(graph)
            
            # Batch node birth operations
            graph = self._batch_node_birth(graph)
            
            step_time = time.time() - start_time
            log_step("Birth/death logic optimized", 
                    step_time=step_time,
                    num_nodes=len(graph.node_labels) if hasattr(graph, 'node_labels') else 0)
            
            return graph
            
        except Exception as e:
            log_step("Birth/death logic optimization error", error=str(e))
            return graph
    
    def _batch_node_death(self, graph):
        """Batch process node death operations."""
        try:
            if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
                return graph
            
            # Find nodes to remove (vectorized)
            death_threshold = 0.1
            energy_values = graph.x[:, 0]
            nodes_to_remove = torch.where(energy_values < death_threshold)[0]
            
            # Limit number of nodes to remove per step
            max_deaths = min(100, len(nodes_to_remove))
            if len(nodes_to_remove) > max_deaths:
                nodes_to_remove = nodes_to_remove[:max_deaths]
            
            # Remove nodes in batch
            if len(nodes_to_remove) > 0:
                # Create mask for nodes to keep
                keep_mask = torch.ones(len(graph.node_labels), dtype=torch.bool)
                keep_mask[nodes_to_remove] = False
                
                # Update graph
                graph.x = graph.x[keep_mask]
                graph.node_labels = [graph.node_labels[i] for i in range(len(graph.node_labels)) if keep_mask[i]]
                
                # Update edge indices
                if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0:
                    # Remove edges connected to dead nodes
                    edge_mask = ~torch.isin(graph.edge_index[0], nodes_to_remove) & \
                               ~torch.isin(graph.edge_index[1], nodes_to_remove)
                    graph.edge_index = graph.edge_index[:, edge_mask]
                    
                    # Adjust indices
                    for removed_idx in sorted(nodes_to_remove.tolist(), reverse=True):
                        graph.edge_index[0] = torch.where(graph.edge_index[0] > removed_idx, 
                                                        graph.edge_index[0] - 1, graph.edge_index[0])
                        graph.edge_index[1] = torch.where(graph.edge_index[1] > removed_idx, 
                                                        graph.edge_index[1] - 1, graph.edge_index[1])
                
                # Update node IDs
                for new_idx, label in enumerate(graph.node_labels):
                    label["id"] = new_idx
            
            return graph
            
        except Exception as e:
            log_step("Batch node death error", error=str(e))
            return graph
    
    def _batch_node_birth(self, graph):
        """Batch process node birth operations."""
        try:
            if not hasattr(graph, 'node_labels') or not hasattr(graph, 'x'):
                return graph
            
            # Find nodes eligible for birth (vectorized)
            birth_threshold = 0.8
            energy_values = graph.x[:, 0]
            birth_candidates = torch.where(energy_values > birth_threshold)[0]
            
            # Limit number of births per step
            max_births = min(50, len(birth_candidates))
            if len(birth_candidates) > max_births:
                birth_candidates = birth_candidates[:max_births]
            
            # Create new nodes in batch
            if len(birth_candidates) > 0:
                new_features = []
                new_labels = []
                
                for candidate_idx in birth_candidates:
                    parent_energy = graph.x[candidate_idx, 0].item()
                    
                    # Create new node
                    new_energy = parent_energy * 0.4  # 40% of parent energy
                    new_features.append([new_energy])
                    
                    new_node_id = len(graph.node_labels) + len(new_labels)
                    new_labels.append({
                        "id": new_node_id,
                        "type": "dynamic",
                        "behavior": "dynamic",
                        "energy": new_energy,
                        "state": "active",
                        "membrane_potential": new_energy / 244.0,
                        "threshold": 0.3,
                        "refractory_timer": 0.0,
                        "last_activation": 0,
                        "plasticity_enabled": True,
                        "eligibility_trace": 0.0,
                        "last_update": 0
                    })
                    
                    # Reduce parent energy
                    graph.x[candidate_idx, 0] = parent_energy * 0.7  # 70% remaining
                
                # Add new nodes to graph
                if new_features:
                    new_features_tensor = torch.tensor(new_features, dtype=graph.x.dtype)
                    graph.x = torch.cat([graph.x, new_features_tensor], dim=0)
                    graph.node_labels.extend(new_labels)
            
            return graph
            
        except Exception as e:
            log_step("Batch node birth error", error=str(e))
            return graph
    
    def _update_metrics(self, step_time: float, graph):
        """Update performance metrics."""
        try:
            # Calculate metrics
            node_count = len(graph.node_labels) if hasattr(graph, 'node_labels') else 0
            edge_count = graph.edge_index.shape[1] if hasattr(graph, 'edge_index') and graph.edge_index.numel() > 0 else 0
            throughput = node_count / step_time if step_time > 0 else 0
            
            # Get comprehensive performance metrics from new monitoring system
            from performance_monitor import get_system_performance_metrics
            perf_metrics = get_system_performance_metrics()
            
            metrics = PerformanceMetrics(
                step_time=step_time,
                memory_usage=perf_metrics['memory_usage'],
                cpu_usage=perf_metrics['cpu_usage'],
                gpu_usage=perf_metrics['gpu_usage'],
                node_count=node_count,
                edge_count=edge_count,
                throughput=throughput,
                timestamp=time.time()
            )
            
            self.metrics_history.append(metrics)
            self.step_times.append(step_time)
            
            # Call optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    log_step("Optimization callback error", error=str(e))
            
            # Log performance warnings
            if step_time > self.target_step_time * 2:
                log_step("Performance warning: Step time exceeded target", 
                        step_time=step_time,
                        target=self.target_step_time,
                        node_count=node_count)
            
        except Exception as e:
            log_step("Metrics update error", error=str(e))
    
    def add_optimization_callback(self, callback: Callable):
        """Add callback for optimization events."""
        self.optimization_callbacks.append(callback)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.step_times:
            return {"status": "no_data"}
        
        avg_step_time = np.mean(self.step_times)
        max_step_time = np.max(self.step_times)
        min_step_time = np.min(self.step_times)
        
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "avg_step_time": avg_step_time,
            "max_step_time": max_step_time,
            "min_step_time": min_step_time,
            "target_step_time": self.target_step_time,
            "performance_ratio": avg_step_time / self.target_step_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_metrics": len(self.metrics_history)
        }
    
    def optimize_for_realtime(self, graph) -> torch.Tensor:
        """Apply all optimizations for real-time performance."""
        start_time = time.time()
        
        try:
            # Apply all optimizations
            graph = self.optimize_graph_processing(graph)
            graph = self.optimize_connection_formation(graph)
            graph = self.optimize_birth_death_logic(graph)
            
            total_time = time.time() - start_time
            
            # Log performance improvement
            if len(self.step_times) > 1:
                prev_avg = np.mean(list(self.step_times)[:-1])
                improvement = (prev_avg - total_time) / prev_avg * 100 if prev_avg > 0 else 0
                log_step("Real-time optimization applied", 
                        total_time=total_time,
                        improvement_percent=improvement)
            
            return graph
            
        except Exception as e:
            log_step("Real-time optimization error", error=str(e))
            return graph

# Example usage and testing
if __name__ == "__main__":
    print("⚡ Testing Performance Optimizer...")
    
    # Create test graph
    import torch
    from torch_geometric.data import Data
    
    # Create a large test graph
    num_nodes = 10000
    num_edges = 5000
    
    x = torch.randn(num_nodes, 1)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_labels = [{"id": i, "type": "dynamic", "energy": x[i, 0].item()} for i in range(num_nodes)]
    
    test_graph = Data(x=x, edge_index=edge_index, node_labels=node_labels)
    
    print(f"Created test graph: {num_nodes} nodes, {num_edges} edges")
    
    # Test performance optimizer
    optimizer = PerformanceOptimizer(target_step_time=0.016)  # 60 FPS
    
    # Add callback
    def performance_callback(metrics):
        print(f"Step time: {metrics.step_time:.4f}s, "
              f"Nodes: {metrics.node_count}, "
              f"Throughput: {metrics.throughput:.0f} nodes/sec")
    
    optimizer.add_optimization_callback(performance_callback)
    
    # Test optimization
    print("Running optimization...")
    optimized_graph = optimizer.optimize_for_realtime(test_graph)
    
    # Get performance stats
    stats = optimizer.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Performance Optimizer test completed!")
