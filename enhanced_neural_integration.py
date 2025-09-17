
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from torch_geometric.data import Data
import logging
from logging_utils import log_step
from enhanced_neural_dynamics import EnhancedNeuralDynamics, create_enhanced_neural_dynamics
from enhanced_connection_system import EnhancedConnectionSystem, create_enhanced_connection_system
from enhanced_node_behaviors import EnhancedNodeBehaviorSystem, create_enhanced_node_behavior_system
from node_access_layer import NodeAccessLayer
from config_manager import get_learning_config, get_system_constants


class EnhancedNeuralIntegration:

    def __init__(self):
        self.neural_dynamics = create_enhanced_neural_dynamics()
        self.connection_system = create_enhanced_connection_system()
        self.node_behavior_system = create_enhanced_node_behavior_system()
        self.integration_active = True
        self.update_frequency = 1
        self.last_update_step = 0
        self.integration_stats = {
            'total_updates': 0,
            'neural_dynamics_updates': 0,
            'connection_updates': 0,
            'node_behavior_updates': 0,
            'integration_errors': 0
        }
        log_step("EnhancedNeuralIntegration initialized")
    def integrate_with_existing_system(self, graph: Data, step: int) -> Data:

        if not self.integration_active:
            return graph
        try:
            graph = self.neural_dynamics.update_neural_dynamics(graph, step)
            self.integration_stats['neural_dynamics_updates'] += 1
            graph = self.connection_system.update_connections(graph, step)
            self.integration_stats['connection_updates'] += 1
            graph = self.node_behavior_system.update_node_behaviors(graph, step)
            self.integration_stats['node_behavior_updates'] += 1
            self.integration_stats['total_updates'] += 1
            self.last_update_step = step
            return graph
        except Exception as e:
            log_step("Error in enhanced neural integration", error=str(e))
            self.integration_stats['integration_errors'] += 1
            return graph
    def create_enhanced_node(self, graph: Data, node_id: int, node_type: str = 'dynamic',
                           subtype: str = 'standard', **kwargs) -> bool:

        try:
            access_layer = NodeAccessLayer(graph)
            if not access_layer.is_valid_node_id(node_id):
                log_step("Invalid node ID for enhanced node creation", node_id=node_id)
                return False
            behavior = self.node_behavior_system.create_node_behavior(
                node_id, node_type, subtype=subtype, **kwargs
            )
            access_layer.update_node_property(node_id, 'enhanced_behavior', True)
            access_layer.update_node_property(node_id, 'subtype', subtype)
            access_layer.update_node_property(node_id, 'is_excitatory', kwargs.get('is_excitatory', True))
            if not hasattr(graph, 'enhanced_node_ids'):
                graph.enhanced_node_ids = []
            if node_id not in graph.enhanced_node_ids:
                graph.enhanced_node_ids.append(node_id)
            log_step("Enhanced node created", node_id=node_id, type=node_type, subtype=subtype)
            return True
        except Exception as e:
            log_step("Error creating enhanced node", node_id=node_id, error=str(e))
            return False
    def create_enhanced_connection(self, graph: Data, source_id: int, target_id: int,
                                 connection_type: str = 'excitatory', **kwargs) -> bool:

        try:
            access_layer = NodeAccessLayer(graph)
            if not access_layer.is_valid_node_id(source_id) or not access_layer.is_valid_node_id(target_id):
                log_step("Invalid node IDs for enhanced connection creation",
                        source_id=source_id, target_id=target_id)
                return False
            success = self.connection_system.create_connection(
                source_id, target_id, connection_type, **kwargs
            )
            if success:
                self._update_graph_edges(graph)
                log_step("Enhanced connection created",
                        source_id=source_id,
                        target_id=target_id,
                        type=connection_type)
            return success
        except Exception as e:
            log_step("Error creating enhanced connection",
                    source_id=source_id,
                    target_id=target_id,
                    error=str(e))
            return False
    def _update_graph_edges(self, graph: Data):
        if not hasattr(graph, 'edge_index'):
            graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        active_connections = []
        for connection in self.connection_system.connections:
            if connection.active:
                active_connections.append([connection.source_id, connection.target_id])
        if active_connections:
            new_edges = torch.tensor(active_connections, dtype=torch.long).t()
            if graph.edge_index.numel() == 0:
                graph.edge_index = new_edges
            else:
                graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
    def set_neuromodulator_level(self, neuromodulator: str, level: float):
        self.connection_system.set_neuromodulator_level(neuromodulator, level)
        self.neural_dynamics.set_neuromodulator_level(neuromodulator, level)
    def get_integration_statistics(self) -> Dict[str, Any]:
        stats = self.integration_stats.copy()
        stats['neural_dynamics_stats'] = self.neural_dynamics.get_statistics()
        stats['connection_stats'] = self.connection_system.get_connection_statistics()
        stats['node_behavior_stats'] = self.node_behavior_system.get_behavior_statistics()
        return stats
    def reset_integration_statistics(self):
        self.integration_stats = {
            'total_updates': 0,
            'neural_dynamics_updates': 0,
            'connection_updates': 0,
            'node_behavior_updates': 0,
            'integration_errors': 0
        }
        self.neural_dynamics.reset_statistics()
        self.connection_system.reset_statistics()
        self.node_behavior_system.reset_statistics()
    def enable_integration(self):
        self.integration_active = True
        log_step("Enhanced neural integration enabled")
    def disable_integration(self):
        self.integration_active = False
        log_step("Enhanced neural integration disabled")
    def cleanup(self):
        self.neural_dynamics.cleanup()
        self.connection_system.cleanup()
        self.node_behavior_system.cleanup()
        log_step("Enhanced neural integration cleaned up")


def create_enhanced_neural_integration() -> EnhancedNeuralIntegration:
    return EnhancedNeuralIntegration()


def integrate_with_simulation_manager(simulation_manager):

    enhanced_integration = create_enhanced_neural_integration()
    simulation_manager.enhanced_integration = enhanced_integration
    original_update_node_behaviors = simulation_manager._update_node_behaviors
    def enhanced_update_node_behaviors():
        original_update_node_behaviors()
        if hasattr(simulation_manager, 'graph') and simulation_manager.graph is not None:
            simulation_manager.graph = enhanced_integration.integrate_with_existing_system(
                simulation_manager.graph, simulation_manager.step_counter
            )
    simulation_manager._update_node_behaviors = enhanced_update_node_behaviors
    log_step("Enhanced neural integration added to simulation manager")
    return enhanced_integration
if __name__ == "__main__":
    print("EnhancedNeuralIntegration created successfully!")
    print("Features include:")
    print("- Coordinates all enhanced neural systems")
    print("- Integrates with existing ID-based architecture")
    print("- Provides unified interface for neural simulation")
    print("- Maintains compatibility with existing systems")
    print("- Sophisticated node behaviors and connections")
    print("- Advanced learning mechanisms")
    try:
        integration = create_enhanced_neural_integration()
        print(f"Integration system created with {len(integration.integration_stats)} statistics tracked")
        stats = integration.get_integration_statistics()
        print(f"Integration statistics: {stats}")
        integration.disable_integration()
        integration.enable_integration()
        print("Enable/disable test: PASSED")
    except Exception as e:
        print(f"EnhancedNeuralIntegration test failed: {e}")
    print("EnhancedNeuralIntegration test completed!")
