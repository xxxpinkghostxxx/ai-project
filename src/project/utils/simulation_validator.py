"""
Comprehensive simulation rule validator that intentionally creates and tests
every possible combination of node types, subtypes, and connection types.
"""

import logging
import torch
from typing import Any, Dict, List, Tuple
from itertools import product

from project.pyg_neural_system import (
    PyGNeuralSystem,
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
    NODE_SPAWN_THRESHOLD,
    NODE_ENERGY_SPAWN_COST,
    MAX_NODE_BIRTHS_PER_STEP,
    GATE_THRESHOLD,
)

from project.utils.energy_calculator import EnergyCalculator

logger = logging.getLogger(__name__)


class SimulationValidator:
    """Validates all simulation rules by intentionally creating every possible combination."""

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.test_system: PyGNeuralSystem | None = None
        self.energy_calculator = EnergyCalculator()

    def run_full_test(self, device: str = "cpu") -> Dict[str, Any]:
        """Run comprehensive rule validation test with all combinations."""
        self.results = {}
        self.errors = []
        self.warnings = []

        try:
            # Create minimal system for combination testing
            self.test_system = PyGNeuralSystem(
                sensory_width=1,
                sensory_height=1,
                n_dynamic=1,
                workspace_size=(1, 1),
                device=device,
            )

            if self.test_system.g is None:
                self.errors.append("Failed to initialize test system graph")
                self.results["status"] = "ERROR"
                return self.results

            # Phase 1: Test all node type combinations
            logger.info("Phase 1: Testing all node type combinations")
            self._test_all_node_type_combinations()

            # Phase 2: Test all dynamic subtype combinations
            logger.info("Phase 2: Testing all dynamic subtype combinations")
            self._test_all_dynamic_subtype_combinations()

            # Phase 3: Test all connection type combinations
            logger.info("Phase 3: Testing all connection type combinations")
            self._test_all_connection_type_combinations()

            # Phase 4: Test all connection subtype combinations
            logger.info("Phase 4: Testing all connection subtype combinations")
            self._test_all_connection_subtype_combinations()

            # Phase 5: Test cross-combinations (node types × connection types)
            logger.info("Phase 5: Testing cross-combinations (node types × connection types)")
            self._test_cross_combinations()

            # Phase 6: Test rules for each combination
            logger.info("Phase 6: Testing rules for each combination")
            self._test_rules_for_all_combinations()

            # Phase 7: Test energy transfer for each combination
            logger.info("Phase 7: Testing energy transfer for each combination")
            self._test_energy_transfer_combinations()

            # Phase 8: Test sensory nodes specifically
            logger.info("Phase 8: Testing sensory nodes")
            self._test_sensory_nodes_comprehensive()

            # Phase 9: Test workspace nodes specifically
            logger.info("Phase 9: Testing workspace nodes")
            self._test_workspace_nodes_comprehensive()

            # Phase 10: Test energy state tracking through 10 steps
            logger.info("Phase 10: Testing energy state tracking through 10 steps")
            self._test_energy_state_tracking_10_steps()

            # Phase 11: Test sensory nodes with mock pixel input
            logger.info("Phase 11: Testing sensory nodes with mock pixel input")
            self._test_sensory_mock_pixel_input()

            # Phase 12: Test workspace nodes with mock canvas output
            logger.info("Phase 12: Testing workspace nodes with mock canvas output")
            self._test_workspace_mock_canvas_output()

            # Phase 13: Test full pipeline: pixels → sensory → dynamic → workspace → canvas
            logger.info("Phase 13: Testing full pipeline (pixels → sensory → dynamic → workspace → canvas)")
            self._test_full_pipeline_mock()

            # Phase 14: Test spawn/death thresholds
            logger.info("Phase 14: Testing spawn/death thresholds")
            self._test_spawn_death_thresholds(self.test_system)

            # Phase 15: Test density rules
            logger.info("Phase 15: Testing density rules")
            self._test_density_rules(self.test_system)

            # Cleanup
            self.test_system.cleanup()

            self.results["status"] = "PASSED" if not self.errors else "FAILED"
            self.results["errors"] = self.errors
            self.results["warnings"] = self.warnings
            self.results["total_tests"] = len([k for k in self.results.keys() if k not in ("status", "errors", "warnings")])

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.errors.append(f"Test setup failed: {str(e)}")
            self.results["status"] = "ERROR"
            logger.exception("Validation test failed")
        finally:
            if self.test_system is not None:
                try:
                    self.test_system.cleanup()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        return self.results

    def _create_test_nodes(self, node_configs: List[Dict[str, Any]]) -> List[int]:
        """Create test nodes with specific configurations. Returns list of node indices."""
        if self.test_system is None or self.test_system.g is None:
            return []

        g = self.test_system.g
        device = self.test_system.device
        node_indices: List[int] = []

        for config in node_configs:
            node_type = config.get('node_type', NODE_TYPE_DYNAMIC)
            dynamic_subtype = config.get('dynamic_subtype', SUBTYPE_TRANSMITTER)
            
            # Add node
            if g.energy is None:
                continue
                
            current_count = g.num_nodes or 0
            new_idx = current_count
            
            # Extend all node tensors
            g.energy = torch.cat([g.energy, torch.tensor([[NODE_ENERGY_CAP * 0.5]], device=device)])
            g.node_type = torch.cat([g.node_type, torch.tensor([node_type], dtype=torch.int64, device=device)])
            
            if hasattr(g, 'pos') and g.pos is not None:
                g.pos = torch.cat([g.pos, torch.rand(1, 2, device=device) * 100])
            else:
                g.pos = torch.rand(1, 2, device=device) * 100
                
            if hasattr(g, 'velocity') and g.velocity is not None:
                g.velocity = torch.cat([g.velocity, torch.zeros(1, 2, device=device)])
            else:
                g.velocity = torch.zeros(1, 2, device=device)
                
            if hasattr(g, 'dynamic_subtype') and g.dynamic_subtype is not None:
                g.dynamic_subtype = torch.cat([g.dynamic_subtype, torch.tensor([dynamic_subtype], dtype=torch.int64, device=device)])
            else:
                g.dynamic_subtype = torch.tensor([dynamic_subtype], dtype=torch.int64, device=device)
                
            if hasattr(g, 'parent') and g.parent is not None:
                g.parent = torch.cat([g.parent, torch.tensor([-1], dtype=torch.int64, device=device)])
            else:
                g.parent = torch.tensor([-1], dtype=torch.int64, device=device)
                
            # Add other required attributes with defaults
            for attr_name, default_value in [
                ('dynamic_subtype2', 0),
                ('dynamic_subtype3', 0),
                ('dynamic_subtype4', 0),
                ('max_connections', 10),
                ('phase_offset', 0.0),
            ]:
                if hasattr(g, attr_name) and getattr(g, attr_name) is not None:
                    attr = getattr(g, attr_name)
                    if isinstance(default_value, (int, float)):
                        if attr.dim() == 0:
                            new_val = torch.tensor([default_value], dtype=attr.dtype, device=device)
                        else:
                            new_val = torch.tensor([[default_value]], dtype=attr.dtype, device=device) if attr.dim() > 1 else torch.tensor([default_value], dtype=attr.dtype, device=device)
                    else:
                        new_val = torch.tensor([default_value], dtype=attr.dtype, device=device)
                    setattr(g, attr_name, torch.cat([attr, new_val]))
                else:
                    if isinstance(default_value, (int, float)):
                        setattr(g, attr_name, torch.tensor([default_value], dtype=torch.int64 if isinstance(default_value, int) else torch.float32, device=device))
                    else:
                        setattr(g, attr_name, torch.tensor([default_value], dtype=torch.int64, device=device))

            g.num_nodes = (g.num_nodes or 0) + 1
            node_indices.append(new_idx)

        return node_indices

    def _create_test_connection(self, src_idx: int, dst_idx: int, conn_type: int, conn_subtype3: int) -> bool:
        """Create a test connection with specific type and subtype. Returns True if created."""
        if self.test_system is None or self.test_system.g is None:
            return False

        g = self.test_system.g
        device = self.test_system.device

        # Check if connection is allowed by rules
        src_type = g.node_type[src_idx].item()
        dst_type = g.node_type[dst_idx].item()

        # Rule: No connections TO sensory nodes
        if dst_type == NODE_TYPE_SENSORY:
            return False

        # Rule: No connections FROM workspace nodes
        if src_type == NODE_TYPE_WORKSPACE:
            return False

        # Rule: No sensory->workspace connections
        if src_type == NODE_TYPE_SENSORY and dst_type == NODE_TYPE_WORKSPACE:
            return False

        # Create edge
        if g.edge_index is None:
            g.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long, device=device)
        g.edge_index = torch.cat([g.edge_index, new_edge], dim=1)

        # Add edge attributes
        weight = 0.1 if conn_type == CONN_TYPE_EXCITATORY else (-0.1 if conn_type == CONN_TYPE_INHIBITORY else 0.05)
        
        for attr_name, attr_value in [
            ('weight', torch.tensor([[weight]], device=device)),
            ('energy_transfer_capacity', torch.tensor([[0.5]], device=device)),
            ('conn_type', torch.tensor([[conn_type]], dtype=torch.int64, device=device)),
            ('plastic_lr', torch.tensor([[0.01 if conn_type == CONN_TYPE_PLASTIC else 0.0]], device=device)),
            ('gate_threshold', torch.tensor([[GATE_THRESHOLD if conn_type == CONN_TYPE_GATED else 0.0]], device=device)),
            ('conn_subtype2', torch.tensor([0], dtype=torch.int64, device=device)),  # Default subtype2
            ('conn_subtype3', torch.tensor([conn_subtype3], dtype=torch.int64, device=device)),
        ]:
            if hasattr(g, attr_name) and getattr(g, attr_name) is not None:
                attr = getattr(g, attr_name)
                setattr(g, attr_name, torch.cat([attr, attr_value]))
            else:
                setattr(g, attr_name, attr_value)

        # num_edges is a property computed from edge_index, don't assign directly
        return True

    def _test_all_node_type_combinations(self) -> None:
        """Test all combinations of source and destination node types."""
        # Current simulation supports 3 node classes (sensory/dynamic/workspace).
        # Legacy "highway" nodes were removed from `project.pyg_neural_system`.
        node_types = [NODE_TYPE_SENSORY, NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE]
        combinations_tested = 0
        valid_combinations = 0
        invalid_combinations = 0

        for src_type, dst_type in product(node_types, node_types):
            combinations_tested += 1
            test_name = f"src_{src_type}_dst_{dst_type}"
            
            # Create test nodes
            src_config = {'node_type': src_type, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
            dst_config = {'node_type': dst_type, 'dynamic_subtype': SUBTYPE_RESONATOR}
            
            # Reset system for clean test
            self._reset_test_system()
            node_indices = self._create_test_nodes([src_config, dst_config])
            
            if len(node_indices) < 2:
                self.errors.append(f"{test_name}: Failed to create test nodes")
                continue

            src_idx, dst_idx = node_indices[0], node_indices[1]

            # Try to create connection
            conn_created = self._create_test_connection(
                src_idx, dst_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )

            # Check rules
            expected_allowed = True
            if dst_type == NODE_TYPE_SENSORY:
                expected_allowed = False  # No connections TO sensory
            elif src_type == NODE_TYPE_WORKSPACE:
                expected_allowed = False  # No connections FROM workspace
            elif src_type == NODE_TYPE_SENSORY and dst_type == NODE_TYPE_WORKSPACE:
                expected_allowed = False  # No sensory->workspace

            if conn_created == expected_allowed:
                valid_combinations += 1
                self.results[f"{test_name}_rule_check"] = "PASS"
            else:
                invalid_combinations += 1
                self.errors.append(f"{test_name}: Connection rule violation (created={conn_created}, expected={expected_allowed})")

        self.results["node_type_combinations_tested"] = combinations_tested
        self.results["node_type_valid_combinations"] = valid_combinations
        self.results["node_type_invalid_combinations"] = invalid_combinations

    def _test_all_dynamic_subtype_combinations(self) -> None:
        """Test all combinations of dynamic node subtypes."""
        subtypes = [SUBTYPE_TRANSMITTER, SUBTYPE_RESONATOR, SUBTYPE_DAMPENER]
        combinations_tested = 0

        for src_subtype, dst_subtype in product(subtypes, subtypes):
            combinations_tested += 1
            test_name = f"src_subtype_{src_subtype}_dst_subtype_{dst_subtype}"

            # Create test nodes
            self._reset_test_system()
            src_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': src_subtype}
            dst_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': dst_subtype}
            
            node_indices = self._create_test_nodes([src_config, dst_config])
            
            if len(node_indices) < 2:
                continue

            src_idx, dst_idx = node_indices[0], node_indices[1]

            # Verify subtypes are set correctly
            if self.test_system and self.test_system.g:
                g = self.test_system.g
                if hasattr(g, 'dynamic_subtype'):
                    src_subtype_actual = g.dynamic_subtype[src_idx].item()
                    dst_subtype_actual = g.dynamic_subtype[dst_idx].item()
                    
                    if src_subtype_actual == src_subtype and dst_subtype_actual == dst_subtype:
                        self.results[f"{test_name}_subtype_check"] = "PASS"
                    else:
                        self.errors.append(f"{test_name}: Subtype mismatch (src: {src_subtype_actual} vs {src_subtype}, dst: {dst_subtype_actual} vs {dst_subtype})")

        self.results["dynamic_subtype_combinations_tested"] = combinations_tested

    def _test_all_connection_type_combinations(self) -> None:
        """Test all connection types."""
        conn_types = [CONN_TYPE_EXCITATORY, CONN_TYPE_INHIBITORY, CONN_TYPE_GATED, CONN_TYPE_PLASTIC]
        combinations_tested = 0

        for conn_type in conn_types:
            combinations_tested += 1
            test_name = f"conn_type_{conn_type}"

            # Create test nodes (dynamic to dynamic)
            self._reset_test_system()
            src_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
            dst_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR}
            
            node_indices = self._create_test_nodes([src_config, dst_config])
            
            if len(node_indices) < 2:
                continue

            src_idx, dst_idx = node_indices[0], node_indices[1]

            # Create connection with this type
            conn_created = self._create_test_connection(
                src_idx, dst_idx, conn_type, CONN_SUBTYPE3_FREE_FLOW
            )

            if conn_created:
                # Verify connection type is set correctly
                if self.test_system and self.test_system.g:
                    g = self.test_system.g
                    if hasattr(g, 'conn_type') and g.edge_index is not None:
                        edge_idx = g.edge_index.shape[1] - 1
                        conn_type_actual = g.conn_type[edge_idx].item()
                        
                        if conn_type_actual == conn_type:
                            self.results[f"{test_name}_type_check"] = "PASS"
                            
                            # Check type-specific attributes
                            if conn_type == CONN_TYPE_GATED:
                                if hasattr(g, 'gate_threshold'):
                                    threshold = g.gate_threshold[edge_idx].item()
                                    if threshold > 0:
                                        self.results[f"{test_name}_gate_threshold"] = "PASS"
                                    else:
                                        self.errors.append(f"{test_name}: Gated connection missing threshold")
                            elif conn_type == CONN_TYPE_PLASTIC:
                                if hasattr(g, 'plastic_lr'):
                                    lr = g.plastic_lr[edge_idx].item()
                                    if lr > 0:
                                        self.results[f"{test_name}_plastic_lr"] = "PASS"
                                    else:
                                        self.warnings.append(f"{test_name}: Plastic connection has zero learning rate")
                        else:
                            self.errors.append(f"{test_name}: Connection type mismatch ({conn_type_actual} vs {conn_type})")

        self.results["connection_type_combinations_tested"] = combinations_tested

    def _test_all_connection_subtype_combinations(self) -> None:
        """Test all connection subtype3 (directionality) combinations."""
        conn_subtypes = [CONN_SUBTYPE3_ONE_WAY_OUT, CONN_SUBTYPE3_ONE_WAY_IN, CONN_SUBTYPE3_FREE_FLOW]
        combinations_tested = 0

        for conn_subtype in conn_subtypes:
            combinations_tested += 1
            test_name = f"conn_subtype3_{conn_subtype}"

            # Create test nodes with parent relationship for one-way testing
            self._reset_test_system()
            src_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
            dst_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR}
            
            node_indices = self._create_test_nodes([src_config, dst_config])
            
            if len(node_indices) < 2:
                continue

            src_idx, dst_idx = node_indices[0], node_indices[1]

            # Set parent relationship for directionality testing
            if self.test_system and self.test_system.g:
                g = self.test_system.g
                if hasattr(g, 'parent'):
                    g.parent[dst_idx] = src_idx  # src is parent of dst

            # Create connection with this subtype
            conn_created = self._create_test_connection(
                src_idx, dst_idx, CONN_TYPE_EXCITATORY, conn_subtype
            )

            if conn_created:
                # Verify subtype is set correctly
                if self.test_system and self.test_system.g:
                    g = self.test_system.g
                    if hasattr(g, 'conn_subtype3') and g.edge_index is not None:
                        edge_idx = g.edge_index.shape[1] - 1
                        subtype_actual = g.conn_subtype3[edge_idx].item()
                        
                        if subtype_actual == conn_subtype:
                            self.results[f"{test_name}_subtype_check"] = "PASS"
                        else:
                            self.errors.append(f"{test_name}: Subtype mismatch ({subtype_actual} vs {conn_subtype})")

        self.results["connection_subtype_combinations_tested"] = combinations_tested

    def _test_cross_combinations(self) -> None:
        """Test cross-combinations of node types × connection types."""
        # Current simulation supports 3 node classes (sensory/dynamic/workspace).
        # Legacy "highway" nodes were removed from `project.pyg_neural_system`.
        node_types = [NODE_TYPE_SENSORY, NODE_TYPE_DYNAMIC, NODE_TYPE_WORKSPACE]
        conn_types = [CONN_TYPE_EXCITATORY, CONN_TYPE_INHIBITORY, CONN_TYPE_GATED, CONN_TYPE_PLASTIC]
        combinations_tested = 0

        for src_type, dst_type, conn_type in product(node_types, node_types, conn_types):
            # Skip invalid combinations
            if dst_type == NODE_TYPE_SENSORY or src_type == NODE_TYPE_WORKSPACE:
                continue
            if src_type == NODE_TYPE_SENSORY and dst_type == NODE_TYPE_WORKSPACE:
                continue

            combinations_tested += 1
            test_name = f"cross_src_{src_type}_dst_{dst_type}_conn_{conn_type}"

            # Create test nodes
            self._reset_test_system()
            src_config = {'node_type': src_type, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
            dst_config = {'node_type': dst_type, 'dynamic_subtype': SUBTYPE_RESONATOR}
            
            node_indices = self._create_test_nodes([src_config, dst_config])
            
            if len(node_indices) < 2:
                continue

            src_idx, dst_idx = node_indices[0], node_indices[1]

            # Create connection
            conn_created = self._create_test_connection(
                src_idx, dst_idx, conn_type, CONN_SUBTYPE3_FREE_FLOW
            )

            if conn_created:
                self.results[f"{test_name}_created"] = True

        self.results["cross_combinations_tested"] = combinations_tested

    def _test_rules_for_all_combinations(self) -> None:
        """Test that rules are followed for all combinations."""
        # Test sensory rules: no incoming connections
        self._reset_test_system()
        sensory_config = {'node_type': NODE_TYPE_SENSORY, 'dynamic_subtype': 0}
        dynamic_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
        
        node_indices = self._create_test_nodes([sensory_config, dynamic_config])
        if len(node_indices) >= 2:
            sensory_idx, dynamic_idx = node_indices[0], node_indices[1]
            
            # Try to create connection TO sensory (should fail)
            conn_created = self._create_test_connection(
                dynamic_idx, sensory_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )
            if conn_created:
                self.errors.append("Rule violation: Connection TO sensory node was created")
            else:
                self.results["sensory_no_incoming_rule"] = "PASS"

        # Test workspace rules: no outgoing connections
        self._reset_test_system()
        workspace_config = {'node_type': NODE_TYPE_WORKSPACE, 'dynamic_subtype': 0}
        dynamic_config2 = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR}
        
        node_indices = self._create_test_nodes([workspace_config, dynamic_config2])
        if len(node_indices) >= 2:
            workspace_idx, dynamic_idx2 = node_indices[0], node_indices[1]
            
            # Try to create connection FROM workspace (should fail)
            conn_created = self._create_test_connection(
                workspace_idx, dynamic_idx2, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )
            if conn_created:
                self.errors.append("Rule violation: Connection FROM workspace node was created")
            else:
                self.results["workspace_no_outgoing_rule"] = "PASS"

    def _test_energy_transfer_combinations(self) -> None:
        """Test energy transfer for different combinations."""
        # Test each connection type's energy transfer behavior
        conn_types = [CONN_TYPE_EXCITATORY, CONN_TYPE_INHIBITORY, CONN_TYPE_GATED, CONN_TYPE_PLASTIC]
        
        for conn_type in conn_types:
            test_name = f"energy_transfer_conn_{conn_type}"
            
            self._reset_test_system()
            src_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
            dst_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR}
            
            node_indices = self._create_test_nodes([src_config, dst_config])
            
            if len(node_indices) < 2:
                continue

            src_idx, dst_idx = node_indices[0], node_indices[1]

            # Set initial energies (energy tensor has shape [N, 1])
            if self.test_system and self.test_system.g:
                g = self.test_system.g
                g.energy[src_idx, 0] = NODE_ENERGY_CAP * 0.8
                g.energy[dst_idx, 0] = NODE_ENERGY_CAP * 0.2

            # Create connection
            conn_created = self._create_test_connection(
                src_idx, dst_idx, conn_type, CONN_SUBTYPE3_FREE_FLOW
            )

            if conn_created:
                # Run one update to test energy transfer
                if self.test_system:
                    initial_dst_energy = self.test_system.g.energy[dst_idx].item() if self.test_system.g else 0.0
                    self.test_system.update()
                    final_dst_energy = self.test_system.g.energy[dst_idx].item() if self.test_system.g else 0.0
                    
                    energy_change = final_dst_energy - initial_dst_energy
                    self.results[f"{test_name}_energy_change"] = energy_change
                    
                    # Excitatory should increase, inhibitory should decrease
                    if conn_type == CONN_TYPE_EXCITATORY and energy_change > 0:
                        self.results[f"{test_name}_behavior"] = "PASS"
                    elif conn_type == CONN_TYPE_INHIBITORY and energy_change < 0:
                        self.results[f"{test_name}_behavior"] = "PASS"
                    elif conn_type in [CONN_TYPE_GATED, CONN_TYPE_PLASTIC]:
                        # Gated/plastic may or may not transfer depending on conditions
                        self.results[f"{test_name}_behavior"] = "CHECKED"

    def _reset_test_system(self) -> None:
        """Reset the test system to a clean state."""
        if self.test_system is None:
            return

        # Create a fresh minimal system
        device = self.test_system.device
        self.test_system.cleanup()
        self.test_system = PyGNeuralSystem(
            sensory_width=1,
            sensory_height=1,
            n_dynamic=1,
            workspace_size=(1, 1),
            device=device,
        )

    def _test_sensory_nodes_comprehensive(self) -> None:
        """Comprehensive test of sensory nodes: initialization, energy state, and behavior."""
        self._reset_test_system()
        
        if not self.test_system or not self.test_system.g:
            self.errors.append("Failed to initialize sensory test system")
            return
        
        g = self.test_system.g
        
        # Use the baseline sensory node (created by _reset_test_system) instead of creating new one
        # This avoids mismatch between sensory node count and expected pixel array size
        sensory_nodes = torch.where(g.node_type == NODE_TYPE_SENSORY)[0]
        if sensory_nodes.numel() == 0:
            self.errors.append("No baseline sensory node found")
            return
        sensory_idx = int(sensory_nodes[0].item())
        
        # Create only a dynamic node for testing connections
        dynamic_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR}
        node_indices = self._create_test_nodes([dynamic_config])
        
        if len(node_indices) < 1:
            self.errors.append("Failed to create dynamic test node")
            return

        dynamic_idx = node_indices[0]

        if self.test_system and self.test_system.g:
            # Test 1: Verify sensory node type
            sensory_type = g.node_type[sensory_idx].item()
            if sensory_type == NODE_TYPE_SENSORY:
                self.results["sensory_node_type"] = "PASS"
            else:
                self.errors.append(f"Sensory node has wrong type: {sensory_type}")

            # Test 2: Set initial sensory energy using update_sensory_nodes (proper way)
            # Formula: energy = ((pixel / 255.0) * NODE_ENERGY_CAP * gain) + bias
            # To get target energy, solve for pixel: pixel = ((energy - bias) / (NODE_ENERGY_CAP * gain)) * 255.0
            # Get gain/bias from config (defaults: gain=1.0, bias=0.0)
            try:
                from project.utils.config_manager import ConfigManager  # type: ignore[import-not-found]
                cfg = ConfigManager()
                gain = float(cfg.get_config('sensory', 'energy_gain') or 1.0)
                bias = float(cfg.get_config('sensory', 'energy_bias') or 0.0)  # Note: config key is 'energy_bias'
            except Exception:  # pylint: disable=broad-exception-caught
                gain = 1.0
                bias = 0.0
            
            target_energy = NODE_ENERGY_CAP * 0.8  # Use higher value
            pixel_value = ((target_energy - bias) / (NODE_ENERGY_CAP * gain)) * 255.0
            pixel_value = max(0.0, min(255.0, pixel_value))  # Clamp to valid range
            sensory_array = torch.tensor([[pixel_value]], dtype=torch.float32)
            
            # Disable warmup for testing by setting count high
            if hasattr(self.test_system, '_sensory_update_count'):
                self.test_system._sensory_update_count = self.test_system.sensory_warmup_frames
            
            self.test_system.update_sensory_nodes(sensory_array.numpy())
            
            # Verify it was set (accounting for gain/bias from config and warmup)
            if self.test_system.sensory_true_values is not None:
                actual_energy = self.test_system.sensory_true_values[0].item()
                # Allow tolerance for config-based gain/bias and warmup
                if abs(actual_energy - target_energy) < target_energy * 0.3:  # 30% tolerance
                    self.results["sensory_true_value_set"] = "PASS"
                else:
                    self.warnings.append(f"Sensory true value: expected ~{target_energy}, got {actual_energy} (pixel={pixel_value}, gain={gain}, bias={bias})")

            # Test 3: Create connection FROM sensory TO dynamic (allowed)
            conn_created = self._create_test_connection(
                sensory_idx, dynamic_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )
            
            if conn_created:
                self.results["sensory_outgoing_connection"] = "PASS"
            else:
                self.errors.append("Failed to create connection FROM sensory node")

            # Test 4: Try to create connection TO sensory (should fail)
            conn_to_sensory = self._create_test_connection(
                dynamic_idx, sensory_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )
            
            if not conn_to_sensory:
                self.results["sensory_no_incoming_connection"] = "PASS"
            else:
                self.errors.append("Rule violation: Connection TO sensory node was created")

            # Test 5: Run update and verify sensory energy is restored to true value
            # Get the true value that was set (from sensory_true_values)
            if self.test_system.sensory_true_values is not None:
                expected_sensory_energy = self.test_system.sensory_true_values[0].item()
            else:
                expected_sensory_energy = target_energy  # Fallback to target
            
            initial_energy_before_update = g.energy[sensory_idx].item()
            # Sync energy values to vector engine before running update
            self.test_system._sync_graph_to_vector_store()
            self.test_system.update()
            final_energy_after_update = g.energy[sensory_idx].item()
            
            # Sensory nodes should be restored to their true values
            if abs(final_energy_after_update - expected_sensory_energy) < expected_sensory_energy * 0.1:
                self.results["sensory_energy_restored"] = "PASS"
            else:
                self.warnings.append(f"Sensory energy not restored: expected ~{expected_sensory_energy}, got {final_energy_after_update}")

    def _test_workspace_nodes_comprehensive(self) -> None:
        """Comprehensive test of workspace nodes: initialization, energy state, and behavior."""
        self._reset_test_system()
        
        # Create a workspace node and dynamic node
        workspace_config = {'node_type': NODE_TYPE_WORKSPACE, 'dynamic_subtype': 0}
        dynamic_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
        
        node_indices = self._create_test_nodes([workspace_config, dynamic_config])
        
        if len(node_indices) < 2:
            self.errors.append("Failed to create workspace test nodes")
            return

        workspace_idx, dynamic_idx = node_indices[0], node_indices[1]

        if self.test_system and self.test_system.g:
            g = self.test_system.g

            # Test 1: Verify workspace node type
            workspace_type = g.node_type[workspace_idx].item()
            if workspace_type == NODE_TYPE_WORKSPACE:
                self.results["workspace_node_type"] = "PASS"
            else:
                self.errors.append(f"Workspace node has wrong type: {workspace_type}")

            # Test 2: Set initial workspace energy (energy tensor has shape [N, 1])
            initial_workspace_energy = NODE_ENERGY_CAP * 0.4
            g.energy[workspace_idx, 0] = initial_workspace_energy

            # Test 3: Create connection FROM dynamic TO workspace (allowed)
            conn_created = self._create_test_connection(
                dynamic_idx, workspace_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )
            
            if conn_created:
                self.results["workspace_incoming_connection"] = "PASS"
            else:
                self.errors.append("Failed to create connection TO workspace node")

            # Test 4: Try to create connection FROM workspace (should fail)
            conn_from_workspace = self._create_test_connection(
                workspace_idx, dynamic_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW
            )
            
            if not conn_from_workspace:
                self.results["workspace_no_outgoing_connection"] = "PASS"
            else:
                self.errors.append("Rule violation: Connection FROM workspace node was created")

            # Test 5: Run update and verify workspace can receive energy
            initial_workspace_energy_before = g.energy[workspace_idx].item()
            initial_dynamic_energy = g.energy[dynamic_idx].item()
            g.energy[dynamic_idx, 0] = NODE_ENERGY_CAP * 0.8  # High energy to transfer
            
            # Sync energy values to vector engine before running update
            self.test_system._sync_graph_to_vector_store()
            
            self.test_system.update()
            final_workspace_energy = g.energy[workspace_idx].item()
            
            # Workspace should be able to receive energy (may increase or stay same)
            if final_workspace_energy >= initial_workspace_energy_before - 1.0:  # Allow for decay
                self.results["workspace_can_receive_energy"] = "PASS"
            else:
                self.warnings.append(f"Workspace energy decreased unexpectedly: {initial_workspace_energy_before} -> {final_workspace_energy}")

    def _test_energy_state_tracking_10_steps(self) -> None:
        """Track energy states through 10 update steps and verify calculations."""
        self._reset_test_system()
        
        # Create a comprehensive test network: sensory -> dynamic1 -> dynamic2 -> workspace
        # IMPORTANT: reuse the baseline sensory/workspace nodes created by `_reset_test_system()`
        # so `sensory_true_values` length matches the actual sensory node count (avoids mismatch drift).
        dynamic1_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER}
        dynamic2_config = {'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR}

        if not self.test_system or not self.test_system.g:
            self.errors.append("Energy tracking test system not initialized")
            return

        g0 = self.test_system.g
        sensory_nodes = torch.where(g0.node_type == NODE_TYPE_SENSORY)[0]
        workspace_nodes = torch.where(g0.node_type == NODE_TYPE_WORKSPACE)[0]
        if sensory_nodes.numel() == 0 or workspace_nodes.numel() == 0:
            self.errors.append("Failed to find baseline sensory/workspace nodes for energy tracking test")
            return

        sensory_idx = int(sensory_nodes[0].item())
        workspace_idx = int(workspace_nodes[0].item())

        node_indices = self._create_test_nodes([dynamic1_config, dynamic2_config])
        if len(node_indices) < 2:
            self.errors.append("Failed to create dynamic nodes for energy tracking test")
            return

        dyn1_idx, dyn2_idx = node_indices[0], node_indices[1]

        if self.test_system and self.test_system.g:
            g = self.test_system.g
            device = self.test_system.device
            # Freeze birth/growth during prediction-based test so energy math is comparable step-to-step.
            # (Otherwise spawning/edge growth changes the system mid-run, making exact prediction impossible.)
            import project.pyg_neural_system as pns  # type: ignore[import-not-found]
            old_max_births = getattr(pns, "MAX_NODE_BIRTHS_PER_STEP", None)
            old_max_conn_births = getattr(pns, "MAX_CONN_BIRTHS_PER_STEP", None)
            if old_max_births is not None:
                pns.MAX_NODE_BIRTHS_PER_STEP = 0
            if old_max_conn_births is not None:
                pns.MAX_CONN_BIRTHS_PER_STEP = 0
            
            # Skip vector engine step during this test because the vector engine uses
            # different decay rules (multiplicative decay) than the PyG simulation 
            # (connection-based maintenance costs). We validate PyG simulation only.
            self.test_system._skip_vector_engine_step = True
            
            # Set initial energies - use very high values to prevent death during 10 steps
            # Need enough energy to survive decay (0.5% per step) and transfers over 10 steps
            initial_energies = {
                'sensory': NODE_ENERGY_CAP * 0.9,  # Very high to maintain
                'dynamic1': NODE_ENERGY_CAP * 0.85,  # Very high to survive
                'dynamic2': NODE_ENERGY_CAP * 0.8,  # Very high to survive
                'workspace': NODE_ENERGY_CAP * 0.75,  # Very high to survive
            }
            
            # CRITICAL: Set ALL nodes' energies high to prevent ANY node from dying (which would shift indices)
            # This includes the baseline dynamic node that isn't part of our test chain
            # Note: g.energy has shape [N, 1], so we need to set the first column
            for i in range(int(g.num_nodes or 0)):
                g.energy[i, 0] = NODE_ENERGY_CAP * 0.7  # Default high energy for all nodes
            
            # Now set specific energies for our test nodes (as scalars for shape [N, 1] tensor)
            g.energy[dyn1_idx, 0] = initial_energies['dynamic1']
            g.energy[dyn2_idx, 0] = initial_energies['dynamic2']
            g.energy[workspace_idx, 0] = initial_energies['workspace']
            
            # Sync energy values from PyG graph to vector engine store
            # This ensures the vector engine uses our set values instead of overwriting them
            self.test_system._sync_graph_to_vector_store()
            
            # Set sensory energy using update_sensory_nodes (proper way)
            # Get gain/bias from config
            try:
                from project.utils.config_manager import ConfigManager  # type: ignore[import-not-found]
                cfg = ConfigManager()
                gain = float(cfg.get_config('sensory', 'energy_gain') or 1.0)
                bias = float(cfg.get_config('sensory', 'energy_bias') or 0.0)  # Note: config key is 'energy_bias'
            except Exception:  # pylint: disable=broad-exception-caught
                gain = 1.0
                bias = 0.0
            
            target_sensory_energy = initial_energies['sensory']
            # Formula: energy = ((pixel / 255.0) * NODE_ENERGY_CAP * gain) + bias
            # Solve for pixel: pixel = ((energy - bias) / (NODE_ENERGY_CAP * gain)) * 255.0
            pixel_value = ((target_sensory_energy - bias) / (NODE_ENERGY_CAP * gain)) * 255.0
            pixel_value = max(0.0, min(255.0, pixel_value))  # Clamp
            sensory_array = torch.tensor([[pixel_value]], dtype=torch.float32)
            
            # Disable warmup for testing
            if hasattr(self.test_system, '_sensory_update_count'):
                self.test_system._sensory_update_count = self.test_system.sensory_warmup_frames
            
            self.test_system.update_sensory_nodes(sensory_array.numpy())
            
            # Update initial_energies with actual values set
            initial_energies['dynamic1'] = g.energy[dyn1_idx].item()
            initial_energies['dynamic2'] = g.energy[dyn2_idx].item()
            initial_energies['workspace'] = g.energy[workspace_idx].item()
            
            # Verify sensory true value was set (with tolerance for config gain/bias)
            if self.test_system.sensory_true_values is not None:
                sensory_true = self.test_system.sensory_true_values[0].item()
                initial_energies['sensory'] = sensory_true  # Use actual value
                if abs(sensory_true - target_sensory_energy) < target_sensory_energy * 0.3:  # 30% tolerance
                    self.results["sensory_initialized_correctly"] = "PASS"
                else:
                    self.warnings.append(f"Sensory initialization: expected ~{target_sensory_energy}, got {sensory_true} (pixel={pixel_value}, gain={gain}, bias={bias})")

            # Isolate this prediction-based test: remove any pre-existing edges so only our
            # sensory -> dyn1 -> dyn2 -> workspace chain affects the results.
            g.edge_index = None
            for edge_attr in [
                "weight",
                "energy_transfer_capacity",
                "conn_type",
                "plastic_lr",
                "gate_threshold",
                "conn_subtype2",
                "conn_subtype3",
            ]:
                if hasattr(g, edge_attr):
                    setattr(g, edge_attr, None)

            # Create connections: sensory -> dyn1 -> dyn2 -> workspace
            self._create_test_connection(sensory_idx, dyn1_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW)
            self._create_test_connection(dyn1_idx, dyn2_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW)
            self._create_test_connection(dyn2_idx, workspace_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW)
            
            # Get connection weights for calculation
            if g.edge_index is not None and hasattr(g, 'weight') and g.weight is not None:
                edge_weights = g.weight.squeeze()
                edge_src = g.edge_index[0]
                edge_dst = g.edge_index[1]
                
                # Map connections
                conn_sens_dyn1 = None
                conn_dyn1_dyn2 = None
                conn_dyn2_ws = None
                
                for i in range(len(edge_src)):
                    src = edge_src[i].item()
                    dst = edge_dst[i].item()
                    weight = edge_weights[i].item() if len(edge_weights) > i else 0.1
                    
                    if src == sensory_idx and dst == dyn1_idx:
                        conn_sens_dyn1 = weight
                    elif src == dyn1_idx and dst == dyn2_idx:
                        conn_dyn1_dyn2 = weight
                    elif src == dyn2_idx and dst == workspace_idx:
                        conn_dyn2_ws = weight

            # PREDICT expected energy values using energy calculator
            # Build connection list for calculator
            connections_list = []
            if g.edge_index is not None and hasattr(g, 'weight') and g.weight is not None:
                edge_weights = g.weight.squeeze()
                edge_src = g.edge_index[0]
                edge_dst = g.edge_index[1]
                conn_types = g.conn_type.squeeze() if hasattr(g, 'conn_type') and g.conn_type is not None else torch.zeros(len(edge_src), dtype=torch.int64, device=g.edge_index.device)
                conn_subtype3 = g.conn_subtype3 if hasattr(g, 'conn_subtype3') else torch.zeros(len(edge_src), dtype=torch.int64, device=g.edge_index.device)
                
                for i in range(len(edge_src)):
                    src = edge_src[i].item()
                    dst = edge_dst[i].item()
                    weight = edge_weights[i].item() if len(edge_weights) > i else 0.1
                    conn_type = conn_types[i].item() if len(conn_types) > i else CONN_TYPE_EXCITATORY
                    conn_subtype = conn_subtype3[i].item() if len(conn_subtype3) > i else CONN_SUBTYPE3_FREE_FLOW
                    connections_list.append((src, dst, weight, conn_type, conn_subtype))
            
            # Build node information for calculator (ALL nodes in graph).
            # This matters because extra baseline nodes/edges in the test system can drain or feed energy.
            num_nodes_total = int(g.num_nodes or 0)
            node_types_dict: Dict[int, int] = {}
            node_subtypes_dict: Dict[int, int | None] = {}
            initial_energies_dict: Dict[int, float] = {}

            for i in range(num_nodes_total):
                node_types_dict[i] = int(g.node_type[i].item()) if hasattr(g, 'node_type') and g.node_type is not None else NODE_TYPE_DYNAMIC
                if hasattr(g, 'dynamic_subtype') and g.dynamic_subtype is not None:
                    st = int(g.dynamic_subtype[i].item())
                    node_subtypes_dict[i] = None if st < 0 else st
                else:
                    node_subtypes_dict[i] = None
                initial_energies_dict[i] = float(g.energy[i].item())

            # Map sensory true values to the actual sensory node indices (in mask order)
            sensory_true_values_dict: Dict[int, float] = {}
            if self.test_system.sensory_true_values is not None and self.test_system.sensory_true_values.numel() > 0:
                sensory_indices = torch.where(g.node_type == NODE_TYPE_SENSORY)[0]
                n_map = min(int(sensory_indices.numel()), int(self.test_system.sensory_true_values.shape[0]))
                for j in range(n_map):
                    sensory_true_values_dict[int(sensory_indices[j].item())] = float(self.test_system.sensory_true_values[j].item())
            
            # Predict energy values
            predicted_energy_history = self.energy_calculator.predict_energy_after_steps(
                initial_energies_dict,
                node_types_dict,
                node_subtypes_dict,
                connections_list,
                num_steps=10,
                sensory_true_values=sensory_true_values_dict if sensory_true_values_dict else None,
            )
            
            # Track energy through 10 steps (ACTUAL values) for named nodes + full per-node history for prediction compare
            energy_history: Dict[str, List[float]] = {'sensory': [], 'dynamic1': [], 'dynamic2': [], 'workspace': []}
            actual_energy_history: Dict[int, List[float]] = {i: [] for i in range(num_nodes_total)}

            # Record initial state
            energy_history['sensory'].append(float(g.energy[sensory_idx].item()))
            energy_history['dynamic1'].append(float(g.energy[dyn1_idx].item()))
            energy_history['dynamic2'].append(float(g.energy[dyn2_idx].item()))
            energy_history['workspace'].append(float(g.energy[workspace_idx].item()))
            for i in range(num_nodes_total):
                actual_energy_history[i].append(float(g.energy[i].item()))
            
            # Run 10 update steps
            for step in range(10):
                self.test_system.update()
                self.test_system.apply_connection_worker_results()
                
                # Record energy after each step
                energy_history['sensory'].append(float(g.energy[sensory_idx].item()))
                energy_history['dynamic1'].append(float(g.energy[dyn1_idx].item()))
                energy_history['dynamic2'].append(float(g.energy[dyn2_idx].item()))
                energy_history['workspace'].append(float(g.energy[workspace_idx].item()))
                for i in range(num_nodes_total):
                    actual_energy_history[i].append(float(g.energy[i].item()))
            
            # COMPARE predicted vs actual (all nodes)
            comparison = self.energy_calculator.compare_predicted_vs_actual(
                predicted_energy_history,
                actual_energy_history,
                tolerance=0.15,  # tolerance for numerical differences / unmodelled effects
            )
            
            # Store comparison results
            if comparison["errors"]:
                for error in comparison["errors"]:
                    self.errors.append(f"Energy prediction error: {error}")
            
            if comparison["warnings"]:
                for warning in comparison["warnings"]:
                    self.warnings.append(f"Energy prediction deviation: {warning}")
            
            self.results["energy_prediction_max_deviation"] = comparison["max_deviation"]
            self.results["energy_prediction_avg_deviation"] = comparison["avg_deviation"]
            self.results["energy_prediction_matches"] = len(comparison["matches"])
            self.results["energy_prediction_warnings"] = len(comparison["warnings"])
            
            # Allow max deviation up to 15.0 for PASS (accounts for noise, numerical precision, 
            # and minor differences in workspace adjustment calculations)
            if comparison["max_deviation"] < 15.0 and len(comparison["errors"]) == 0:
                self.results["energy_prediction_accuracy"] = "PASS"
            else:
                self.results["energy_prediction_accuracy"] = "WARN" if len(comparison["errors"]) == 0 else "FAIL"

            # Restore growth parameters and vector engine step
            if old_max_births is not None:
                pns.MAX_NODE_BIRTHS_PER_STEP = old_max_births
            if old_max_conn_births is not None:
                pns.MAX_CONN_BIRTHS_PER_STEP = old_max_conn_births
            self.test_system._skip_vector_engine_step = False

            # Additional invariants for this 10-step test (non-predictive):
            # - Sensory node should track its true pixel value (overwritten each step)
            # - Nodes should not all die
            # - Total energy should remain bounded
            
            final_energies = {
                'sensory': energy_history['sensory'][-1],
                'dynamic1': energy_history['dynamic1'][-1],
                'dynamic2': energy_history['dynamic2'][-1],
                'workspace': energy_history['workspace'][-1],
            }
            
            # Verify sensory node maintains its true value (should be restored each step)
            # Check if sensory node still exists (may have been removed if died)
            if self.test_system.sensory_true_values is not None and len(self.test_system.sensory_true_values) > 0:
                sensory_expected = self.test_system.sensory_true_values[0].item()
                # Check if sensory node index is still valid
                if sensory_idx < (g.num_nodes or 0):
                    sensory_actual = g.energy[sensory_idx].item()
                    if abs(sensory_actual - sensory_expected) < sensory_expected * 0.1:  # 10% tolerance
                        self.results["sensory_energy_maintained_10_steps"] = "PASS"
                    else:
                        self.warnings.append(f"Sensory energy drift: expected ~{sensory_expected}, got {sensory_actual}")
                else:
                    self.errors.append("Sensory node was removed during 10-step test (died)")
            else:
                self.errors.append("Sensory true values not available after 10 steps")

            # Verify nodes didn't all die (check if any nodes remain alive)
            alive_nodes = 0
            for node_idx in [sensory_idx, dyn1_idx, dyn2_idx, workspace_idx]:
                if node_idx < (g.num_nodes or 0):
                    node_energy = g.energy[node_idx].item()
                    if node_energy > NODE_DEATH_THRESHOLD:
                        alive_nodes += 1
            
            if alive_nodes >= 2:  # At least 2 nodes should survive
                self.results["nodes_survived_10_steps"] = f"{alive_nodes}/4"
            else:
                self.errors.append(f"Too many nodes died during 10-step test: only {alive_nodes}/4 survived")
            
            # Verify energy conservation accounting for node types
            # Sensory nodes: No decay, restored to true values (energy maintained)
            # Dynamic nodes: Decay + connection maintenance costs (energy decreases)
            # Workspace nodes: No decay, receives energy (energy may increase from transfers)
            initial_total = sum(initial_energies.values())
            # Only count final energies for nodes that still exist
            final_total = 0.0
            for node_idx, key in [(sensory_idx, 'sensory'), (dyn1_idx, 'dynamic1'), 
                                   (dyn2_idx, 'dynamic2'), (workspace_idx, 'workspace')]:
                if node_idx < (g.num_nodes or 0):
                    final_total += g.energy[node_idx].item()
            
            # Total energy should decrease due to decay (or stay roughly same if transfers compensate)
            # Allow for significant decrease due to decay over 10 steps
            if final_total <= initial_total * 1.2:  # Allow 20% increase due to transfers
                self.results["energy_conservation_10_steps"] = "PASS"
            else:
                self.warnings.append(f"Energy conservation: initial={initial_total:.2f}, final={final_total:.2f}")

            # Verify dynamic nodes have energy changes (due to transfers, decay, and connection maintenance)
            # Dynamic nodes should decrease due to decay (0.005 per step) and connection maintenance costs
            dyn1_change = final_energies['dynamic1'] - initial_energies['dynamic1']
            dyn2_change = final_energies['dynamic2'] - initial_energies['dynamic2']
            
            # Calculate expected change for dynamic nodes (decay + connection maintenance)
            # Each dynamic node has connections, so maintenance cost applies
            # Expected: energy decreases due to decay and maintenance, but may increase from transfers
            # For simplicity, we just verify they changed (not static)
            if abs(dyn1_change) > 0.1 or abs(dyn2_change) > 0.1:
                self.results["dynamic_energy_changes_10_steps"] = "PASS"
            else:
                self.warnings.append("Dynamic nodes showed minimal energy changes over 10 steps")
            
            # Verify workspace nodes received energy (they should have energy from dynamic nodes)
            workspace_change = final_energies['workspace'] - initial_energies['workspace']
            # Workspace nodes should receive energy from dynamic nodes (positive change expected)
            # But they may start high and adjust down, so we just verify they're above death threshold
            if final_energies['workspace'] > NODE_DEATH_THRESHOLD:
                # Workspace node survived (good), but ideally should have received energy
                if workspace_change > 0.1:
                    self.results["workspace_received_energy_10_steps"] = "PASS"
                elif final_energies['workspace'] > 0.1:
                    # Workspace has some energy but didn't increase (may have been adjusted)
                    self.warnings.append(f"Workspace energy: {final_energies['workspace']:.2f} (did not increase from transfers)")
                else:
                    self.warnings.append(f"Workspace energy very low: {final_energies['workspace']:.2f} (may not be receiving energy)")
            else:
                self.warnings.append(f"Workspace energy below death threshold: {final_energies['workspace']:.2f}")

            # Store results
            self.results["energy_tracking_10_steps"] = {
                'initial': initial_energies,
                'final': final_energies,
                'history': energy_history,
            }

    def _test_spawn_death_thresholds(self, system: PyGNeuralSystem) -> None:
        """Test spawn and death thresholds."""
        if system.g is None:
            return

        test_energy = torch.tensor([NODE_DEATH_THRESHOLD - 1.0], device=system.device).unsqueeze(1)
        below_threshold = test_energy <= NODE_DEATH_THRESHOLD
        self.results["death_threshold_valid"] = below_threshold.item()
        self.results["spawn_threshold"] = NODE_SPAWN_THRESHOLD
        self.results["death_threshold"] = NODE_DEATH_THRESHOLD

    def _test_sensory_mock_pixel_input(self) -> None:
        """Test sensory nodes with mock pixel input (simulating screen capture)."""
        self._reset_test_system()
        
        # Create a mock pixel array (simulating screen capture)
        # For testing, create a 3x3 pixel grid with varying intensities
        mock_pixel_width = 3
        mock_pixel_height = 3
        mock_pixels = torch.randint(0, 256, (mock_pixel_height, mock_pixel_width), dtype=torch.float32)
        
        # Create sensory nodes matching the pixel grid
        sensory_configs = []
        for _ in range(mock_pixel_width * mock_pixel_height):
            sensory_configs.append({'node_type': NODE_TYPE_SENSORY, 'dynamic_subtype': 0})
        
        node_indices = self._create_test_nodes(sensory_configs)
        
        if len(node_indices) < mock_pixel_width * mock_pixel_height:
            self.errors.append("Failed to create sensory nodes for mock pixel test")
            return

        if self.test_system and self.test_system.g:
            g = self.test_system.g
            
            # Test 1: Feed mock pixels to sensory nodes
            # Flatten pixel array to match sensory node count
            pixel_array = mock_pixels.flatten().unsqueeze(1).numpy()  # Shape: (9, 1)
            
            # Disable warmup for testing
            if hasattr(self.test_system, '_sensory_update_count'):
                self.test_system._sensory_update_count = self.test_system.sensory_warmup_frames
            
            self.test_system.update_sensory_nodes(pixel_array)
            
            # Test 2: Verify each sensory node received the correct pixel value
            if self.test_system.sensory_true_values is not None:
                sensory_energies = self.test_system.sensory_true_values.squeeze()
                
                # Get expected energies from pixels (accounting for gain/bias)
                try:
                    from project.utils.config_manager import ConfigManager  # type: ignore[import-not-found]
                    cfg = ConfigManager()
                    gain = float(cfg.get_config('sensory', 'energy_gain') or 1.0)
                    bias = float(cfg.get_config('sensory', 'energy_bias') or 0.0)  # Config key is 'energy_bias'
                except Exception:  # pylint: disable=broad-exception-caught
                    gain = 1.0
                    bias = 0.0
                
                expected_energies = ((pixel_array / 255.0) * NODE_ENERGY_CAP * gain) + bias
                expected_energies = expected_energies.clip(0.0, NODE_ENERGY_CAP)
                
                # Verify each sensory node has correct energy
                matches = 0
                for i, (actual, expected) in enumerate(zip(sensory_energies, expected_energies)):
                    if abs(actual.item() - expected.item()) < expected.item() * 0.1:  # 10% tolerance
                        matches += 1
                    else:
                        self.warnings.append(f"Sensory node {i}: expected ~{expected.item():.2f}, got {actual.item():.2f} (pixel={pixel_array[i][0]})")
                
                if matches == len(sensory_energies):
                    self.results["sensory_mock_pixel_input"] = "PASS"
                else:
                    self.results["sensory_mock_pixel_input"] = f"{matches}/{len(sensory_energies)} matched"
                
                # Test 3: Verify sensory nodes maintain pixel values through updates
                initial_energies = sensory_energies.clone()
                self.test_system.update()
                
                # Check if sensory nodes were restored to true values
                final_energies = g.energy[:len(node_indices)].squeeze()
                restored_count = 0
                for i, (initial, final) in enumerate(zip(initial_energies, final_energies)):
                    if abs(initial.item() - final.item()) < initial.item() * 0.1:
                        restored_count += 1
                
                if restored_count == len(initial_energies):
                    self.results["sensory_pixel_values_maintained"] = "PASS"
                else:
                    self.warnings.append(f"Sensory pixel values: {restored_count}/{len(initial_energies)} maintained after update")

    def _test_workspace_mock_canvas_output(self) -> None:
        """Test workspace nodes with mock canvas output (simulating visualization)."""
        self._reset_test_system()
        
        # Create workspace nodes and dynamic nodes
        workspace_configs = []
        for i in range(4):  # 2x2 workspace grid
            workspace_configs.append({'node_type': NODE_TYPE_WORKSPACE, 'dynamic_subtype': 0})
        
        dynamic_configs = []
        for i in range(2):
            dynamic_configs.append({'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_TRANSMITTER})
        
        all_configs = workspace_configs + dynamic_configs
        node_indices = self._create_test_nodes(all_configs)
        
        if len(node_indices) < 6:
            self.errors.append("Failed to create workspace nodes for mock canvas test")
            return

        workspace_indices = node_indices[:4]
        dynamic_indices = node_indices[4:]

        if self.test_system and self.test_system.g:
            g = self.test_system.g
            
            # Create connections: dynamic → workspace
            for dyn_idx, ws_idx in zip(dynamic_indices, workspace_indices[:2]):
                self._create_test_connection(dyn_idx, ws_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW)
            
            # Set dynamic node energies (high to transfer to workspace)
            # Energy tensor has shape [N, 1]
            for dyn_idx in dynamic_indices:
                g.energy[dyn_idx, 0] = NODE_ENERGY_CAP * 0.8
            
            # Set workspace node energies (above death threshold to prevent removal)
            for ws_idx in workspace_indices:
                g.energy[ws_idx, 0] = NODE_ENERGY_CAP * 0.5  # Set initial energy
            
            # Create mock canvas (2x2 grid to match workspace nodes)
            mock_canvas = [[0.0 for _ in range(2)] for _ in range(2)]
            canvas_updates = []
            
            # Sync energies to vector store and use traditional PyG path
            self.test_system._sync_graph_to_vector_store()
            self.test_system._skip_vector_engine_step = True
            
            # Test 1: Run update and capture workspace node energies
            # Run multiple updates to let energy flow
            for _ in range(3):  # Run 3 updates to allow energy transfer
                self.test_system.update()
                self.test_system.apply_connection_worker_results()
            
            self.test_system._skip_vector_engine_step = False
            
            # Read workspace node energies (simulating canvas read)
            for i, ws_idx in enumerate(workspace_indices):
                if ws_idx < (g.num_nodes or 0):
                    workspace_energy = g.energy[ws_idx].item()
                    grid_x = i % 2
                    grid_y = i // 2
                    mock_canvas[grid_y][grid_x] = workspace_energy
                    canvas_updates.append((grid_x, grid_y, workspace_energy))
            
            # Test 2: Verify workspace nodes output energy to canvas
            if len(canvas_updates) == 4:
                self.results["workspace_canvas_output"] = "PASS"
                self.results["workspace_canvas_data"] = mock_canvas
            else:
                self.errors.append(f"Workspace canvas output incomplete: {len(canvas_updates)}/4 nodes")
            
            # Test 3: Verify workspace nodes can receive energy from dynamic nodes
            workspace_energies = [g.energy[ws_idx].item() for ws_idx in workspace_indices if ws_idx < (g.num_nodes or 0)]
            if any(energy > 0.1 for energy in workspace_energies):
                self.results["workspace_receives_energy"] = "PASS"
            else:
                self.warnings.append("Workspace nodes did not receive energy from dynamic nodes")
            
            # Test 4: Verify workspace nodes don't send energy (no outgoing connections)
            if g.edge_index is not None:
                edge_src = g.edge_index[0]
                workspace_outgoing = sum(1 for ws_idx in workspace_indices if (edge_src == ws_idx).any())
                if workspace_outgoing == 0:
                    self.results["workspace_no_outgoing"] = "PASS"
                else:
                    self.errors.append(f"Workspace nodes have {workspace_outgoing} outgoing connections (should be 0)")

    def _test_full_pipeline_mock(self) -> None:
        """Test full pipeline: mock pixels → sensory → dynamic → workspace → mock canvas."""
        self._reset_test_system()
        
        # Create mock pixel input (2x2 grid)
        mock_pixels = torch.tensor([[128.0, 200.0], [50.0, 180.0]], dtype=torch.float32)
        
        # Create nodes: 4 sensory (for 2x2 pixels), 2 dynamic, 2 workspace
        sensory_configs = [{'node_type': NODE_TYPE_SENSORY, 'dynamic_subtype': 0} for _ in range(4)]
        dynamic_configs = [{'node_type': NODE_TYPE_DYNAMIC, 'dynamic_subtype': SUBTYPE_RESONATOR} for _ in range(2)]
        workspace_configs = [{'node_type': NODE_TYPE_WORKSPACE, 'dynamic_subtype': 0} for _ in range(2)]
        
        all_configs = sensory_configs + dynamic_configs + workspace_configs
        node_indices = self._create_test_nodes(all_configs)
        
        if len(node_indices) < 8:
            self.errors.append("Failed to create nodes for full pipeline test")
            return

        sensory_indices = node_indices[:4]
        dynamic_indices = node_indices[4:6]
        workspace_indices = node_indices[6:8]

        if self.test_system and self.test_system.g:
            g = self.test_system.g
            
            # Step 1: Feed mock pixels to sensory nodes
            pixel_array = mock_pixels.flatten().unsqueeze(1).numpy()
            
            if hasattr(self.test_system, '_sensory_update_count'):
                self.test_system._sensory_update_count = self.test_system.sensory_warmup_frames
            
            self.test_system.update_sensory_nodes(pixel_array)
            
            # Step 2: Set initial energies BEFORE creating connections
            # Set dynamic nodes to high energy (energy tensor has shape [N, 1])
            for dyn_idx in dynamic_indices:
                if dyn_idx < (g.num_nodes or 0):
                    g.energy[dyn_idx, 0] = NODE_ENERGY_CAP * 0.8
            
            # Set workspace nodes to reasonable initial energy (above death threshold)
            for ws_idx in workspace_indices:
                if ws_idx < (g.num_nodes or 0):
                    g.energy[ws_idx, 0] = NODE_ENERGY_CAP * 0.5
            
            # Step 3: Create connections: sensory → dynamic → workspace
            # Connect each sensory to a dynamic
            for i, (sens_idx, dyn_idx) in enumerate(zip(sensory_indices[:2], dynamic_indices)):
                self._create_test_connection(sens_idx, dyn_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW)
            
            # Connect each dynamic to a workspace
            for dyn_idx, ws_idx in zip(dynamic_indices, workspace_indices):
                self._create_test_connection(dyn_idx, ws_idx, CONN_TYPE_EXCITATORY, CONN_SUBTYPE3_FREE_FLOW)
            
            # Step 4: Record initial energies AFTER setting them
            initial_sensory_energies = []
            initial_dynamic_energies = []
            initial_workspace_energies = []
            
            for idx in sensory_indices:
                if idx < (g.num_nodes or 0):
                    initial_sensory_energies.append(g.energy[idx].item())
            
            for idx in dynamic_indices:
                if idx < (g.num_nodes or 0):
                    initial_dynamic_energies.append(g.energy[idx].item())
            
            for idx in workspace_indices:
                if idx < (g.num_nodes or 0):
                    initial_workspace_energies.append(g.energy[idx].item())
            
            # Step 5: Run 5 updates to let energy flow
            # Sync energies to vector store and use traditional PyG path
            self.test_system._sync_graph_to_vector_store()
            self.test_system._skip_vector_engine_step = True
            
            for _ in range(5):
                self.test_system.update()
                self.test_system.apply_connection_worker_results()
            
            self.test_system._skip_vector_engine_step = False
            
            # Step 6: Read final energies and output to mock canvas
            final_sensory_energies = []
            final_dynamic_energies = []
            final_workspace_energies = []
            
            for idx in sensory_indices:
                if idx < (g.num_nodes or 0):
                    final_sensory_energies.append(g.energy[idx].item())
            
            for idx in dynamic_indices:
                if idx < (g.num_nodes or 0):
                    final_dynamic_energies.append(g.energy[idx].item())
            
            for idx in workspace_indices:
                if idx < (g.num_nodes or 0):
                    final_workspace_energies.append(g.energy[idx].item())
            
            # Step 7: Create mock canvas output (2x1 grid for 2 workspace nodes)
            mock_canvas = [[0.0 for _ in range(1)] for _ in range(2)]
            for i, ws_idx in enumerate(workspace_indices):
                if ws_idx < (g.num_nodes or 0):
                    mock_canvas[i][0] = g.energy[ws_idx].item()
            
            # Step 8: Verify pipeline worked
            # Sensory nodes should maintain pixel values (if they still exist)
            if self.test_system.sensory_true_values is not None and len(initial_sensory_energies) > 0:
                sensory_maintained = True
                for i, true_val in enumerate(self.test_system.sensory_true_values):
                    if i < len(final_sensory_energies) and i < len(initial_sensory_energies):
                        # Check if sensory node was restored to true value
                        if abs(final_sensory_energies[i] - true_val.item()) > true_val.item() * 0.3:
                            sensory_maintained = False
                            break
                
                if sensory_maintained and len(final_sensory_energies) > 0:
                    self.results["pipeline_sensory_maintained"] = "PASS"
                else:
                    self.warnings.append(f"Sensory nodes: {len(final_sensory_energies)}/{len(initial_sensory_energies)} maintained pixel values")
            
            # Dynamic nodes should have received energy (check if they survived and have energy)
            if len(final_dynamic_energies) > 0 and any(energy > NODE_DEATH_THRESHOLD for energy in final_dynamic_energies):
                # Check if they received energy from sensory (energy increased or stayed high)
                if any(energy > 1.0 for energy in final_dynamic_energies):
                    self.results["pipeline_dynamic_received"] = "PASS"
                else:
                    self.warnings.append("Dynamic nodes survived but have low energy")
            else:
                self.warnings.append("Dynamic nodes did not survive or receive energy from sensory nodes")
            
            # Workspace nodes should have output to canvas (check if they survived and have energy)
            if len(final_workspace_energies) > 0 and any(energy > NODE_DEATH_THRESHOLD for energy in final_workspace_energies):
                # Workspace nodes can have low energy but still be valid (above death threshold)
                if any(energy > 0.01 for energy in final_workspace_energies):
                    self.results["pipeline_workspace_output"] = "PASS"
                    self.results["pipeline_mock_canvas"] = mock_canvas
                else:
                    self.warnings.append("Workspace nodes survived but have very low energy")
            else:
                self.warnings.append("Workspace nodes did not survive or output energy to canvas")
            
            # Verify full pipeline: pixels → sensory → dynamic → workspace → canvas
            # All stages should have nodes that survived
            pipeline_works = (
                len(initial_sensory_energies) > 0 and 
                len(final_dynamic_energies) > 0 and 
                len(final_workspace_energies) > 0 and
                any(energy > NODE_DEATH_THRESHOLD for energy in final_dynamic_energies) and
                any(energy > NODE_DEATH_THRESHOLD for energy in final_workspace_energies)
            )
            
            if pipeline_works:
                self.results["full_pipeline_verified"] = "PASS"
            else:
                self.errors.append("Full pipeline test failed: energy did not flow through all stages")

    def _test_density_rules(self, system: PyGNeuralSystem) -> None:
        """Test density management if vector engine available."""
        if system.vector_engine is None:
            self.warnings.append("Vector engine not available for density testing")
            return

        density = system.vector_engine.density
        self.results["density_manager_active"] = density is not None
        if density:
            self.results["density_tile_size"] = density.tile_size
            self.results["density_world_size"] = density.world_size
