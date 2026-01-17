"""
Unit tests for workspace node system.
"""

import unittest
import time
from unittest.mock import Mock, MagicMock
import numpy as np

from src.project.workspace.workspace_system import WorkspaceNodeSystem
from src.project.workspace.config import EnergyReadingConfig
from src.project.workspace.workspace_node import WorkspaceNode


class MockNeuralSystem:
    """Mock neural system for testing."""
    
    def __init__(self):
        self.sensory_width = 16
        self.sensory_height = 16
        self.energies = np.random.rand(256) * 244.0
    
    def get_node_energy(self, node_id: int) -> float:
        """Mock energy getter."""
        if 0 <= node_id < len(self.energies):
            return float(self.energies[node_id])
        return 0.0


class TestWorkspaceNode(unittest.TestCase):
    """Test workspace node functionality."""
    
    def setUp(self):
        self.node = WorkspaceNode(node_id=0, grid_x=0, grid_y=0)
    
    def test_node_initialization(self):
        """Test workspace node initialization."""
        self.assertEqual(self.node.node_id, 0)
        self.assertEqual(self.node.grid_position, (0, 0))
        self.assertEqual(self.node.current_energy, 0.0)
        self.assertEqual(len(self.node.energy_history), 0)
    
    def test_energy_history_management(self):
        """Test energy history management."""
        # Add energy readings
        for i in range(10):
            self.node.update_energy(i * 10.0)
        
        # History should be managed
        self.assertLessEqual(len(self.node.energy_history), 100)
        self.assertEqual(self.node.current_energy, 90.0)
    
    def test_energy_trend_calculation(self):
        """Test energy trend calculation."""
        # Stable trend
        for i in range(5):
            self.node.update_energy(50.0)
        self.assertEqual(self.node.get_energy_trend(), "stable")
        
        # Increasing trend
        for i in range(10):
            self.node.update_energy(50.0 + i)
        self.assertEqual(self.node.get_energy_trend(), "increasing")
        
        # Decreasing trend
        for i in range(10):
            self.node.update_energy(50.0 - i)
        self.assertEqual(self.node.get_energy_trend(), "decreasing")


class TestWorkspaceSystem(unittest.TestCase):
    """Test workspace system functionality."""
    
    def setUp(self):
        self.mock_neural_system = MockNeuralSystem()
        self.config = EnergyReadingConfig()
        self.config.grid_size = (4, 4)  # Smaller grid for testing
        self.workspace_system = WorkspaceNodeSystem(
            self.mock_neural_system, self.config
        )
    
    def test_system_initialization(self):
        """Test workspace system initialization."""
        self.assertEqual(len(self.workspace_system.workspace_nodes), 16)
        self.assertEqual(len(self.workspace_system.mapping), 16)
    
    def test_sensory_mapping(self):
        """Test sensory-to-workspace mapping."""
        mapping = self.workspace_system.mapping
        
        # Verify mapping structure
        self.assertIsInstance(mapping, dict)
        self.assertEqual(len(mapping), 16)
        
        # Verify mapping content
        total_mapped_sensory = sum(len(sensory_list) for sensory_list in mapping.values())
        self.assertEqual(total_mapped_sensory, 256)  # 16x16 sensory grid
    
    def test_energy_reading(self):
        """Test energy reading functionality."""
        # Update workspace system
        self.workspace_system.update()
        
        # Verify energy reading
        node_energies = [node.current_energy for node in self.workspace_system.workspace_nodes]
        self.assertGreater(sum(node_energies), 0)
    
    def test_observer_pattern(self):
        """Test observer pattern implementation."""
        # Create mock observer
        mock_observer = Mock()
        
        # Add observer
        self.workspace_system.add_observer(mock_observer)
        
        # Update system
        self.workspace_system.update()
        
        # Verify observer was called
        self.assertTrue(mock_observer.on_workspace_update.called)
    
    def test_system_health(self):
        """Test system health monitoring."""
        health = self.workspace_system.get_system_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('running', health)
        self.assertIn('node_count', health)
        self.assertIn('mapping_coverage', health)


if __name__ == '__main__':
    unittest.main()