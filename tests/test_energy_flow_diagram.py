"""
Comprehensive tests for EnergyFlowDiagram.

This module contains unit tests, integration tests, edge cases, and performance tests
for the EnergyFlowDiagram class, covering diagram creation, visualization, and analysis.
"""

import unittest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.energy.energy_flow_diagram import EnergyFlowDiagram, create_energy_flow_visualizations


class TestEnergyFlowDiagramInitialization(unittest.TestCase):
    """Unit tests for EnergyFlowDiagram initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_initialization(self):
        """Test diagram initialization."""
        self.assertIsInstance(self.diagram.G, type(self.diagram.G))
        self.assertIsInstance(self.diagram.node_positions, dict)
        self.assertIsInstance(self.diagram.node_colors, dict)
        self.assertIsInstance(self.diagram.edge_colors, dict)
        self.assertIsInstance(self.diagram.node_sizes, dict)

    def test_initialization_empty_collections(self):
        """Test that collections are initialized empty."""
        self.assertEqual(len(self.diagram.node_positions), 0)
        self.assertEqual(len(self.diagram.node_colors), 0)
        self.assertEqual(len(self.diagram.edge_colors), 0)
        self.assertEqual(len(self.diagram.node_sizes), 0)


class TestSystemArchitectureDiagram(unittest.TestCase):
    """Unit tests for system architecture diagram creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_create_system_architecture_diagram(self):
        """Test system architecture diagram creation."""
        self.diagram.create_system_architecture_diagram()

        # Check that nodes were added
        self.assertGreater(len(self.diagram.G.nodes()), 0)
        self.assertGreater(len(self.diagram.node_positions), 0)

        # Check expected layers
        expected_layers = ['input', 'conversion', 'distribution', 'processing', 'output']
        node_labels = [self.diagram.G.nodes[node]['label'] for node in self.diagram.G.nodes()]

        # Should contain expected node types
        self.assertIn('Visual Input', node_labels)
        self.assertIn('Neural Dynamics', node_labels)
        self.assertIn('Behaviors', node_labels)

    def test_system_architecture_node_positions(self):
        """Test that nodes have proper positions."""
        self.diagram.create_system_architecture_diagram()

        # Check that all nodes have positions
        for node in self.diagram.G.nodes():
            self.assertIn(node, self.diagram.node_positions)
            pos = self.diagram.node_positions[node]
            self.assertIsInstance(pos, tuple)
            self.assertEqual(len(pos), 2)

    def test_system_architecture_colors(self):
        """Test node coloring in system architecture."""
        self.diagram.create_system_architecture_diagram()

        # Check that nodes have colors
        for node in self.diagram.G.nodes():
            self.assertIn(node, self.diagram.node_colors)
            color = self.diagram.node_colors[node]
            self.assertIsInstance(color, str)

    def test_system_architecture_edges(self):
        """Test edge creation in system architecture."""
        self.diagram.create_system_architecture_diagram()

        # Should have edges
        self.assertGreater(len(self.diagram.G.edges()), 0)

        # Check edge colors
        for edge in self.diagram.G.edges():
            self.assertIn(edge, self.diagram.edge_colors)

    def test_system_architecture_feedback_loops(self):
        """Test feedback loops in system architecture."""
        self.diagram.create_system_architecture_diagram()

        # Should have feedback edges (different color)
        feedback_edges = [edge for edge, color in self.diagram.edge_colors.items()
                         if color == 'darkred']
        self.assertGreater(len(feedback_edges), 0)


class TestEnergyBehaviorDiagram(unittest.TestCase):
    """Unit tests for energy behavior diagram creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_create_energy_behavior_diagram(self):
        """Test energy behavior diagram creation."""
        self.diagram.create_energy_behavior_diagram()

        # Check behaviors are present
        node_labels = [self.diagram.G.nodes[node]['label'] for node in self.diagram.G.nodes()]
        expected_behaviors = ['Oscillator', 'Integrator', 'Relay', 'Highway', 'Dynamic']

        for behavior in expected_behaviors:
            self.assertIn(behavior, node_labels)

        # Check sensory input
        self.assertIn('Sensory\nInput', node_labels)

    def test_behavior_diagram_connections(self):
        """Test connections in behavior diagram."""
        self.diagram.create_energy_behavior_diagram()

        # Should have connections between behaviors
        edges = list(self.diagram.G.edges())
        self.assertGreater(len(edges), 0)

        # Check that sensory input connects to all behaviors
        sensory_connections = [edge for edge in edges if 'sensory_input' in str(edge)]
        self.assertEqual(len(sensory_connections), 5)  # One to each behavior

    def test_behavior_diagram_colors(self):
        """Test coloring in behavior diagram."""
        self.diagram.create_energy_behavior_diagram()

        # Behavior nodes should have specific color
        for node in self.diagram.G.nodes():
            if self.diagram.G.nodes[node]['label'] in ['Oscillator', 'Integrator', 'Relay', 'Highway', 'Dynamic']:
                self.assertEqual(self.diagram.node_colors[node], 'lightcoral')


class TestLearningIntegrationDiagram(unittest.TestCase):
    """Unit tests for learning integration diagram creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_create_learning_integration_diagram(self):
        """Test learning integration diagram creation."""
        self.diagram.create_learning_integration_diagram()

        node_labels = [self.diagram.G.nodes[node]['label'] for node in self.diagram.G.nodes()]

        # Check learning components
        expected_components = ['STDP Engine', 'Hebbian Learning', 'Eligibility Traces',
                             'Weight Updates', 'Memory Consolidation', 'Homeostasis Controller', 'Energy\nSource']

        for component in expected_components:
            self.assertIn(component, node_labels)

    def test_learning_integration_flow(self):
        """Test learning flow connections."""
        self.diagram.create_learning_integration_diagram()

        # Check energy to learning connections
        energy_edges = [edge for edge in self.diagram.G.edges() if 'energy_source' in str(edge)]
        self.assertEqual(len(energy_edges), 6)  # Energy to each learning component

        # Check learning flow
        learning_flow_edges = []
        for edge in self.diagram.G.edges():
            source_label = self.diagram.G.nodes[edge[0]]['label']
            target_label = self.diagram.G.nodes[edge[1]]['label']
            if (source_label in ['STDP Engine', 'Hebbian Learning', 'Eligibility Traces', 'Weight Updates'] and
                target_label in ['Weight Updates', 'Memory Consolidation']):
                learning_flow_edges.append(edge)

        self.assertGreater(len(learning_flow_edges), 0)


class TestDiagramDrawing(unittest.TestCase):
    """Unit tests for diagram drawing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any created files
        import glob
        for pattern in ['*.png', '*_diagram.png']:
            for file in glob.glob(os.path.join(self.temp_dir, pattern)):
                try:
                    os.remove(file)
                except:
                    pass

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_draw_diagram_basic(self, mock_show, mock_savefig):
        """Test basic diagram drawing."""
        self.diagram.create_system_architecture_diagram()
        self.diagram.draw_diagram("Test Diagram")

        # Should call savefig and show
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_draw_diagram_with_custom_title(self, mock_show, mock_savefig, mock_figure):
        """Test diagram drawing with custom title."""
        mock_fig = Mock()
        mock_figure.return_value = mock_fig

        self.diagram.create_energy_behavior_diagram()
        self.diagram.draw_diagram("Custom Test Title")

        mock_savefig.assert_called_once_with('custom_test_title.png',
                                           dpi=300, bbox_inches='tight')

    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_labels')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_draw_diagram_networkx_calls(self, mock_show, mock_savefig, mock_labels,
                                        mock_edges, mock_nodes):
        """Test that NetworkX drawing functions are called correctly."""
        self.diagram.create_system_architecture_diagram()
        self.diagram.draw_diagram()

        # Should call all NetworkX drawing functions
        mock_nodes.assert_called_once()
        mock_edges.assert_called_once()
        mock_labels.assert_called_once()

    def test_draw_empty_diagram(self):
        """Test drawing empty diagram."""
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'):
            # Should handle empty diagram gracefully
            self.diagram.draw_diagram("Empty Diagram")


class TestCentralityAnalysis(unittest.TestCase):
    """Unit tests for centrality analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_create_energy_centrality_analysis(self):
        """Test centrality analysis creation."""
        self.diagram.create_system_architecture_diagram()
        result = self.diagram.create_energy_centrality_analysis()

        # Should return centrality measures
        expected_keys = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], dict)

    def test_centrality_analysis_with_empty_graph(self):
        """Test centrality analysis with empty graph."""
        # Should handle empty graph
        result = self.diagram.create_energy_centrality_analysis()

        # Should return empty dicts
        self.assertIsInstance(result, dict)

    def test_centrality_analysis_values(self):
        """Test that centrality values are reasonable."""
        self.diagram.create_system_architecture_diagram()
        result = self.diagram.create_energy_centrality_analysis()

        # Check that values are between 0 and 1
        for centrality_type in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']:
            for value in result[centrality_type].values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestCreateEnergyFlowVisualizations(unittest.TestCase):
    """Unit tests for create_energy_flow_visualizations function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import glob
        for file in glob.glob(os.path.join(self.temp_dir, '*.png')):
            try:
                os.remove(file)
            except:
                pass

    @patch('energy.energy_flow_diagram.EnergyFlowDiagram')
    @patch('builtins.print')
    def test_create_energy_flow_visualizations_success(self, mock_print, mock_diagram_class):
        """Test successful visualization creation."""
        # Mock diagram instance
        mock_diagram = Mock()
        mock_diagram_class.return_value = mock_diagram
        mock_diagram.create_energy_centrality_analysis.return_value = {
            'degree_centrality': {}, 'betweenness_centrality': {}, 'closeness_centrality': {}
        }

        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'):
            result = create_energy_flow_visualizations()

        # Should create 3 diagrams
        self.assertEqual(mock_diagram_class.call_count, 3)
        self.assertEqual(mock_diagram.create_system_architecture_diagram.call_count, 1)
        self.assertEqual(mock_diagram.create_energy_behavior_diagram.call_count, 1)
        self.assertEqual(mock_diagram.create_learning_integration_diagram.call_count, 1)
        self.assertEqual(mock_diagram.draw_diagram.call_count, 3)
        self.assertEqual(mock_diagram.create_energy_centrality_analysis.call_count, 1)

        # Should return centrality analysis
        self.assertIsInstance(result, dict)

    @patch('builtins.print')
    def test_create_energy_flow_visualizations_matplotlib_error(self, mock_print):
        """Test visualization creation with matplotlib error."""
        with patch.dict('sys.modules', {'matplotlib.pyplot': None}):
            with self.assertRaises(ImportError):
                create_energy_flow_visualizations()

    @patch('builtins.print')
    def test_create_energy_flow_visualizations_networkx_error(self, mock_print):
        """Test visualization creation with networkx error."""
        with patch.dict('sys.modules', {'networkx': None}):
            with self.assertRaises(ImportError):
                create_energy_flow_visualizations()


class TestEnergyFlowDiagramIntegration(unittest.TestCase):
    """Integration tests for EnergyFlowDiagram."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_full_diagram_workflow(self):
        """Test complete diagram creation and analysis workflow."""
        # Create all diagram types
        self.diagram.create_system_architecture_diagram()
        self.diagram.create_energy_behavior_diagram()
        self.diagram.create_learning_integration_diagram()

        # Should have combined all diagrams
        self.assertGreater(len(self.diagram.G.nodes()), 10)
        self.assertGreater(len(self.diagram.G.edges()), 10)

        # Perform centrality analysis
        centrality = self.diagram.create_energy_centrality_analysis()
        self.assertIsInstance(centrality, dict)

        # Test drawing (with mocked matplotlib)
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'):
            self.diagram.draw_diagram("Integration Test")

    def test_diagram_data_consistency(self):
        """Test that diagram data structures are consistent."""
        self.diagram.create_system_architecture_diagram()

        # All nodes should have positions, colors, and sizes
        for node in self.diagram.G.nodes():
            self.assertIn(node, self.diagram.node_positions)
            self.assertIn(node, self.diagram.node_colors)
            self.assertIn(node, self.diagram.node_sizes)

        # All edges should have colors
        for edge in self.diagram.G.edges():
            self.assertIn(edge, self.diagram.edge_colors)

    def test_multiple_diagram_creation(self):
        """Test creating multiple diagrams in sequence."""
        diagrams = []

        # Create separate diagram instances
        for i in range(3):
            diagram = EnergyFlowDiagram()
            if i == 0:
                diagram.create_system_architecture_diagram()
            elif i == 1:
                diagram.create_energy_behavior_diagram()
            else:
                diagram.create_learning_integration_diagram()
            diagrams.append(diagram)

        # Each should have different content
        node_counts = [len(d.G.nodes()) for d in diagrams]
        self.assertEqual(len(set(node_counts)), 3)  # All different


class TestEnergyFlowDiagramEdgeCases(unittest.TestCase):
    """Edge case tests for EnergyFlowDiagram."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_draw_diagram_with_special_characters(self):
        """Test drawing diagram with special characters in title."""
        self.diagram.create_system_architecture_diagram()

        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.show'):
            self.diagram.draw_diagram("Test: Diagram (with) Special-Chars_123")

            # Should sanitize filename
            mock_savefig.assert_called_once()
            args = mock_savefig.call_args
            filename = args[0][0]
            self.assertTrue(filename.endswith('.png'))
            self.assertNotIn(':', filename)
            self.assertNotIn('(', filename)

    def test_centrality_analysis_with_isolated_nodes(self):
        """Test centrality analysis with isolated nodes."""
        # Create diagram with isolated nodes
        self.diagram.G.add_node('isolated1', label='Isolated 1')
        self.diagram.G.add_node('isolated2', label='Isolated 2')
        self.diagram.node_positions['isolated1'] = (0, 0)
        self.diagram.node_positions['isolated2'] = (1, 1)
        self.diagram.node_colors['isolated1'] = 'blue'
        self.diagram.node_colors['isolated2'] = 'red'
        self.diagram.node_sizes['isolated1'] = 1000
        self.diagram.node_sizes['isolated2'] = 1000

        # Should handle isolated nodes
        result = self.diagram.create_energy_centrality_analysis()
        self.assertIsInstance(result, dict)

    def test_diagram_with_no_edges(self):
        """Test diagram with nodes but no edges."""
        self.diagram.G.add_node('node1', label='Node 1')
        self.diagram.G.add_node('node2', label='Node 2')
        self.diagram.node_positions['node1'] = (0, 0)
        self.diagram.node_positions['node2'] = (1, 1)
        self.diagram.node_colors['node1'] = 'blue'
        self.diagram.node_colors['node2'] = 'red'
        self.diagram.node_sizes['node1'] = 1000
        self.diagram.node_sizes['node2'] = 1000

        # Should handle gracefully
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'):
            self.diagram.draw_diagram("No Edges Test")

    def test_centrality_with_single_node(self):
        """Test centrality analysis with single node."""
        self.diagram.G.add_node('single', label='Single Node')
        self.diagram.node_positions['single'] = (0, 0)
        self.diagram.node_colors['single'] = 'blue'
        self.diagram.node_sizes['single'] = 1000

        result = self.diagram.create_energy_centrality_analysis()
        self.assertIsInstance(result, dict)

        # Single node should have centrality of 0 (or undefined, but not crash)
        self.assertIn('single', result['degree_centrality'])


class TestEnergyFlowDiagramPerformance(unittest.TestCase):
    """Performance tests for EnergyFlowDiagram."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_large_diagram_creation_performance(self):
        """Test performance of creating large diagrams."""
        import time

        start_time = time.time()
        self.diagram.create_system_architecture_diagram()
        creation_time = time.time() - start_time

        # Should create quickly (< 0.1 seconds)
        self.assertLess(creation_time, 0.1)

    def test_centrality_analysis_performance(self):
        """Test performance of centrality analysis."""
        self.diagram.create_system_architecture_diagram()

        import time
        start_time = time.time()
        result = self.diagram.create_energy_centrality_analysis()
        analysis_time = time.time() - start_time

        # Should analyze quickly (< 0.1 seconds)
        self.assertLess(analysis_time, 0.1)
        self.assertIsInstance(result, dict)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_drawing_performance(self, mock_show, mock_savefig):
        """Test performance of diagram drawing."""
        self.diagram.create_system_architecture_diagram()

        import time
        start_time = time.time()
        self.diagram.draw_diagram("Performance Test")
        drawing_time = time.time() - start_time

        # Should draw quickly (< 0.5 seconds)
        self.assertLess(drawing_time, 0.5)


class TestEnergyFlowDiagramRealWorldUsage(unittest.TestCase):
    """Real-world usage tests for EnergyFlowDiagram."""

    def setUp(self):
        """Set up test fixtures."""
        self.diagram = EnergyFlowDiagram()

    def test_system_architecture_for_documentation(self):
        """Test system architecture diagram for documentation purposes."""
        self.diagram.create_system_architecture_diagram()

        # Should have all expected layers
        layers_found = set()
        for node in self.diagram.G.nodes():
            node_data = self.diagram.G.nodes[node]
            if 'layer' in node_data:
                layers_found.add(node_data['layer'])

        expected_layers = {'input', 'conversion', 'distribution', 'processing', 'output'}
        self.assertTrue(expected_layers.issubset(layers_found))

    def test_behavior_diagram_for_analysis(self):
        """Test behavior diagram for behavioral analysis."""
        self.diagram.create_energy_behavior_diagram()

        # Should show energy flow between behaviors
        behavior_nodes = []
        for node in self.diagram.G.nodes():
            label = self.diagram.G.nodes[node]['label']
            if label in ['Oscillator', 'Integrator', 'Relay', 'Highway', 'Dynamic']:
                behavior_nodes.append(node)

        # Should have connections forming a cycle
        edges = list(self.diagram.G.edges())
        behavior_edges = []
        for edge in edges:
            source_label = self.diagram.G.nodes[edge[0]]['label']
            target_label = self.diagram.G.nodes[edge[1]]['label']
            if (source_label in ['Oscillator', 'Integrator', 'Relay', 'Highway', 'Dynamic'] and
                target_label in ['Oscillator', 'Integrator', 'Relay', 'Highway', 'Dynamic']):
                behavior_edges.append(edge)

        # Should have cyclic connections
        self.assertGreater(len(behavior_edges), 3)

    def test_learning_integration_for_research(self):
        """Test learning integration diagram for research purposes."""
        self.diagram.create_learning_integration_diagram()

        # Should show energy-driven learning components
        learning_components = []
        for node in self.diagram.G.nodes():
            label = self.diagram.G.nodes[node]['label']
            if 'Learning' in label or 'STDP' in label or 'Weight' in label or 'Memory' in label:
                learning_components.append(label)

        expected_components = ['STDP Engine', 'Hebbian Learning', 'Weight Updates', 'Memory Consolidation']
        for component in expected_components:
            self.assertIn(component, learning_components)

    def test_visualization_export_workflow(self):
        """Test complete visualization export workflow."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.show') as mock_show:

            # Create and save all visualizations
            centrality = create_energy_flow_visualizations()

            # Should have created 3 PNG files
            self.assertEqual(mock_savefig.call_count, 3)

            # Should have performed centrality analysis
            self.assertIsInstance(centrality, dict)
            self.assertIn('degree_centrality', centrality)
            self.assertIn('betweenness_centrality', centrality)
            self.assertIn('closeness_centrality', centrality)

    def test_diagram_analysis_for_system_design(self):
        """Test diagram analysis for system design decisions."""
        self.diagram.create_system_architecture_diagram()
        centrality = self.diagram.create_energy_centrality_analysis()

        # Find most central nodes
        degree_cent = centrality['degree_centrality']
        betweenness_cent = centrality['betweenness_centrality']

        if degree_cent:
            most_connected = max(degree_cent.items(), key=lambda x: x[1])
            most_central_bridge = max(betweenness_cent.items(), key=lambda x: x[1])

            # Most connected node should be a processing or distribution node
            most_connected_label = self.diagram.G.nodes[most_connected[0]]['label']
            self.assertIn(most_connected_label, ['Neural Dynamics', 'Connection Logic',
                                               'Dynamic Nodes', 'Enhanced Nodes'])

            # Most central bridge should facilitate energy flow
            bridge_label = self.diagram.G.nodes[most_central_bridge[0]]['label']
            # Could be any key component in the flow


if __name__ == '__main__':
    unittest.main()






