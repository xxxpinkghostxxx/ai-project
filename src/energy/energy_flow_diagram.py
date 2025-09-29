"""
Energy Flow Diagram Generator
Creates visual representations of energy flow through the neural simulation system.
"""

import re


class EnergyFlowDiagram:
    """Generates visual diagrams of energy flow in the neural system."""

    def __init__(self):
        import networkx as nx
        self.G = nx.DiGraph()
        self.node_positions = {}
        self.node_colors = {}
        self.edge_colors = {}
        self.node_sizes = {}

    def create_system_architecture_diagram(self):
        """Create the overall system architecture with energy flow."""

        # Define system layers
        layers = {
            'input': ['Visual Input', 'Audio Input', 'Sensor Input'],
            'conversion': ['Visual→Energy Bridge', 'Audio→Energy Bridge', 'Sensor→Energy Bridge'],
            'distribution': ['Sensory Nodes', 'Dynamic Nodes', 'Enhanced Nodes'],
            'processing': ['Neural Dynamics', 'Learning Engine', 'Connection Logic'],
            'output': ['Behaviors', 'Memory', 'Actions']
        }

        # Add nodes with positions
        y_positions = [4, 3, 2, 1, 0]
        for i, (layer_name, nodes) in enumerate(layers.items()):
            for j, node in enumerate(nodes):
                node_id = f"{layer_name}_{j}"
                self.G.add_node(node_id, label=node, layer=layer_name)
                self.node_positions[node_id] = (j * 2, y_positions[i])

                # Color coding by layer
                if layer_name == 'input':
                    self.node_colors[node_id] = 'lightblue'
                elif layer_name == 'conversion':
                    self.node_colors[node_id] = 'lightgreen'
                elif layer_name == 'distribution':
                    self.node_colors[node_id] = 'orange'
                elif layer_name == 'processing':
                    self.node_colors[node_id] = 'red'
                else:
                    self.node_colors[node_id] = 'purple'

                self.node_sizes[node_id] = 2000

        # Add energy flow edges
        energy_flows = [
            # Input to conversion
            ('input_0', 'conversion_0'), ('input_1', 'conversion_1'), ('input_2', 'conversion_2'),

            # Conversion to distribution
            ('conversion_0', 'distribution_0'), ('conversion_1', 'distribution_0'),
            ('conversion_2', 'distribution_0'), ('conversion_0', 'distribution_1'),
            ('conversion_1', 'distribution_1'), ('conversion_2', 'distribution_1'),

            # Distribution to processing
            ('distribution_0', 'processing_0'), ('distribution_1', 'processing_0'),
            ('distribution_2', 'processing_0'), ('distribution_0', 'processing_1'),
            ('distribution_1', 'processing_1'), ('distribution_2', 'processing_1'),
            ('distribution_0', 'processing_2'), ('distribution_1', 'processing_2'),
            ('distribution_2', 'processing_2'),

            # Processing to output
            ('processing_0', 'output_0'), ('processing_1', 'output_0'),
            ('processing_2', 'output_0'), ('processing_0', 'output_1'),
            ('processing_1', 'output_1'), ('processing_2', 'output_1'),
            ('processing_0', 'output_2'), ('processing_1', 'output_2'),
            ('processing_2', 'output_2'),
        ]

        for source, target in energy_flows:
            self.G.add_edge(source, target, weight=2)
            self.edge_colors[(source, target)] = 'darkblue'

        # Add feedback loops
        feedback_loops = [
            ('output_0', 'distribution_1'), ('output_1', 'distribution_2'),
            ('processing_1', 'conversion_0'), ('processing_2', 'conversion_1'),
        ]

        for source, target in feedback_loops:
            self.G.add_edge(source, target, weight=1)
            self.edge_colors[(source, target)] = 'darkred'

    def create_energy_behavior_diagram(self):
        """Create diagram showing different energy behaviors."""

        behaviors = [
            'Oscillator', 'Integrator', 'Relay', 'Highway', 'Dynamic'
        ]

        # Add behavior nodes
        for i, behavior in enumerate(behaviors):
            node_id = f"behavior_{i}"
            self.G.add_node(node_id, label=behavior, type='behavior')
            self.node_positions[node_id] = (i * 2, 2)
            self.node_colors[node_id] = 'lightcoral'
            self.node_sizes[node_id] = 1500

        # Add energy flow between behaviors
        behavior_connections = [
            ('behavior_0', 'behavior_1'), ('behavior_1', 'behavior_2'),
            ('behavior_2', 'behavior_3'), ('behavior_3', 'behavior_4'),
            ('behavior_4', 'behavior_0'),  # Cycle back
        ]

        for source, target in behavior_connections:
            self.G.add_edge(source, target, weight=1.5)
            self.edge_colors[(source, target)] = 'darkgreen'

        # Add sensory input
        self.G.add_node('sensory_input', label='Sensory\nInput', type='input')
        self.node_positions['sensory_input'] = (-2, 3)
        self.node_colors['sensory_input'] = 'lightblue'
        self.node_sizes['sensory_input'] = 1200

        # Connect sensory to all behaviors
        for i in range(5):
            self.G.add_edge('sensory_input', f'behavior_{i}')
            self.edge_colors[('sensory_input', f'behavior_{i}')] = 'blue'

    def create_learning_integration_diagram(self):
        """Create diagram showing energy integration with learning."""

        # Learning components
        learning_nodes = [
            'STDP Engine', 'Hebbian Learning', 'Eligibility Traces',
            'Weight Updates', 'Memory Consolidation', 'Homeostasis Controller'
        ]

        # Add learning nodes
        for i, node in enumerate(learning_nodes):
            node_id = f"learning_{i}"
            self.G.add_node(node_id, label=node, type='learning')
            self.node_positions[node_id] = (i * 1.5, 1)
            self.node_colors[node_id] = 'gold'
            self.node_sizes[node_id] = 1800

        # Add energy source
        self.G.add_node('energy_source', label='Energy\nSource', type='energy')
        self.node_positions['energy_source'] = (-1, 2)
        self.node_colors['energy_source'] = 'red'
        self.node_sizes['energy_source'] = 1000

        # Connect energy to learning
        for i in range(len(learning_nodes)):
            self.G.add_edge('energy_source', f'learning_{i}')
            self.edge_colors[('energy_source', f'learning_{i}')] = 'red'

        # Add learning flow
        learning_flow = [
            ('learning_0', 'learning_3'), ('learning_1', 'learning_3'),
            ('learning_2', 'learning_3'), ('learning_3', 'learning_4'),
        ]

        for source, target in learning_flow:
            self.G.add_edge(source, target, weight=2)
            self.edge_colors[(source, target)] = 'orange'

    def draw_diagram(self, title: str = "Energy Flow Architecture"):
        """Draw the energy flow diagram."""

        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure(figsize=(15, 10))

        # Get node colors and sizes
        node_color_list = [self.node_colors.get(node, 'lightgray') for node in self.G.nodes()]
        node_size_list = [self.node_sizes.get(node, 1000) for node in self.G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, self.node_positions,
            node_color=node_color_list,
            node_size=node_size_list,
            alpha=0.8
        )

        # Draw edges
        edges = list(self.G.edges())
        edge_colors = [self.edge_colors.get(edge, 'gray') for edge in edges]

        nx.draw_networkx_edges(
            self.G, self.node_positions,
            edgelist=edges,
            edge_color=edge_colors,
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=20
        )

        # Draw labels
        labels = {node: self.G.nodes[node]['label'] for node in self.G.nodes()}
        nx.draw_networkx_labels(
            self.G, self.node_positions,
            labels,
            font_size=8,
            font_weight='bold'
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        filename = re.sub(r'[^a-zA-Z0-9_]', '_', title.lower()) + '.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def create_energy_centrality_analysis(self):
        """Analyze how central energy is to the system."""

        import networkx as nx

        if len(self.G.nodes()) == 0:
            return {
                'degree_centrality': {},
                'betweenness_centrality': {},
                'closeness_centrality': {}
            }

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.G)
        betweenness_centrality = nx.betweenness_centrality(self.G)
        closeness_centrality = nx.closeness_centrality(self.G)

        # Find most central nodes
        most_degree = max(degree_centrality.items(), key=lambda x: x[1])
        most_betweenness = max(betweenness_centrality.items(), key=lambda x: x[1])
        most_closeness = max(closeness_centrality.items(), key=lambda x: x[1])

        print("Energy Centrality Analysis:")
        print("=" * 50)
        print(f"Most Connected Node: {self.G.nodes[most_degree[0]]['label']} (degree: {most_degree[1]:.3f})")
        print(f"Most Central Bridge: {self.G.nodes[most_betweenness[0]]['label']} (betweenness: {most_betweenness[1]:.3f})")
        print(f"Most Efficient Hub: {self.G.nodes[most_closeness[0]]['label']} (closeness: {most_closeness[1]:.3f})")

        return {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality
        }

def create_energy_flow_visualizations():
    """Create all energy flow visualizations."""

    import matplotlib.pyplot as plt
    import networkx as nx

    print("Creating Energy Flow Visualizations...")
    print("=" * 50)

    # 1. System Architecture
    print("1. Creating System Architecture Diagram...")
    diagram1 = EnergyFlowDiagram()
    diagram1.create_system_architecture_diagram()
    diagram1.draw_diagram("Energy Flow System Architecture")

    # 2. Energy Behaviors
    print("2. Creating Energy Behavior Diagram...")
    diagram2 = EnergyFlowDiagram()
    diagram2.create_energy_behavior_diagram()
    diagram2.draw_diagram("Energy Behavior Flow")

    # 3. Learning Integration
    print("3. Creating Learning Integration Diagram...")
    diagram3 = EnergyFlowDiagram()
    diagram3.create_learning_integration_diagram()
    diagram3.draw_diagram("Energy-Driven Learning")

    # 4. Centrality Analysis
    print("4. Performing Centrality Analysis...")
    centrality = diagram1.create_energy_centrality_analysis()

    print("\nEnergy Flow Diagrams Created Successfully!")
    print("Files saved:")
    print("- energy_flow_system_architecture.png")
    print("- energy_behavior_flow.png")
    print("- energy-driven_learning.png")

    return centrality

if __name__ == "__main__":
    try:
        centrality_analysis = create_energy_flow_visualizations()
        print("\nCentrality Analysis Results:")
        print(centrality_analysis)
    except ImportError as e:
        print(f"Visualization requires matplotlib and networkx: {e}")
        print("Install with: pip install matplotlib networkx")
    except Exception as e:
        print(f"Error creating visualizations: {e}")






