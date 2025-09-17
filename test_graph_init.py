import torch
from torch_geometric.data import Data
from energy.node_id_manager import get_id_manager
from neural.dynamic_nodes import add_dynamic_nodes

# Create a dummy pixel_graph with some sensory nodes
id_manager = get_id_manager()
num_sensory = 10
x = torch.randn(num_sensory, 1)
node_labels = []
for i in range(num_sensory):
    node_id = id_manager.generate_unique_id("sensory")
    node_labels.append({
        'id': node_id,
        'type': 'sensory',
        'behavior': 'sensory',
        'energy': float(x[i, 0]),
        'state': 'active',
        'membrane_potential': 0.0,
        'threshold': 0.3,
        'refractory_timer': 0.0,
        'last_activation': 0,
        'plasticity_enabled': True,
        'eligibility_trace': 0.0,
        'last_update': 0
    })
    id_manager.register_node_index(node_id, i)

pixel_graph = Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long), node_labels=node_labels)

# Initialize graph
graph = add_dynamic_nodes(pixel_graph)
print(f"Graph initialized successfully with {len(graph.x)} nodes")
print("Logging from add_dynamic_nodes executed without error")