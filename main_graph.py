from src.ui.screen_graph import capture_screen, create_pixel_gray_graph, RESOLUTION_SCALE
from src.neural.dynamic_nodes import add_dynamic_nodes
from src.utils.pattern_consolidation_utils import create_workspace_node, create_sensory_node, create_dynamic_node
import torch
import logging
import threading

 


def create_workspace_grid():

    from src.energy.node_id_manager import get_id_manager
    id_manager = get_id_manager()
    grid_size = 16
    num_nodes = grid_size * grid_size
    x = torch.zeros((num_nodes, 1), dtype=torch.float32)
    node_labels = []
    for y in range(grid_size):
        for x_coord in range(grid_size):
            node_id = id_manager.generate_unique_id("workspace", {"x": x_coord, "y": y})
            node_index = y * grid_size + x_coord
            node_label = create_workspace_node(
                node_id=node_id,
                x=x_coord,
                y=y,
                energy=0.0,
                state="inactive",
                membrane_potential=0.0,
                threshold=0.6,
                refractory_timer=0.0,
                last_activation=0,
                plasticity_enabled=True,
                eligibility_trace=0.0,
                last_update=0,
                workspace_capacity=5.0,
                workspace_creativity=1.5,
                workspace_focus=3.0
            )
            node_labels.append(node_label)
            id_manager.register_node_index(node_id, node_index)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    from torch_geometric.data import Data
    workspace_graph = Data(
        x=x,
        edge_index=edge_index,
        node_labels=node_labels,
        h=grid_size,
        w=grid_size
    )
    return workspace_graph


def merge_graphs(graph1, graph2):

    from torch_geometric.data import Data
    from src.energy.node_id_manager import get_id_manager
    combined_x = torch.cat([graph1.x, graph2.x], dim=0)
    combined_labels = graph1.node_labels + graph2.node_labels
    id_manager = get_id_manager()
    offset = graph1.x.size(0)
    for i, node in enumerate(graph2.node_labels):
        node_id = node.get('id')
        if node_id is not None:
            new_index = offset + i
            # Use proper method instead of direct attribute access
            if not id_manager.register_node_index(node_id, new_index):
                # If registration fails, generate a new ID
                new_node_id = id_manager.generate_unique_id(node.get('type', 'unknown'))
                node['id'] = new_node_id
                id_manager.register_node_index(new_node_id, new_index)
    if hasattr(graph2, 'edge_index') and graph2.edge_index.numel() > 0:
        adjusted_edge_index = graph2.edge_index + offset
        combined_edge_index = torch.cat([graph1.edge_index, adjusted_edge_index], dim=1)
    else:
        combined_edge_index = graph1.edge_index
    merged_graph = Data(
        x=combined_x,
        edge_index=combined_edge_index,
        node_labels=combined_labels
    )
    for attr in ['h', 'w', 'step']:
        if hasattr(graph1, attr):
            setattr(merged_graph, attr, getattr(graph1, attr))
    return merged_graph


def create_test_graph(num_sensory=100, num_dynamic=20):
    """Create a test graph with sensory and dynamic nodes for testing purposes."""
    import numpy as np
    from torch_geometric.data import Data
    from src.energy.node_id_manager import get_id_manager
    
    id_manager = get_id_manager()
    x = torch.abs(torch.randn(num_sensory + num_dynamic, 1)) * 100  # Positive initial energies 0-100
    node_labels = []
    
    # Create sensory nodes
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
    
    # Create dynamic nodes
    for i in range(num_sensory, num_sensory + num_dynamic):
        node_id = id_manager.generate_unique_id("dynamic")
        node_labels.append({
            'id': node_id,
            'type': 'dynamic',
            'behavior': 'dynamic',
            'energy': float(x[i, 0]),
            'state': 'inactive',
            'membrane_potential': 0.0,
            'threshold': 0.5,
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': 0
        })
        id_manager.register_node_index(node_id, i)
    
    # Create random connections
    edge_list = []
    for i in range(min(50, num_sensory + num_dynamic)):
        source = np.random.randint(0, num_sensory + num_dynamic)
        target = np.random.randint(0, num_sensory + num_dynamic)
        if source != target:
            edge_list.append([source, target])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty((2, 0), dtype=torch.long)
    
    return Data(
        x=x,
        edge_index=edge_index,
        node_labels=node_labels
    )


def create_test_graph_with_workspace(num_sensory=100, num_dynamic=20, num_workspace=100):
    """Create a test graph with sensory, dynamic, and workspace nodes for testing purposes."""
    import numpy as np
    from torch_geometric.data import Data
    from src.energy.node_id_manager import get_id_manager
    
    id_manager = get_id_manager()
    x = torch.abs(torch.randn(num_sensory + num_dynamic + num_workspace, 1)) * 100  # Positive initial
    node_labels = []
    
    # Create sensory nodes with spatial coordinates
    for i in range(num_sensory):
        node_id = id_manager.generate_unique_id("sensory")
        x_coord = i % 10
        y_coord = i // 10
        node_labels.append({
            'id': node_id,
            'type': 'sensory',
            'behavior': 'sensory',
            'energy': float(x[i, 0]),
            'state': 'active',
            'x': x_coord,
            'y': y_coord,
            'membrane_potential': 0.0,
            'threshold': 0.3,
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': False,
            'eligibility_trace': 0.0,
            'last_update': 0
        })
        id_manager.register_node_index(node_id, i)
    
    # Create dynamic nodes
    for i in range(num_sensory, num_sensory + num_dynamic):
        node_id = id_manager.generate_unique_id("dynamic")
        node_labels.append({
            'id': node_id,
            'type': 'dynamic',
            'behavior': 'dynamic',
            'energy': float(x[i, 0]),
            'state': 'inactive',
            'membrane_potential': 0.0,
            'threshold': 0.5,
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': 0
        })
        id_manager.register_node_index(node_id, i)
    
    # Create workspace nodes
    for i in range(num_sensory + num_dynamic, num_sensory + num_dynamic + num_workspace):
        node_id = id_manager.generate_unique_id("workspace")
        x_coord = (i - num_sensory - num_dynamic) % 10
        y_coord = (i - num_sensory - num_dynamic) // 10
        node_labels.append({
            'id': node_id,
            'type': 'workspace',
            'behavior': 'workspace',
            'energy': float(x[i, 0]),
            'state': 'inactive',
            'x': x_coord,
            'y': y_coord,
            'membrane_potential': 0.0,
            'threshold': 0.6,
            'refractory_timer': 0.0,
            'last_activation': 0,
            'plasticity_enabled': True,
            'eligibility_trace': 0.0,
            'last_update': 0,
            'workspace_capacity': 5.0,
            'workspace_creativity': 1.5,
            'workspace_focus': 3.0
        })
        id_manager.register_node_index(node_id, i)
    
    # Create random connections
    edge_list = []
    for i in range(min(50, num_sensory + num_dynamic + num_workspace)):
        source = np.random.randint(0, num_sensory + num_dynamic + num_workspace)
        target = np.random.randint(0, num_sensory + num_dynamic + num_workspace)
        if source != target:
            edge_list.append([source, target])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty((2, 0), dtype=torch.long)
    
    return Data(
        x=x,
        edge_index=edge_index,
        node_labels=node_labels
    )


def initialize_main_graph(scale=RESOLUTION_SCALE):

    arr = capture_screen(scale=scale)
    pixel_graph = create_pixel_gray_graph(arr)
    # Register pixel graph node IDs in the ID manager
    from src.energy.node_id_manager import get_id_manager
    id_manager = get_id_manager()
    for idx, node in enumerate(pixel_graph.node_labels):
        node_id = node.get('id')
        if node_id is not None and not id_manager.is_valid_id(node_id):
            # Add to active IDs since they were created externally
            id_manager._active_ids.add(node_id)
            id_manager._node_type_map[node_id] = node.get('type', 'sensory')
            id_manager.register_node_index(node_id, idx)
    main_graph = add_dynamic_nodes(pixel_graph)
    workspace_graph = create_workspace_grid()
    full_graph = merge_graphs(main_graph, workspace_graph)
    if len(full_graph.node_labels) == 0:
        full_graph = add_dynamic_nodes(full_graph, num_dynamic=5)

    # Ensure all node IDs are registered in the ID manager
    from src.energy.node_id_manager import get_id_manager
    id_manager = get_id_manager()
    for idx, node in enumerate(full_graph.node_labels):
        node_id = node.get('id')
        if node_id is not None and id_manager.get_node_index(node_id) is None:
            id_manager.register_node_index(node_id, idx)

    # Fast initialization of sparse random edges with bounded average out-degree
    import numpy as np
    from src.neural.connection_logic import EnhancedEdge

    num_nodes = full_graph.x.shape[0]
    edge_list = []

    # Ensure edge attribute containers exist
    if not hasattr(full_graph, 'edge_attributes'):
        full_graph.edge_attributes = []
    if not hasattr(full_graph, '_edge_attributes_lock'):
        full_graph._edge_attributes_lock = threading.Lock()

    # Parameters for faster initialization with adaptive scaling
    avg_out_degree = min(4, max(2, num_nodes // 1000))  # Adaptive degree based on graph size
    # Soft cap to prevent pathological startup times on very large graphs
    max_initial_edges = max(10000, min(num_nodes * avg_out_degree, 50000))
    seen = set()

    if num_nodes > 1:
        logging.info(f"[INIT] Generating initial edges with avg_out_degree={avg_out_degree}, num_nodes={num_nodes}")
        # Pre-allocate edge types and weights for better performance
        edge_types = ['excitatory', 'inhibitory', 'modulatory']
        type_weights = [0.6, 0.3, 0.1]  # Favor excitatory connections

        for i in range(num_nodes):
            # Sample an out-degree around the target average; clamp within valid range
            deg = int(np.random.poisson(avg_out_degree))
            if deg <= 0:
                continue
            deg = min(deg, num_nodes - 1)

            # Choose distinct targets excluding self efficiently
            # Sample from [0, num_nodes-2], then shift indices >= i by +1 to skip self
            targets = np.random.choice(num_nodes - 1, size=min(deg, num_nodes - 1), replace=False)
            targets = targets + (targets >= i)

            for j in targets:
                key = (i, int(j))
                if key in seen:
                    continue
                seen.add(key)

                edge_list.append([i, int(j)])
                # Use weighted random choice for edge types
                edge_type = np.random.choice(edge_types, p=type_weights)
                weight = float(np.random.uniform(0.1, 1.0))
                edge = EnhancedEdge(i, int(j), weight=weight, edge_type=edge_type)
                with full_graph._edge_attributes_lock:
                    full_graph.edge_attributes.append(edge)

                # Periodic progress logging and soft cap enforcement
                if len(edge_list) % 10000 == 0:
                    logging.info(f"[INIT] Edge generation progress: {len(edge_list)} edges")
                if len(edge_list) >= max_initial_edges:
                    break

            if len(edge_list) >= max_initial_edges:
                break

    full_graph.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long).contiguous()
    edge_count = full_graph.edge_index.shape[1]
    density = (edge_count / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
    logging.info(f"Graph created with {len(full_graph.node_labels)} nodes and {edge_count} initial edges (density: {density:.6f})")
    return full_graph
if __name__ == "__main__":
    import time
    print("Press Ctrl+C to stop.")
    try:
        max_iterations = 1000  # Add graceful exit condition
        iteration_count = 0
        while iteration_count < max_iterations:
            graph = initialize_main_graph()
            print(f"Graph: {len(graph.x)} nodes (including dynamic nodes)")
            print(f"First 6 node labels: {graph.node_labels[:6]}")
            print(f"Last 6 node labels: {graph.node_labels[-6:]}")
            time.sleep(0.5)
            iteration_count += 1
        print(f"Test completed after {max_iterations} iterations")
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        time.sleep(1.0)  # Brief pause before potential restart







