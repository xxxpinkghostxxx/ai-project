import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import faulthandler
faulthandler.enable()  # Enable fault handler for segfault tracebacks
from core.simulation_manager import get_simulation_manager
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')  # More verbose for diagnosis

async def test_sim():
    print('Starting test_sim')
    try:
        print('Using pre-imported simulation_manager')
    except Exception as e:
        print(f'Import error: {e}')
        import traceback
        traceback.print_exc()
        return

    try:
        manager = get_simulation_manager()
        manager.headless = True
        print('Got simulation manager (headless mode)')
    except Exception as e:
        print(f'Manager creation error: {e}')
        import traceback
        traceback.print_exc()
        return

    try:
        manager.initialize_graph()
        print('Graph initialized')
    except Exception as e:
        print(f'Initialization error: {e}')
        import traceback
        traceback.print_exc()
        return

    try:
        for i in range(500):
            await manager.run_single_step()
            if i % 10 == 0:
                node_count = len(manager.graph.node_labels) if hasattr(manager, 'graph') and manager.graph is not None and hasattr(manager.graph, 'node_labels') else 'N/A'
                print(f'Step {i}: Node count ~{node_count}')
            if i % 50 == 0:
                print(f'Completed step {i}')
                # Log key metrics
                if hasattr(manager, 'network_metrics') and manager.network_metrics and hasattr(manager, 'graph') and manager.graph:
                    metrics = manager.network_metrics.calculate_comprehensive_metrics(manager.graph)
                    avg_energy = metrics.get('energy_balance', {}).get('avg_energy', 0.0) if 'energy_balance' in metrics else 0.0
                    plasticity = metrics.get('plasticity', 0.01)
                    connectivity = metrics.get('connectivity', {}).get('density', 0.0) if 'connectivity' in metrics else 0.0
                    print(f'  Avg Energy: {avg_energy:.2f}, Plasticity: {plasticity:.3f}, Connectivity: {connectivity:.3f}')
        print('Simulation steps complete')
    except Exception as e:
        print(f'Run error at step {i if "i" in locals() else "unknown"}: {e}')
        import traceback
        traceback.print_exc()

asyncio.run(test_sim())