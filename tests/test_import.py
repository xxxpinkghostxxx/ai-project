import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.energy.energy_behavior import get_node_energy_cap
    print('Energy cap:', get_node_energy_cap())
    print('Import test: PASSED')
except Exception as e:
    print('Import failed:', e)
    import traceback
    traceback.print_exc()
    print('Import test: FAILED')






