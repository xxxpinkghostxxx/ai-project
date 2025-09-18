try:
    from energy.energy_behavior import get_node_energy_cap
    print('Energy cap:', get_node_energy_cap())
except Exception as e:
    print('Import failed:', e)
    import traceback
    traceback.print_exc()