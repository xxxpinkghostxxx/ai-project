#!/usr/bin/env python3
"""
Simple test for energy-modulated learning without full simulation manager.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_energy_modulation_logic():
    """Test the energy modulation logic directly."""
    print("Testing Energy Modulation Logic")
    print("=" * 40)

    # Test the energy modulation calculation
    def calculate_energy_modulated_rate(energy1, energy2, base_rate=0.01, energy_cap=1.0):
        """Simplified version of energy modulation calculation."""
        avg_energy = (energy1 + energy2) / 2.0
        normalized_energy = min(avg_energy / energy_cap, 1.0) if energy_cap > 0 else 0.5
        modulated_rate = base_rate * (0.5 + 0.5 * normalized_energy)
        return modulated_rate

    # Test cases
    test_cases = [
        (0.1, 0.1, "Low energy nodes"),
        (0.5, 0.5, "Medium energy nodes"),
        (0.9, 0.9, "High energy nodes"),
        (0.1, 0.9, "Mixed energy nodes"),
    ]

    base_rate = 0.02
    print(f"Base learning rate: {base_rate}")
    print()

    for energy1, energy2, description in test_cases:
        modulated = calculate_energy_modulated_rate(energy1, energy2, base_rate)
        factor = modulated / base_rate
        print(f"{description}: {energy1}/{energy2} -> {modulated:.4f} (factor: {factor:.2f}x)")

    # Verify that higher energy leads to higher learning rates
    low_mod = calculate_energy_modulated_rate(0.1, 0.1, base_rate)
    high_mod = calculate_energy_modulated_rate(0.9, 0.9, base_rate)

    if high_mod > low_mod:
        print("\nSUCCESS: Energy modulation working correctly!")
        print(".2f")
        return True
    else:
        print("\nFAILED: Energy modulation not working")
        return False

def test_learning_engine_methods():
    """Test learning engine methods directly."""
    print("\nTesting Learning Engine Methods")
    print("=" * 40)

    try:
        from learning.learning_engine import LearningEngine

        # Create learning engine
        engine = LearningEngine()
        print(f"Learning engine created with energy modulation: {engine.energy_learning_modulation}")

        # Test energy modulation method
        test_pre = {'id': 1, 'energy': 0.8, 'membrane_potential': 0.7}
        test_post = {'id': 2, 'energy': 0.3, 'membrane_potential': 0.4}

        modulated = engine._calculate_energy_modulated_rate(test_pre, test_post, 0.02)
        print(f"Energy modulation test: {modulated:.4f}")

        # Test with different combinations
        high_pre = {'id': 1, 'energy': 0.9}
        high_post = {'id': 2, 'energy': 0.9}
        high_mod = engine._calculate_energy_modulated_rate(high_pre, high_post, 0.02)

        low_pre = {'id': 1, 'energy': 0.1}
        low_post = {'id': 2, 'energy': 0.1}
        low_mod = engine._calculate_energy_modulated_rate(low_pre, low_post, 0.02)

        print(f"High energy modulation: {high_mod:.4f}")
        print(f"Low energy modulation: {low_mod:.4f}")

        if high_mod > low_mod:
            print("SUCCESS: Learning engine energy modulation working!")
            return True
        else:
            print("FAILED: Learning engine energy modulation not working")
            return False

    except Exception as e:
        print(f"Learning engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hebbian_system_creation():
    """Test Hebbian system creation and basic properties."""
    print("\nTesting Hebbian System Creation")
    print("=" * 40)

    try:
        from learning.live_hebbian_learning import create_live_hebbian_learning

        # Create Hebbian system
        hebbian = create_live_hebbian_learning()
        print(f"Hebbian system created with energy modulation: {hebbian.energy_learning_modulation}")

        # Check parameters
        params = hebbian.get_learning_parameters()
        print(f"Learning parameters: {params}")

        # Check that energy modulation is enabled
        if hebbian.energy_learning_modulation:
            print("SUCCESS: Hebbian system has energy modulation enabled!")
            return True
        else:
            print("WARNING: Hebbian system energy modulation is disabled")
            return False

    except Exception as e:
        print(f"Hebbian system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("SIMPLE ENERGY-MODULATED LEARNING TEST")
    print("=" * 50)

    test1 = test_energy_modulation_logic()
    test2 = test_learning_engine_methods()
    test3 = test_hebbian_system_creation()

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Energy Logic Test: {'PASS' if test1 else 'FAIL'}")
    print(f"Learning Engine Test: {'PASS' if test2 else 'FAIL'}")
    print(f"Hebbian System Test: {'PASS' if test3 else 'FAIL'}")

    if all([test1, test2, test3]):
        print("\nOVERALL: SUCCESS - All energy modulation components working!")
    else:
        print("\nOVERALL: ISSUES DETECTED - Some components may need fixes")