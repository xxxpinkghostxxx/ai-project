#!/usr/bin/env python3
"""
Simple test to verify energy modulation fixes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_energy_cap():
    """Test that energy cap is correctly set to 5.0"""
    try:
        from energy.energy_behavior import get_node_energy_cap
        cap = get_node_energy_cap()
        print(f"Energy cap: {cap}")
        assert cap == 5.0, f"Expected energy cap 5.0, got {cap}"
        print("PASS: Energy cap test passed")
        return True
    except Exception as e:
        print(f"FAIL: Energy cap test failed: {e}")
        return False

def test_energy_modulation():
    """Test that energy modulation is working"""
    try:
        from learning.live_hebbian_learning import create_live_hebbian_learning
        learning = create_live_hebbian_learning()

        # Test with low energy (should give lower learning rate)
        low_rate = learning._calculate_energy_modulated_learning_rate(0, 1)  # Low energy nodes
        print(f"Low energy learning rate: {low_rate}")

        # Test with high energy (should give higher learning rate)
        high_rate = learning._calculate_energy_modulated_learning_rate(8, 9)  # High energy nodes
        print(f"High energy learning rate: {high_rate}")

        # High energy should give higher rate than low energy
        assert high_rate > low_rate, f"High energy rate {high_rate} should be > low energy rate {low_rate}"
        print("PASS: Energy modulation test passed")
        return True
    except Exception as e:
        print(f"FAIL: Energy modulation test failed: {e}")
        return False

def test_learning_engine_modulation():
    """Test learning engine energy modulation"""
    try:
        from learning.learning_engine import LearningEngine
        engine = LearningEngine()

        # Create mock nodes with different energies
        low_energy_node = {'energy': 0.5}
        high_energy_node = {'energy': 4.0}

        # Test modulation
        low_rate = engine._calculate_energy_modulated_rate(low_energy_node, low_energy_node, 0.01)
        high_rate = engine._calculate_energy_modulated_rate(high_energy_node, high_energy_node, 0.01)

        print(f"Learning engine - Low energy rate: {low_rate}")
        print(f"Learning engine - High energy rate: {high_rate}")

        assert high_rate > low_rate, f"High energy rate {high_rate} should be > low energy rate {low_rate}"
        print("PASS: Learning engine modulation test passed")
        return True
    except Exception as e:
        print(f"FAIL: Learning engine modulation test failed: {e}")
        return False

def main():
    print("Testing Energy System Fixes")
    print("=" * 40)

    tests = [
        test_energy_cap,
        test_energy_modulation,
        test_learning_engine_modulation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All energy modulation fixes are working!")
        return True
    else:
        print("FAILURE: Some tests failed. Energy modulation needs more work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)