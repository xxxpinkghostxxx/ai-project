#!/usr/bin/env python3
"""
Test script to verify that the critical bug fixes are working.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that the fixed imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from core.main_graph import create_workspace_grid, merge_graphs
        print("✓ Core imports working")
    except ImportError as e:
        print(f"✗ Core import failed: {e}")
        return False
    
    try:
        # Test neural imports
        from neural.enhanced_neural_dynamics import EnhancedNeuralDynamics
        from neural.enhanced_connection_system import EnhancedConnectionSystem
        from neural.enhanced_node_behaviors import EnhancedNodeBehaviorSystem
        print("✓ Neural imports working")
    except ImportError as e:
        print(f"✗ Neural import failed: {e}")
        return False
    
    try:
        # Test utils imports
        from utils.error_handler import ErrorHandler
        print("✓ Utils imports working")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the fixed modules."""
    print("\nTesting basic functionality...")
    
    try:
        # Test error handler
        error_handler = ErrorHandler()
        print("✓ ErrorHandler created successfully")
        
        # Test neural dynamics
        dynamics = EnhancedNeuralDynamics()
        print("✓ EnhancedNeuralDynamics created successfully")
        
        # Test connection system
        conn_system = EnhancedConnectionSystem()
        print("✓ EnhancedConnectionSystem created successfully")
        
        # Test node behavior system
        behavior_system = EnhancedNodeBehaviorSystem()
        print("✓ EnhancedNodeBehaviorSystem created successfully")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    return True

def test_cleanup():
    """Test that cleanup methods work properly."""
    print("\nTesting cleanup methods...")
    
    try:
        # Test cleanup methods
        dynamics = EnhancedNeuralDynamics()
        dynamics.cleanup()
        print("✓ Neural dynamics cleanup working")
        
        conn_system = EnhancedConnectionSystem()
        conn_system.cleanup()
        print("✓ Connection system cleanup working")
        
        behavior_system = EnhancedNodeBehaviorSystem()
        behavior_system.cleanup()
        print("✓ Node behavior system cleanup working")
        
    except Exception as e:
        print(f"✗ Cleanup test failed: {e}")
        return False
    
    return True

def test_input_validation():
    """Test input validation."""
    print("\nTesting input validation...")
    
    try:
        # Test connection system input validation
        conn_system = EnhancedConnectionSystem()
        
        # Test invalid inputs
        result = conn_system.create_connection(-1, 0, 'excitatory')
        assert result == False, "Should reject negative source_id"
        
        result = conn_system.create_connection(0, -1, 'excitatory')
        assert result == False, "Should reject negative target_id"
        
        result = conn_system.create_connection(0, 0, 'excitatory')
        assert result == False, "Should reject self-connection"
        
        result = conn_system.create_connection(0, 1, 'invalid_type')
        assert result == False, "Should reject invalid connection type"
        
        print("✓ Input validation working")
        
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running bug fix verification tests...\n")
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_cleanup,
        test_input_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Critical bugs have been fixed.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
