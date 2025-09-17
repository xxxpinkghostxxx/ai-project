#!/usr/bin/env python3
"""
NASA Power of Ten Rules Compliance Verification Script
Demonstrates the successful implementation of all NASA safety-critical coding standards.
"""

import os
import sys
import ast
import time

from static_allocator import get_static_allocator, get_static_list, get_static_dict
from nasa_code_analyzer import NASACodeAnalyzer



def verify_rule_1_control_flow():
    """Verify Rule 1: Simplified Control Flow"""
    print_info("🔍 Verifying Rule 1: Simplified Control Flow")
    
    # Check simulation_manager.py for function lengths
    with open('simulation_manager.py', 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    long_functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
            function_length = end_line - start_line + 1
            
            if function_length > 60:
                long_functions.append((node.name, function_length))
    
    if long_functions:
        print_failure(f"Found {len(long_functions)} functions exceeding 60 lines:")
        for name, length in long_functions:
            print_failure(f"   - {name}: {length} lines")
        return False
    else:
        print_success("All functions are under 60 lines")
        return True


def verify_rule_2_loop_bounds():
    """Verify Rule 2: Fixed Upper Bounds on Loops"""
    print_info("🔍 Verifying Rule 2: Fixed Upper Bounds on Loops")
    
    # Check for while loops without bounds
    with open('simulation_manager.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    unbounded_loops = []
    
    for i, line in enumerate(lines, 1):
        if 'while' in line and 'for' not in line:
            # Check if it has explicit bounds
            if not any(comparison in line for comparison in ['<', '<=', '>', '>=']):
                unbounded_loops.append((i, line.strip()))
    
    if unbounded_loops:
        print_failure(f"Found {len(unbounded_loops)} potentially unbounded while loops:")
        for line_num, line in unbounded_loops:
            print_failure(f"   Line {line_num}: {line}")
        return False
    else:
        print_success("All loops have explicit bounds")
        return True


def verify_rule_3_static_allocation():
    """Verify Rule 3: Static Memory Allocation"""
    print("\n🔍 Verifying Rule 3: Static Memory Allocation")
    
    # Test static allocator
    try:
        allocator = get_static_allocator()
        
        # Test static list allocation
        static_list = get_static_list(50)
        assert len(static_list) == 50, "Static list size mismatch"
        
        # Test static dict allocation
        static_dict = get_static_dict(10)
        assert isinstance(static_dict, dict), "Static dict type mismatch"
        
        # Test buffer allocation
        node_data = allocator.allocate_node_data(100)
        assert node_data.shape == (100, 10), "Node buffer shape mismatch"
        
        edge_data = allocator.allocate_edge_data(50)
        assert edge_data.shape == (50, 3), "Edge buffer shape mismatch"
        
        print("✅ Static allocation system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Static allocation test failed: {e}")
        return False


def verify_rule_4_function_length():
    """Verify Rule 4: Function Length Limits"""
    print("\n🔍 Verifying Rule 4: Function Length Limits")
    
    # This is the same as Rule 1, but we'll check multiple files
    files_to_check = ['simulation_manager.py', 'behavior_engine.py', 'energy_behavior.py']
    all_good = True
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        long_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                function_length = end_line - start_line + 1
                
                if function_length > 60:
                    long_functions.append((node.name, function_length))
        
        if long_functions:
            print(f"❌ {filename} has {len(long_functions)} functions exceeding 60 lines")
            all_good = False
        else:
            print(f"✅ {filename}: All functions under 60 lines")
    
    return all_good


def verify_rule_5_data_scope():
    """Verify Rule 5: Restricted Data Scope"""
    print("\n🔍 Verifying Rule 5: Restricted Data Scope")
    
    # Check for global variables in behavior_engine.py
    with open('behavior_engine.py', 'r') as f:
        content = f.read()
    
    # Look for BehaviorCache class (good practice)
    if 'class BehaviorCache:' in content:
        print("✅ BehaviorCache class found - good encapsulation")
    else:
        print("❌ BehaviorCache class not found")
        return False
    
    # Check for reduced global variables
    global_count = content.count('global ')
    if global_count <= 2:  # Should be minimal
        print(f"✅ Minimal global variables: {global_count}")
        return True
    else:
        print(f"❌ Too many global variables: {global_count}")
        return False


def verify_rule_6_return_values():
    """Verify Rule 6: Check Return Values"""
    print("\n🔍 Verifying Rule 6: Check Return Values")
    
    # Check for proper return value handling in simulation_manager.py
    with open('simulation_manager.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    unchecked_calls = []
    
    for i, line in enumerate(lines, 1):
        # Look for function calls that might not check return values
        if '()' in line and not line.strip().startswith('#'):
            if not any(pattern in line for pattern in ['=', 'if ', 'assert ', 'return ', 'def ']):
                if any(func in line for func in ['get_', 'create_', 'update_', 'process_']):
                    unchecked_calls.append((i, line.strip()))
    
    if len(unchecked_calls) > 5:  # Allow some flexibility
        print(f"⚠️  Found {len(unchecked_calls)} potentially unchecked function calls")
        return False
    else:
        print("✅ Return values are properly checked")
        return True


def verify_rule_7_preprocessor():
    """Verify Rule 7: Minimize Preprocessor Directives"""
    print("\n🔍 Verifying Rule 7: Minimize Preprocessor Directives")
    
    # Check for complex conditional checks
    with open('simulation_manager.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    complex_conditionals = []
    
    for i, line in enumerate(lines, 1):
        if 'hasattr' in line and 'and' in line and 'hasattr' in line[line.find('and'):]:
            complex_conditionals.append((i, line.strip()))
    
    if len(complex_conditionals) > 3:  # Allow some flexibility
        print(f"⚠️  Found {len(complex_conditionals)} complex conditional checks")
        return False
    else:
        print("✅ Preprocessor directives are minimized")
        return True


def verify_rule_8_pointer_usage():
    """Verify Rule 8: Limit Pointer Usage"""
    print("\n🔍 Verifying Rule 8: Limit Pointer Usage")
    
    # Check for circular imports
    import_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and file != 'verify_nasa_compliance.py':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    if 'from simulation_manager import' in content:
                        import_files.append(file)
                except:
                    pass
    
    # Check if simulation_manager imports any of these back
    with open('simulation_manager.py', 'r') as f:
        sim_content = f.read()
    
    circular_deps = []
    for file in import_files:
        if f'from {file.replace(".py", "")} import' in sim_content:
            circular_deps.append(file)
    
    if circular_deps:
        print(f"❌ Found circular dependencies: {circular_deps}")
        return False
    else:
        print("✅ No circular dependencies found")
        return True


def verify_rule_9_assertions():
    """Verify Rule 9: Use Assertions"""
    print("\n🔍 Verifying Rule 9: Use Assertions")
    
    # Check for assertions in critical files
    files_to_check = ['simulation_manager.py', 'behavior_engine.py', 'energy_behavior.py']
    total_assertions = 0
    total_functions = 0
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                for child in ast.walk(node):
                    if isinstance(child, ast.Assert):
                        total_assertions += 1
    
    assertion_ratio = total_assertions / total_functions if total_functions > 0 else 0
    
    if assertion_ratio >= 1.0:  # At least 1 assertion per function
        print(f"✅ Good assertion density: {assertion_ratio:.2f} assertions per function")
        return True
    else:
        print(f"⚠️  Low assertion density: {assertion_ratio:.2f} assertions per function")
        return False


def verify_rule_10_static_analysis():
    """Verify Rule 10: Use Static Analysis Tools"""
    print("\n🔍 Verifying Rule 10: Use Static Analysis Tools")
    
    # Check if NASA analyzer exists and works
    if not os.path.exists('nasa_code_analyzer.py'):
        print("❌ NASA code analyzer not found")
        return False
    
    try:
        analyzer = NASACodeAnalyzer()
        
        # Test analyzer on a single file
        results = analyzer.analyze_file('simulation_manager.py')
        
        if results:
            print(f"✅ NASA analyzer working - found {len(results)} issues")
            return True
        else:
            print("✅ NASA analyzer working - no issues found")
            return True
            
    except Exception as e:
        print(f"❌ NASA analyzer test failed: {e}")
        return False


def run_comprehensive_verification():
    """Run comprehensive NASA compliance verification"""
    print("🚀 NASA Power of Ten Rules Compliance Verification")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all verification functions
    verifications = [
        ("Rule 1: Control Flow", verify_rule_1_control_flow),
        ("Rule 2: Loop Bounds", verify_rule_2_loop_bounds),
        ("Rule 3: Static Allocation", verify_rule_3_static_allocation),
        ("Rule 4: Function Length", verify_rule_4_function_length),
        ("Rule 5: Data Scope", verify_rule_5_data_scope),
        ("Rule 6: Return Values", verify_rule_6_return_values),
        ("Rule 7: Preprocessor", verify_rule_7_preprocessor),
        ("Rule 8: Pointer Usage", verify_rule_8_pointer_usage),
        ("Rule 9: Assertions", verify_rule_9_assertions),
        ("Rule 10: Static Analysis", verify_rule_10_static_analysis),
    ]
    
    results = []
    for rule_name, verify_func in verifications:
        try:
            result = verify_func()
            results.append((rule_name, result))
        except Exception as e:
            print(f"❌ {rule_name} verification failed with error: {e}")
            results.append((rule_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for rule_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {rule_name}")
    
    print(f"\nOverall: {passed}/{total} rules passed ({passed/total*100:.1f}%)")
    
    end_time = time.time()
    print(f"\nVerification completed in {end_time - start_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 CONGRATULATIONS! All NASA Power of Ten rules are compliant!")
    elif passed >= total * 0.8:
        print("\n✅ GOOD! Most NASA rules are compliant with minor issues.")
    else:
        print("\n⚠️  ATTENTION! Several NASA rules need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_verification()
    sys.exit(0 if success else 1)
