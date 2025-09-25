import subprocess
import os
import sys

def run_all_tests():
    """Run all test files and collect results."""
    test_dir = 'tests'
    results = {
        'total_files': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': []
    }

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found")
        return results

    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]

    for test_file in sorted(test_files):
        results['total_files'] += 1
        test_path = os.path.join(test_dir, test_file)

        print(f"Running {test_file}...")
        try:
            # Increase timeout for performance-heavy tests
            timeout = 300 if test_file in ['test_neural_performance.py', 'test_neural_simulation_scenarios.py', 'test_memory_pool_manager.py'] else 60
            result = subprocess.run([sys.executable, '-m', 'pytest', test_path, '--tb=no', '-q'],
                                  capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                results['passed'] += 1
                results['details'].append(f"{test_file}: PASSED")
                print(f"  PASSED")
            else:
                results['failed'] += 1
                results['details'].append(f"{test_file}: FAILED\n{result.stdout}\n{result.stderr}")
                print(f"  FAILED")
        except subprocess.TimeoutExpired:
            results['errors'] += 1
            results['details'].append(f"{test_file}: TIMEOUT")
            print(f"  TIMEOUT")
        except Exception as e:
            results['errors'] += 1
            results['details'].append(f"{test_file}: ERROR - {str(e)}")
            print(f"  ERROR - {str(e)}")

    return results

if __name__ == "__main__":
    results = run_all_tests()

    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Total test files: {results['total_files']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")

    if results['details']:
        print("\nDETAILS:")
        for detail in results['details']:
            print(f"  {detail}")

    # Save to file
    with open('test_run_summary.txt', 'w') as f:
        f.write(f"Total test files: {results['total_files']}\n")
        f.write(f"Passed: {results['passed']}\n")
        f.write(f"Failed: {results['failed']}\n")
        f.write(f"Errors: {results['errors']}\n\n")
        for detail in results['details']:
            f.write(f"{detail}\n\n")