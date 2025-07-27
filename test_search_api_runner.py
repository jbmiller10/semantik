#!/usr/bin/env python3
"""Simple test runner to check search API tests"""

import subprocess
import sys

# List of tests to run in order
tests = [
    "tests/unit/test_search_api.py::TestSearchAPI::test_generate_mock_embedding",
    "tests/unit/test_search_api.py::TestSearchAPI::test_get_or_create_metric",
    "tests/unit/test_search_api.py::TestSearchAPI::test_model_status",
    "tests/unit/test_search_api.py::TestSearchAPI::test_root_endpoint",
    "tests/unit/test_search_api.py::TestSearchAPI::test_health_endpoint",
    "tests/unit/test_search_api.py::TestSearchAPI::test_search_post_endpoint",
    "tests/unit/test_search_api.py::TestSearchAPI::test_embed_endpoint",
    "tests/unit/test_search_api.py::TestSearchAPI::test_list_models_endpoint",
]

def run_test(test_path):
    """Run a single test and return result"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-xvs", test_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

if __name__ == "__main__":
    print("Running search API tests...")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning: {test}")
        success, stdout, stderr = run_test(test)
        
        if success:
            print("✓ PASSED")
            passed += 1
        else:
            print("✗ FAILED")
            failed += 1
            if stderr:
                print(f"Error: {stderr}")
            if "FAILED" in stdout:
                # Print the failure details
                lines = stdout.split('\n')
                for i, line in enumerate(lines):
                    if "FAILED" in line or "ERROR" in line:
                        # Print context around the failure
                        start = max(0, i - 5)
                        end = min(len(lines), i + 10)
                        print('\n'.join(lines[start:end]))
                        break
    
    print("\n" + "=" * 80)
    print(f"Summary: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)