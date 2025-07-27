#!/usr/bin/env python3
"""Quick test runner to verify search_api tests work."""

import subprocess
import sys


def run_tests():
    """Run the search_api unit tests."""
    print("Running search_api unit tests...")

    # Run tests with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/test_search_api.py",
        "tests/unit/test_search_api_edge_cases.py",
        "-v",
        "--tb=short",
        "--cov=packages.vecpipe.search_api",
        "--cov-report=term-missing",
        "-x",  # Stop on first failure
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
