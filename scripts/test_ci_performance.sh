#!/bin/bash
#
# Run CI-friendly performance tests locally
# This helps verify that CI tests will pass before pushing

set -e

echo "Setting up CI-like environment..."
export CI=true
export TESTING=true
export USE_MOCK_EMBEDDINGS=true
export CUDA_VISIBLE_DEVICES=""

echo "Running CI environment tests..."
poetry run pytest tests/test_ci_environment.py -v

echo "Running CI-friendly performance tests..."
poetry run pytest tests/performance/test_ci_chunking_performance.py -v --timeout=60

echo "Running advanced performance tests with CI settings..."
poetry run pytest tests/performance/test_advanced_chunking_benchmarks.py -v --timeout=60 -k "not stress_test"

echo "CI performance tests completed successfully!"