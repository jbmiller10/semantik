#!/bin/bash
# Run critical privacy tests for Semantik
# These tests ensure NO data leaves the user's system

echo "Running Semantik Privacy Tests..."
echo "================================="
echo "Testing that semantic chunking NEVER sends data to external APIs"
echo ""

# Set environment for production-like testing
export TESTING=false
export ENABLE_OPENAI_EMBEDDINGS=false

# Run privacy tests with verbose output
pytest -xvs tests/security/test_embedding_privacy.py \
    --tb=short \
    -k "test_no_network_requests or test_openai_embeddings_properly_disabled or test_local_embedding" \
    || { echo "CRITICAL: Privacy tests failed! Data might be leaking."; exit 1; }

echo ""
echo "Running all security tests..."
pytest -xvs tests/security/ -m "security or privacy" \
    || { echo "Security tests failed!"; exit 1; }

echo ""
echo "✓ All privacy tests passed - No data leaves your system!"