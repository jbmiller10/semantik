#!/bin/bash
# Test script to verify test discovery in CI environment

set -euo pipefail

echo "=== Verifying test file discovery ==="
echo "Current directory: $(pwd)"
echo ""

# List test files
echo "Test files found:"
find src/components/__tests__ -name "*.test.tsx" -type f | sort
echo ""

# Test glob pattern matching
echo "=== Testing vitest glob patterns ==="

echo "1. Testing specific file patterns:"
npm test -- --run --reporter=verbose 'src/components/__tests__/CollectionCard.test.tsx' || true

echo ""
echo "2. Testing wildcard patterns:"
npm test -- --run --reporter=verbose 'src/components/__tests__/*Collection*.test.tsx' || true

echo ""
echo "3. Testing multiple specific files:"
npm test -- --run --reporter=verbose \
  'src/components/__tests__/CollectionCard.test.tsx' \
  'src/components/__tests__/CreateCollectionModal.test.tsx' || true

echo ""
echo "=== Test discovery complete ==="