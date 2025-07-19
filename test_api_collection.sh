#!/bin/bash

# Test API collection creation
API_URL="http://localhost:8080/api"

echo "1. Logging in as testuser..."
LOGIN_RESPONSE=$(curl -s -X POST "$API_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }')

# Extract access token
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token')

if [ "$ACCESS_TOKEN" == "null" ] || [ -z "$ACCESS_TOKEN" ]; then
  echo "Login failed. Response:"
  echo "$LOGIN_RESPONSE" | jq .
  exit 1
fi

echo "Login successful! Token obtained."
echo

echo "2. Creating a new collection via API..."
CREATE_RESPONSE=$(curl -s -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "name": "Test Collection API",
    "description": "Created via API for testing",
    "directory_path": "/mnt/docs",
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16",
    "chunk_size": 600,
    "chunk_overlap": 200,
    "batch_size": 96,
    "scan_subdirs": true
  }')

# Check if creation was successful
if echo "$CREATE_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
  JOB_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')
  echo "Collection created successfully!"
  echo "Job ID: $JOB_ID"
  echo
  echo "Full response:"
  echo "$CREATE_RESPONSE" | jq .
else
  echo "Failed to create collection. Response:"
  echo "$CREATE_RESPONSE" | jq .
  exit 1
fi

echo
echo "3. Checking collections list..."
COLLECTIONS_RESPONSE=$(curl -s -X GET "$API_URL/collections" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "Collections:"
echo "$COLLECTIONS_RESPONSE" | jq .