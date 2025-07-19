#!/bin/bash

# Test API v2 collection creation
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

echo "2. Creating a new collection via v2 API..."
CREATE_RESPONSE=$(curl -s -X POST "$API_URL/v2/collections" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "name": "Test Collection API V2",
    "description": "Created via API v2 for testing",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "is_public": false
  }')

# Check if creation was successful
if echo "$CREATE_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
  COLLECTION_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')
  echo "Collection created successfully!"
  echo "Collection ID: $COLLECTION_ID"
  echo "Full response:"
  echo "$CREATE_RESPONSE" | jq .
else
  echo "Failed to create collection. Response:"
  echo "$CREATE_RESPONSE" | jq .
  
  # Check if it's because collection already exists
  if echo "$CREATE_RESPONSE" | jq -r '.detail' | grep -q "already exists"; then
    echo
    echo "Collection already exists. Let's list existing collections..."
    COLLECTIONS_RESPONSE=$(curl -s -X GET "$API_URL/v2/collections" \
      -H "Authorization: Bearer $ACCESS_TOKEN")
    echo "$COLLECTIONS_RESPONSE" | jq .
    
    # Try to find the existing collection
    EXISTING_ID=$(echo "$COLLECTIONS_RESPONSE" | jq -r '.collections[] | select(.name == "Test Collection API V2") | .id' | head -1)
    if [ -n "$EXISTING_ID" ]; then
      COLLECTION_ID="$EXISTING_ID"
      echo
      echo "Using existing collection ID: $COLLECTION_ID"
    else
      exit 1
    fi
  else
    exit 1
  fi
fi

echo
echo "3. Adding source directory to the collection..."
ADD_SOURCE_RESPONSE=$(curl -s -X POST "$API_URL/v2/collections/$COLLECTION_ID/sources" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "source_path": "/mnt/docs",
    "config": {
      "metadata": {
        "source": "API test"
      }
    }
  }')

if echo "$ADD_SOURCE_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
  OPERATION_ID=$(echo "$ADD_SOURCE_RESPONSE" | jq -r '.id')
  echo "Source added successfully!"
  echo "Operation ID: $OPERATION_ID"
  echo "Full response:"
  echo "$ADD_SOURCE_RESPONSE" | jq .
else
  echo "Failed to add source. Response:"
  echo "$ADD_SOURCE_RESPONSE" | jq .
fi

echo
echo "4. Checking operation status..."
if [ -n "$OPERATION_ID" ]; then
  OPERATION_STATUS=$(curl -s -X GET "$API_URL/v2/operations/$OPERATION_ID" \
    -H "Authorization: Bearer $ACCESS_TOKEN")
  echo "Operation status:"
  echo "$OPERATION_STATUS" | jq .
fi

echo
echo "5. Listing all collections..."
COLLECTIONS_RESPONSE=$(curl -s -X GET "$API_URL/v2/collections" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "Collections:"
echo "$COLLECTIONS_RESPONSE" | jq .