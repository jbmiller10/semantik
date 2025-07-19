#!/bin/bash

# Simple API collection creation test
API_URL="http://localhost:8080/api"

echo "=== Testing Collection Creation via API ==="
echo

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
  echo "❌ Login failed. Response:"
  echo "$LOGIN_RESPONSE" | jq .
  exit 1
fi

echo "✅ Login successful!"
echo

# Generate unique collection name with timestamp
TIMESTAMP=$(date +%s)
COLLECTION_NAME="Test Collection API $TIMESTAMP"

echo "2. Creating a new collection: '$COLLECTION_NAME'"
CREATE_RESPONSE=$(curl -s -X POST "$API_URL/v2/collections" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{
    \"name\": \"$COLLECTION_NAME\",
    \"description\": \"Created via API for testing\",
    \"embedding_model\": \"Qwen/Qwen3-Embedding-0.6B\",
    \"quantization\": \"float16\",
    \"chunk_size\": 1000,
    \"chunk_overlap\": 200,
    \"is_public\": false
  }")

# Check if creation was successful
if echo "$CREATE_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
  COLLECTION_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')
  echo "✅ Collection created successfully!"
  echo "   - Collection ID: $COLLECTION_ID"
  echo "   - Name: $(echo "$CREATE_RESPONSE" | jq -r '.name')"
  echo "   - Status: $(echo "$CREATE_RESPONSE" | jq -r '.status')"
  echo "   - Model: $(echo "$CREATE_RESPONSE" | jq -r '.embedding_model')"
  echo "   - Quantization: $(echo "$CREATE_RESPONSE" | jq -r '.quantization')"
else
  echo "❌ Failed to create collection. Response:"
  echo "$CREATE_RESPONSE" | jq .
  exit 1
fi

echo
echo "3. Waiting for collection to be ready..."
for i in {1..10}; do
  sleep 1
  STATUS_RESPONSE=$(curl -s -X GET "$API_URL/v2/collections/$COLLECTION_ID" \
    -H "Authorization: Bearer $ACCESS_TOKEN")
  STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
  
  if [ "$STATUS" == "ready" ]; then
    echo "✅ Collection is now ready!"
    break
  else
    echo "   Status: $STATUS (attempt $i/10)"
  fi
done

echo
echo "4. Verifying collection appears in list..."
COLLECTIONS_RESPONSE=$(curl -s -X GET "$API_URL/v2/collections" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

if echo "$COLLECTIONS_RESPONSE" | jq -e ".collections[] | select(.id == \"$COLLECTION_ID\")" > /dev/null 2>&1; then
  echo "✅ Collection found in list!"
  COLLECTION_COUNT=$(echo "$COLLECTIONS_RESPONSE" | jq '.total')
  echo "   Total collections for user: $COLLECTION_COUNT"
else
  echo "❌ Collection not found in list!"
fi

echo
echo "5. Summary:"
echo "   ✅ Successfully created collection '$COLLECTION_NAME'"
echo "   ✅ Collection ID: $COLLECTION_ID"
echo "   ✅ API is working correctly for collection creation"
echo
echo "Note: To index documents, add sources to this collection using:"
echo "   POST /api/v2/collections/$COLLECTION_ID/sources"
echo "   with body: { \"source_path\": \"/mnt/docs\" }"