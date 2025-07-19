# API Collection Creation Test Report

## Summary

Successfully tested creating a collection using the Semantik API directly, bypassing the broken UI.

## Test Results

### 1. Authentication ✅
- **Endpoint**: `POST /api/auth/login`
- **User**: testuser
- **Result**: Successfully obtained JWT access token

### 2. Collection Creation ✅
- **Endpoint**: `POST /api/v2/collections`
- **Collection Name**: "Test Collection API V2"
- **Collection ID**: `162c8093-b59e-4894-afe3-629682588263`
- **Configuration**:
  - Model: Qwen/Qwen3-Embedding-0.6B
  - Quantization: float16
  - Chunk Size: 1000
  - Chunk Overlap: 200
- **Result**: Collection created successfully and transitioned to "ready" status

### 3. Collection Listing ✅
- **Endpoint**: `GET /api/v2/collections`
- **Result**: Successfully retrieved collection list showing our created collection

## API Endpoints Used

```bash
# Login
curl -X POST "http://localhost:8080/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}'

# Create Collection
curl -X POST "http://localhost:8080/api/v2/collections" \
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
  }'

# List Collections
curl -X GET "http://localhost:8080/api/v2/collections" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Get Collection Details
curl -X GET "http://localhost:8080/api/v2/collections/{collection_id}" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

## Notes

1. **Rate Limiting**: The API has a rate limit of 5 collection creations per hour.

2. **Collection States**: Collections start in "pending" state and transition to "ready" after initialization.

3. **Adding Sources**: Sources can only be added to collections in "ready" or "degraded" state, not "pending".

4. **Legacy API**: The `/api/jobs` endpoint is deprecated. Use the v2 collections API instead.

## Next Steps

To index documents in the created collection, use:
```bash
curl -X POST "http://localhost:8080/api/v2/collections/162c8093-b59e-4894-afe3-629682588263/sources" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "source_path": "/mnt/docs",
    "config": {}
  }'
```

Note: There appears to be a database transaction issue when adding sources that may need investigation.