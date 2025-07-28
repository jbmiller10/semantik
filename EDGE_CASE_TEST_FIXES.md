# Edge Case Test Fixes Summary

## Issues Fixed in `tests/unit/test_search_api_edge_cases.py`

### 1. Import Errors
- **Problem**: Tests were importing models directly from `packages.vecpipe.search_api` that are actually defined in `shared.contracts.search`
- **Fix**: Added proper imports from `shared.contracts.search` for:
  - `BatchSearchRequest`
  - `BatchSearchResponse`
  - `SearchRequest`
  - `SearchResponse`

### 2. Duplicate Decorators
- **Problem**: Multiple test methods had duplicate `@pytest.mark.asyncio` decorators
- **Fix**: Removed duplicate decorators from all async test methods

### 3. Incorrect Mock Paths
- **Problem**: Tests were patching import paths that don't exist in the module namespace
- **Fix**: Updated mock paths:
  - `packages.vecpipe.search_api.QdrantClient` → `qdrant_client.QdrantClient`
  - `packages.vecpipe.search_api.get_collection_metadata` → `shared.database.collection_metadata.get_collection_metadata`

## Changes Made

1. **Line 14-24**: Fixed imports to properly import from both `search_api` and `shared.contracts.search`
2. **Line 412**: Removed `BatchSearchRequest` from the import since it's already imported at the top
3. **Multiple lines**: Removed duplicate `@pytest.mark.asyncio` decorators
4. **Line 427-428**: Fixed mock patch paths for QdrantClient and get_collection_metadata

## Expected Results

After these fixes, the tests should:
- Import all required modules correctly
- Run without decorator conflicts
- Mock the correct import paths

The tests should now run properly and either PASS or FAIL based on the actual test logic, rather than ERROR due to import/setup issues.