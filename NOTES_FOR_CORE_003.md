# Notes for CORE-003: Move Embedding Service to Shared Package

## Context from TEST-002 Implementation

While implementing the integration test for search_api's use of the embedding service, I discovered the following dependency chain:

### Current Dependency Flow
1. `packages/vecpipe/search_api.py` imports `EmbeddingService` from `webui.embedding_service`
2. The search endpoint calls `generate_embedding_async` (local function)
3. `generate_embedding_async` calls `model_manager.generate_embedding_async`
4. `packages/vecpipe/model_manager.py` also imports `EmbeddingService` from `webui.embedding_service`
5. `model_manager.generate_embedding_async` calls `embedding_service.generate_single_embedding`

### Key Files and Import Locations

1. **packages/vecpipe/search_api.py**
   - Line 26: `from webui.embedding_service import EmbeddingService`
   - Line 192: Initializes embedding_service in lifespan function

2. **packages/vecpipe/model_manager.py**
   - Line 13: `from webui.embedding_service import EmbeddingService`
   - Line 54: Creates EmbeddingService instance
   - Lines 164-166: Calls `embedding_service.generate_single_embedding`

### Integration Test Details

The integration test created in `tests/integration/test_search_api_integration.py` mocks:
- `packages.webui.embedding_service.embedding_service.generate_single_embedding`
- `packages.vecpipe.model_manager.EmbeddingService`

The test includes a TODO comment on line 9:
```python
# TODO: Update patch path in this test after CORE-003 is merged
```

### Required Updates for CORE-003

When moving the embedding service to the shared package, you'll need to:

1. **Update import statements** in:
   - `packages/vecpipe/search_api.py` (line 26)
   - `packages/vecpipe/model_manager.py` (line 13)
   - Any other files that import from `webui.embedding_service`

2. **Update the integration test** mock paths:
   - Change `packages.webui.embedding_service.embedding_service.generate_single_embedding` to the new shared package path
   - Change `packages.vecpipe.model_manager.EmbeddingService` to use the new import path

3. **Consider other imports**: The embedding service may be imported in other locations not covered by this test.

### Additional Observations

- The search_api doesn't directly use the embedding_service for generating embeddings; it goes through model_manager
- The model_manager acts as an intermediary, providing lazy loading and automatic unloading of models
- The embedding_service is also used for status checks in the search_api (lines 316-327, 1039-1041, 1054-1056, 1119-1136)

### Testing Challenges Discovered

During TEST-002 implementation, we discovered several architectural issues that make testing difficult:

1. **Settings loaded at module import time**: The `settings` object in `config.py` is instantiated when the module is imported, making it impossible to override `USE_MOCK_EMBEDDINGS` in tests after import.

2. **Current test limitations**: 
   - Tests cannot easily control whether mock or real embeddings are used
   - The `model_used` field in responses depends on environment configuration
   - Integration tests are more like "flow tests" rather than true integration tests

3. **Test files created**:
   - `test_search_api_integration.py`: Original test with mocking challenges
   - `test_search_api_embedding_flow.py`: Alternative approach that documents current constraints

### Recommendations for CORE-003

1. **Dependency Injection**: Consider making settings injectable rather than globally imported
2. **Factory Pattern**: Use factory functions to create services with configurable settings
3. **Better Testability**: Design the shared package with testing in mind from the start

### Testing After Refactor

After completing CORE-003, run:
```bash
poetry run pytest tests/integration/test_search_api_integration.py tests/integration/test_search_api_embedding_flow.py -v
```

This will verify that the refactored structure maintains the expected behavior and hopefully allows for better test isolation.