# Search Reranking Tests

Tests for search reranking (cross-encoder re-scoring). All tests run without GPU or model dependencies.

## Test Files

**Backend:**
- `tests/webui/api/v2/test_search.py` - API endpoint tests
- `tests/webui/services/test_search_service_reranking.py` - Service layer tests
- `tests/integration/test_search_reranking_integration.py` - Integration tests

**Frontend:**
- `apps/webui-react/src/components/__tests__/SearchInterface.reranking.test.tsx`
- `apps/webui-react/src/stores/__tests__/searchStore.reranking.test.ts`

## Running Tests

```bash
# Backend
uv run pytest tests/ -k "rerank" -v

# Frontend
cd apps/webui-react && npm test -- SearchInterface.reranking
```

## Fixtures

`tests/fixtures/search_reranking_fixtures.py`:
- `create_mock_collection()`, `create_search_result()`, `create_vecpipe_response()`

All external deps (vecpipe, DB, Qdrant) are mocked.
