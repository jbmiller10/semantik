# Phase 1 Ticket: Qdrant Usage Metrics & Hybrid Search Contract (Target: October 27 – November 7, 2025)

## Background
The audit and internal TODOs noted placeholder usage metrics (`packages/webui/services/resource_manager.py:203`) and mismatched hybrid search payloads causing HTTP 422 errors. This undermines Semantik’s portfolio pitch since the dashboards show zero usage and hybrid search reranking appears broken. This ticket focuses on the Qdrant integration and search contract alignment while the sibling ticket handles sharing/quotas.

## Objectives
1. Report accurate vector usage/storage metrics by querying Qdrant.
2. Complete collection rename flows to keep Postgres and Qdrant in sync.
3. Align hybrid search request/response contracts and reranking logic across backend and frontend.

## Requirements
### 1. Qdrant Usage Reporting
- Implement the placeholder in `packages/webui/services/resource_manager.py` so the settings/analytics endpoints report actual vector counts and storage per collection.
  - Use existing Qdrant client utilities (check `packages/shared/managers/qdrant_manager.py`).
  - Cache responses appropriately to avoid performance issues (short TTL or reuse existing caching layer).
- Add error handling for unavailable Qdrant instances, returning a graceful fallback message.
- Update tests (`tests/webui/services/test_resource_manager.py`) using a mocked Qdrant client to cover success and failure modes.

### 2. Collection Rename Completeness
- Address audit notes about incomplete Qdrant rename support. Ensure renaming a collection in our API updates both the relational metadata and Qdrant collection name.
- Provide transactional safety: rollback or compensate if Qdrant rename fails.
- Add integration tests validating rename behavior and ensuring usage metrics follow the new name.

### 3. Hybrid Search Contract & Reranking
- Backend (`packages/webui/services/search_service.py`):
  - Map legacy form values (`weighted`, `bm25`) to the accepted enums (`hybrid_mode: rerank`, `keyword_mode: any`).
  - Ensure all request payloads match Pydantic models to prevent 422 errors.
  - Sort results using `reranked_score` when present, falling back to `score`.
- Frontend (`apps/webui-react/src/services/api/v2/types.ts`, related hooks):
  - Update TypeScript types and request builders to use the new enums/fields.
  - Ensure UI defaults align with backend expectations.
- Tests:
  - Python integration test hitting `/api/v2/search` for hybrid mode verifying HTTP 200 and proper reranked ordering (`tests/webui/api/v2/test_search_hybrid.py`).
  - Unit test ensuring `search_service` sorts by `reranked_score`.
  - Frontend test confirming the request object matches the contract.

## Acceptance Criteria
- Settings/analytics endpoints show real Qdrant usage values rather than zeros; errors are handled gracefully when Qdrant is down.
- Collection rename updates both DB and Qdrant collections, with tests covering success and failure scenarios.
- Hybrid search requests succeed without validation errors, and reranked results are sorted by rerank score by default.
- `uv run pytest tests -k "qdrant or hybrid"` passes, and relevant frontend tests compile/run (`npm test --prefix apps/webui-react -- SearchInterface`).

## Validation Steps
1. `uv run pytest tests/webui/services/test_resource_manager.py::TestQdrantUsage`
2. `uv run pytest tests/webui/api/v2/test_search_hybrid.py -v`
3. Manual API smoke test executing a hybrid search via `curl` or Postman to confirm fields and ordering.
4. `npm test --prefix apps/webui-react -- SearchInterface` (or equivalent) verifying updated contract.

## Coordination Notes
- Coordinate with the sharing/quotas ticket to avoid conflicting edits in `resource_manager.py`.
- Inform DevOps if Qdrant client configuration changes (env vars, timeouts).
- Align with frontend engineers so UI updates release in tandem.

## Out of Scope
- Building new analytics dashboards beyond exposing accurate metrics.
- Implementing new search modalities (semantic-only, keyword-only changes already exist).
- Large-scale performance optimization of Qdrant queries beyond what’s required for correctness.
- UI work beyond updating request builders/tests for the new contract (covered in companion frontend ticket).
