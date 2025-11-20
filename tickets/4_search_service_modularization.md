Title: Modularize vecpipe search service and remove legacy search params

Background
- `packages/vecpipe/search_api.py` (>1.5k LOC) mixes routing, service logic, metrics, rerank, and upserts in one file.
- WebUI `SearchService` still normalizes legacy `hybrid_search_mode/keyword_mode`; prerelease allows removal.
- Shared contracts exist but are loosely enforced.

Goal
Split vecpipe into cohesive modules, enforce a single search contract, and drop legacy fields end-to-end.

Scope
- Extract modules: router, service/core logic, metrics, rerank helpers, models/schemas; leave entrypoint <300 LOC.
- Finalize contract in `shared/contracts/search.py`; remove `hybrid_search_mode`/`keyword_mode` legacy fields from server and WebUI client; adjust payload builders.
- Add tests (FastAPI TestClient/httpx async) for semantic and hybrid search, rerank on/off, and error/status mapping.
- Update docs (`docs/API_ARCHITECTURE.md` or new SEARCH.md) with request/response examples for each mode.

Out of Scope
- Changing model selection policy or Qdrant schema.

Suggested Steps
1) Create new modules and refactor entrypoint to delegate.
2) Update shared contract types; refactor WebUI `SearchService` to use canonical fields only.
3) Add targeted tests covering success/error paths and rerank toggle; ensure metrics middleware still registers.
4) Refresh docs with examples and removal of legacy fields.

Acceptance Criteria
- Entry file small and delegates to modules; typing passes.
- No code accepts or forwards `hybrid_search_mode`/legacy fields; WebUI sends canonical fields only.
- Tests cover semantic/hybrid/rerank paths and error mapping; CI passes.
- Documentation reflects the simplified contract with examples.
