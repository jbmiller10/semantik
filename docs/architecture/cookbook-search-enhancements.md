# Extension Cookbook: Search Enhancements

> **Audience:** Engineers extending search behavior in Semantik
> **Prerequisites:** Familiarity with VecPipe search flow and WebUI request handling

---

## Table of Contents

1. [Add a New Search Type](#1-add-a-new-search-type)
2. [Add or Extend Reranking](#2-add-or-extend-reranking)
3. [Add Search Filters](#3-add-search-filters)
4. [Add Search Analytics](#4-add-search-analytics)

---

## 1. Add a New Search Type

### Overview
Search types are defined by `search_type` in `SearchRequest` and routed in VecPipe. Current types are `semantic`, `question`, `code`, and `hybrid` (with legacy alias `vector`).

### Steps
1. Update validation in `packages/shared/contracts/search.py` (`SearchRequest.validate_search_type`).
2. Add any per-type embedding instruction in `packages/vecpipe/search/service.py` (`SEARCH_INSTRUCTIONS`).
3. Route the new type in `packages/vecpipe/search/service.py` (`perform_search`).
4. Update route descriptions in `packages/vecpipe/search/router.py` (query descriptions and docs).
5. Expose the new type in the UI:
   - `apps/webui-react/src/components/search/SearchForm.tsx`
   - `apps/webui-react/src/stores/searchStore.ts`
   - `apps/webui-react/src/services/api/v2/types.ts`
6. Update `docs/SEARCH.md` and `docs/API_REFERENCE.md` to reflect the new type.

### Validation
- Add/extend tests in `tests/` or `packages/vecpipe` to cover routing and request validation.

---

## 2. Add or Extend Reranking

### Overview
Reranking is implemented in VecPipe via a cross-encoder model and managed by the model manager. The request contract exposes `use_reranker`, `rerank_model`, and `rerank_quantization`.

### Steps
1. Implement or extend the reranker in `packages/vecpipe/reranker.py`.
2. Wire it through `packages/vecpipe/model_manager.py` (loading and `rerank_async`).
3. Update model selection defaults in `packages/vecpipe/qwen3_search_config.py` if needed.
4. Ensure any new config is surfaced via `packages/shared/config` and documented in `docs/CONFIGURATION.md`.
5. Update UI controls in `apps/webui-react/src/components/search/SearchOptions.tsx` and state in `apps/webui-react/src/stores/searchStore.ts`.

### Validation
- Add tests for `rerank_async` and search responses in `packages/vecpipe` or `tests/`.

---

## 3. Add Search Filters

### Overview
`SearchRequest.filters` accepts a raw Qdrant filter dict and is passed directly in the VecPipe search call.

### Steps
1. Define/validate any filter schema you want to support in `packages/shared/contracts/search.py` (or a helper module).
2. Ensure the WebUI passes `filters` through to the search request (`apps/webui-react/src/components/search/SearchForm.tsx`).
3. Confirm the filter payload aligns with Qdrant’s filter schema.

### Validation
- Add a unit/integration test that asserts the filter dict reaches Qdrant with the expected shape.

---

## 4. Add Search Analytics

### Overview
VecPipe already emits Prometheus counters and timings for search requests and latency.

### Steps
1. Add new metrics in `packages/vecpipe/search/metrics.py` or `packages/shared/metrics/prometheus.py` as needed.
2. Emit metrics in `packages/vecpipe/search/service.py` at the relevant points.
3. If exposing metrics in the UI, update WebUI endpoints and UI components accordingly.

### Validation
- Verify the metric appears on the Prometheus endpoint and increments as expected.
