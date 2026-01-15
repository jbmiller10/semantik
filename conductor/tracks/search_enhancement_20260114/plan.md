# Implementation Plan - Search System Enhancement

## Phase 1: Backend Support for Reranking and Hybrid Tuning

- [ ] Task: Update Shared Search Schemas
    - [ ] Create/Update Pydantic models in `packages/shared` to include `alpha` (hybrid weight) and `rerank` (boolean) configuration.
    - [ ] Add `rerank_limit` and `rerank_model` configuration options.

- [ ] Task: Implement Reranking Logic in Vecpipe
    - [ ] Add dependency for cross-encoders (if not present).
    - [ ] Implement `Reranker` class in `packages/vecpipe`.
    - [ ] Add endpoint or logic to rerank a list of (query, document) pairs.
    - [ ] Write unit tests for the reranker.

- [ ] Task: Integrate Hybrid Tuning in Search Service
    - [ ] Modify the search service in `packages/vecpipe` or `packages/shared` to use the `alpha` parameter for combining sparse and dense scores.
    - [ ] Ensure `alpha=0` (keyword only), `alpha=1` (semantic only), and intermediate values work correctly.
    - [ ] Write unit tests for score combination logic.

- [ ] Task: Expose New Parameters in WebUI API
    - [ ] Update `packages/webui/api/v2/search.py` endpoint to accept the new search configuration schema.
    - [ ] Pass these parameters down to the search service/vecpipe.
    - [ ] Write integration tests for the updated search API.

- [ ] Task: Conductor - User Manual Verification 'Backend Support for Reranking and Hybrid Tuning' (Protocol in workflow.md)

## Phase 2: Frontend UI Controls

- [ ] Task: Update Search API Client
    - [ ] Update frontend API client types and functions to match the new backend search API signature.

- [ ] Task: Add Hybrid Search Controls
    - [ ] Create a "Search Tuning" component in `apps/webui-react`.
    - [ ] Add a slider for `alpha` (Keyword <-> Semantic).
    - [ ] Add a tooltip explaining the trade-off.

- [ ] Task: Add Reranking Controls
    - [ ] Add a toggle switch for "Rerank Results".
    - [ ] (Optional) Add a dropdown to select reranking model if multiple are available.

- [ ] Task: Visualize Search Scores
    - [ ] Update the search result list item component to display the final score.
    - [ ] (Optional) Show breakdown of sparse vs. dense scores if helpful for debugging/transparency.

- [ ] Task: Conductor - User Manual Verification 'Frontend UI Controls' (Protocol in workflow.md)

## Phase 3: Integration and Performance Tuning

- [ ] Task: End-to-End Testing
    - [ ] Write an E2E test (Playwright) that performs a search with specific tuning and verifies the results change as expected.
    - [ ] Verify reranking improves relevance on a known test set (if possible).

- [ ] Task: Performance Benchmarking
    - [ ] Measure latency impact of reranking with different `rerank_limit` values.
    - [ ] Optimize batch sizes or model selection if latency is too high.

- [ ] Task: Conductor - User Manual Verification 'Integration and Performance Tuning' (Protocol in workflow.md)
