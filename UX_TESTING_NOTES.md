# Semantik UX Testing Report

**Date**: 2026-02-05
**Tester**: Claude (automated UX testing via Chrome)
**App Version**: feature/agent-sdk-migration branch
**Environment**: Docker (all services up; qdrant and worker reported unhealthy)
**URL**: http://localhost:8080

---

## Executive Summary

Semantik has a polished, professional UI with a dark theme and well-structured navigation. The core flows (login, collection creation, settings) are generally smooth. However, several significant UX issues were found, primarily around **form validation**, **error messaging**, and **state inconsistencies**.

**Overall UX Rating: 7/10** - Good foundation with several areas needing improvement.

---

## Test Results

### 1. Login & Authentication

| Test | Status | Notes |
|------|--------|-------|
| Login page load | PASS | Clean, centered form with Semantik branding. Professional look. |
| Login with valid credentials | PASS | Smooth redirect to Collections page. No delay. |
| "Remember me" checkbox | PASS | Present and checked by default. |
| Sign out button | PASS | Visible in top-right header. |
| "Sign up" link | PRESENT | Available below login form. |

**UX Positives:**
- Clean, minimal login page with good visual hierarchy
- Clear "Welcome back / Sign in to continue" messaging
- Password visibility toggle (eye icon)

**UX Issues:** None found.

---

### 2. Navigation & Layout

| Test | Status | Notes |
|------|--------|-------|
| Tab navigation (Collections, Active Operations, Benchmarks, Search) | PASS | Clear tab bar with active state indicator |
| Settings page access | PASS | Via "Settings" link in header |
| Back to Home from Settings | MINOR ISSUE | Returns to last active tab, not always Collections |
| Logo/brand link | NOT TESTED | Should link to home page |

**UX Positives:**
- Consistent top navigation bar across all pages
- Clean tab-based architecture
- "Signed in as [username]" always visible
- Settings page has well-organized sub-tabs (Preferences, Admin, System, Plugins, MCP Profiles, API Keys, Models)

**UX Issues:**
- **(LOW)** "Back to Home" from Settings returns to the last active tab (e.g., Search), not the Collections (default) tab. May confuse first-time users.

---

### 3. Collections Dashboard

| Test | Status | Notes |
|------|--------|-------|
| Collection cards display | PASS | 3-column grid layout, good information density |
| Collection search bar | PRESENT | Placeholder "Search collections..." |
| Status filter dropdown | PARTIAL | Dropdown opens with options (All Status, Pending, Ready, Processing, Error) |
| Error badge display | PASS | Red error count badges visible on cards |
| "Manage" link on cards | PASS | Navigates to collection detail view |

**UX Positives:**
- Cards show key info at a glance: name, description, model badge, doc/vector counts, last updated
- Status badges (Ready, Processing, Pending) are color-coded and clear
- Error count badges are prominent
- "+ New Collection" button is well-positioned

**UX Issues:**
- **(MEDIUM)** Status filter may not differentiate between "Ready with errors" and "Error" status. Many collections show "Ready" with error badges, but filtering by "Error" doesn't surface them. The distinction is unclear to users.
- **(LOW)** No sort options (by name, date, size, status). Only filter by status.
- **(LOW)** No pagination - appears to use infinite scroll for large number of collections.
- **(LOW)** No bulk operations (multi-select, bulk delete) for managing many test collections.

---

### 4. Collection Creation Wizard

| Test | Status | Notes |
|------|--------|-------|
| 3-step wizard flow | PASS | Clear progress indicator (Basics & Source -> Mode Selection -> Configure Pipeline) |
| Name & description fields | PASS | Required name, optional description |
| Source type selection (None, Local Directory, Git Repo, Email IMAP) | PASS | Visual card selection with icons |
| Local Directory config | PASS | Path input, recursive toggle, include/exclude patterns |
| Mode selection (Assisted vs Manual) | PASS | Clear descriptions for each option |
| Pipeline visual builder | PASS | Interactive flow: Source -> Parser -> Chunker -> Embedder |
| Pipeline node config panel | PASS | Right-side panel shows node-specific settings |

**UX Positives:**
- Visual pipeline builder is impressive and intuitive
- Source type cards have clear icons and descriptions
- Default include patterns (*.md, *.txt) are sensible
- "Switch to Assisted mode" link available on manual pipeline page
- Chunker and Embedder nodes have + buttons for alternatives

**UX Issues:**
- **(CRITICAL)** **Missing form validation**: The wizard allows creating a collection with no embedding model selected. When no models are installed, the embedder config shows "No embedding models installed" in red text, but the "Create Collection" button is NOT disabled. Clicking it results in: a timeout error ("Timed out waiting for collection to become ready"), AND the collection is partially created in a broken "Pending" state with 0 docs/0 vectors. The form should validate model selection and either disable the button or show a blocking error.
- **(HIGH)** **Misleading error message**: The timeout error ("Timed out waiting for collection to become ready") doesn't tell the user the actual problem (no model selected). It should say something like "No embedding model configured. Install a model from Settings > Models first."
- **(MEDIUM)** **Orphaned collection**: After the creation error, the collection still exists on the dashboard in a "Pending" state that can never progress. Users would need to manually delete it.

---

### 5. Collection Detail / Management

| Test | Status | Notes |
|------|--------|-------|
| Overview tab (stats) | PASS | Documents, Vectors, Total Size, Operations cards |
| Jobs tab | NOT TESTED | |
| Files tab | PASS | Table with name, size, status, created date |
| Visualize tab | PASS | UMAP/t-SNE/PCA projection options |
| Settings tab | PASS | Shows model, chunking config, re-index, sparse indexing |
| Add Data button | PRESENT | |
| Rename button | PRESENT | |
| Delete button | PRESENT | Red/destructive styling |

**UX Positives:**
- Files tab shows clear table with status badges (completed/failed) per file
- "RETRY ALL FAILED" button for batch retry
- Individual retry button per file
- Visualize tab offers multiple projection methods with descriptions
- Settings tab clearly shows chunking parameters
- Re-index warning is prominent with destructive action styling
- Sparse indexing toggle is clearly presented

**UX Issues:**
- **(MEDIUM)** Files tab shows no explanation for WHY a file failed. Hover tooltip or expandable error detail would help. (e.g., PNG files failed but user doesn't know it's because images can't be embedded as text)
- **(LOW)** No file search/filter within the files list.
- **(LOW)** Collection detail opens as a modal overlay, which works well for quick access but limits screen real estate for complex operations.

---

### 6. Search

| Test | Status | Notes |
|------|--------|-------|
| Search page layout | PASS | Clean form with query input, collection picker, mode selection |
| Collection multi-select picker | EXCELLENT | Search filter, Select All/Clear All, shows doc/vector counts and model per collection |
| Search mode toggle (Dense/Sparse/Hybrid) | PASS | Visual card selection with icons and descriptions |
| Embedding mode dropdown | PASS | "General" default |
| Cross-Encoder Reranking checkbox | PASS | With info tooltip |
| HyDE Query Expansion checkbox | PASS | With info tooltip |
| Advanced Options collapsible | PASS | Expandable section |
| Search with unavailable service | PARTIAL | Shows error but could be more helpful |

**UX Positives:**
- Collection picker is excellent: filters, counts, model info, multi-select with checkboxes
- Note: "Only showing collections that are ready with indexed vectors" is very helpful
- Search mode cards have clear icons and concise descriptions
- Advanced options are hidden by default to avoid overwhelming new users

**UX Issues:**
- **(MEDIUM)** Search error message ("Search service unavailable for collection X (status: 503)") is technically accurate but doesn't help users. Should suggest: "The embedding model could not be loaded. Check GPU memory or restart the VecPipe service."
- **(LOW)** No loading indicator visible during search (or it was very brief). Users may not know the search is in progress.
- **(LOW)** "Search Results: Found 0 results across 0 collections" shown even when all searches failed - this is misleading. Should show "No successful searches" instead.
- **(LOW)** No keyboard shortcut to trigger search (Enter key from query field should submit).

---

### 7. Settings Pages

| Test | Status | Notes |
|------|--------|-------|
| Preferences tab | PASS | Collapsible sections for Search, Collection Defaults, LLM Config, Interface |
| Models tab (Model Manager) | PASS | Tabs for Embedding/Local LLM/Reranker/SPLADE with install counts |
| Search preferences | PASS | Results count, mode, reranker, threshold, HyDE defaults |

**UX Positives:**
- Well-organized with collapsible sections
- Model Manager shows RAM requirements and download buttons
- Filter buttons (All/Installed/Available) for models
- Search functionality for models
- Cache usage monitoring available
- Help text under each setting field

**UX Issues:** None significant found.

---

## Infrastructure Issues (Not UX bugs, but affect UX)

| Issue | Impact | Notes |
|-------|--------|-------|
| VecPipe GPU memory exhaustion | CRITICAL | 91% GPU used, 24GB reserved but 0 models loaded. All search returns 503. |
| Qdrant unhealthy status | HIGH | May affect search results even with working embedder |
| Worker unhealthy status | MEDIUM | May affect document indexing operations |
| "Processing" collections but no active operations | CONFUSING | Collections show "Processing" but Active Operations tab shows empty |

---

## Summary of Issues by Severity

### Critical (1)
1. **Collection creation without model validation** - Allows creating a broken collection with no model, leaving orphaned records

### High (1)
1. **Misleading error on collection creation** - "Timed out" error instead of telling user the real problem

### Medium (4)
1. **Status filter confusion** - "Ready with errors" vs "Error" status distinction unclear
2. **File failure reasons hidden** - No explanation for why individual files failed to process
3. **Search error not actionable** - 503 error doesn't suggest what user can do
4. **Processing/Active Ops inconsistency** - Collections show "Processing" but no active operations listed

### Low (7)
1. No collection sort options
2. No pagination for collections
3. No bulk collection operations
4. "Back to Home" returns to last tab, not default
5. No file search/filter in collection detail
6. "0 results across 0 collections" when all searches failed
7. No loading indicator or Enter-key search submit

---

## Code Investigation Findings

*Investigated by a 4-agent team analyzing the codebase in parallel.*

### Finding 1: Collection Creation Validation Gap (CRITICAL)

**Root Cause:** Three layers of silent fallback, zero validation.

| Layer | File | Issue |
|-------|------|-------|
| Frontend wizard | `apps/webui-react/src/components/wizard/CollectionWizard.tsx:286-291` | `isNextDisabled` only checks `name.trim()` for step 0. **Zero validation on step 2** (Configure Pipeline). |
| Frontend fallback | `CollectionWizard.tsx:220` | `handleCreate` falls back: `embedderNode?.config?.model || embedderNode?.plugin_id || 'sentence-transformers/all-MiniLM-L6-v2'` - cascades to invalid values. |
| Backend fallback | `packages/webui/services/collection_service.py:156` | Falls back to `Qwen/Qwen3-Embedding-0.6B` if model is empty, **never validates model is installed**. |
| Orphaned record | `collection_service.py:271` | Creates DB record + Celery task. The task fails later when model can't load, but collection remains in "Pending" forever. |

**Key files to fix:**
- `apps/webui-react/src/components/wizard/CollectionWizard.tsx` (validation in `isNextDisabled` + `handleCreate`)
- `apps/webui-react/src/components/pipeline/NodeConfigEditor.tsx:30-96` (ModelSelectorField should propagate error state)
- `packages/webui/services/collection_service.py:126-294` (server-side model validation)
- `packages/webui/api/schemas.py:83` (Pydantic schema has default, no installed-model validator)

**Fix approach:** Disable "Create Collection" when embedder has no model. Add backend validation that model is installed before creating records.

---

### Finding 2: Error Messaging Issues (HIGH/MEDIUM)

**"Timed out" error in collection creation:**
- Generated in the frontend wizard's `handleCreate` method which polls for collection readiness after creation
- The real failure (model not installed) happens in the Celery worker, not returned to the frontend
- The frontend only sees "collection not ready after N seconds" and shows a generic timeout

**Search 503 error:**
- VecPipe returns 503 when it can't load the embedding model (GPU OOM)
- The webui proxies this error but only surfaces the HTTP status code
- VecPipe logs contain detailed info (memory stats, model requirements) that is NOT returned in the response body

**"Found 0 results across 0 collections":**
- The search results component renders this summary regardless of whether searches succeeded or all failed
- Should differentiate between "searched and found nothing" vs "all searches failed"

---

### Finding 3: Status Filter & Processing/Active Ops Inconsistency (MEDIUM)

**Status filter confusion:**
- Collection status is a database enum: `pending`, `ready`, `processing`, `error`
- The `error_count` badge on cards is separate - it counts **failed documents/operations**, not the collection status
- A collection can be `ready` (all operations complete) but have `error_count > 0` (some files failed)
- The "Error" filter only matches `status = 'error'` (collection-level failure), NOT `error_count > 0`
- Users see error badges and expect the "Error" filter to capture them

**Processing/Active Ops inconsistency:**
- Collections stuck in "Processing" likely have operations where the Celery worker died (unhealthy worker)
- Active Operations queries for operations with `status IN ('pending', 'running')` but the worker may have crashed without updating the operation status
- The operation record might still say "running" in Postgres but there's no actual Celery task executing
- Active Operations may filter by a different criteria (e.g., recently updated) causing stale operations to not appear

---

### Finding 4: File Failure Details (MEDIUM)

**The backend ALREADY stores error details - the frontend just doesn't display them.**

| Data Available | Where Stored | Surfaced to UI? |
|---------------|-------------|-----------------|
| `error_message` (text) | `documents.error_message` column | **Returned by API but NOT displayed** in Files tab |
| `error_category` (transient/permanent/unknown) | `documents.error_category` column | Not returned by list endpoint (omitted in manual response construction) |
| `retry_count` | `documents.retry_count` column | Not returned |
| Per-stage failure details | `PipelineFailure` table (stage, error type, traceback) | No API endpoint |

**Quick win:** `CollectionDetailsModal.tsx:421-427` renders the status badge but ignores `doc.error_message` which is **already in the API response**. Just add a tooltip.

**API fix needed:** `packages/webui/api/v2/collections.py:550-568` manually constructs `DocumentResponse` but omits `error_category`, `retry_count`, `last_retry_at` fields. The Pydantic schema already has these fields.

**Collections dashboard:** Sort is hardcoded to `updated_at desc` in `CollectionsDashboard.tsx:36-37`. No user-facing sort options exist.

---

## Recommendations

1. **Immediate**: Add form validation to collection creation wizard - disable "Create Collection" when no model is selected, or show a blocking validation error.
2. **Immediate**: Improve error messages to be actionable - tell users what went wrong and what they can do.
3. **Quick win**: Show `error_message` tooltip on failed files in the Files tab - data already available in the API response, just needs frontend display.
4. **Quick win**: Pass `error_category` and `retry_count` in `list_collection_documents` API response (fields already defined in schema, just omitted).
5. **Short-term**: Add "Has Errors" option to the status filter, or add a separate "Errors" filter that matches `error_count > 0`.
6. **Short-term**: Add sort options (name, date, size) to the collections dashboard.
7. **Medium-term**: Add bulk operations for collection management.
8. **Medium-term**: Add GPU memory status indicator visible to users, so they understand when search will be unavailable.
9. **Medium-term**: Handle stale "Processing" operations - detect when a Celery worker has died and mark operations as failed.
