You are absolutely right to call that out. My apologies. In my focus on refining the backend plan, I truncated the list and didn't carry over the subsequent phases. That was a significant oversight on my part.

A plan that stops after the backend implementation is an incomplete plan. Let's correct that immediately.

Here is the **complete, fully-revised Master Task List**, including all phases from start to finish, incorporating all the feedback and refinements we've discussed. This version is a comprehensive, end-to-end roadmap.

***

# **Collections Refactor - Master Task List**

**Document Version:** 3.1 (Complete)
**Status:** Final
**Date:** 2025-07-15
**Estimated Timeline:** 9-11 weeks

## **1. Overview**

This document breaks down the Collection-Centric Architecture refactor into a sequence of executable developer tasks. The primary goal is to re-architect the application around the concept of "Collections" as persistent, user-managed knowledge bases, moving away from the previous job-centric model.

Given that Semantik is pre-release, this plan prioritizes building the optimal architecture without the constraints of backwards compatibility or data migration.

**Total Tasks:** 32
**Critical Path:** Tasks marked with 🔴 are on the critical path and block subsequent work.

---

## **Phase 1: Project Scaffolding & Database (Week 1)**

### TASK-000: Project Branching & Tooling Setup 🔴
**Priority:** Critical | **Effort:** 2 hours | **Assignee:** Tech Lead
**Context:** Establish the foundational structure for the refactoring work.
**Requirements:**
1.  Create a new long-lived feature branch: `feature/collections-refactor`.
2.  Update the root `README.md` on this branch with a prominent notice about the refactor.
3.  Create a new, blank Alembic revision file: `alembic revision -m "init_collections_schema"`.
4.  Set up a project board (e.g., GitHub Projects) to track all tasks.
**Acceptance Criteria:** Branch exists, board is created, blank migration file is committed.
**How it fits:** Sets up the workspace for the entire refactor.

---

### TASK-001: Establish Initial Database Schema 🔴
**Priority:** Critical | **Effort:** 4 hours | **Dependencies:** TASK-000 | **Assignee:** Backend Developer
**Context:** Implement the new, optimized schema from scratch via an Alembic migration.
**Requirements:**
1.  In the new migration file, script the dropping of all old tables (`jobs`, `files`, etc.) using `op.drop_table(..., if_exists=True)`.
2.  Implement the new schema for `collections`, `collection_sources`, `operations`, and `documents` tables as defined in the execution plan.
3.  Use `sa.DateTime(timezone=True)` for all timestamp fields.
4.  Define all foreign keys, `ON DELETE CASCADE` constraints, and indexes.
**Acceptance Criteria:** `alembic upgrade head` runs successfully on a fresh database; `downgrade` is functional.
**How it fits:** Creates the database foundation for the new architecture.

---

### TASK-002: Implement SQLAlchemy Models 🔴
**Priority:** Critical | **Effort:** 6 hours | **Dependencies:** TASK-001 | **Assignee:** Backend Developer
**Context:** Create SQLAlchemy models in `packages/shared/database/models.py` that map to the new database schema.
**Requirements:**
1.  Use `sqlalchemy.DateTime(timezone=True)`.
2.  Define `relationship()` with correct `back_populates` and `cascade` options.
3.  Use Python `Enum` for all `status` fields.
**Acceptance Criteria:** Models for `Collection`, `CollectionSource`, `Operation`, `Document` are implemented and match the schema precisely.
**How it fits:** Provides the ORM layer for all database interactions.

---

### TASK-003: Implement Collection & Operation Repositories
**Priority:** High | **Effort:** 8 hours | **Dependencies:** TASK-002 | **Assignee:** Backend Developer
**Context:** Create the repository layer for the new `collections` and `operations` tables.
**Requirements:**
1.  Create `CollectionRepository` and `OperationRepository` in `packages/shared/database/repositories/`.
2.  Implement all necessary CRUD methods (`create`, `get_by_uuid`, `update_status`, etc.).
3.  Ensure methods are atomic and enforce `user_id` permission checks.
4.  The `OperationRepository.set_task_id` method must be an immediate, high-priority update.
**Acceptance Criteria:** Repositories provide a clean, async interface for database operations, raising custom exceptions for errors.
**How it fits:** The repository layer will be used by the `CollectionService` to interact with the database.

---

### TASK-004: Implement Document Repository with Deduplication
**Priority:** High | **Effort:** 6 hours | **Dependencies:** TASK-002 | **Assignee:** Backend Developer
**Context:** Create the repository for the `documents` table, including the logic for content-based deduplication.
**Requirements:**
1.  Create `DocumentRepository` in `packages/shared/database/repositories/`.
2.  Implement an atomic `create_if_not_exists` method that computes a file's SHA-256 hash and checks for existence within the same `collection_id`.
3.  The method should return a tuple: `(Document, bool)` where the boolean indicates if the document was newly created.
**Acceptance Criteria:** Correctly identifies and skips duplicate files within a collection's scope.
**How it fits:** Prevents redundant data processing and storage.

---

## **Phase 2: Core Services (Week 2)**

### TASK-005: Implement `get_user_collection` Security Dependency 🔴
**Priority:** Critical | **Effort:** 3 hours | **Dependencies:** TASK-003 | **Assignee:** Backend Developer
**Context:** Standardize security by creating a reusable FastAPI dependency to fetch a collection and verify user ownership.
**Requirements:**
1.  Create a new function, `get_collection_for_user`, in `packages/webui/dependencies.py`.
2.  The function must accept a `collection_uuid` and `current_user` from `Depends`.
3.  It must raise an `HTTPException(404)` if the collection is not found and an `HTTPException(403)` if the user does not have access.
4.  On success, it returns the `Collection` ORM object.
**Acceptance Criteria:** All collection-specific API endpoints will use this dependency to enforce access control.
**How it fits:** Centralizes security logic, making the API more robust and DRY.

---

### TASK-006: Implement Collection Service
**Priority:** High | **Effort:** 8 hours | **Dependencies:** TASK-003, TASK-004 | **Assignee:** Backend Developer
**Context:** Create the `CollectionService` to contain the business logic for managing collections.
**Requirements:**
1.  Create `packages/webui/services/collection_service.py`.
2.  Implement methods for `create_collection`, `add_source`, `reindex_collection`, and `delete_collection`.
3.  Enforce state transitions (e.g., prevent re-indexing a collection that is already `indexing`).
4.  All methods that trigger background work must create an `Operation` record and dispatch the corresponding Celery task.
**Acceptance Criteria:** The service correctly validates inputs and collection states before performing actions.
**How it fits:** This is the core business logic layer separating the API from data manipulation.

---

### TASK-007: Implement Qdrant Management Service
**Priority:** High | **Effort:** 6 hours | **Dependencies:** None | **Assignee:** Backend Developer
**Context:** Create a service to manage Qdrant resources, supporting the blue-green deployment strategy.
**Requirements:**
1.  Create `packages/shared/managers/qdrant_manager.py`.
2.  Implement `create_staging_collection(base_name)` to create uniquely named collections.
3.  Implement `cleanup_orphaned_collections(active_collections)` to safely delete old collections.
**Acceptance Criteria:** Staging collections are created correctly; cleanup logic is robust and safe.
**How it fits:** Enables zero-downtime reindexing at the vector storage layer.

---

### TASK-008: Implement File Scanning Service
**Priority:** High | **Effort:** 6 hours | **Dependencies:** TASK-004 | **Assignee:** Backend Developer
**Context:** Create a service for scanning source directories and coordinating deduplication.
**Requirements:**
1.  Create `packages/webui/services/file_scanning_service.py`.
2.  Implement `scan_directory_and_register_documents` which uses `DocumentRepository.create_if_not_exists`.
3.  The service must return detailed statistics (`total_files_found`, `new_files_registered`, `duplicate_files_skipped`).
**Acceptance Criteria:** The service correctly identifies new vs. duplicate files and returns accurate statistics.
**How it fits:** This is the first step in any indexing or append operation.

---

## **Phase 3: Task Processing & API (Weeks 3-4)**

### TASK-009: Refactor Celery Task Structure 🔴
**Priority:** Critical | **Effort:** 12 hours | **Dependencies:** TASK-006, TASK-007, TASK-008 | **Assignee:** Backend Developer
**Context:** Create a unified Celery task to handle all collection operations.
**Requirements:**
1.  Create a new task `process_collection_operation(operation_id)` in `packages/webui/tasks.py`.
2.  Set `acks_late=True` and define `soft_time_limit`/`time_limit` in the decorator.
3.  **The first action** in the task must be to update the `operations` record with `celery_task_id = self.request.id`.
4.  Implement a `try...finally` block to guarantee final status updates (`completed` or `failed`).
**Acceptance Criteria:** A single, robust Celery task entry point is implemented that correctly routes to different handlers.
**How it fits:** This is the core background processing engine.

---

### TASK-010: Implement Blue-Green Staging Creation
**Priority:** High | **Effort:** 6 hours | **Dependencies:** TASK-009 | **Assignee:** Backend Developer
**Context:** Implement the first part of the re-indexing flow: creating the "green" (staging) Qdrant collections.
**Requirements:**
1.  Create a `reindex_handler` function called by the main Celery task.
2.  Use `QdrantManager` to create new, unique staging collections.
3.  Store the list of new staging collection names in the `collections.qdrant_staging` database field.
**Acceptance Criteria:** Staging collections are created in Qdrant and their names are persisted to the database.
**How it fits:** The first critical step of the zero-downtime re-indexing process.

---

### TASK-011: Implement Re-indexing to Staging
**Priority:** High | **Effort:** 8 hours | **Dependencies:** TASK-010 | **Assignee:** Backend Developer
**Context:** Implement the core processing loop for re-indexing into the staging collections.
**Requirements:**
1.  The `reindex_handler` will fetch all documents for the collection.
2.  It will process each document using the *new* configuration (if provided) and ingest vectors into the staging collections.
3.  Progress must be continuously updated in the `operations` table.
**Acceptance Criteria:** All documents are re-processed and indexed into the correct staging collections.
**How it fits:** Builds the new, updated index in the background.

---

### TASK-012: Implement Atomic Switch & Cleanup
**Priority:** High | **Estimated Effort:** 6 hours | **Dependencies:** TASK-011 | **Assignee:** Backend Developer
**Context:** Implement the final steps of the re-indexing process: the atomic switch and scheduling the cleanup of old resources.
**Requirements:**
1.  On successful completion, the Celery task will call a new, internal-only API endpoint: `POST /api/internal/complete_reindex`.
2.  This internal API endpoint will perform a single, atomic database transaction to:
    *   Copy the list of collection names from `qdrant_staging` to `qdrant_collections_active`.
    *   Set `qdrant_staging` to `NULL`.
    *   Update the `collections.status` to `ready`.
3.  The API will return the list of the old, now-orphaned Qdrant collection names.
4.  The Celery task will then dispatch a *new* cleanup task, scheduled to run after a delay.
**Acceptance Criteria:** The switch from staging to active collections is atomic and safe.
**How it fits:** This makes the new index live with zero downtime.

---

### TASK-013: Implement Operation Failure Handlers
**Priority:** High | **Estimated Effort:** 4 hours | **Dependencies:** TASK-009 | **Assignee:** Backend Developer
**Context:** Implement robust error handling that updates collection and operation status based on the type of failure.
**Requirements:**
1.  In the Celery task's `on_failure` handler, update the `operations` table to `failed` with the error message.
2.  Update the parent `collections` table to an appropriate state (`degraded` for failed re-index, `error` for failed initial index).
3.  Ensure staging resources are cleaned up immediately on a failed re-index.
**Acceptance Criteria:** The system gracefully handles failures, communicates the state clearly, and does not get stuck in a processing state.
**How it fits:** Ensures system resilience and provides clear feedback on failures.

---

### TASK-014: Implement Resource Cleanup Task
**Priority:** Medium | **Estimated Effort:** 4 hours | **Dependencies:** TASK-009 | **Assignee:** Backend Developer
**Context:** Create a Celery task for cleaning up orphaned Qdrant collections after a successful re-index.
**Requirements:**
1.  The task `cleanup_qdrant_collections(collection_names: List[str])` will be called with a delay.
2.  It will safely delete the specified Qdrant collections.
3.  It must include checks to prevent accidental deletion of active collections.
**Acceptance Criteria:** Old collections are deleted successfully and safely after a grace period.
**How it fits:** Completes the blue-green deployment lifecycle by freeing up resources.

---

## **Phase 4: API Implementation (Weeks 5-6)**

### TASK-015: Create Collection API Routes 🔴
**Priority:** Critical | **Estimated Effort:** 8 hours | **Dependencies:** TASK-005, TASK-006 | **Assignee:** Backend Developer
**Context:** Implement the new RESTful API endpoints for collection management.
**Requirements:**
1.  Create endpoints for all collection CRUD operations under `/api/collections/`.
2.  Use the `get_collection_for_user` dependency for all endpoints operating on a specific collection.
3.  Return consistent, typed responses using Pydantic models.
**Acceptance Criteria:** All collection management functionalities are exposed via a secure, RESTful API.
**How it fits:** Provides the new API interface for the frontend.

---

### TASK-016: Implement Multi-Collection Search API 🔴
**Priority:** Critical | **Estimated Effort:** 8 hours | **Dependencies:** TASK-015 | **Assignee:** Senior Backend Developer
**Context:** Implement the search endpoint to query across multiple collections, aggregate results, and perform mandatory re-ranking.
**Requirements:**
1.  The `/api/search` endpoint will accept a list of `collection_uuids`.
2.  Implement the federated search logic: parallel queries to Qdrant, aggregation, deduplication, and final re-ranking.
3.  If collections use different models, re-ranking is mandatory.
4.  Gracefully handle partial failures (e.g., one Qdrant source is down).
**Acceptance Criteria:** The endpoint provides relevant, globally ranked results from multiple distinct collections.
**How it fits:** This is a core value proposition of the new architecture.

---

### TASK-017: Implement WebSocket Progress Updates
**Priority:** High | **Estimated Effort:** 6 hours | **Dependencies:** TASK-009 | **Assignee:** Backend Developer
**Context:** Update the WebSocket implementation to track progress based on `operation_id` instead of `job_id`.
**Requirements:**
1.  Create a new WebSocket endpoint: `/ws/operations/{operation_id}`.
2.  The endpoint must authenticate the user and verify they have permission to view the operation's parent collection.
3.  The backend will publish progress updates to a Redis channel (e.g., `operation:{operation_id}`).
4.  The WebSocket handler will subscribe to this channel and stream updates to the client.
**Acceptance Criteria:** The frontend can receive real-time progress updates for any long-running collection operation.
**How it fits:** Provides essential real-time feedback to the user.

---

## **Phase 5: Frontend Implementation (Weeks 7-8)**

### TASK-018: Create Collection Store & LocalStorage Migration 🔴
**Priority:** Critical | **Estimated Effort:** 8 hours | **Dependencies:** TASK-015 | **Assignee:** Frontend Developer
**Context:** Create a new Zustand store for collections and implement a mechanism to handle the breaking change in `localStorage`.
**Requirements:**
1.  Create `stores/collectionStore.ts` to manage collections, operations, and their states.
2.  Implement optimistic updates for a responsive UI.
3.  **Implement a `localStorage` versioning check** on app startup. If the stored version doesn't match the new version, clear all app-related `localStorage` and force a reload.
**Acceptance Criteria:** Frontend state management is aligned with the new collection model; users with old local data have their state reset cleanly.
**How it fits:** This is the state management foundation for the new UI.

---

### TASK-019: Implement Collection Dashboard
**Priority:** High | **Effort:** 8 hours | **Dependencies:** TASK-018 | **Assignee:** Frontend Developer
**Context:** Build the new main "Collections" dashboard view.
**Requirements:**
1.  A grid of `CollectionCard` components.
2.  Search and filter controls for collections.
3.  A prominent "Create Collection" button that launches the creation modal.
4.  A helpful empty state for new users.
**Acceptance Criteria:** The dashboard is the new primary landing page and effectively displays all user collections.
**How it fits:** Replaces the old `Jobs` tab as the application's central hub.

---

### TASK-020: Implement Collection Card & Details Panel
**Priority:** High | **Effort:** 10 hours | **Dependencies:** TASK-019 | **Assignee:** Frontend Developer
**Context:** Build the `CollectionCard` for the dashboard and the comprehensive `CollectionDetailsPanel` modal.
**Requirements:**
1.  `CollectionCard` must show status, key stats, and an indicator for active operations.
2.  `CollectionDetailsPanel` must have the four tabs: Overview, Jobs (Operation History), Files, and Settings.
3.  The "Re-index" button in the Settings tab must be disabled until a change is made and require typed confirmation.
**Acceptance Criteria:** Users can view high-level status on the dashboard and drill down into a detailed management view.
**How it fits:** These are the core UI components for interacting with collections.

---

### TASK-021: Implement Create/Add/Re-index Modals
**Priority:** High | **Effort:** 8 hours | **Dependencies:** TASK-020 | **Assignee:** Frontend Developer
**Context:** Build the modals for all major collection operations.
**Requirements:**
1.  `CreateCollectionModal`: Simple, single-step form with an "Advanced Settings" accordion.
2.  `AddDataModal`: Allows adding a new source directory to an existing collection.
3.  `ReindexModal`: Clearly shows current vs. new settings and the implications of the operation.
**Acceptance Criteria:** All modals provide a clear, user-friendly interface for their respective operations.
**How it fits:** These components provide the user interface for all write/update operations on collections.

---

### TASK-022: Update Search UI & Logic
**Priority:** High | **Effort:** 6 hours | **Dependencies:** TASK-018 | **Assignee:** Frontend Developer
**Context:** Update the search interface to support selecting multiple collections and displaying grouped results.
**Requirements:**
1.  Replace the single collection dropdown with a multi-select component.
2.  Search results should be grouped by the collection they came from.
3.  The UI must handle the `warnings` field from the search API response to notify users of partial search failures.
**Acceptance Criteria:** Users can easily search across one or more collections and understand the origin of their results.
**How it fits:** Aligns the search functionality with the new multi-source collection model.

---

### TASK-023: Implement Active Operations Tab
**Priority:** Medium | **Effort:** 4 hours | **Dependencies:** TASK-017 | **Assignee:** Frontend Developer
**Context:** Convert the old "Jobs" tab into the new "Active Operations" queue.
**Requirements:**
1.  The view should only show operations with `processing` or `queued` status.
2.  It should display a global list of all active operations across all of the user's collections.
3.  Each item should link to its parent `CollectionDetailsPanel`.
**Acceptance Criteria:** Users have a single place to see what the system is currently doing.
**How it fits:** Provides essential system-wide visibility.

---

## **Phase 6: Testing, Documentation & Polish (Weeks 9-11)**

### TASK-024: Write Backend Integration Tests 🔴
**Priority:** Critical | **Effort:** 12 hours | **Dependencies:** All backend tasks | **Assignee:** QA/Backend Developer
**Context:** Create comprehensive integration tests for the new collection system.
**Requirements:** Test the full collection lifecycle, deduplication, blue-green re-indexing, multi-collection search, and error handling scenarios.
**Acceptance Criteria:** The backend is validated with a robust suite of tests covering all new functionality.

---

### TASK-025: Write Frontend E2E Tests
**Priority:** High | **Effort:** 8 hours | **Dependencies:** All frontend tasks | **Assignee:** QA/Frontend Developer
**Context:** Create Playwright end-to-end tests for the main user flows.
**Requirements:** Write tests for:
1.  Creating a new collection.
2.  Adding data to an existing collection.
3.  Searching across multiple collections.
4.  Initiating a re-index and verifying the UI state changes.
**Acceptance Criteria:** The primary user journeys are validated through automated E2E tests.

---

### TASK-026: Performance & Load Testing
**Priority:** Medium | **Effort:** 8 hours | **Dependencies:** TASK-024 | **Assignee:** QA/Backend Developer
**Context:** Test the system's performance under realistic load.
**Requirements:** Test concurrent operations, large collection re-indexing, and multi-collection search performance. Measure resource usage.
**Acceptance Criteria:** Performance meets defined metrics (e.g., search < 500ms for 5 collections).

---

### TASK-027: Update All Documentation 🔴
**Priority:** Critical | **Effort:** 8 hours | **Dependencies:** All implementation tasks | **Assignee:** Technical Writer/Developer
**Context:** Update all user and developer documentation to reflect the new collection-centric architecture.
**Requirements:** Update `README.md`, `docs/ARCH.md`, `docs/API_REFERENCE.md`, and create a new `docs/COLLECTION_MANAGEMENT.md`.
**Acceptance Criteria:** All documentation is accurate, comprehensive, and reflects the final state of the refactored application.

---

### TASK-028 - TASK-031: *Reserved for emergent tasks, bug fixes, and polish discovered during implementation.*

---

### TASK-032: Final Review and Merge to Main 🔴
**Priority:** Critical | **Effort:** 4 hours | **Dependencies:** All other tasks | **Assignee:** Tech Lead
**Context:** Perform a final review of all code, tests, and documentation before merging the feature branch into `main`.
**Requirements:**
1.  Code review all pull requests.
2.  Ensure all tests are passing in the CI pipeline.
3.  Remove all deprecated code and feature flags.
4.  Tag a new release version (e.g., `v2.0.0`).
**Acceptance Criteria:** The `feature/collections-refactor` branch is successfully merged into `main`. The project is stable and ready for release.