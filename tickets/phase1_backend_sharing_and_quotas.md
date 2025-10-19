# Phase 1 Ticket: Deliver Collection Sharing & Tiered Quotas (Target: October 27 – November 7, 2025)

## Background
Phase 0 secures the stack; Phase 1 must unlock collaboration scenarios for demos. The audit (see `CODEBASE_REVIEW_REPORT.md`) flagged unfinished `CollectionPermission` stubs and hard-coded usage limits in `packages/webui/services/resource_manager.py`. Reviewers expect to create a collection, invite teammates, and watch tier-aware limits. This ticket equips a stateless agent to deliver the sharing model and quota enforcement.

## Objectives
1. Implement collection sharing permissions end-to-end.
2. Introduce per-user resource quotas with reasonable tier defaults and error messaging.
3. Expose quota metadata via the settings API for frontend consumption.

## Requirements
### 1. Collection Sharing System
- **Schema**: Confirm the presence of a `collection_permissions` table; if missing or incomplete, create/extend via Alembic under `alembic/versions/`. Fields should include `id`, `collection_id`, `user_id`, `role` (`owner`, `editor`, `viewer`), audit columns, and uniqueness constraints to prevent duplicates.
- **Repositories**: Flesh out stubs in `packages/shared/database/repositories/collection_repository.py` and `operation_repository.py` with helpers like `list_shared_with_user`, `get_user_role`, `upsert_permission`, `remove_permission`.
- **Service Layer & API**:
  - Update FastAPI routes (likely `packages/webui/api/v2/collections.py`) to allow owners to add/remove collaborators and to enforce authorization when reading/updating collections and operations.
  - Ensure background operation polling respects permissions; unauthorized users must receive 403.
- **Testing**: Add integration tests covering invite, list, remove, and access denial flows (`tests/webui/api/v2/test_collection_sharing.py`). Include at least one scenario verifying that editors can trigger operations but viewers cannot.

### 2. Per-User Resource Quotas
- **Data Model**: Introduce per-user quota storage. If a model doesn’t exist, create `packages/shared/database/models/user_limits.py` (or extend existing user settings) to track limits such as `max_collections`, `max_storage_bytes`, `max_documents`.
- **Service Logic**: Refactor `packages/webui/services/resource_manager.py` to read limits from the new model or configuration rather than hard-coded constants. Implement checks for collection creation, file uploads, and other capacity-sensitive operations.
- **Error Handling**: Return localized messages explaining which limit was exceeded and current usage vs allowed. Integrate with API error schema.
- **Administration**: Provide defaults (e.g., `free`, `pro` tiers). Document how to override via environment variables or admin endpoints.
- **Testing**: Add unit tests for the resource manager (e.g., `tests/webui/services/test_resource_manager.py::TestQuotaLimits`) covering success, near-limit, and over-limit cases.

### 3. Settings API Exposure
- Update `/api/v1/settings` (or equivalent) to include per-user quota metadata (limits and current usage). Ensure responses remain backwards compatible or document the change for frontend consumers.
- Update or create tests validating the new fields.

## Acceptance Criteria
- Sharing endpoints support inviting/removing users with role enforcement, and unauthorized access returns 403.
- Quota enforcement prevents over-limit actions with informative error responses; limits persist per user in the database.
- Settings API returns quota metadata consumed by the frontend (coordinate with the companion frontend ticket).
- `uv run alembic upgrade head` succeeds on a fresh database with any new migrations.
- Tests added above pass under `uv run pytest tests -k "sharing or quota"`.

## Validation Steps
1. `uv run alembic upgrade head`
2. `uv run pytest tests/webui/api/v2/test_collection_sharing.py -v`
3. `uv run pytest tests/webui/services/test_resource_manager.py::TestQuotaLimits -v`
4. Manual API smoke test: invite a secondary user, verify permissions and quota messaging through the API.

## Coordination Notes
- Coordinate with the frontend Phase 1 ticket to ensure UI surfaces new quota metadata.
- Sync with data/DevOps if new migrations or env vars require deployment adjustments.
- Document any default tier values for use in documentation later in Phase 3.

## Out of Scope
- Implementing payment/subscription systems tied to quotas.
- Adding new roles beyond owner/editor/viewer.
- Building UI for managing quotas/sharing (handled in separate frontend ticket).
- Bulk migrations for legacy datasets beyond ensuring forward compatibility of schema.
