# Ticket #001: Implement Re-indexing Functionality

**Priority**: CRITICAL
**Type**: Feature Implementation
**Component**: Backend + Frontend
**Blocks**: Phase 6 Testing

## Summary
The re-indexing functionality, a core feature of the collection-centric refactor, is completely missing from the implementation. The Settings tab in the collection management panel only displays read-only configuration values with no ability to modify embedding settings or trigger re-indexing operations.

## Context
According to the architectural plan, re-indexing is a critical feature that enables:
- Zero-downtime updates to embedding settings
- Changing chunk size, overlap, or embedding models without losing data
- Maintaining search availability during re-indexing operations

This feature is essential for production use cases where collections need to be optimized or updated without service interruption.

## Current State
- Settings tab shows only read-only values
- No form fields for editing configuration
- No "Re-index" button present
- No backend endpoints appear to exist for triggering re-indexing

## Expected Behavior
1. Settings tab should have editable form fields for:
   - Embedding model selection
   - Chunk size
   - Chunk overlap
   - Other relevant embedding parameters

2. Changes to these fields should enable a "Re-index" button

3. Clicking "Re-index" should:
   - Show a confirmation dialog explaining the operation
   - Create a new re-indexing operation
   - Use staging collections in Qdrant for zero-downtime
   - Maintain search availability on the old index during processing
   - Atomically swap collections when complete

## Technical Requirements

### Frontend (React/TypeScript)
1. Update `CollectionSettings.tsx` or relevant component to:
   - Convert read-only displays to form inputs
   - Add state management for edited values
   - Implement change detection to enable "Re-index" button
   - Add confirmation dialog with clear messaging
   - Handle operation status updates

### Backend (Python/FastAPI)
1. Create API endpoint: `POST /api/collections/{id}/reindex`
   - Validate changed settings
   - Create new operation with type "REINDEX"
   - Return operation ID for tracking

2. Implement Celery task for re-indexing:
   - Create staging Qdrant collection
   - Re-process all documents with new settings
   - Implement atomic swap mechanism
   - Clean up old collection after success

3. Update CollectionService to:
   - Handle re-indexing state transitions
   - Manage concurrent operations correctly
   - Update collection metadata after completion

### Database
1. Ensure operations table supports "REINDEX" type
2. Add fields for tracking staging collection names
3. Consider adding reindex_count to collections table

## Testing Requirements
1. Unit tests for re-indexing logic
2. Integration tests for the full re-indexing flow
3. E2E tests verifying zero-downtime behavior
4. Performance tests with large collections

## Acceptance Criteria
- [ ] Settings tab has editable form fields
- [ ] Re-index button appears when settings are changed
- [ ] Confirmation dialog clearly explains the operation
- [ ] Re-indexing operation appears in Active Operations
- [ ] Search remains available during re-indexing
- [ ] Collection automatically uses new settings after completion
- [ ] Old Qdrant collection is cleaned up after success
- [ ] Proper error handling for failures
- [ ] Status updates are real-time via WebSocket or polling

## Related Code Locations
- Frontend: `apps/webui-react/src/components/collections/` (Settings components)
- Backend API: `packages/webui/api/v2/collections.py`
- Celery Tasks: `packages/webui/tasks.py`
- Collection Service: `packages/webui/services/collection_service.py`
- Database Models: `packages/shared/database/models/`

## Notes
This is a blocking issue for the collection-centric refactor. Without re-indexing, users cannot optimize their collections after creation, which significantly limits the system's usefulness in production scenarios.