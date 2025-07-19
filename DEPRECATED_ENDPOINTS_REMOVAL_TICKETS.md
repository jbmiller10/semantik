# Deprecated Job-Centric Endpoints Removal Tickets

These tickets track the complete removal of deprecated job-centric API endpoints after the v2.0 release.

## TICKET-001: Remove Deprecated Job CRUD Endpoints
**Priority**: Medium
**Target Release**: Post v2.0
**Dependencies**: None

### Description
Remove the following deprecated endpoints from `/packages/webui/api/jobs.py`:
- `POST /api/jobs` - Create job endpoint
- `GET /api/jobs` - List jobs endpoint
- `GET /api/jobs/{job_id}` - Get job details endpoint
- `DELETE /api/jobs/{job_id}` - Delete job endpoint

### Acceptance Criteria
- [ ] Remove endpoint implementations
- [ ] Remove associated request/response models if no longer used
- [ ] Update tests to remove deprecated endpoint coverage
- [ ] Ensure no internal code references these endpoints

## TICKET-002: Remove Job-Centric WebSocket Handler
**Priority**: Low
**Target Release**: Post v2.0
**Dependencies**: TICKET-001

### Description
Remove the legacy job-based WebSocket handler at `/ws/{job_id}` in favor of the operation-based handler.

### Acceptance Criteria
- [ ] Remove `websocket_endpoint` function from jobs.py
- [ ] Update main.py to remove the WebSocket route mounting
- [ ] Update frontend to exclusively use operation-based WebSocket

## TICKET-003: Clean Up Job-Centric Utility Functions
**Priority**: Low
**Target Release**: Post v2.0
**Dependencies**: TICKET-001, TICKET-002

### Description
Remove any remaining job-centric utility functions and helpers that are no longer needed after the API removal.

### Acceptance Criteria
- [ ] Audit codebase for unused job-specific utilities
- [ ] Remove deprecated functions
- [ ] Update any remaining references to use collection-based alternatives

## Notes
- Since Semantik is pre-release, we have flexibility on timing
- Monitor usage of deprecated endpoints before removal
- Ensure all functionality is available through collection-based endpoints before removal