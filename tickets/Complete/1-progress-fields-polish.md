# [1] Progress Field Polish – Operation Status in Projection Meta

## Context
- The UI now consumes `operation_id` but still lacks consistent `operation_status` in GET responses.
- Some responses (list/get) may omit the field or return stale data, making progress banner less reliable.
- Need to ensure metadata includes up-to-date operation status for all projection endpoints.

## Acceptance Criteria
- `operation_status` returned in POST, GET (single), and GET (list) responses.
- Status reflects latest associated `operations` row (PROCESSING/COMPLETED/FAILED/etc.).
- Frontend updates to fall back on status when websocket not connected.
- Regression tests cover presence of both operation fields.

## Implementation Outline
1. Ensure `_encode_projection` attaches `operation.status.value`.
2. Update `_to_metadata_response` mapping (already partly done) to propagate field.
3. Frontend: display status chip or message when `pendingOperationId` null but `operation_status` known.
4. Add unit tests verifying schema serialization includes the field.

## Affected Files / Areas
- `packages/webui/services/projection_service.py`
- `packages/webui/api/v2/projections.py`
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
- Related TypeScript types (`StartProjectionResponse`, etc.).

## API & Data Contracts
- Document `operation_id` + `operation_status` in API docs for projection metadata.

## Test Notes
- Backend serialization tests.
- Frontend render test consuming mocked response with status.
