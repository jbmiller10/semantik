# BE-WS-003: Add WebSocket Cancel Support for Preview/Comparison

## Context
Frontend supports REST cancellation, but WebSocket preview/comparison lacks a cancel path. Introduce a cancel message so server-side processing can be stopped, preventing wasted compute and stale updates.

## Requirements
- Extend WebSocket protocol to support cancel by `requestId`:
  - New inbound message: `{ type: 'cancel_request', requestId: '...' }`.
  - On cancel, stop any in-flight processing for that `requestId` and send a final notification:
    - Either `{ type: 'preview_error', data: { message: 'Cancelled', code: 'CANCELLED' }, requestId }`, or
    - A special `{ type: 'preview_complete', data: {...}, requestId }` indicating cancellation.
  - Apply the same semantics for comparison requests.
- Ensure resources are released (unsubscribe from streams, free buffers, close handles) to prevent leaks.
- Add tests covering cancel before start, mid-stream, and after completion.

## Acceptance Criteria
- Cancel message stops processing within ~200ms for simulated workload.
- Server emits a single final message for the cancelled `requestId` and no further messages are sent for it.
- No resource leaks (verified via logs/metrics in tests).

## Technical Notes
- If using Redis Pub/Sub for progress, ensure cancel signal propagates to the worker processing the request.
- Document message types and sequences in a protocol doc.

## Suggested Files
- WS handler (example): `packages/.../ws/chunking.py` or equivalent.
- Protocol doc: `docs/websocket-chunking-protocol.md` (add cancel section).
- Tests: `tests/backend/test_websocket_chunking_cancel.py`.

## Test Commands
- Python: `pytest tests/backend/test_websocket_chunking_cancel.py -v`

## Dependencies
- None. Enables FE-WS-004.
