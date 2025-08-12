# FE-WS-004: Wire WebSocket Cancel in Frontend + UI Controls

## Context
With BE cancel support, the FE must expose cancel controls during WebSocket-based preview/comparison and stop processing/UI updates for the active `requestId`.

## Requirements
- Add a Cancel control in `ChunkingPreviewPanel.tsx` when `wsProgress` is present:
  - Clicking Cancel calls a new `cancel()` from `useChunkingWebSocket` that sends `{ type: 'cancel_request', requestId }`.
  - On confirmation/cancel complete, clear progress and show “Cancelled” or return to idle.
- Ensure subsequent messages for the cancelled `requestId` are ignored.
- Do not use disconnect as a proxy for cancel; cancel must be per-request.
- Add unit tests for hook behavior and UI.

## Acceptance Criteria
- Cancel button appears only while a WS preview/comparison is active, disappears on completion or cancellation.
- UI stops receiving/applying messages for cancelled `requestId`.
- Hook and component tests pass.

## Suggested Files
- `apps/webui-react/src/hooks/useChunkingWebSocket.ts` (add `cancel()` and message handling).
- `apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx` (add cancel UI and states).
- Tests in `apps/webui-react/src/components/chunking/__tests__/` and `apps/webui-react/src/hooks/__tests__/`.

## Test Commands
- `cd apps/webui-react && npm run test:ci`

## Dependencies
- Depends on BE-WS-003.
