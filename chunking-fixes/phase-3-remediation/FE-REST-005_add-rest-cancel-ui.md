# FE-REST-005: Add REST Cancel UI Control (Fallback Mode)

## Context
REST preview already supports cancellation via `cancelAllRequests()` and request IDs. Expose a cancel control in the UI for REST mode or when WebSocket is offline.

## Requirements
- In `ChunkingPreviewPanel.tsx`, when using REST mode (or WS not connected) and `previewLoading` is true:
  - Show a “Cancel” button that calls `useChunkingStore().cancelActiveRequests()`.
  - Stop showing loading and return to idle or a “Cancelled” state.
- Ensure canceled promises are handled (no unhandled rejections).
- Add unit tests simulating an inflight REST preview and verify the cancel button stops loading.

## Acceptance Criteria
- Cancel button appears during REST previews and reliably cancels inflight requests.
- UI returns to idle/canceled state without residual loading or errors.
- Tests pass.

## Suggested Files
- `apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx` (add cancel UI for REST mode).
- `apps/webui-react/src/stores/chunkingStore.ts` (confirm `cancelActiveRequests()` behavior and state reset if needed).
- Tests: `apps/webui-react/src/components/chunking/__tests__/ChunkingPreviewPanel.test.tsx`.

## Test Commands
- `cd apps/webui-react && npm run test:ci`

## Dependencies
- Independent. Can be implemented in parallel with BE-WS-003/FE-WS-004.
