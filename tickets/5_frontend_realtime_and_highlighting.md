Title: Finish realtime chunking UX and restore test coverage

Background
- WebSocket service tests have TODOs/failures (`apps/webui-react/src/services/__tests__/websocket.test.ts`).
- Chunk highlighting/scroll is stubbed in `components/DocumentViewer.tsx`; search result chunk-detail click is stubbed in `components/SearchResults.tsx`.
- These gaps reduce confidence in the key chunking preview flow.

Goal
Deliver working realtime chunking UX (auth/heartbeat/reconnect) and chunk navigation, with automated tests.

Scope
- Stabilize WebSocket service: auth request/response, heartbeat/pong, reconnect/backoff, clean close handling. Fix/complete tests using the mock WebSocket.
- Implement chunk highlighting + scroll-to-chunk when `chunkId` is provided to DocumentViewer; handle long documents.
- Wire search result “open chunk detail” to viewer/side panel; manage selection state.
- Add component/integration tests (Vitest + RTL; optional Cypress) covering the flows above.

Out of Scope
- Visual redesign or theming.

Suggested Steps
1) Improve TestMockWebSocket to cover open/auth/heartbeat/reconnect/close; finish tests, remove TODOs.
2) Implement highlight/scroll logic in DocumentViewer; add a test feeding a `chunkId` and asserting scroll + highlight.
3) Connect SearchResults click to viewer state; add a test ensuring click surfaces chunk content/context.
4) Run `npm test --prefix apps/webui-react`; add a focused e2e spec if time.

Acceptance Criteria
- WebSocket tests pass with coverage of auth, heartbeat, reconnect, and close handling; no TODOs remain.
- DocumentViewer scrolls/highlights the provided chunk; covered by automated test.
- Search result click opens chunk detail UI; covered by test.
- `npm test --prefix apps/webui-react` (and any added e2e) passes.
