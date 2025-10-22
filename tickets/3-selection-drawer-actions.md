# [3] Selection Drawer Actions – Document Viewer & Similar Search

## Context
- The projection selection drawer currently renders “Open” and “Find Similar” buttons as disabled placeholders.
- Product PRD calls for users to jump from a selected embedding point to the corresponding document chunk and to launch a relevance search seeded by that chunk.
- Back-end selection API already returns `document_id`, `chunk_id`, `chunk_index`, and `content_preview`, so the UI has the data needed to wire these actions.

## Acceptance Criteria
- Clicking **Open** opens the existing DocumentViewer (or modal) focused on the selected chunk/doc.
- Clicking **Find Similar** submits the chunk as a query to the existing search API and renders results in-context (drawer or existing search UI).
- Multiple selections should be handled gracefully (e.g., enable actions on the first/primary item and surface feedback if multiple).
- Analytics event recorded for both actions.
- Errors surface non-blocking toasts without breaking the drawer state.

## Implementation Outline
1. **Selection focus logic** – decide primary item when multiple indices are selected (e.g., first item in `selectionState.items`).
2. **Open action**
   - Reuse DocumentViewer component (from collections dashboard) or trigger router navigation with query params (`collectionId`, `documentId`, `chunkIndex`).
   - Ensure viewer scrolls/highlights the chunk.
3. **Find Similar action**
   - POST to existing `/api/v2/search` (or relevant hook) with seed query payload (chunk text + metadata).
   - Display results in a sub-panel, or reuse the global search drawer if available.
4. **UI states**
   - Enable buttons only when selection metadata is present.
   - Show loading spinners while actions in-flight; handle errors with toast + logging.
5. **Telemetry**
   - Emit analytics events (`projection_selection_open`, `projection_selection_find_similar`).
6. **Documentation/tooltips**
   - Add tooltips describing the actions for accessibility.

## Affected Files / Areas
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
- DocumentViewer component & routing helpers (`apps/webui-react/src/components/DocumentViewer/…`)
- Search API client (`apps/webui-react/src/services/api/v2/search.ts` or equivalent)
- Analytics/telemetry utilities (`apps/webui-react/src/lib/telemetry.ts`)

## API & Data Contracts
- Selection API already returns `document_id`, `chunk_id`, `content_preview`; ensure contract documented so backend changes remain backward compatible.
- Search request payload should include `collection_id`, seed text/IDs, and optional filters; confirm with search team.

## Test Notes
- Cypress/e2e: select point → click Open → verify DocumentViewer highlights chunk.
- Unit: mock selection state and assert buttons call navigation/search handlers.
- Integration: verify search API receives expected payload; handle error and empty states.
