# Implementation Plan: HyDE Search Integration

## Phase 1: Backend Implementation [checkpoint: 8f1e627]
- [x] Task: Update Search API with HyDE Logic
    - [x] Modify `CollectionSearchRequest` in `packages/webui/api/v2/schemas.py` to include `hyde_enabled: bool = False`.
    - [x] Modify `SearchService.multi_collection_search` in `packages/webui/services/search_service.py` to handle the HyDE logic.
    - [x] Implement `_generate_hypothetical_document` method in `SearchService` using `LLMServiceFactory`.
    - [x] Implement fallback logic (catch `LLMNotConfiguredError`, `LLMAuthenticationError`, timeouts) to revert to standard search.
    - [x] [5aacab0]
- [x] Task: Conductor - User Manual Verification 'Backend Implementation' (Protocol in workflow.md)

## Phase 2: Configuration & Settings
- [ ] Task: Add User Settings for HyDE
    - [ ] Update `UserSettings` model (or equivalent) to include:
        - `hyde_enabled_default`: bool
        - `hyde_llm_tier`: LLMQualityTier (default: LOW)
    - [ ] Update MCP Server configuration endpoints/schema to include `hyde_enabled_default` for MCP clients.
- [ ] Task: Conductor - User Manual Verification 'Configuration & Settings' (Protocol in workflow.md)

## Phase 3: Frontend Integration
- [ ] Task: Update Search UI
    - [ ] Update `SearchBox` component to include the "Smart Search" (HyDE) toggle.
    - [ ] Display the generated hypothetical document (collapsible "Show AI Thought" section) in the search results area if HyDE was used.
    - [ ] Update `SettingsPage` to allow configuring the default LLM tier for HyDE.
- [ ] Task: Conductor - User Manual Verification 'Frontend Integration' (Protocol in workflow.md)
