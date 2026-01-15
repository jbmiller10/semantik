# Specification: HyDE (Hypothetical Document Embeddings) Search

## Overview
Implement HyDE (Hypothetical Document Embeddings) to improve search retrieval quality by using an LLM to generate a hypothetical answer to a query before performing vector search. This track leverages the recently implemented LLM Provider Integration and Local LLM features.

## Goals
- Enhance semantic search accuracy for short or vague queries.
- Provide user control over when HyDE is active.
- Allow users to inspect the "mental model" of the search by viewing the generated hypothetical document.
- Enable HyDE configuration for external tools via MCP (Model Context Protocol).

## Functional Requirements
- **Search Logic Integration:**
    - Update the search pipeline to intercept queries when HyDE is enabled.
    - Use `LLMServiceFactory` to generate a hypothetical passage based on the user's query.
    - Use the generated passage for the embedding and vector search step instead of (or in addition to) the raw query.
- **UI Controls:**
    - **Toggle:** Add a "Smart Search" (HyDE) toggle switch directly in the search bar/filter area.
    - **Inspection:** Add a UI element (e.g., an expandable "Show AI Thought" section) that displays the generated hypothetical document.
- **Settings & Configuration:**
    - The LLM tier used for HyDE (High Quality, Low Quality, or Local) must be configurable in the user settings.
    - **MCP Defaults:** Allow the user to configure whether HyDE is enabled by default for MCP server requests.
- **Error Handling & Fallback:**
    - If the LLM service is unavailable, unconfigured, or times out, the system must silently fall back to a standard semantic search using the original query.
    - Log warnings for failed HyDE attempts without interrupting the user's search.

## Non-Functional Requirements
- **Latency:** Target overhead for HyDE generation < 2.0s for cloud providers and < 5.0s for local models.
- **User Experience:** The UI should indicate that the search is being "enhanced" while the LLM call is in progress.

## Acceptance Criteria
- [ ] Search API accepts a `hyde_enabled` flag.
- [ ] Results returned with HyDE enabled are measurably different (and ideally more relevant) than raw query search.
- [ ] User can see the generated text in the UI via a toggle.
- [ ] System falls back gracefully to standard search if LLM fails.
- [ ] MCP configuration interface allows toggling HyDE default status.

## Out of Scope
- Fine-tuning models specifically for HyDE generation.
- Multi-stage HyDE (generating multiple hypothetical documents).