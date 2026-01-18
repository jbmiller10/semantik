# Specification: Search System Enhancement

## Overview
This track focuses on improving the search quality and flexibility of Semantik. It involves integrating a cross-encoder reranking step to refine search results and adding configuration options for hybrid search (balancing keyword vs. semantic scores).

## Goals
- Implement cross-encoder reranking to improve result relevance.
- Provide user-configurable parameters for hybrid search tuning (alpha parameter).
- Visualize the impact of reranking and hybrid tuning in the UI.

## Requirements

### Functional Requirements
- **Reranking:**
  - Integrate a cross-encoder model (e.g., from `sentence-transformers` or Hugging Face) into the search pipeline.
  - Allow enabling/disabling reranking per search query or globally.
  - Support configuration of the number of top-k results to rerank.
- **Hybrid Search Tuning:**
  - Expose an `alpha` parameter (0.0 to 1.0) to control the weight between sparse (keyword) and dense (semantic) search scores.
  - Update the API to accept `alpha` and reranking parameters.
- **UI:**
  - Add controls in the search interface to adjust `alpha`.
  - Add a toggle for reranking.
  - Display search scores (and reranked scores) to the user for transparency.

### Non-Functional Requirements
- **Performance:** Reranking should not add excessive latency (target < 500ms for reranking top 50 results).
- **Extensibility:** The reranking logic should be modular (potentially a plugin).
- **Usability:** Default settings should provide good results without manual tuning.

## Technical Design
- **Backend:**
  - Update `packages/vecpipe` to support reranking models.
  - Modify `packages/webui/api/v2/search.py` to accept new parameters.
  - Update `packages/shared` search logic to handle hybrid scoring combinations.
- **Frontend:**
  - Update `apps/webui-react` search components to include new controls.
  - Visualize score distribution.

## Data Models
- No major schema changes expected, but configuration objects passed to search API will need updating.
