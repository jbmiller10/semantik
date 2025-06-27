# Unified Search Implementation

## What We Did

We unified the search implementation between the web UI and search API to reduce code duplication and maintenance burden.

### Changes Made

1. **Created `vecpipe/search_utils.py`** - Shared utilities for search operations:
   - `search_qdrant()` - Performs vector search against Qdrant
   - `parse_search_results()` - Parses Qdrant results into standard format

2. **Updated Search API** (`vecpipe/search_api.py`):
   - Added `collection` parameter to support job-specific collections
   - Now uses shared `search_qdrant()` utility
   - Maintains backward compatibility

3. **Updated Web UI** (`webui/app.py`):
   - Now uses shared `search_qdrant()` utility
   - Removed duplicate Qdrant search code
   - Still handles job-specific model/quantization logic

## Benefits

1. **Single Search Implementation**: Both APIs now use the same core search logic
2. **Easier Maintenance**: Changes to search behavior only need to be made in one place
3. **Consistent Results**: Both interfaces return identical search results
4. **Reduced Code**: Eliminated ~20 lines of duplicate code

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Search API    │     │    Web UI       │
│  /search?q=...  │     │  /api/search    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       │ (handles auth & job lookup)
         │                       │
         └───────┬───────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ search_utils  │
         │ search_qdrant │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │    Qdrant     │
         └───────────────┘
```

## Key Differences Remain

While search is unified, the two endpoints serve different purposes:

### Search API (`/search`)
- Public API for general search
- Uses configured embedding model (env var)
- Simple query parameters
- No authentication required

### Web UI (`/api/search`)
- Internal API for web interface
- Uses job-specific model/quantization
- Requires authentication
- Tracks which job created the embeddings

## Testing

Run the test script to verify the unified implementation:
```bash
python test_unified_search.py
```

## Future Improvements

1. Could further unify by having search API optionally accept model/quantization parameters
2. Could add caching layer in search_utils for frequently searched queries
3. Could add search result ranking/reranking in search_utils