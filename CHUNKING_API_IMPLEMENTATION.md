# Chunking Strategy API Implementation

This document summarizes the implementation of chunking strategy selection through the API.

## Overview

Users can now select from 6 different chunking strategies when creating collections:
- **character**: Simple character-based chunking
- **recursive**: Recursive text splitting (default)
- **markdown**: Markdown-aware chunking
- **semantic**: Semantic similarity-based chunking
- **hierarchical**: Multi-level hierarchical chunking
- **hybrid**: Combines multiple strategies based on content

## Changes Made

### 1. API Schema Updates (`packages/webui/api/schemas.py`)

- Added `ChunkingStrategyEnum` with all 6 strategies
- Added `chunking_strategy` field to `CollectionBase` (default: "recursive")
- Added `chunking_params` field for strategy-specific parameters
- Added validation logic for strategy-specific parameters
- Updated `CollectionResponse.from_collection()` to handle chunking fields

### 2. Database Schema Updates

- Created migration `20250803150601_add_chunking_strategy_to_collections.py`
- Added `chunking_strategy` column (string, default: "recursive")
- Added `chunking_params` column (JSON, nullable)
- Added index on `chunking_strategy` for performance

### 3. Database Models (`packages/shared/database/models.py`)

- Added `chunking_strategy` and `chunking_params` columns to Collection model

### 4. Collection Repository (`packages/shared/database/repositories/collection_repository.py`)

- Updated `create()` method to accept chunking_strategy and chunking_params
- Updated method signature and documentation

### 5. Collection Service (`packages/webui/services/collection_service.py`)

- Updated `create_collection()` to pass chunking fields to repository
- Updated collection dictionary response to include chunking fields
- Updated `reindex_collection()` to preserve chunking configuration

### 6. API Routes (`packages/webui/api/v2/collections.py`)

- Updated create collection endpoint to pass chunking_strategy and params
- Updated response to include chunking fields

### 7. Task Processing (`packages/webui/tasks.py`)

- Replaced `TokenChunker` with `ChunkingFactory`
- Updated `_process_append_operation()` to use selected chunking strategy
- Updated `_process_reindex_operation()` to use selected chunking strategy
- Properly converts ChunkResult objects to expected dictionary format

## API Usage Examples

### 1. Create Collection with Semantic Chunking

```json
POST /api/v2/collections
{
  "name": "Technical Documentation",
  "description": "API and developer docs",
  "chunking_strategy": "semantic",
  "chunking_params": {
    "breakpoint_percentile_threshold": 90,
    "buffer_size": 2,
    "max_chunk_size": 2000
  },
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
}
```

### 2. Create Collection with Hierarchical Chunking

```json
POST /api/v2/collections
{
  "name": "Research Papers",
  "description": "Academic papers with nested structure",
  "chunking_strategy": "hierarchical",
  "chunking_params": {
    "chunk_sizes": [2048, 512, 128],
    "chunk_overlap": 50
  }
}
```

### 3. Create Collection with Hybrid Chunking

```json
POST /api/v2/collections
{
  "name": "Mixed Content",
  "description": "Various document types",
  "chunking_strategy": "hybrid",
  "chunking_params": {
    "semantic_threshold": 0.7,
    "length_threshold": 0.8
  }
}
```

## Validation Rules

### Semantic Strategy Parameters
- `breakpoint_percentile_threshold`: 0-100 (percentage)
- `buffer_size`: >= 0 (integer)
- `max_chunk_size`: > 0 (integer)

### Hierarchical Strategy Parameters
- `chunk_sizes`: List of positive integers in descending order
- `chunk_overlap`: >= 0 (integer)

### Hybrid Strategy Parameters
- `semantic_threshold`: 0.0-1.0 (float)
- `length_threshold`: 0.0-1.0 (float)

### Character, Recursive, and Markdown Strategies
- Use standard `chunk_size` and `chunk_overlap` from collection config

## Backward Compatibility

- Default strategy is "recursive" to maintain compatibility
- Existing collections without chunking_strategy will use "recursive"
- `chunk_size` and `chunk_overlap` continue to work as before

## Testing

Use the provided `test_chunking_api.py` script to test all chunking strategies:

```bash
python test_chunking_api.py
```

## Migration Instructions

1. Run the database migration:
   ```bash
   poetry run alembic upgrade head
   ```

2. Existing collections will automatically use "recursive" strategy

3. Update any API clients to include chunking_strategy if needed

## Future Enhancements

1. Add chunking strategy recommendations based on file types
2. Allow strategy changes during reindexing
3. Add preview endpoint to test chunking before collection creation
4. Implement auto-detection of optimal strategy based on content analysis