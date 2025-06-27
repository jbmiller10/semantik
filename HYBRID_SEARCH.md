# Hybrid Search Implementation

This document describes the hybrid search functionality that combines vector similarity search with text-based keyword matching in Qdrant.

## Overview

Hybrid search improves search accuracy by combining:
1. **Vector Similarity**: Semantic understanding through embeddings
2. **Keyword Matching**: Exact text matching for specific terms

## Architecture Changes

### 1. Data Model Updates

The payload structure now includes full text content:

```python
{
    "doc_id": "document_hash",
    "chunk_id": "doc_chunk_0001", 
    "path": "/path/to/document.pdf",
    "content": "Full text content of the chunk..."  # NEW: Full text for hybrid search
}
```

### 2. New Components

#### `vecpipe/hybrid_search.py`
Core hybrid search engine with:
- Keyword extraction
- Text filtering using Qdrant's MatchText
- Two search modes: "filter" and "rerank"
- Keyword-only search capability

#### Updated APIs
- `/hybrid_search` - Hybrid vector + text search
- `/keyword_search` - Text-only search
- `/api/hybrid_search` - WebUI hybrid search endpoint

## Usage

### Search API Endpoints

#### 1. Hybrid Search
```bash
GET http://localhost:8000/hybrid_search?q=machine+learning+algorithms&k=10&mode=filter&keyword_mode=any
```

Parameters:
- `q`: Search query
- `k`: Number of results (1-100)
- `mode`: "filter" or "rerank"
- `keyword_mode`: "any" or "all"
- `score_threshold`: Optional minimum score

#### 2. Keyword Search
```bash
GET http://localhost:8000/keyword_search?q=python+docker&k=10&mode=any
```

### WebUI Integration

The WebUI now supports hybrid search through:
```bash
POST http://localhost:8080/api/hybrid_search
{
    "query": "search terms",
    "k": 10,
    "job_id": "optional_job_id",
    "mode": "filter",
    "keyword_mode": "any"
}
```

## Search Modes

### Filter Mode
- Uses Qdrant's built-in text filtering
- Faster for large collections
- Keywords act as pre-filters before vector search

### Rerank Mode
- Retrieves more candidates via vector search
- Re-scores based on keyword matches
- Better for precision with smaller result sets
- Combined score: 70% vector + 30% keywords

## Implementation Details

### Keyword Extraction
- Removes common stop words
- Filters words < 3 characters
- Case-insensitive matching

### Text Matching
- Uses Qdrant's MatchText condition
- Supports "any" (OR) and "all" (AND) modes
- Matches against the "content" field in payload

## Migration Notes

For existing data:
1. Re-process documents with updated embedding scripts
2. Or manually update payloads to include "content" field
3. The system gracefully handles missing content fields

## Testing

Run the test suite:
```bash
python test_hybrid_search.py
```

This tests:
- Search API hybrid endpoints
- Keyword-only search
- Direct hybrid search functionality
- Both filter and rerank modes

## Performance Considerations

- **Storage**: Full text in payload increases storage requirements
- **Indexing**: Consider enabling payload indexing for text fields
- **Memory**: Rerank mode loads more candidates into memory
- **Speed**: Filter mode is generally faster for large collections

## Future Enhancements

1. **Fuzzy Matching**: Support approximate keyword matches
2. **Phrase Search**: Support exact phrase matching
3. **Field-Specific Search**: Search in specific metadata fields
4. **Boosting**: Allow boosting certain keywords
5. **Stemming/Lemmatization**: Better keyword normalization