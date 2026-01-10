# Sparse Indexing Guide

This guide explains how to use sparse indexing in Semantik for improved hybrid search quality.

## Overview

Sparse indexing enables **hybrid search** by combining traditional keyword-based retrieval with dense vector search. While dense embeddings excel at semantic similarity, sparse vectors capture exact term matches that dense models may miss.

### Dense vs Sparse vs Hybrid

| Search Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Dense** | Semantic similarity, synonyms, paraphrases | May miss exact terms, proper nouns |
| **Sparse** | Exact matches, rare terms, technical jargon | No semantic understanding |
| **Hybrid** | Best of both approaches | Slightly higher latency |

### BM25 vs SPLADE

Semantik supports two sparse indexing approaches:

| Feature | BM25 | SPLADE |
|---------|------|--------|
| **Type** | Statistical (TF-IDF) | Neural (learned) |
| **State** | Stateful (IDF stats) | Stateless |
| **GPU Required** | No | Yes (recommended) |
| **Throughput** | ~1000 docs/sec | 10-50 docs/sec (GPU) |
| **Quality** | Good for keyword search | Better semantic alignment |
| **Setup** | Simple | Requires model download |

**Recommendation:** Start with BM25 for simplicity. Use SPLADE if you need better semantic term expansion.

---

## Quick Start

### 1. Enable Sparse Indexing

```bash
# Via API: Enable BM25 on an existing collection
curl -X POST "http://localhost:8000/api/v2/collections/{collection_name}/sparse-index" \
  -H "Content-Type: application/json" \
  -d '{
    "plugin_id": "bm25-local",
    "reindex_existing": true
  }'
```

### 2. Search with Hybrid Mode

```bash
# Hybrid search combines dense + sparse with RRF fusion
curl -X POST "http://localhost:8000/api/v2/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning optimization techniques",
    "collection_names": ["my_docs"],
    "search_mode": "hybrid",
    "k": 10
  }'
```

### 3. Check Status

```bash
# Get sparse index status
curl "http://localhost:8000/api/v2/collections/{collection_name}/sparse-index"
```

---

## Search Modes

The `search_mode` parameter controls how search is performed:

### Dense (Default)

```json
{"search_mode": "dense"}
```

Standard vector search using dense embeddings only. Best for semantic similarity queries.

### Sparse

```json
{"search_mode": "sparse"}
```

Sparse vector search only (BM25 or SPLADE). Best for exact keyword matching.

### Hybrid

```json
{"search_mode": "hybrid", "rrf_k": 60}
```

Combines dense and sparse results using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score = 1/(rank_dense + k) + 1/(rank_sparse + k)
```

The `rrf_k` parameter (default: 60) controls fusion behavior:
- **Lower values (20-40)**: More weight to top results
- **Higher values (60-100)**: More balanced fusion

Scores are normalized to [0, 1] for consistent `score_threshold` behavior.

---

## Built-in Indexers

### BM25 (`bm25-local`)

Classic probabilistic retrieval using the BM25 algorithm.

**Configuration:**

```json
{
  "plugin_id": "bm25-local",
  "model_config": {
    "k1": 1.5,
    "b": 0.75,
    "lowercase": true,
    "remove_stopwords": true,
    "min_token_length": 2
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.5 | Term saturation (0-3). Higher = more weight to term frequency |
| `b` | 0.75 | Length normalization (0-1). 0 = none, 1 = full |
| `lowercase` | true | Convert text to lowercase |
| `remove_stopwords` | true | Remove common English stopwords |
| `min_token_length` | 2 | Minimum token length to include |

**IDF Persistence:**

BM25 stores corpus statistics (IDF values) in:
```
data/sparse_indexes/{collection_name}/idf_stats.json
```

These statistics are updated incrementally as documents are added/removed.

**Performance:**
- Throughput: ~1000 documents/second (CPU)
- Query latency: <10ms
- Memory: Scales with vocabulary size

### SPLADE (`splade-local`)

Neural sparse encoder using learned term importance weights.

**Configuration:**

```json
{
  "plugin_id": "splade-local",
  "model_config": {
    "model_name": "naver/splade-cocondenser-ensembledistil",
    "device": "auto",
    "quantization": "float16",
    "batch_size": 32,
    "max_length": 512
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `naver/splade-cocondenser-ensembledistil` | HuggingFace model ID |
| `device` | `auto` | Device selection (`auto`, `cuda`, `cpu`) |
| `quantization` | `float16` | Precision (`float32`, `float16`, `int8`) |
| `batch_size` | 32 | Batch size for inference |
| `max_length` | 512 | Maximum sequence length |
| `top_k_tokens` | None | Optional top-k filtering per vector |

**GPU Memory Recommendations:**

| GPU VRAM | Recommended `batch_size` |
|----------|-------------------------|
| 4GB | 8 |
| 6GB | 16 |
| 8GB | 32 |
| 12GB | 64 |
| 24GB | 128 |

**Performance:**
- Throughput: 10-50 documents/second (GPU)
- Throughput: 0.3-1 documents/second (CPU)
- Query latency: <100ms (GPU), <1s (CPU)
- Model load time: ~30s (GPU), ~60s (CPU)

---

## API Reference

### Enable Sparse Index

```
POST /api/v2/collections/{collection_name}/sparse-index
```

**Request:**
```json
{
  "plugin_id": "bm25-local",
  "model_config": {},
  "reindex_existing": true
}
```

**Response:**
```json
{
  "status": "enabled",
  "plugin_id": "bm25-local",
  "sparse_collection_name": "my_docs_sparse_bm25",
  "reindex_job_id": "abc123"
}
```

### Get Sparse Index Status

```
GET /api/v2/collections/{collection_name}/sparse-index
```

**Response:**
```json
{
  "enabled": true,
  "plugin_id": "bm25-local",
  "sparse_collection_name": "my_docs_sparse_bm25",
  "chunk_count": 1500,
  "last_indexed_at": "2026-01-08T14:30:00Z",
  "model_config": {"k1": 1.5, "b": 0.75}
}
```

### Disable Sparse Index

```
DELETE /api/v2/collections/{collection_name}/sparse-index
```

Removes the sparse index and deletes the sparse Qdrant collection.

### Trigger Reindex

```
POST /api/v2/collections/{collection_name}/sparse-index/reindex
```

Rebuilds the sparse index for all chunks. Returns a job ID for progress tracking.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "queued",
  "total_chunks": 1500
}
```

### Check Reindex Progress

```
GET /api/v2/collections/{collection_name}/sparse-index/reindex/{job_id}
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "in_progress",
  "progress": 0.45,
  "indexed_chunks": 675,
  "total_chunks": 1500,
  "estimated_remaining_seconds": 120
}
```

### Search with Sparse

```
POST /api/v2/search
```

**Request:**
```json
{
  "query": "machine learning",
  "collection_names": ["my_docs"],
  "search_mode": "hybrid",
  "rrf_k": 60,
  "k": 10,
  "score_threshold": 0.5
}
```

**Response includes:**
```json
{
  "results": [...],
  "search_mode_used": "hybrid",
  "sparse_search_time_ms": 15,
  "rrf_fusion_time_ms": 2,
  "warnings": []
}
```

---

## MCP Integration

The MCP server supports sparse search through the `search_mode` parameter:

```json
{
  "tool": "search_my_collection",
  "parameters": {
    "query": "error handling patterns",
    "search_mode": "hybrid",
    "k": 5
  }
}
```

---

## Performance Considerations

### Latency Budget

| Component | Typical Latency |
|-----------|-----------------|
| Dense search | 10-50ms |
| Sparse search (BM25) | 5-15ms |
| Sparse search (SPLADE) | 50-100ms |
| RRF fusion | 1-5ms |
| **Total (hybrid)** | 50-150ms |

Dense and sparse searches run in parallel, so hybrid latency is approximately:
`max(dense_latency, sparse_latency) + fusion_latency`

### Indexing Throughput

| Indexer | Documents/sec | Notes |
|---------|---------------|-------|
| BM25 | ~1000 | CPU-only |
| SPLADE (GPU) | 10-50 | Depends on batch size |
| SPLADE (CPU) | 0.3-1 | Not recommended for large corpora |

### Storage

Sparse indexes are stored in separate Qdrant collections:
- Collection naming: `{base_name}_sparse_{type}` (e.g., `docs_sparse_bm25`)
- Index type: In-memory for fast search
- Point IDs: Match dense collection chunk IDs for RRF alignment

---

## Migration from Legacy Hybrid Search

**Important:** The legacy `hybrid_search.py` (keyword-filter approach) has been removed. The new sparse indexing system replaces it entirely.

### What Changed

| Old API | New API |
|---------|---------|
| `search_type="hybrid"` | `search_mode="hybrid"` |
| Keyword filter matching | True sparse vector search |
| 70/30 weighted score | RRF fusion with configurable k |
| No IDF weighting | Full BM25 or SPLADE scoring |

### Migration Steps

1. **Update API calls:**
   ```python
   # Old
   response = await search(collection, query, search_type="hybrid")

   # New
   response = await search(collection, query, search_mode="hybrid")
   ```

2. **Enable sparse indexing** on existing collections (one-time):
   ```bash
   POST /api/v2/collections/{name}/sparse-index
   {"plugin_id": "bm25-local", "reindex_existing": true}
   ```

3. **Wait for reindex** to complete before using hybrid mode.

4. **Update MCP tools** to use `search_mode` parameter.

---

## Troubleshooting

### Hybrid Search Returns Dense-Only Results

**Symptom:** Response includes warning about sparse unavailable.

**Cause:** Sparse indexing not enabled on the collection.

**Fix:** Enable sparse indexing:
```bash
POST /api/v2/collections/{name}/sparse-index
{"plugin_id": "bm25-local", "reindex_existing": true}
```

### SPLADE Model Loading Fails

**Symptom:** "Insufficient GPU memory" or OOM error.

**Fix:**
1. Reduce `batch_size` in model_config
2. Use `"device": "cpu"` (slower)
3. Use BM25 instead (no GPU required)

### IDF Statistics Drift

**Symptom:** Search quality degrades over time.

**Cause:** Incremental IDF updates can drift from actual corpus statistics.

**Fix:** Trigger full reindex:
```bash
POST /api/v2/collections/{name}/sparse-index/reindex
```

### Slow Sparse Search

**Symptom:** Sparse queries taking >100ms.

**Causes and fixes:**
1. **SPLADE on CPU:** Use GPU or switch to BM25
2. **Large vocabulary:** Consider enabling vocabulary hashing
3. **Qdrant cold start:** First query is slower; subsequent queries are cached

---

## See Also

- [Search System](SEARCH_SYSTEM.md) - Full search architecture
- [Plugin Development](PLUGIN_DEVELOPMENT.md) - Create custom sparse indexers
- [Plugin Protocols](plugin-protocols.md) - SparseIndexerProtocol reference
- [API Reference](API_REFERENCE.md) - Complete API documentation
