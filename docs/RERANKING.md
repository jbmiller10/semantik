# Reranking

Two-stage retrieval: fast vector search + cross-encoder reranking. ~20-30% better relevance.

**Features**: Auto model selection, memory-aware loading, graceful fallback

## How It Works

1. Vector search retrieves k×5 candidates (fast, 10-50ms)
2. Cross-encoder scores query-doc pairs (accurate, 100-800ms)
3. Return top k reranked results

Why? Embeddings are fast but lossy. Cross-encoders are accurate but slow. Two stages = best of both.

## Models

Auto-selects reranker based on embedding model:

| Reranker | Params | Memory (fp16/int8) | Batch | Speed | Use Case |
|----------|--------|-------------------|-------|-------|----------|
| Qwen3-0.6B | 0.6B | 1.2GB / 600MB | 64-256 | Fast | General, real-time |
| Qwen3-4B | 4B | 8GB / 4GB | 16-64 | Medium | Complex queries |
| Qwen3-8B | 8B | 16GB / 8GB | 8-32 | Slow | Research, legal |

**Logic**: Exact match → size-based fallback → default (0.6B)

## Usage

```python
# POST /search (GET doesn't support reranking)
requests.post("/search", json={
    "query": "...",
    "k": 10,
    "use_reranker": True,
    "rerank_model": "Qwen/Qwen3-Reranker-4B",  # optional
    "rerank_quantization": "int8"  # optional
})
```

## Config

| Param | Default | Notes |
|-------|---------|-------|
| `candidate_multiplier` | 5 | Retrieve k×5 candidates |
| `min_candidates` | 20 | Floor |
| `max_candidates` | 200 | Cap |
| `use_hybrid_scoring` | True | 0.3 vector + 0.7 rerank |
| `hybrid_weight` | 0.3 | Vector score weight |

Batch sizes auto-configured per model+quantization.

## Performance

**Memory**: Embedding + reranker loaded. Lazy loading, auto-unload after 5min idle, pre-flight checks.

**Latency**:
- Vector: 10-50ms
- Load model: 2-10s (first time)
- Rerank 50 docs: 100-300ms (0.6B), 300-600ms (4B), 600-1200ms (8B)

**Tips**: Start with 0.6B+int8. Tune `candidate_multiplier` (3-10). Monitor GPU with `nvidia-smi`.

## Implementation

**Format**: `<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}`

**Scoring**: Token probabilities (Yes/No tokens → softmax → P(yes) = score, range 0-1)

**Error handling**: Falls back to vector-only on OOM

## Examples

```python
# Basic
search(..., use_reranker=True)

# Custom model
search(..., use_reranker=True, rerank_model="Qwen/Qwen3-Reranker-4B", rerank_quantization="float16")

# Note: BatchSearchRequest doesn't support reranking - use individual POST requests
```

## When to Use

**Yes**: Complex queries, Q&A, technical docs, precision > speed
**No**: Simple lookups, autocomplete, high volume, limited GPU

## Production Checklist

- Test query volume
- Monitor GPU memory
- Set timeouts
- Configure unload timing
- Test fallback

## Troubleshooting

**OOM**: Use smaller model, int8, reduce batch size
**Slow**: Reduce `candidate_multiplier`, smaller model, quantization
**Cold start**: Pre-load models, increase unload timeout

Debug: `logging.getLogger("semantik.reranker").setLevel(logging.DEBUG)`

## Advanced

**Custom models**: Add mapping, configure batch sizes, implement scoring
**A/B testing**: Compare `use_reranker=True` vs `False`

## Features

**Flash Attention**: Auto-detected (2-3x faster for long docs)
**Collection integration**: Auto-configures from metadata
**Monitoring**: `reranking_time_ms`, `reranker_model`, `candidates_retrieved`, `memory_fallback_count`

## Files

- `reranker.py` - Core logic
- `search/service.py` - API integration
- `qwen3_search_config.py` - Config
- `memory_utils.py` - Memory management