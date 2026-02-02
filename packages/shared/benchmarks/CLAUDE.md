# Benchmark System

IR evaluation framework for measuring search quality against ground truth datasets.

## Dataset JSON Schema

```json
{
  "schema_version": "1.0",
  "metadata": {"name": "Dataset Name", "description": "..."},
  "queries": [
    {
      "query_key": "q1",
      "query_text": "search query",
      "relevant_docs": [
        {"doc_ref": {"uri": "file:///doc.pdf"}, "relevance_grade": 3},
        {"doc_ref": {"document_id": "doc-123"}, "relevance_grade": 2}
      ]
    }
  ]
}
```

**Relevance grades:** 0=not relevant, 1=marginal, 2=relevant, 3=highly relevant

**Doc ref resolution priority:**
1. `document_id` - Direct ID lookup
2. `uri` - Exact URI match
3. `content_hash` - SHA-256 (must be unique in collection)
4. `path` - Treated as URI (legacy)
5. `file_name` - Filename only (must be unique)

**Limits:** 10MB upload, 1000 queries, 100 judgments per query (configurable)

## Metrics Computed

**With k-values (default: 5, 10, 20):**

| Metric | Formula | Notes |
|--------|---------|-------|
| `precision@k` | relevant_in_top_k / k | Fraction of top-k that are relevant |
| `recall@k` | relevant_in_top_k / total_relevant | Fraction of relevant docs found |
| `ndcg@k` | DCG / IDCG | Uses graded relevance (0-3) |

**Without k-values:**

| Metric | Formula | Notes |
|--------|---------|-------|
| `mrr` | 1 / rank_of_first_relevant | Mean Reciprocal Rank |
| `ap` | mean(precision@each_relevant_pos) | Average Precision |

## Critical: Graded vs Binary Relevance

- **nDCG** uses full grades (0, 1, 2, 3) - grade 3 rewarded more than grade 1
- **All other metrics** use binary: grade > 0 = relevant, grade 0 = not relevant

```python
# In metrics.py:
# Precision/Recall/MRR/AP treat grade=1 same as grade=3
relevant_doc_ids = {doc_id for doc_id, grade in judgments.items() if grade > 0}
```

A doc with `relevance_grade=1` and one with `relevance_grade=3` are equivalent for precision, but nDCG correctly differentiates them.

## Configuration Matrix

Define parameter combinations to compare search strategies:

```python
config_matrix = {
    "primary_k": 10,                      # k for final reporting
    "k_values_for_metrics": [5, 10, 20],  # k values computed per query
    "top_k_values": [100, 200],           # Search depth (must be >= max k_values)
    "search_modes": ["dense", "hybrid"],  # Search strategies
    "use_reranker": [False, True],        # Reranking toggle
    "rrf_k_values": [60],                 # RRF constant for hybrid
    "score_thresholds": [None, 0.5],      # Score filtering
}
# Creates: 2 × 2 × 2 × 1 × 2 = 16 runs
```

## Execution Flow

1. **Create benchmark** → Validates config, pre-creates all run combinations
2. **Start benchmark** → Atomic PENDING→RUNNING transition, dispatches Celery task
3. **Execute runs** → Sequential (prevents GPU thrashing), skips COMPLETED/FAILED (idempotent)
4. **Per query** → Search → collapse chunks to documents → compute metrics
5. **Aggregate** → Mean across queries per run
6. **Progress** → Redis pub/sub → WebSocket → UI (every 5 queries or 500ms)

## Chunk-to-Document Collapse

Search returns chunks; same document may appear multiple times. First-hit deduplication:
- Doc at positions 1 and 5 → only position 1 kept
- Applied BEFORE metric computation
- Matches real-world UX (one result per document)

## Common Gotchas

### top_k must be >= max(k_values)
```python
# ❌ FAILS - can't compute precision@20 with only 10 results
{"k_values_for_metrics": [5, 10, 20], "top_k_values": [10]}

# ✅ CORRECT
{"k_values_for_metrics": [5, 10, 20], "top_k_values": [100]}
```

### Ambiguous mappings still run
PARTIAL mapping status (some unresolved) allows execution. Unresolved judgments contribute 0 to metrics - can skew results.

### Empty ground truth returns 0
Query with no relevant docs → recall = 0.0 (no division by zero), metric still computed.

### Idempotent on retry
Runs marked COMPLETED/FAILED are skipped on re-execution. Safe to retry partial failures.

### Content hash / filename must be unique
Multiple docs with same hash or filename → marked "ambiguous", not resolved.

## File Structure

```
shared/benchmarks/
├── metrics.py           # compute_all_metrics(), precision/recall/ndcg/mrr/ap
├── types.py             # RetrievedDocument, QueryResult dataclasses
├── evaluator.py         # ConfigurationEvaluator orchestration
└── utils.py             # k-value parsing, validation helpers

webui/services/
├── benchmark_service.py     # CRUD, start/cancel, results aggregation
├── benchmark_executor.py    # Celery task execution logic
├── dataset_service.py       # Dataset upload, validation
└── mapping_service.py       # Collection mapping, resolution
```

## Testing

```bash
uv run pytest tests/shared/benchmarks/ -v
uv run pytest tests/webui/services/test_benchmark_service.py -v
```
