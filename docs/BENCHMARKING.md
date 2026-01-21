# Benchmarking Tool

Semantik includes an integrated benchmarking system for evaluating search quality. This guide explains how to use it to compare retrieval configurations and identify optimal settings for your collections.

## Overview

The benchmarking tool allows you to:

1. **Upload ground truth datasets** — Define queries and their expected relevant documents
2. **Map datasets to collections** — Link your evaluation data to indexed collections
3. **Configure comparison matrices** — Test multiple search configurations in parallel
4. **Analyze results** — Compare metrics like Precision, Recall, MRR, and nDCG

## Quick Start

1. Navigate to the **Benchmarks** tab in the Semantik UI
2. Upload a dataset file (JSON format)
3. Create a mapping to your target collection
4. Resolve document references
5. Create a benchmark with your desired configuration matrix
6. Start the benchmark and view results

---

## Dataset Format

Benchmark datasets are JSON files containing queries and their relevant documents.

### Basic Schema

```json
{
  "schema_version": "1.0",
  "metadata": {
    "name": "My Evaluation Dataset",
    "description": "Optional description"
  },
  "queries": [
    {
      "query_key": "q1",
      "query_text": "How do I configure authentication?",
      "relevant_docs": [
        {
          "doc_ref": { "uri": "file:///docs/auth-guide.md" },
          "relevance_grade": 3
        },
        {
          "doc_ref": { "uri": "file:///docs/security.md" },
          "relevance_grade": 2
        }
      ]
    }
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `schema_version` | No | Dataset format version (default: "1.0") |
| `metadata` | No | Optional metadata object |
| `queries` | Yes | Array of query objects |

### Query Object

| Field | Required | Description |
|-------|----------|-------------|
| `query_key` | Yes | Unique identifier for the query (also accepts `query_id`) |
| `query_text` | Yes | The search query text (also accepts `query`) |
| `relevant_docs` | Yes | Array of relevant document judgments |
| `metadata` | No | Optional query-level metadata |

### Relevance Judgment Object

| Field | Required | Description |
|-------|----------|-------------|
| `doc_ref` | Yes | Document reference (see below) |
| `relevance_grade` | No | Relevance level 0-3 (default: 2) |

### Document References

Document references tell the system how to match ground truth documents to your indexed collection. Supported formats:

**URI-based reference** (recommended):
```json
{
  "doc_ref": { "uri": "file:///path/to/document.md" }
}
```

**Simple string shorthand**:
```json
{
  "doc_ref": "file:///path/to/document.md"
}
```

The URI should match the document's URI in Semantik. For file-based connectors, this is typically the absolute file path with `file://` prefix.

### Relevance Grades

| Grade | Meaning |
|-------|---------|
| 0 | Not relevant |
| 1 | Marginally relevant |
| 2 | Relevant (default) |
| 3 | Highly relevant |

Graded relevance enables nDCG calculation. For binary evaluation, use grades 0 (not relevant) and 1+ (relevant).

### Dataset Limits

| Limit | Default |
|-------|---------|
| Max file size | 10 MB |
| Max queries | 1,000 |
| Max judgments per query | 100 |

### Example: Minimal Dataset

```json
{
  "queries": [
    {
      "query_key": "auth",
      "query_text": "How do I set up JWT authentication?",
      "relevant_docs": [
        { "doc_ref": "file:///docs/jwt-auth.md", "relevance_grade": 3 }
      ]
    },
    {
      "query_key": "deploy",
      "query_text": "Docker deployment instructions",
      "relevant_docs": [
        { "doc_ref": "file:///docs/docker.md", "relevance_grade": 3 },
        { "doc_ref": "file:///docs/deployment.md", "relevance_grade": 2 }
      ]
    }
  ]
}
```

---

## Mapping Workflow

Before running benchmarks, you must map your dataset to a collection and resolve document references.

### 1. Create Mapping

Link your dataset to a target collection:

```
Dataset → Collection Mapping → Resolved Document IDs
```

The mapping creates a binding between your dataset and a specific collection.

### 2. Resolve References

Resolution matches your `doc_ref` URIs to actual documents in the collection:

- **Resolved**: URI matches a document in the collection
- **Partial**: Some documents matched, others not found
- **Pending**: Resolution not yet attempted

### 3. Handle Unresolved References

If some documents aren't found:

1. Check that the URIs in your dataset match the document paths in Semantik
2. Ensure documents are indexed in the target collection
3. Update your dataset file with correct URIs if needed

You can run benchmarks with partial mappings, but metrics will only reflect the resolved documents.

---

## Configuration Matrix

The configuration matrix defines which search parameters to test. Each unique combination creates a separate benchmark run.

### Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `search_modes` | Search algorithm | `["dense"]`, `["dense", "hybrid"]` |
| `use_reranker` | Enable reranking | `[true]`, `[false]`, `[true, false]` |
| `top_k_values` | Results to retrieve | `[10]`, `[5, 10, 20]` |
| `rrf_k_values` | RRF constant (hybrid only) | `[60]`, `[40, 60, 80]` |
| `score_thresholds` | Minimum score filter | `[null]`, `[0.5]` |

### Search Modes

| Mode | Description | Requirements |
|------|-------------|--------------|
| `dense` | Vector similarity search | Always available |
| `sparse` | BM25/SPLADE keyword search | Requires sparse index |
| `hybrid` | Combined dense + sparse with RRF fusion | Requires sparse index |

### Example Configurations

**Dense vs Hybrid comparison:**
```json
{
  "search_modes": ["dense", "hybrid"],
  "use_reranker": [false],
  "top_k_values": [10],
  "rrf_k_values": [60]
}
```
Creates 2 runs: dense@10 and hybrid@10.

**Reranker A/B test:**
```json
{
  "search_modes": ["dense"],
  "use_reranker": [false, true],
  "top_k_values": [10, 20]
}
```
Creates 4 runs: dense@10, dense@10+rerank, dense@20, dense@20+rerank.

**Top-K sensitivity analysis:**
```json
{
  "search_modes": ["dense"],
  "use_reranker": [false],
  "top_k_values": [5, 10, 20, 50]
}
```
Creates 4 runs testing different result set sizes.

### Configuration Count

The total number of runs is the product of all parameter variations:

```
runs = |search_modes| × |use_reranker| × |top_k_values| × |rrf_k_values| × |score_thresholds|
```

The UI shows warnings when approaching limits:
- **Yellow warning**: >25 configurations
- **Blocked**: >50 configurations

---

## Metrics

The benchmarking tool computes standard information retrieval metrics.

### Precision@K

**What it measures**: Of the top K results, how many are relevant?

```
Precision@K = (relevant docs in top K) / K
```

- Range: 0.0 to 1.0
- Higher is better
- Use when: You care about result quality at fixed positions

### Recall@K

**What it measures**: Of all relevant documents, how many appear in top K?

```
Recall@K = (relevant docs in top K) / (total relevant docs)
```

- Range: 0.0 to 1.0
- Higher is better
- Use when: You need to find all relevant documents

### MRR (Mean Reciprocal Rank)

**What it measures**: How early does the first relevant result appear?

```
MRR = 1 / (rank of first relevant result)
```

- Range: 0.0 to 1.0
- Higher is better (1.0 = first result is relevant)
- Use when: Users typically want just one good answer

### nDCG@K (Normalized Discounted Cumulative Gain)

**What it measures**: Ranking quality with graded relevance

nDCG rewards:
- Highly relevant documents ranked higher than marginally relevant ones
- Relevant documents appearing earlier in results

```
nDCG@K = DCG@K / IDCG@K
```

- Range: 0.0 to 1.0
- Higher is better
- Use when: You have graded relevance judgments

### AP (Average Precision)

**What it measures**: Overall ranking quality across all recall levels

- Range: 0.0 to 1.0
- Higher is better
- Use when: You want a single summary metric

---

## Document-Level Evaluation

Semantik retrieval returns chunks, but benchmarks evaluate at the document level.

### How It Works

1. Search returns ranked chunks with scores
2. Chunks are collapsed to unique documents (first-hit ranking)
3. Metrics are computed on the document ranking

**Example:**
```
Search results (chunks):
  1. doc_a/chunk_3 (score: 0.95)
  2. doc_b/chunk_1 (score: 0.90)
  3. doc_a/chunk_1 (score: 0.85)  ← duplicate, removed
  4. doc_c/chunk_2 (score: 0.80)

After collapsing:
  1. doc_a
  2. doc_b
  3. doc_c
```

This ensures metrics reflect document-level retrieval quality, not chunk distribution.

---

## Running Benchmarks

### Via the UI

1. **Navigate** to the Benchmarks tab
2. **Select** your dataset and mapping
3. **Configure** the parameter matrix
4. **Click** "Start Benchmark"
5. **Monitor** progress in real-time
6. **View** results when complete

### Progress Tracking

While running, you'll see:
- Overall progress (runs completed / total runs)
- Current run progress (queries evaluated)
- Live metrics updates as runs complete

### Cancellation

You can cancel a running benchmark at any time. Completed runs are preserved.

---

## Interpreting Results

### Results Table

The results view shows:
- Configuration parameters for each run
- Metrics at all evaluated K values
- Timing information (search latency)
- Status (completed, failed)

### Best Configuration

The system highlights the best-performing configuration based on your primary metric (default: nDCG@10).

### Per-Query Drill-Down

Click a run to see per-query results:
- Which queries performed well/poorly
- Retrieved document IDs
- Individual query metrics
- Search timing

This helps identify:
- Queries where your system struggles
- Documents that should rank higher
- Potential ground truth errors

---

## Best Practices

### Dataset Quality

1. **Representative queries**: Include queries your users actually ask
2. **Complete judgments**: Label all relevant documents, not just the obvious ones
3. **Graded relevance**: Use all four grades for nuanced nDCG evaluation
4. **Sufficient size**: 50-200 queries provides statistical significance

### Configuration Design

1. **Start simple**: Test one variable at a time
2. **Use presets**: Built-in presets cover common comparisons
3. **Mind the matrix**: Don't create more runs than necessary

### Iteration

1. Run baseline benchmark (dense, no reranker)
2. Test one improvement at a time
3. Compare metrics to identify winners
4. Validate with different query subsets

---

## Troubleshooting

### "Mapping must be resolved"

Your dataset mapping has unresolved document references. Click "Resolve" on the mapping to attempt resolution, then check for any unmatched URIs.

### Low Recall Despite Relevant Documents

- Check that document URIs match exactly
- Verify documents are indexed in the collection
- Review the "unresolved" list in mapping details

### Unexpected Metric Values

- Confirm relevance grades are correct (0-3 scale)
- Check for duplicate document references
- Verify query text matches intended search behavior

### Sparse/Hybrid Not Available

Sparse and hybrid search modes require a sparse index on the collection. Build a sparse index (BM25 or SPLADE) before benchmarking these modes.

---

## API Reference

### Datasets

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/benchmark-datasets` | POST | Upload dataset |
| `/api/v2/benchmark-datasets` | GET | List datasets |
| `/api/v2/benchmark-datasets/{id}` | GET | Get dataset details |
| `/api/v2/benchmark-datasets/{id}` | DELETE | Delete dataset |
| `/api/v2/benchmark-datasets/{id}/mappings` | POST | Create mapping |
| `/api/v2/benchmark-datasets/{id}/mappings` | GET | List mappings |
| `/api/v2/benchmark-datasets/{id}/mappings/{mapping_id}/resolve` | POST | Resolve mapping |

### Benchmarks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/benchmarks` | POST | Create benchmark |
| `/api/v2/benchmarks` | GET | List benchmarks |
| `/api/v2/benchmarks/{id}` | GET | Get benchmark details |
| `/api/v2/benchmarks/{id}/start` | POST | Start execution |
| `/api/v2/benchmarks/{id}/cancel` | POST | Cancel execution |
| `/api/v2/benchmarks/{id}/results` | GET | Get results |
| `/api/v2/benchmarks/{id}/runs/{run_id}/queries` | GET | Get per-query results |
| `/api/v2/benchmarks/{id}` | DELETE | Delete benchmark |

---

## Glossary

| Term | Definition |
|------|------------|
| **Dataset** | Collection of queries with relevance judgments |
| **Mapping** | Link between a dataset and a Semantik collection |
| **Run** | Single benchmark execution with one configuration |
| **Configuration Matrix** | Parameter combinations to test |
| **Ground Truth** | Known-correct relevance judgments |
| **Relevance Grade** | 0-3 score indicating document relevance |
| **RRF** | Reciprocal Rank Fusion (hybrid search algorithm) |
