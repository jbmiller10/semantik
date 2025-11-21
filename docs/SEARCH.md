# Search API Contract

This document summarizes the canonical search contract shared between the vecpipe service, the WebUI backend, and the React client.

- Canonical fields only: the API accepts `hybrid_mode` (`weighted` or `filter`) and `keyword_mode` (`any` or `all`). The legacy `hybrid_search_mode` field and legacy keyword values are rejected.
- Requests with unknown fields are rejected (`extra="forbid"` on all search request models).
- `search_type` supports `semantic`, `question`, `code`, and `hybrid`. The legacy alias `vector` maps to `semantic` only.

## Semantic Search (POST /search)

```json
{
  "query": "quantum networking",
  "k": 5,
  "search_type": "semantic",
  "use_reranker": false
}
```

Response (truncated):

```json
{
  "query": "quantum networking",
  "results": [
    {
      "doc_id": "doc_1",
      "chunk_id": "chunk_42",
      "score": 0.91,
      "path": "/docs/quantum.pdf",
      "metadata": { "page": 3 }
    }
  ],
  "num_results": 5,
  "search_type": "semantic",
  "reranking_used": false
}
```

## Semantic Search with Rerank

```json
{
  "query": "quantum networking",
  "k": 5,
  "search_type": "semantic",
  "use_reranker": true,
  "rerank_model": "Qwen/Qwen3-Reranker-0.6B"
}
```

`reranking_used` is `true` and `reranker_model` reflects the model actually used.

## Hybrid Search (GET /hybrid_search)

```http
GET /hybrid_search?q=distributed+systems&k=5&mode=weighted&keyword_mode=all
```

Response shows keyword matches plus combined scores:

```json
{
  "query": "distributed systems",
  "results": [
    {
      "doc_id": "doc_7",
      "chunk_id": "chunk_3",
      "score": 0.78,
      "matched_keywords": ["distributed", "systems"],
      "keyword_score": 0.66,
      "combined_score": 0.80
    }
  ],
  "search_mode": "weighted",
  "keywords_extracted": ["distributed", "systems"]
}
```

## Batch Search (POST /search/batch)

```json
{
  "queries": ["zero shot learning", "retrieval augmented generation"],
  "k": 3,
  "search_type": "semantic"
}
```

Returns a list of `SearchResponse` objects under `responses`.

## Error Mapping

- Invalid hybrid/keyword modes return HTTP 422 (validation error).
- Dimension mismatches return HTTP 400 with `error="dimension_mismatch"`.
- Qdrant errors propagate as HTTP 502.
- Reranker memory exhaustion returns HTTP 507 with `error="insufficient_memory"`.

## Legacy Fields Removed

- `hybrid_search_mode` is no longer accepted anywhere in the stack (client, WebUI, vecpipe).
- Legacy keyword aliases (e.g., `bm25`) now raise validation errors; use `keyword_mode: "any" | "all"`.
- React client and WebUI payload builders only emit canonical fields.
