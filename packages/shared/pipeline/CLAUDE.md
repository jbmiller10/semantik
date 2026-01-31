# Pipeline DAG System

Document routing system using directed acyclic graphs with predicate-based edge matching.

## Predicate Expression Syntax

Predicates evaluate file attributes to determine routing:

| Pattern | Example | Behavior |
|---------|---------|----------|
| Exact match | `{"mime_type": "application/pdf"}` | String equality |
| Glob | `{"mime_type": "application/*"}` | Wildcard matching |
| Negation | `{"mime_type": "!image/*"}` | Prefix with `!` |
| Numeric | `{"size_bytes": ">10000000"}` | Operators: `>`, `>=`, `<`, `<=`, `==`, `!=` |
| Array (OR) | `{"extension": [".md", ".txt"]}` | Match any element |
| Nested path | `{"metadata.detected.is_code": true}` | Dot notation traversal |
| Multiple fields | `{"mime_type": "text/*", "size_bytes": "<100000"}` | AND logic |
| Catch-all | `None` or `{}` | Matches everything |

**Metadata namespacing:** Use `metadata.source.*`, `metadata.detected.*`, `metadata.parsed.*`

**Legacy format:** `source_metadata.*` auto-translates to `metadata.source.*`

## Validation Rules

DAGs are validated with these 9 rules:

1. **Exactly one EMBEDDER** - Terminal node for vector storage
2. **All edge references exist** - `from_node`/`to_node` must exist (except `_source`)
3. **All nodes reachable** - BFS from `_source` must reach every node
4. **All nodes reach terminal** - Every node must path to EMBEDDER or EXTRACTOR
5. **No cycles** - Acyclic graph required
6. **Non-parallel catch-all from _source** - CRITICAL: Must have `when=None` AND `parallel=False`
7. **Unique node IDs** - No duplicate node identifiers
8. **Valid plugin IDs** - If `known_plugins` provided, validates against registry
9. **Unique path names** - Parallel edges from same node need distinct `path_name`

## Edge Evaluation Order

At each routing stage, edges evaluate in strict order:

1. Parallel predicate edges → all matches fire
2. Exclusive predicate edges → **first match wins** (short-circuit)
3. Parallel catch-all edges → all fire
4. Exclusive catch-all edges → first match wins (fallback)

**Parallel (`parallel=True`):** All matching edges fire together (fan-out)

**Exclusive (`parallel=False`, default):** First match wins, others skipped

## Common Gotchas

### Missing catch-all from _source
```python
# ❌ FAILS - no fallback path
edges = [
    PipelineEdge(from_node="_source", to_node="pdf_parser", when={"mime_type": "application/pdf"}),
]

# ✅ CORRECT - catch-all handles unmatched files
edges = [
    PipelineEdge(from_node="_source", to_node="pdf_parser", when={"mime_type": "application/pdf"}),
    PipelineEdge(from_node="_source", to_node="text_parser"),  # Catch-all
]
```

### Parallel edges need unique path_name
```python
# ❌ FAILS - duplicate path names
PipelineEdge(from_node="_source", to_node="chunker1", parallel=True, path_name="extract"),
PipelineEdge(from_node="_source", to_node="chunker2", parallel=True, path_name="extract"),

# ✅ CORRECT
PipelineEdge(from_node="_source", to_node="chunker1", parallel=True, path_name="detailed"),
PipelineEdge(from_node="_source", to_node="chunker2", parallel=True, path_name="summary"),
```

### All paths must converge to terminal
```python
# ❌ FAILS - chunker1 never reaches embedder
edges = [
    PipelineEdge(from_node="_source", to_node="parser1"),
    PipelineEdge(from_node="parser1", to_node="chunker1"),
    # Missing: PipelineEdge(from_node="chunker1", to_node="embedder")
]
```

### Predicate on missing field returns False
If `file_ref.extension` is `None`, predicate `{"extension": ".pdf"}` won't match. Use catch-all edges.

## Node Types

| Type | Purpose | Example Config |
|------|---------|----------------|
| `PARSER` | Extract text from raw bytes | `{"strategy": "auto"}` |
| `CHUNKER` | Split text into chunks | `{"max_tokens": 1000, "overlap_tokens": 50}` |
| `EXTRACTOR` | Extract metadata/entities | Plugin-specific (optional, for parallel paths) |
| `EMBEDDER` | Generate vector embeddings | `{"model": "..."}` |

## File Structure

```
pipeline/
├── types.py          # DAG, Node, Edge, FileReference dataclasses
├── predicates.py     # matches_predicate(), match_value()
├── router.py         # Edge matching, _evaluate_edges()
├── validation.py     # validate() with 9 rules
├── executor.py       # Pipeline execution orchestrator
├── sniff.py          # Pre-routing content detection
├── defaults.py       # Default pipeline factory
└── templates/        # Pre-configured pipelines (codebase, docs, academic)
```

## Testing

```bash
uv run pytest tests/shared/pipeline/ -v
uv run pytest tests/shared/pipeline/test_router.py::test_specific -v
```
