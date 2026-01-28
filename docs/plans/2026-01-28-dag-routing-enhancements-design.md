# DAG Routing Enhancements Design

**Date:** 2026-01-28
**Status:** Draft
**Context:** External feedback on DAG pipeline architecture identified gaps in routing semantics

## Overview

This design addresses routing gaps in the DAG pipeline architecture, focusing on derived metadata routing while preserving the simplicity of the current edge-based routing model.

## Current State (What We Have)

The existing routing implementation is solid:

| Capability | Implementation | Location |
|------------|----------------|----------|
| Deterministic precedence | First-match-wins | `packages/shared/pipeline/router.py` |
| Catch-all routes | Required by validation | `packages/shared/pipeline/validation.py` |
| Glob patterns | `*`, `?`, `[...]` support | `packages/shared/pipeline/predicates.py` |
| Negation | `!` prefix | `predicates.py` |
| OR lists | Array values | `predicates.py` |
| Numeric comparisons | `>`, `>=`, `<`, `<=`, `==`, `!=` | `predicates.py` |
| Nested field access | Dot notation | `predicates.py` |
| Validation | 8 rules including unreachable detection | `validation.py` |

## Problem Statement

Current routing can only use metadata available at ingest time (extension, source-provided MIME, source_metadata). This limits routing decisions to surface-level attributes.

Real-world routing needs:
- Route scanned PDFs to OCR parser, native PDFs to fast parser
- Route by detected language (Chinese docs to multilingual chunker)
- Route large documents to different chunking strategies
- Route based on content characteristics (has tables, has images)

## Design Decisions

### 1. Built-in Sniff Step

A lightweight analysis step runs before routing decisions, enriching metadata with derived fields.

**Initial sniff capabilities:**

| Field | Cost | How |
|-------|------|-----|
| `detected.mime_type` | ~1ms | Magic bytes (first 4-8KB) |
| `detected.is_scanned_pdf` | ~10-50ms | PDF structure analysis, text layer check |
| `detected.language` | ~5-20ms | Sample content, run classifier |
| `detected.approx_token_count` | ~10-30ms | Sample content, rough tokenization |

**Cost justification:** Total sniff cost (~50-100ms) is ~1-5% of typical pipeline time. Embedding dominates at 1-10s per document. No lazy evaluation needed.

**Extensibility:** New sniff fields can be added in future versions. The design is additive—old DAGs continue working, new DAGs can opt into new fields.

### 2. Metadata Architecture

Single `metadata` dict with namespace conventions replaces the current `source_metadata` field.

```
metadata:
  source:           # From connector/source plugin
    mime_type: "application/octet-stream"
    size_bytes: 1048576
    created_at: "2026-01-28T10:00:00Z"
  detected:         # From sniff step
    mime_type: "application/pdf"
    is_scanned_pdf: true
    language: "en"
    approx_token_count: 15000
  parsed:           # From parser stage (future)
    page_count: 12
    has_tables: true
  custom:           # User-defined
    department: "engineering"
    priority: "high"
```

**Design rationale:**

- **Single namespace over separate fields:** Avoids proliferation of fields on FileReference. Parser metadata, chunker metadata, etc. all fit naturally.
- **Explicit prefixes over merged keys:** Users specify which layer to trust (`detected.mime_type` vs `source.mime_type`). No magic resolution rules to learn.
- **Conflict prevention:** Each stage writes to its own namespace. Single-path execution means no race conditions.

**Predicate examples:**

```python
# Route scanned PDFs to OCR parser
{"metadata.detected.is_scanned_pdf": true}

# Route Chinese documents to multilingual chunker
{"metadata.detected.language": "zh"}

# Route large documents to semantic chunker
{"metadata.detected.approx_token_count": ">10000"}

# Combine conditions (AND logic)
{"metadata.detected.mime_type": "application/pdf", "metadata.detected.is_scanned_pdf": false}
```

### 3. Route Preview (Recommended Enhancement)

A "test this file" feature in the DAG editor UI:

1. User uploads or selects a sample file
2. System runs sniff step in preview mode
3. UI shows actual routing path with predicate evaluation:

```
Sample: quarterly-report.pdf

Metadata detected:
  detected.mime_type = "application/pdf"
  detected.is_scanned_pdf = false
  detected.language = "en"

Edge evaluation:
  ├─ source → ocr_parser
  │    when: {detected.is_scanned_pdf: true}
  │    Result: NOT MATCHED (is_scanned_pdf = false)
  │
  ├─ source → pdf_parser
  │    when: {detected.mime_type: "application/pdf"}
  │    Result: MATCHED ✓
  │
  └─ source → text_parser
       when: null (catch-all)
       Result: SKIPPED (earlier match)

Path: source → pdf_parser → chunker → embedder
```

**Value:** High learnability impact, aids debugging, low implementation complexity.

## What We're Not Doing

### Fan-out (Same File, Multiple Branches)

**Considered:** Allow same document through multiple parallel branches (e.g., dense + sparse embeddings).

**Decision:** Not needed. Hybrid search is already implemented at the ingestion layer, not as DAG fan-out:
- Dense embeddings: DAG pipeline (source → parser → chunker → embedder)
- Sparse embeddings: Parallel side-effect via `_maybe_generate_sparse_vectors()`

This architecture is deliberate—sparse indexing (especially BM25) is stateful and doesn't fit the streaming/per-file paradigm.

**Revisit if:** Other fan-out use cases emerge (multiple chunking strategies, parsing + summarization branches).

### Fallback-on-Failure

**Considered:** If parser A fails, route to parser B as fallback.

**Decision:** Defer. Current fail-fast approach (consecutive failure tracker → halt → investigate) is valid. Fallback routing could mask underlying issues.

**Revisit if:** Real-world failure patterns emerge that would benefit from graceful degradation.

## Implementation Plan

### Phase 1: Metadata Architecture Refactor

1. **Rename `source_metadata` → `metadata`** on FileReference
2. **Restructure existing data** under `metadata.source.*`
3. **Update predicates.py** to handle new structure (backward compatible with old field name during migration)
4. **Update existing DAGs** to use new predicate paths

### Phase 2: Sniff Step Implementation

1. **Create sniff module** in `packages/shared/pipeline/sniff.py`
2. **Implement initial detectors:**
   - MIME via magic bytes (use `python-magic` or similar)
   - PDF scanned detection (check for text layer)
   - Language detection (use `langdetect` or `fasttext`)
   - Token count estimation
3. **Integrate into routing** - sniff runs before `get_entry_node()`
4. **Handle sniff failures gracefully** - missing detected fields don't break routing

### Phase 3: Route Preview UI

1. **Add file upload/select** in DAG editor
2. **Create preview endpoint** that runs sniff + evaluates predicates
3. **Display results** with matched/not-matched visualization
4. **Show final path** through DAG

## Sniff Extensibility

The sniff system is designed for future extension:

### Adding New Sniff Fields

1. Add detector function to `sniff.py`
2. Write to `metadata.detected.{new_field}`
3. Document field in sniff capabilities table
4. No migration needed—existing DAGs unaffected

### Potential Future Fields

| Field | Use Case |
|-------|----------|
| `detected.has_tables` | Route to table-aware parser |
| `detected.has_images` | Route to multimodal pipeline |
| `detected.encoding` | Handle non-UTF8 documents |
| `detected.is_code` | Route to code-specific chunker |
| `detected.document_type` | Invoice, resume, article, etc. |

### Plugin-Extensible Sniff (Future)

If third-party sniff plugins are needed:
- Plugins write to `metadata.detected.{plugin_name}.*`
- Core sniff fields remain in `metadata.detected.*`
- Plugin fields are opt-in per collection

## Migration Strategy

### Backward Compatibility

1. **FileReference** accepts both `source_metadata` (deprecated) and `metadata`
2. **Predicates** check both paths during transition
3. **Existing DAGs** continue working without modification
4. **Deprecation warning** logged when old field used

### Migration Path

1. Deploy code with dual support
2. Migrate existing collections (background task or on-access)
3. Update documentation and examples
4. Remove deprecated field in future version

## Testing Strategy

### Unit Tests

- Sniff detectors with various file types
- Predicate evaluation with nested metadata paths
- Backward compatibility with old metadata structure

### Integration Tests

- End-to-end routing with sniffed metadata
- Preview endpoint returns correct paths
- Migration preserves existing behavior

### Edge Cases

- Files that fail sniffing (corrupted, unsupported)
- Missing detected fields in predicates
- Very large files (sniff timeout handling)

## Open Questions

1. **Sniff timeout:** Should there be a per-file timeout for sniff operations? Suggested: 5s default, configurable.

2. **Sniff caching:** Should sniff results be cached for re-indexing? Likely yes, stored alongside source metadata.

3. **Selective sniffing:** Run all detectors always, or only those referenced in DAG predicates? Recommendation: Run all (cost is negligible), keeps behavior predictable.

## Appendix: Feedback Analysis

Original feedback identified 6 routing requirements. Assessment against our implementation:

| Requirement | Status | Notes |
|-------------|--------|-------|
| Deterministic precedence | ✅ Have | First-match-wins |
| Catch-all + fallback | ✅ Partial | Catch-all required; per-edge fallback deferred |
| Match on derived metadata | 🔨 This design | Sniff step + metadata namespaces |
| Expression support | ✅ Have | Glob, negation, numeric, arrays |
| Fan-out/join semantics | ⏭️ Deferred | Not needed for current use cases |
| Preview + validation | ✅ Partial → 🔨 | Have validation; adding preview |
