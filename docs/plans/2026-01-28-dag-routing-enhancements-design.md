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

### 1. Two-Stage Metadata Enrichment

Metadata is enriched at two points in the pipeline:

1. **Pre-routing sniff** (`metadata.detected.*`) - Minimal, only for parser selection
2. **Parser-emitted metadata** (`metadata.parsed.*`) - Rich metadata discovered during parsing

This avoids duplicating work (sniff then parse same content) while enabling routing at both entry and mid-pipeline stages.

#### Pre-Routing Sniff (Minimal)

Only fields needed to select the right parser:

| Field | Cost | How | Use Case |
|-------|------|-----|----------|
| `detected.is_scanned_pdf` | ~30ms | PDF structure analysis, text layer check | OCR parser vs fast parser |
| `detected.is_code` | ~10ms | Heuristics (shebang, syntax patterns, extension) | Code parser vs prose parser |
| `detected.is_structured_data` | ~5ms | Try parse first N bytes as JSON/CSV/XML | Structured parser vs text parser |

**Total pre-routing cost:** ~45ms worst case (<2% of pipeline time)

#### Parser-Emitted Metadata

Everything else is discovered during parsing and written to `metadata.parsed.*`:

| Field | Emitted By | Use Case |
|-------|------------|----------|
| `parsed.page_count` | PDF parser | Downstream chunking decisions |
| `parsed.has_tables` | PDF/HTML parser | Table-aware chunking |
| `parsed.has_images` | PDF/HTML parser | Multimodal pipeline routing |
| `parsed.detected_language` | Any parser | Multilingual embedder selection |
| `parsed.approx_token_count` | Any parser | Chunking strategy |
| `parsed.has_code_blocks` | Markdown parser | Code-preserving chunking |
| `parsed.text_quality` | OCR parser | Quality threshold routing |

**Routing implications:** Entry routing (source → parser) uses `detected.*`. Mid-pipeline routing (parser → chunker) can use `parsed.*`.

**Extensibility:** New sniff fields can be added in future versions. Parser plugins can emit any `parsed.*` fields they discover. The design is additive—old DAGs continue working, new DAGs can opt into new fields.

### 2. Metadata Architecture

Single `metadata` dict with namespace conventions replaces the current `source_metadata` field.

```
metadata:
  source:           # From connector/source plugin
    mime_type: "application/pdf"
    size_bytes: 1048576
    created_at: "2026-01-28T10:00:00Z"
  detected:         # From pre-routing sniff (minimal)
    is_scanned_pdf: true
    is_code: false
    is_structured_data: false
  parsed:           # From parser stage
    page_count: 12
    has_tables: true
    detected_language: "en"
    approx_token_count: 15000
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
# Entry routing (source → parser) - uses detected.*
# Route scanned PDFs to OCR parser
{"metadata.detected.is_scanned_pdf": true}

# Route code files to code parser
{"metadata.detected.is_code": true}

# Route JSON/CSV to structured parser
{"metadata.detected.is_structured_data": true}

# Mid-pipeline routing (parser → chunker) - uses parsed.*
# Route Chinese documents to multilingual embedder
{"metadata.parsed.detected_language": "zh"}

# Route large documents to semantic chunker
{"metadata.parsed.approx_token_count": ">10000"}

# Route documents with tables to table-aware chunker
{"metadata.parsed.has_tables": true}
```

### 3. Route Preview (Recommended Enhancement)

A "test this file" feature in the DAG editor UI:

1. User uploads or selects a sample file
2. System runs sniff step in preview mode
3. UI shows actual routing path with predicate evaluation:

```
Sample: quarterly-report.pdf

Pre-routing sniff:
  detected.is_scanned_pdf = false
  detected.is_code = false
  detected.is_structured_data = false

Edge evaluation (entry routing):
  ├─ source → ocr_parser
  │    when: {detected.is_scanned_pdf: true}
  │    Result: NOT MATCHED (is_scanned_pdf = false)
  │
  ├─ source → pdf_parser
  │    when: {source.mime_type: "application/pdf"}
  │    Result: MATCHED ✓
  │
  └─ source → text_parser
       when: null (catch-all)
       Result: SKIPPED (earlier match)

Parser-emitted metadata (after pdf_parser):
  parsed.page_count = 12
  parsed.has_tables = true
  parsed.detected_language = "en"
  parsed.approx_token_count = 8500

Edge evaluation (mid-pipeline routing):
  ├─ pdf_parser → table_chunker
  │    when: {parsed.has_tables: true}
  │    Result: MATCHED ✓
  │
  └─ pdf_parser → default_chunker
       when: null (catch-all)
       Result: SKIPPED (earlier match)

Path: source → pdf_parser → table_chunker → embedder
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

### Phase 1: Metadata Architecture Refactor ✅ COMPLETE

**Status:** Implemented

1. ✅ **Rename `source_metadata` → `metadata`** on FileReference (`types.py:64`)
2. ✅ **Restructure existing data** under `metadata.source.*`
3. ✅ **Update predicates.py** - `_translate_legacy_path()` handles backward compatibility
4. ✅ **Connectors updated** - LocalFile, Git, IMAP all use new structure
5. ✅ **Serialization** - `to_dict()`/`from_dict()` handle both formats

### Phase 2: Pre-Routing Sniff Implementation ✅ COMPLETE

**Status:** Implemented

1. ✅ **Create sniff module** in `packages/shared/pipeline/sniff.py` (535 lines)
2. ✅ **Implement minimal detectors:**
   - `is_scanned_pdf`: pypdf text layer check (~30ms)
   - `is_code`: Extension + shebang + syntax patterns (~10ms)
   - `is_structured_data`: JSON/CSV/XML/YAML detection (~5ms)
3. ✅ **Integrate into routing** - executor calls `_sniffer.sniff()` before `get_entry_node()`
4. ✅ **Handle sniff failures gracefully** - logged as warnings, processing continues
5. ✅ **Tests** - `tests/unit/pipeline/test_sniff.py` (937+ lines)

### Phase 3: Parser Metadata Emission ✅ COMPLETE

**Status:** Implemented 2026-01-28

1. ✅ **Define `parsed.*` field conventions** - Added `ParsedMetadata` TypedDict in `parser.py`
2. ✅ **Update parser plugins** to emit metadata during parsing:
   - TextParserPlugin: emits `detected_language`, `approx_token_count`, `line_count`, `has_code_blocks`
   - UnstructuredParserPlugin: emits `page_count`, `has_tables`, `has_images`, `element_types`, `approx_token_count`
3. ✅ **Enable mid-pipeline routing** - executor wires metadata to `file_ref.metadata["parsed"]` via `_enrich_parsed_metadata()`
4. ✅ **Document emitted fields** - See `ParsedMetadata` schema and routing examples below

#### Implementation Details

**Metadata Flow:**
1. Parser computes metadata during `parse_bytes()`
2. Returns metadata in `ParserOutput.metadata`
3. Executor copies recognized fields to `file_ref.metadata["parsed"]` via `_enrich_parsed_metadata()`
4. Subsequent routing uses `metadata.parsed.*` predicates

**Emitted Fields by Parser:**

| Field | TextParser | UnstructuredParser | Description |
|-------|------------|-------------------|-------------|
| `detected_language` | ✓ | - | ISO 639-1 code via langdetect (optional dependency) |
| `approx_token_count` | ✓ | ✓ | Word count approximation |
| `line_count` | ✓ | - | Number of text lines |
| `has_code_blocks` | ✓ | - | Markdown code fences detected via regex |
| `page_count` | - | ✓ | Maximum page number found in elements |
| `has_tables` | - | ✓ | True if any element has category="Table" |
| `has_images` | - | ✓ | True if any element has category="Image" |
| `element_types` | - | ✓ | Sorted list of unique element categories |

**Mid-Pipeline Routing Examples:**

```yaml
# Route Chinese documents to multilingual embedder
edges:
  - from_node: parser
    to_node: multilingual_embedder
    when:
      metadata.parsed.detected_language: zh

# Route large documents to semantic chunker
edges:
  - from_node: parser
    to_node: semantic_chunker
    when:
      metadata.parsed.approx_token_count: ">10000"

# Route documents with tables to table-aware chunker
edges:
  - from_node: parser
    to_node: table_chunker
    when:
      metadata.parsed.has_tables: true

# Route code-heavy documents to code chunker
edges:
  - from_node: parser
    to_node: code_chunker
    when:
      metadata.parsed.has_code_blocks: true
```

**Testing:**
- Unit tests: `tests/unit/plugins/test_parser_plugins.py` (14 tests for metadata emission)
- Integration tests: `tests/integration/pipeline/test_parsed_metadata_routing.py` (7 tests for routing)

**Dependencies:**
- `langdetect` added as optional dependency in `pyproject.toml` for language detection

### Phase 4: Route Preview UI ✅ COMPLETE

1. **Add file upload/select** in DAG editor
2. **Create preview endpoint** that runs sniff + evaluates predicates
3. **Display results** with matched/not-matched visualization
4. **Show final path** through DAG

### Phase 5: Dynamic Field Discovery (Plugin Contract)

**Problem:** The edge predicate editor currently shows all possible metadata fields, but:
- Not all parsers emit all fields (TextParser vs UnstructuredParser emit different fields)
- Future plugins may emit custom fields
- Showing unavailable fields creates a confusing UX

**Solution:** Parsers declare which fields they emit. UI dynamically shows available fields based on context.

#### Backend Changes

1. **Add `EMITTED_FIELDS` to parser plugin base class:**

```python
# packages/shared/plugins/types/parser.py
class ParserPlugin(SemanticPlugin, ABC):
    PLUGIN_TYPE = "parser"

    # Parsers declare which parsed.* fields they emit
    EMITTED_FIELDS: ClassVar[list[str]] = []

    @abstractmethod
    async def parse_bytes(self, ...) -> ParserOutput:
        ...
```

2. **Update existing parsers to declare their fields:**

```python
# packages/shared/plugins/builtins/text_parser.py
class TextParserPlugin(ParserPlugin):
    EMITTED_FIELDS = [
        "detected_language",
        "approx_token_count",
        "line_count",
        "has_code_blocks",
    ]

# packages/shared/plugins/builtins/unstructured_parser.py
class UnstructuredParserPlugin(ParserPlugin):
    EMITTED_FIELDS = [
        "page_count",
        "has_tables",
        "has_images",
        "element_types",
        "approx_token_count",
    ]
```

3. **Add endpoint to query available fields for an edge:**

```
POST /api/v2/pipeline/available-predicate-fields
Body: { dag: PipelineDAG, from_node: string }

Response: {
  fields: [
    { value: "metadata.source.mime_type", label: "MIME Type", category: "source" },
    { value: "metadata.detected.is_scanned_pdf", label: "Is Scanned PDF", category: "detected" },
    { value: "metadata.parsed.has_tables", label: "Has Tables", category: "parsed" },
    ...
  ]
}
```

**Logic:**
- Always include `source.*` fields (from connector)
- Always include `detected.*` fields (from sniff step)
- For edges from `_source`: No `parsed.*` fields (parser hasn't run yet)
- For edges from parser nodes: Include `parsed.*` fields from that parser's `EMITTED_FIELDS`

4. **Add field discovery to plugin registry:**

```python
# packages/shared/plugins/registry.py
def get_parser_emitted_fields(plugin_id: str) -> list[str]:
    """Get the parsed.* fields emitted by a parser plugin."""
    plugin_cls = self.find_by_id(plugin_id)
    if hasattr(plugin_cls, 'EMITTED_FIELDS'):
        return plugin_cls.EMITTED_FIELDS
    return []
```

#### Frontend Changes

1. **Update EdgePredicateEditor to fetch available fields:**

```typescript
// Fetch fields when edge is selected
const { data: availableFields, isLoading } = useQuery({
  queryKey: ['predicate-fields', edge.from_node, dagHash],
  queryFn: () => pipelineApi.getAvailablePredicateFields(dag, edge.from_node),
});
```

2. **Show loading state while fetching**

3. **Fall back to all fields if fetch fails** (graceful degradation)

4. **Add "Custom field" option** for advanced users to type arbitrary paths

#### Plugin Documentation

Update plugin development docs to include:

```markdown
## Parser Plugins

### Declaring Emitted Fields

Parser plugins should declare which `parsed.*` metadata fields they emit.
This enables the UI to show relevant routing options.

class MyCustomParser(ParserPlugin):
    PLUGIN_ID = "my-custom-parser"
    EMITTED_FIELDS = [
        "custom_field_1",
        "custom_field_2",
    ]

    async def parse_bytes(self, content, file_ref, config):
        # ... parsing logic ...
        return ParserOutput(
            content=text,
            metadata={
                "custom_field_1": value1,
                "custom_field_2": value2,
            }
        )

Fields in EMITTED_FIELDS should match the keys returned in metadata.
```

#### Migration

- Existing parsers updated to declare `EMITTED_FIELDS`
- No breaking changes - field declaration is additive
- Plugins without `EMITTED_FIELDS` show all fields (backward compatible)

#### Testing

- Unit test: Parser plugins have `EMITTED_FIELDS` matching actual emitted keys
- Integration test: Endpoint returns correct fields based on DAG structure
- Frontend test: EdgePredicateEditor shows dynamic fields

## Extensibility

The metadata system is designed for future extension at both layers:

### Adding Pre-Routing Sniff Fields

Only add to `detected.*` if the field is needed for **parser selection**:

1. Add detector function to `sniff.py`
2. Write to `metadata.detected.{new_field}`
3. Document field in sniff capabilities table
4. No migration needed—existing DAGs unaffected

**Criteria for adding to sniff:** Does this field determine which parser to use? If yes, add to sniff. If it's discovered during parsing, it belongs in `parsed.*`.

### Potential Future Sniff Fields

| Field | Use Case | Why Pre-Routing |
|-------|----------|-----------------|
| `detected.is_encrypted` | Route to decrypt handler | Can't parse without decryption |
| `detected.encoding` | Route to encoding converter | Parser needs correct encoding |
| `detected.dominant_script` | Route to script-specific parser | CJK vs Latin handling |

### Adding Parser-Emitted Fields

Parsers can emit any `parsed.*` fields they discover:

1. Update parser plugin to write fields during parsing
2. Document fields in parser plugin documentation
3. Fields become available for mid-pipeline routing

**Example:** A new "invoice parser" might emit `parsed.is_invoice`, `parsed.vendor_name`, `parsed.total_amount`.

### Plugin-Extensible Metadata (Future)

If third-party plugins need custom metadata:
- Sniff plugins write to `metadata.detected.{plugin_name}.*`
- Parser plugins write to `metadata.parsed.{plugin_name}.*`
- Core fields remain unprefixed within their namespace

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

3. **Selective sniffing:** Run all detectors always, or only those referenced in DAG predicates? Recommendation: Run all (cost is ~45ms, negligible), keeps behavior predictable.

4. **Parser metadata schema:** Should parsers declare which `parsed.*` fields they emit? Would enable UI to show available fields for mid-pipeline routing.

5. **Mid-pipeline routing validation:** Should validation check that `parsed.*` predicates only appear on edges from nodes that emit those fields?

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
