# Parser Metadata Conventions

Parser plugins emit standardized metadata fields that enable mid-pipeline routing decisions in DAG pipelines.

## Metadata Namespace

Parser metadata is written to `FileReference.metadata["parsed"]` after parsing and can be used in routing predicates.

## Standard Fields

### ParsedMetadata Schema

All fields are optional (`NotRequired`) as parsers emit only fields they can compute.

| Field | Type | Description | Emitted By |
|-------|------|-------------|------------|
| `page_count` | int | Number of pages (PDF, DOCX) | UnstructuredParser |
| `line_count` | int | Number of text lines | TextParser |
| `has_tables` | bool | Contains table elements | UnstructuredParser |
| `has_images` | bool | Contains images | UnstructuredParser |
| `has_code_blocks` | bool | Contains markdown code fences | TextParser |
| `detected_language` | str \| None | ISO 639-1 language code (en, zh, etc.) | TextParser |
| `approx_token_count` | int | Approximate word/token count | Both |
| `element_types` | list[str] | Unique element types found | UnstructuredParser |
| `text_quality` | float | OCR confidence (0.0-1.0) | (future) |

## Parser-Specific Behavior

### TextParserPlugin

**Emits:**
- `detected_language`: Uses langdetect library (optional dependency, graceful fallback if missing)
- `approx_token_count`: Simple word count via `len(text.split())`
- `line_count`: `text.count('\n') + 1`
- `has_code_blocks`: Regex detection of triple backticks (```...```)

**Example:**
```python
from shared.plugins.builtins.text_parser import TextParserPlugin

parser = TextParserPlugin()
result = parser.parse_bytes(b"# Example\n```python\ncode\n```")

assert result.metadata["has_code_blocks"] is True
assert result.metadata["approx_token_count"] > 0
assert result.metadata["line_count"] >= 1
# detected_language may be 'en' or None if langdetect not installed
```

### UnstructuredParserPlugin

**Emits:**
- `page_count`: Maximum page number from element metadata
- `has_tables`: True if any element has `category="Table"`
- `has_images`: True if any element has `category="Image"`
- `element_types`: Sorted list of unique element categories found
- `approx_token_count`: Word count from concatenated element text

**Example:**
```python
from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

parser = UnstructuredParserPlugin()
result = parser.parse_bytes(pdf_bytes, mime_type="application/pdf")

assert "page_count" in result.metadata
assert "has_tables" in result.metadata
assert "element_types" in result.metadata  # e.g., ["Table", "Title", "Text"]
assert result.metadata["approx_token_count"] > 0
```

## Usage in Routing Predicates

Routing predicates can reference `metadata.parsed.*` fields for mid-pipeline decisions:

### Example 1: Route by Language

```yaml
# DAG edge configuration
edges:
  - from_node: parser
    to_node: multilingual_chunker
    when:
      metadata.parsed.detected_language: zh  # Chinese documents
```

### Example 2: Route by Size

```yaml
edges:
  - from_node: parser
    to_node: semantic_chunker
    when:
      metadata.parsed.approx_token_count: ">10000"  # Large documents
```

### Example 3: Route by Content Type

```yaml
edges:
  - from_node: parser
    to_node: table_aware_chunker
    when:
      metadata.parsed.has_tables: true  # Documents with tables
```

### Example 4: Route Code-Heavy Documents

```yaml
edges:
  - from_node: parser
    to_node: code_chunker
    when:
      metadata.parsed.has_code_blocks: true  # Markdown with code
```

### Fallback Behavior

Missing fields evaluate to `None`/`False`, causing routing to fall through to catch-all edges:

```yaml
edges:
  - from_node: parser
    to_node: specialized_chunker
    when:
      metadata.parsed.has_code_blocks: true  # Only if present and true

  - from_node: parser
    to_node: default_chunker  # Catch-all (no when clause)
```

## Implementation Architecture

### 1. Parser Metadata Emission

Parsers compute metadata during `parse_bytes()`:

```python
# In parser plugin
def parse_bytes(self, content: bytes, ...) -> ParserOutput:
    text = decode(content)

    # Compute metadata
    metadata = {
        "parser": "text",
        "approx_token_count": len(text.split()),
        "has_code_blocks": bool(re.search(r'```', text)),
        # ...
    }

    return ParserOutput(text=text, metadata=metadata)
```

### 2. Executor Enrichment

Executor's `_enrich_parsed_metadata()` copies recognized fields to `file_ref.metadata["parsed"]`:

```python
def _enrich_parsed_metadata(
    self,
    file_ref: FileReference,
    parse_metadata: dict[str, Any],
) -> None:
    # Recognized field names from ParsedMetadata schema
    parsed_fields = {
        "page_count", "has_tables", "has_images", "has_code_blocks",
        "detected_language", "approx_token_count", "line_count",
        "element_types", "text_quality",
    }

    if "parsed" not in file_ref.metadata:
        file_ref.metadata["parsed"] = {}

    # Copy only recognized fields
    for key, value in parse_metadata.items():
        if key in parsed_fields:
            file_ref.metadata["parsed"][key] = value
```

### 3. Routing Predicate Evaluation

Router evaluates predicates against enriched `file_ref.metadata`:

```python
# Predicate: metadata.parsed.has_code_blocks == true
has_code = file_ref.metadata.get("parsed", {}).get("has_code_blocks")
if has_code is True:
    # Route to code_chunker
```

## Adding Custom Fields

To add a new parsed metadata field:

1. **Add field to `ParsedMetadata` TypedDict** in `shared/plugins/types/parser.py`:
   ```python
   class ParsedMetadata(TypedDict, total=False):
       # ... existing fields ...
       my_custom_field: NotRequired[bool]
       """Description of custom field"""
   ```

2. **Add field name to `_enrich_parsed_metadata()`** in executor:
   ```python
   parsed_fields = {
       # ... existing fields ...
       "my_custom_field",
   }
   ```

3. **Emit field in parser's `ParserOutput.metadata`**:
   ```python
   metadata["my_custom_field"] = compute_custom_field(text)
   ```

4. **Document field behavior** in this guide

5. **Add tests** for field emission and routing

## Backward Compatibility

- Parsers that don't emit parsed metadata continue to work
- Missing fields in predicates evaluate to `None`/`False`
- Routing falls through to catch-all edges when fields are missing
- Non-standard fields are filtered out during enrichment

## Testing

**Unit Tests:**
- `tests/unit/plugins/test_parser_plugins.py::TestTextParserMetadataEmission`
- `tests/unit/plugins/test_parser_plugins.py::TestUnstructuredParserMetadataEmission`

**Integration Tests:**
- `tests/integration/pipeline/test_parsed_metadata_routing.py`

**Example Test:**
```python
def test_parser_emits_metadata():
    parser = TextParserPlugin()
    result = parser.parse_bytes(b"```python\ncode\n```")

    assert result.metadata["has_code_blocks"] is True
    assert result.metadata["approx_token_count"] > 0
```

## Performance Considerations

- **Language detection**: Optional dependency (langdetect) with graceful fallback
- **Code block detection**: Fast regex pattern matching
- **Token counting**: Simple word splitting (`len(text.split())`)
- **Element tracking**: Incremental during parse loop (no extra pass)

All metadata computation happens during parsing with minimal overhead.

## See Also

- **Design Document**: `docs/plans/2026-01-28-dag-routing-enhancements-design.md`
- **Parser Plugin Base**: `packages/shared/plugins/types/parser.py`
- **Routing System**: `packages/shared/pipeline/router.py`
- **Pre-Routing Sniff**: `packages/shared/pipeline/sniff.py`
