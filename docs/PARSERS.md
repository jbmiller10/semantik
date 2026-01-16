# Parser System

Semantik uses a unified parser system to extract text from documents. The `parse_content()` function provides a single entry point with automatic format detection and fallback behavior.

## Quick Start

```python
from shared.text_processing.parsers import parse_content

# Parse bytes with metadata
result = parse_content(
    pdf_bytes,
    filename="guide.pdf",
    file_extension=".pdf",
    metadata={"source_type": "git", "source_path": "docs/guide.pdf"},
)
print(result.text)      # Extracted text
print(result.metadata)  # Merged metadata

# Parse pre-decoded text (string input)
result = parse_content(
    "# README\n\nContent here",
    filename="README.md",
    file_extension=".md",
)
```

## `parse_content()` API Reference

```python
def parse_content(
    content: bytes | str,
    *,
    filename: str | None = None,
    file_extension: str | None = None,
    mime_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_elements: bool = False,
    parser_overrides: dict[str, str] | None = None,
    parser_configs: dict[str, dict[str, Any]] | None = None,
) -> ParseResult:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | `bytes \| str` | Document content. Strings are treated as pre-decoded text. |
| `filename` | `str \| None` | Filename hint for format detection. |
| `file_extension` | `str \| None` | Extension hint (e.g., `.pdf`). Primary selector for parser. |
| `mime_type` | `str \| None` | MIME type hint (used by UnstructuredParser). |
| `metadata` | `dict[str, Any] \| None` | Metadata to merge into result. |
| `include_elements` | `bool` | If `True`, populates `ParseResult.elements`. Default: `False`. |
| `parser_overrides` | `dict[str, str] \| None` | Extension-to-parser overrides (e.g., `{".html": "text"}`). |
| `parser_configs` | `dict[str, dict[str, Any]] \| None` | Per-parser configuration. |

### Return Type: `ParseResult`

```python
@dataclass(frozen=True)
class ParseResult:
    text: str                        # Concatenated text content
    elements: list[ParsedElement]    # Per-element metadata (if include_elements=True)
    metadata: dict[str, Any]         # Document-level metadata
```

### Exceptions

| Exception | When Raised |
|-----------|-------------|
| `UnsupportedFormatError` | No parser can handle the content. |
| `ExtractionFailedError` | Parser failed (corrupt file, missing dependency, etc.). |
| `ParserConfigError` | Invalid configuration options passed. |

## Parser Selection Rules

The parser selection algorithm uses file extension to choose candidates:

### Text-First Extensions

For known text/code formats, `TextParser` is tried first:

`.txt`, `.text`, `.md`, `.markdown`, `.mdown`, `.mkd`, `.mdx`, `.rst`, `.adoc`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.go`, `.rs`, `.rb`, `.php`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.sh`, `.bash`, `.zsh`, `.sql`, `.graphql`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.xml`, `.csv`, `.css`, `.scss`, `.less`

### Unstructured-First Extensions

For binary/structured document formats, `UnstructuredParser` is tried first:

`.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.eml`, `.msg`, `.html`, `.htm`

### Unknown/No Extension

If the extension is unknown or absent, `UnstructuredParser` is tried first (binary-first assumption).

### Fallback Behavior

When a parser raises `UnsupportedFormatError`, the next candidate is tried. Other exceptions (`ExtractionFailedError`) do not trigger fallback.

```
Extension: .pdf
Candidates: [unstructured, text]
1. Try UnstructuredParser → succeeds → return result
   OR raises UnsupportedFormatError → try next
2. Try TextParser → succeeds or fails
```

## Parser Configuration

Use `parser_configs` to pass options to specific parsers:

```python
result = parse_content(
    html_bytes,
    file_extension=".html",
    parser_configs={
        "unstructured": {"strategy": "fast"},
        "text": {"encoding": "latin-1"},
    },
)
```

### TextParser Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `encoding` | `str` | `"utf-8"` | Text encoding for decoding bytes. |
| `errors` | `str` | `"replace"` | Encoding error handling: `"replace"`, `"ignore"`, `"strict"`. |

**Binary detection:** TextParser rejects content containing NUL bytes or with >30% non-printable characters.

### UnstructuredParser Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategy` | `str` | `"auto"` | Parsing strategy: `"auto"`, `"fast"`, `"hi_res"`, `"ocr_only"`. |
| `include_page_breaks` | `bool` | `True` | Track page numbers in metadata. |
| `infer_table_structure` | `bool` | `True` | Preserve table structure during extraction. |

## Parser Overrides

Force a specific parser for certain extensions:

```python
# Force text parser for HTML (bypass unstructured)
result = parse_content(
    html_bytes,
    file_extension=".html",
    parser_overrides={".html": "text"},
)
```

## Metadata Contract

### Parser-Set Keys

All parsers set these keys in `ParseResult.metadata`:

| Key | Description |
|-----|-------------|
| `filename` | Document filename (from parameter or `"document"` default). |
| `file_extension` | Normalized extension (lowercase, with dot). |
| `file_type` | Extension without leading dot. |
| `mime_type` | MIME type (if provided or detected). |
| `parser` | Parser name that produced the result (`"text"` or `"unstructured"`). |

### Caller-Set Keys (Standardized)

Connectors should pass these in the `metadata` parameter:

| Key | Description |
|-----|-------------|
| `source_type` | Connector type (e.g., `"git"`, `"directory"`, `"imap"`). |
| `source_path` | Original path/identifier within the source. |
| `local_file_path` | Local filesystem path (for file-based sources). |

### UnstructuredParser Element Metadata

When `include_elements=True`, each `ParsedElement` may include:

| Key | Description |
|-----|-------------|
| `page_number` | Page number (for paginated documents). |
| `element_type` | Unstructured element category (e.g., `"Title"`, `"NarrativeText"`). |

## `include_elements` Behavior

By default (`include_elements=False`), `ParseResult.elements` is empty and only `text` is populated. This is more efficient for most use cases.

When `include_elements=True`:
- TextParser: Returns a single element containing the full text.
- UnstructuredParser: Returns one element per parsed document element with per-element metadata.

```python
result = parse_content(pdf_bytes, file_extension=".pdf", include_elements=True)

for element in result.elements:
    print(f"Page {element.metadata.get('page_number')}: {element.text[:50]}...")
```

## Built-in Parsers Reference

### TextParser

Lightweight parser for text/code files. No external dependencies.

**Supported Extensions:**
`.txt`, `.text`, `.md`, `.markdown`, `.mdown`, `.mkd`, `.mdx`, `.rst`, `.adoc`, `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.go`, `.rs`, `.rb`, `.php`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.sh`, `.bash`, `.zsh`, `.sql`, `.graphql`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.xml`, `.csv`, `.css`, `.scss`, `.less`

**Config Options:**
- `encoding` (default: `"utf-8"`)
- `errors` (default: `"replace"`)

### UnstructuredParser

Full-featured parser using the [unstructured](https://github.com/Unstructured-IO/unstructured) library. Handles complex document formats.

**Supported Extensions:**
`.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.html`, `.htm`, `.eml`, `.msg`, `.txt`, `.md`, `.rst`

**Supported MIME Types:**
`application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `application/msword`, `application/vnd.openxmlformats-officedocument.presentationml.presentation`, `text/html`, `message/rfc822`, `text/plain`, `text/markdown`

**Config Options:**
- `strategy` (default: `"auto"`): `"auto"`, `"fast"`, `"hi_res"`, `"ocr_only"`
- `include_page_breaks` (default: `True`)
- `infer_table_structure` (default: `True`)

## Adding a New Parser

1. Create a parser class extending `BaseParser`:

```python
from shared.text_processing.parsers.base import BaseParser, ParseResult, ParsedElement
from shared.text_processing.parsers.exceptions import UnsupportedFormatError, ExtractionFailedError

class CustomParser(BaseParser):
    @classmethod
    def supported_extensions(cls) -> frozenset[str]:
        return frozenset({".custom", ".cst"})

    @classmethod
    def get_config_options(cls) -> list[dict[str, Any]]:
        return [
            {"name": "option1", "type": "text", "label": "Option 1", "default": "value"},
        ]

    def parse_file(self, file_path: str, metadata: dict | None = None, *, include_elements: bool = False) -> ParseResult:
        # Read file and delegate to parse_bytes
        content = Path(file_path).read_bytes()
        return self.parse_bytes(content, filename=Path(file_path).name, metadata=metadata, include_elements=include_elements)

    def parse_bytes(self, content: bytes, *, filename: str | None = None, file_extension: str | None = None, mime_type: str | None = None, metadata: dict | None = None, include_elements: bool = False) -> ParseResult:
        # Implement extraction logic
        try:
            text = self._extract(content)
        except SomeLibraryError as e:
            raise ExtractionFailedError(f"Failed to parse: {e}", cause=e)

        return ParseResult(
            text=text,
            elements=[ParsedElement(text=text, metadata={})] if include_elements else [],
            metadata={"parser": "custom", "filename": filename, **(metadata or {})},
        )
```

2. Register in `registry.py`:

```python
# In packages/shared/text_processing/parsers/registry.py

def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from .text import TextParser
    from .unstructured import UnstructuredParser
    from .custom import CustomParser  # Add import

    register_parser("text", TextParser)
    register_parser("unstructured", UnstructuredParser)
    register_parser("custom", CustomParser)  # Register
    _REGISTERED = True
```

3. Update `DEFAULT_PARSER_MAP` if needed:

```python
DEFAULT_PARSER_MAP: dict[str, str] = {
    # ...existing entries...
    ".custom": "custom",
    ".cst": "custom",
}
```

## Registry Functions

For advanced use cases, the parser registry provides:

```python
from shared.text_processing.parsers import (
    list_parsers,                    # List all registered parser names
    list_parsers_for_extension,      # List parsers supporting an extension
    get_parser,                      # Get configured parser instance
    parser_candidates_for_extension, # Get priority-ordered candidates
)

# List all parsers
>>> list_parsers()
['text', 'unstructured']

# Find parsers for .html
>>> list_parsers_for_extension(".html")
['unstructured']

# Get candidates in priority order
>>> parser_candidates_for_extension(".html")
['unstructured', 'text']

>>> parser_candidates_for_extension(".py")
['text', 'unstructured']
```
