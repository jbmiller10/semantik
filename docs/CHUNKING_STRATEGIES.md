# Chunking Strategies

Semantik ships with several built-in chunking strategies and supports plugins that add new ones at runtime. Strategy IDs are strings; treat the list as extensible.

To discover what is available in your environment (including plugins), call:
`GET /api/v2/chunking/strategies`.

## Built-in Strategies (Core)

These are the core strategies exposed by the API today. Availability may grow as the platform evolves.

- `fixed_size` (aka character): Fast, predictable chunk sizes; good for simple text and structured data.
- `recursive`: Splits by a hierarchy of separators; strong general-purpose default.
- `markdown`: Markdown-aware splitting for headings/sections.
- `semantic`: Embedding-driven boundaries for better semantic coherence (slower, higher compute).
- `hierarchical`: Multi-level chunking (large-to-small) for long documents.
- `hybrid`: Combines primary + fallback strategies (e.g., recursive + character).

## Legacy / Compatibility Strategies

These exist for backward compatibility and may map to modern implementations:

- `sliding_window`
- `document_structure`

## Plugins & Extensibility

Chunking strategies can be extended via the unified plugin system (`semantik.plugins` entry points). Plugins register their own IDs and metadata at startup. See `docs/CHUNKING_PLUGINS.md` for packaging and metadata requirements.

## Practical Selection Tips

- Start with `recursive` if unsure.
- Use `markdown` for markdown-heavy docs.
- Use `semantic` when search quality matters most and extra compute is acceptable.
- Use `fixed_size` for speed or highly structured inputs.
