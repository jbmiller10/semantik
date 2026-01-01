# Chunking Plugins

Load external chunking strategies at runtime without forking core.

## Quick Start

1) Publish a Python package that exposes an entry point:

```toml
[project.entry-points."semantik.plugins"]
my_fancy = "my_plugin.module:MyFancyChunker"
```

2) Implement the strategy class (domain-style):

```python
from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.value_objects import ChunkConfig
from packages.shared.chunking.domain.entities import Chunk

class MyFancyChunker(ChunkingStrategy):
    INTERNAL_NAME = "my_fancy"
    API_ID = "my_fancy"  # shown to clients
    METADATA = {
        "display_name": "My Fancy Chunking",
        "description": "Uses signal X to cut text",
        "best_for": ["pdf", "md"],
        "pros": ["Great for reports"],
        "cons": ["Slower on short text"],
        "performance_characteristics": {"speed": "moderate", "quality": "high"},
        "manager_defaults": {"chunk_size": 600, "chunk_overlap": 80},
        "factory_defaults": {"chunk_size": 600, "chunk_overlap": 80},
        "aliases": ["fancy"],
    }

    def __init__(self) -> None:
        super().__init__(name=self.INTERNAL_NAME)

    def chunk(self, content: str, config: ChunkConfig, progress_callback=None) -> list[Chunk]:
        # Implement chunking here; return list[Chunk]
        ...
```

3) Install the plugin package in the same environment as Semantik. On startup, the loader registers your class in the factory and adds metadata.

## Visual Example (Required)

Plugins **must** provide a visual preview:

```python
METADATA = {
  ...,
  "visual_example": {
    "url": "https://cdn.example.com/my-fancy-chunker.png",  # HTTPS only
    "caption": "How My Fancy Chunking splits markdown"      # optional
  }
}
```

Plugins without valid `visual_example.url` are skipped. Keep images lightweight, HTTPS only.

## Runtime

**Loader**: `packages/shared/plugins/loader.py`
**Entry point**: `semantik.plugins`
**Toggle**: `SEMANTIK_ENABLE_CHUNKING_PLUGINS` (default `true`, gated by `SEMANTIK_ENABLE_PLUGINS`)
**Loaded in**: webui startup + Celery workers

## API Visibility

Plugins aren't in v2 typed enums yet. Use string strategy ID (`"my_fancy"`) in server-side calls. Future API revision will expose plugins properly.

## Config

Put plugin-specific options under `config.metadata` to avoid schema changes.

## Safety

Plugins run in-process. Review before installing untrusted code.
