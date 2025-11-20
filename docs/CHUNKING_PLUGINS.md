# Chunking Plugins (Preview)

This project can load external chunking strategies at runtime so 3rd parties can extend Semantik without forking the core.

## Quick contract

1) Publish a Python package that exposes an entry point:

```toml
[project.entry-points."semantik.chunking_strategies"]
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

3) Install the plugin package in the same environment as Semantik. On startup, the loader will:
   - register the class in the chunking factory/registry,
   - add metadata/defaults so it appears in listing helpers,
   - keep core strategies unchanged.

## Required visual example

To keep the chunking strategy guide consistent, plugins **must** provide a visual preview:

```python
METADATA = {
  ...,
  "visual_example": {
    "url": "https://cdn.example.com/my-fancy-chunker.png",  # HTTPS only
    "caption": "How My Fancy Chunking splits markdown"      # optional
  }
}
```

Plugins without a valid `visual_example.url` are skipped at load time. Keep images lightweight and hosted over HTTPS.

## Runtime behavior

- Loader: `packages/shared/chunking/plugin_loader.py`
- Entry point group: `semantik.chunking_strategies`
- Enable/disable via env: `SEMANTIK_ENABLE_PLUGINS` (`true` by default; set to `false`/`0` to skip loading).
- Loaded in both web startup (`packages/webui/startup_tasks.py`) and Celery worker import path (`packages/webui/chunking_tasks.py`).

## API/UI visibility

- Core v2 API schemas are enum-backed; plugin strategies are discoverable via internal helpers (e.g., `strategy_registry.list_strategy_metadata()`) but are not yet exposed through typed v2 endpoints.
- For now, call server-side services with the string `strategy` ID (`"my_fancy"`) in contexts that accept free-form identifiers.
  A future API revision can add an untyped endpoint that lists and accepts plugin strategies.

## Config parameters

- Plugin-specific options should live under `config.metadata` (or `custom_attributes`) to avoid schema changes.
- `METADATA["factory_defaults"]` seeds the internal factory defaults; `manager_defaults` feeds UI defaults.

## Safety

- Plugins run in-process and are trusted code. Do not install untrusted plugins without review.
