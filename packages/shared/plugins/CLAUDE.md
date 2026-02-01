<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding new plugin types or protocols
     - Changing plugin lifecycle or registration
     - Modifying state file format or sync behavior
     - Updating required class variables or methods
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>Plugin System</name>
  <purpose>Unified plugin discovery, loading, registration, and state management</purpose>
  <location>packages/shared/plugins/</location>
</component>

<architecture>
  <flow>Entry points (semantik.plugins) → Loader → Registry → State File → VecPipe reads</flow>
  <key-principle>Protocol-based validation allows both ABC and duck-typed plugins</key-principle>
</architecture>

<plugin-types>
  7 types with distinct protocols and required methods:

  | Type | Required Methods | Notes |
  |------|------------------|-------|
  | embedding | embed_texts, get_definition, supports_model | GPU memory managed by VecPipe |
  | chunking | chunk, validate_content, estimate_chunks | Strategy pattern |
  | connector | authenticate, load_documents, get_config_fields | Yields IngestedDocument |
  | reranker | rerank, get_capabilities | Cross-encoder scoring |
  | extractor | extract, supported_extractions | Metadata/entity extraction |
  | parser | parse_file, parse_bytes, supported_extensions | SYNC methods (Celery requirement) |
  | sparse_indexer | encode_documents, encode_query, remove_documents | BM25 or SPLADE |
</plugin-types>

<lifecycle>
  1. Discovery: Entry points (semantik.plugins) scanned, or built-ins registered
  2. Loading: Protocol validation via _satisfies_protocol()
  3. Registration: Thread-safe PluginRegistry stores PluginRecord
  4. State sync: WebUI writes /data/plugin_state.json, VecPipe reads
  5. Activation: Type-specific factory registration (e.g., EmbeddingProviderFactory)
</lifecycle>

<key-files>
  <file path="base.py">SemanticPlugin ABC with PLUGIN_TYPE, PLUGIN_ID, PLUGIN_VERSION</file>
  <file path="loader.py">load_plugins() - idempotent, type-filtered loading</file>
  <file path="registry.py">Thread-safe PluginRegistry singleton (plugin_registry)</file>
  <file path="protocols.py">Protocol classes for structural typing validation</file>
  <file path="state.py">Plugin state file I/O for cross-service config</file>
  <file path="manifest.py">PluginManifest dataclass for discovery/UI</file>
  <file path="discovery.py">Agent-facing discovery APIs (find_plugins_for_input)</file>
  <file path="types/*.py">Type-specific plugin base classes</file>
  <file path="builtins/*.py">Built-in implementations (bm25, splade, text_parser, etc.)</file>
</key-files>

<creating-plugin>
  Required class variables:
    PLUGIN_TYPE: ClassVar[str]       # e.g., "connector"
    PLUGIN_ID: ClassVar[str]         # Globally unique across ALL types
    PLUGIN_VERSION: ClassVar[str]    # Semver, default "0.0.0"

  Required methods:
    @classmethod
    def get_manifest(cls) -> PluginManifest: ...

  Optional methods:
    @classmethod
    def get_config_schema(cls) -> dict | None: ...  # JSON Schema
    @classmethod
    async def health_check(cls, config: dict | None) -> bool: ...
    async def initialize(self, config: dict | None) -> None: ...
    async def cleanup(self) -> None: ...

  Entry point registration (pyproject.toml):
    [project.entry-points."semantik.plugins"]
    my_connector = "my_package.connector:MyConnectorPlugin"
</creating-plugin>

<state-file-sync>
  WebUI writes plugin state to /data/plugin_state.json:
  {
    "disabled_ids": ["plugin-id-1"],
    "configs": {
      "plugin-id": {"key": "value"}
    }
  }

  VecPipe reads this file on startup and when creating providers.
  State sync is ONE-WAY: WebUI writes, VecPipe reads.
</state-file-sync>

<gotchas>
  <gotcha>Plugin IDs must be globally unique across ALL types (not just within type)</gotcha>
  <gotcha>Parser methods are SYNC (not async) for billiard.Pool compatibility in Celery</gotcha>
  <gotcha>External chunking plugins require METADATA.visual_example with https:// URL</gotcha>
  <gotcha>Sparse indexers require SPARSE_TYPE in ('bm25', 'splade')</gotcha>
  <gotcha>load_plugins() is idempotent - safe to call multiple times per type</gotcha>
  <gotcha>Use PROTOCOL_BY_TYPE dict to look up protocol for validation</gotcha>
</gotchas>

<env-flags>
  SEMANTIK_ENABLE_PLUGINS=true          # Global enable/disable
  SEMANTIK_ENABLE_EMBEDDING_PLUGINS=true  # Per-type flags
  SEMANTIK_ENABLE_CHUNKING_PLUGINS=true
  SEMANTIK_ENABLE_CONNECTOR_PLUGINS=true
  # ... etc for each type
</env-flags>

<testing>
  <command>uv run pytest tests/unit/plugins/ -v</command>
  <fixtures>
    - clean_registry: Reset registry between tests
    - dummy_plugin_class: Mock plugin for testing
  </fixtures>
</testing>
