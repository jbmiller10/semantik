# Semantik Plugin Extensibility Roadmap
---

## Table of Contents

1. [Migration Policy](#migration-policy)
2. [Current State Analysis](#current-state-analysis)
3. [Strategic Recommendations](#strategic-recommendations)
4. [Quick Wins](#quick-wins)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Revision History](#revision-history)

---

## Migration Policy

> **No Backwards Compatibility Required.** This roadmap represents a clean-break migration. Upon completion:
>
> - All deprecated code paths will be **removed**, not just deprecated
> - Legacy plugin patterns will be **deleted** once adapters are in place
> - `connector_catalog.py` **has been removed** (replaced by `get_config_fields()` on connectors) ✅
> - Old entry point groups **have been consolidated** into `semantik.plugins` ✅
> - No shims, compatibility layers, or re-exports will be maintained
>
> **Rationale:** Maintaining backwards compatibility adds complexity and technical debt. Since this is a major version release, we take the opportunity to clean up fully.

---

## Design Philosophy

> **Primary Goal:** Enable rapid experimentation with new AI/ML techniques without modifying core code.
>
> **Target User:** The developer themselves (and power users who want to experiment with different techniques).
>
> **Key Principles:**
> 1. **Experimentation-first** - Minimize friction from "I want to try X" to "X is running"
> 2. **Extensibility as differentiator** - Others can extend without forking
> 3. **Clean contracts** - Well-documented interfaces enable future AI-assisted plugin authoring
> 4. **Pragmatic scope** - Build infrastructure for realistic use cases, not hypothetical ecosystems
>
> **What this is NOT:**
> - A platform for a thriving third-party plugin marketplace
> - Enterprise-grade plugin sandboxing for untrusted code
> - A framework where plugins are the primary development model
>
> **Expected plugin authorship:** Primarily the core developer, with occasional community contributions.

---

## Current State Analysis

### What's Working Well

| Component | Pattern | Strengths |
|-----------|---------|-----------|
| **Embedding Providers** | Entry points (`semantik.embedding_providers`) | Clean abstract base, auto-detection via `supports_model()`, metadata registry |
| **Chunking Strategies** | Entry points (`semantik.chunking_strategies`) | Well-defined contract, METADATA for UI, factory pattern |
| **Connectors** | Manual factory registration | Lazy loading, config validation |
| **Documentation** | `docs/EMBEDDING_PLUGINS.md`, `docs/CHUNKING_PLUGINS.md`, `docs/CONNECTORS.md` | Clear guides for plugin authors |

### Current Plugin Contracts

#### Embedding Plugins (`packages/shared/embedding/plugin_base.py`)

```python
class BaseEmbeddingPlugin(BaseEmbeddingService):
    # Required class attributes
    INTERNAL_NAME: ClassVar[str]
    API_ID: ClassVar[str]
    PROVIDER_TYPE: ClassVar[str]  # "local", "remote", "hybrid"
    METADATA: ClassVar[dict]

    # Required methods
    async def initialize(model_name: str, **kwargs) -> None
    async def embed_texts(texts: list[str], batch_size: int, mode: EmbeddingMode = None) -> list[list[float]]
    async def embed_single(text: str, mode: EmbeddingMode = None) -> list[float]
    def get_dimension() -> int
    def get_model_info() -> dict
    async def cleanup() -> None

    # Required class methods
    @classmethod def get_definition() -> EmbeddingProviderDefinition
    @classmethod def supports_model(model_name: str) -> bool
```

#### Chunking Plugins (`packages/shared/chunking/domain/services/chunking_strategies/base.py`)

```python
class ChunkingStrategy(ABC):
    # Required class attributes
    INTERNAL_NAME: str
    API_ID: str | None
    METADATA: dict  # Must include visual_example.url (HTTPS) for external plugins

    # Required methods
    def chunk(content: str, config: ChunkConfig, progress_callback=None) -> list[Chunk]
    def validate_content(content: str) -> tuple[bool, str | None]
    def estimate_chunks(content_length: int, config: ChunkConfig) -> int
```

#### Connector Plugins (`packages/shared/connectors/base.py`)

```python
class BaseConnector(ABC):
    def __init__(self, config: dict[str, Any])

    # Required methods
    async def authenticate() -> bool
    async def load_documents(source_id=None) -> AsyncIterator[IngestedDocument]
    def validate_config() -> None  # Optional override
```

### Current Gaps

| Gap | Impact | Priority | Status |
|-----|--------|----------|--------|
| ~~**No connector entry points**~~ | ~~Connectors require code changes to add~~ | ~~High~~ | ✅ **Fixed in Phase 1** |
| **No plugin versioning** | Cannot manage compatibility or upgrades | High | Planned |
| **🔴 No sandboxing** | Plugins run with FULL process privileges (secrets, network, filesystem) | **Critical** | Must address early |
| ~~🔴 Chunking loader not idempotent~~ | ~~Can re-register plugins~~ | ~~Critical~~ | ✅ **Fixed in Phase 0** |
| ~~🟠 Registry not thread-safe~~ | ~~Race conditions in plugin registration~~ | ~~High~~ | ✅ **Fixed in Phase 0** |
| ~~🟠 Connectors lack metadata~~ | ~~No `PLUGIN_ID`, `METADATA` on connector classes~~ | ~~High~~ | ✅ **Fixed in Phase 0** |
| ~~**🟠 Three incompatible plugin patterns**~~ | ~~Different contracts for embedding/chunking/connector~~ | ~~**High**~~ | ✅ **Fixed in Phase 1** (adapter layer) |
| **"First registered wins" conflicts** | Unpredictable behavior with multiple providers for same model | Medium | Planned |
| ~~**No plugin marketplace/registry**~~ | ~~Manual discovery and installation~~ | ~~Medium~~ | ✅ **Fixed in Phase 4** |
| ~~**No plugin CLI tools**~~ | ~~Higher barrier for plugin developers~~ | ~~Medium~~ | ✅ **Fixed in Phase 1.4** |
| **No centralized hook/event system** | Can't extend application behavior | Medium | Phase 3.3 |
| **Visual example requirement** | Chunking plugins must provide HTTPS image URL | Low | Quick win |
| **No hot-reload** | Requires restart to load new plugins | Low | Future |
| **Limited testing utilities** | Plugin authors must build their own test harness | Medium | Phase 3.2 |

> **⚠️ CRITICAL SECURITY FINDING:** Current plugins have access to all environment variables (including `JWT_SECRET_KEY`, `POSTGRES_PASSWORD`), unrestricted network access, full container filesystem, and GPU resources. A malicious plugin could exfiltrate all data and credentials. Phase 5 (Security) priority elevated.

### Current Loading Sequence

1. `webui/main.py` imports FastAPI app
2. `shared.embedding.providers/__init__.py` auto-registers built-in providers
3. `webui.services.factory` calls `load_embedding_plugins()` (entry points)
4. `webui.services.factory` calls `load_chunking_plugins()` (entry points)
5. `ConnectorFactory` lazy-loads connectors on first use (manual registry)

**Built-ins registered FIRST**, then external plugins.

> **⚠️ BUG FOUND:** Embedding loader is idempotent (has `_plugins_loaded` flag), but chunking loader is NOT - it can be called multiple times causing re-registration. Additionally, plugin loading happens at 4+ different points: WebUI startup, Celery worker startup, VecPipe service startup, and ModelManager on-demand.

### Major Discovery: Reranking Already Exists!

**IMPORTANT:** The codebase analysis revealed that reranking infrastructure **already exists** in VecPipe:

| Component | Location | Status |
|-----------|----------|--------|
| `use_reranker` parameter | `search_service.py` | ✅ Exists |
| `rerank_model` parameter | `search_service.py` | ✅ Exists |
| Qwen3 reranker config | `qwen3_search_config.py` | ✅ Exists |
| Cross-encoder support | `reranker.py` in VecPipe | ✅ Exists |
| 5x candidate multiplier | Bounds 20-200 | ✅ Exists |

**Impact:** Phase 2.1 (Reranker Plugins) scope reduced by ~80%. Only need to create plugin interface wrapper around existing code.

---

## Strategic Recommendations

### 1. Unify the Plugin System

Create a common base for all plugin types:

```python
# packages/shared/plugins/base.py
from abc import ABC, abstractmethod
from typing import ClassVar
from dataclasses import dataclass

@dataclass
class PluginManifest:
    id: str
    version: str
    type: str  # "embedding", "chunking", "connector", "reranker", etc.
    display_name: str
    description: str
    author: str | None = None
    license: str | None = None
    homepage: str | None = None
    requires: list[str] = field(default_factory=list)  # ["torch>=2.0"]
    semantik_version: str | None = None  # ">=2.0.0,<3.0.0"
    capabilities: dict = field(default_factory=dict)

class SemanticPlugin(ABC):
    """Universal base for all Semantik plugins"""
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str] = "0.0.0"

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin metadata for discovery and UI"""
        ...

    @classmethod
    def get_config_schema(cls) -> dict | None:
        """Optional: JSON Schema for plugin configuration"""
        return None

    async def health_check(self) -> bool:
        """Optional: Check if plugin is healthy"""
        return True
```

**Unified entry point group:**
```toml
[project.entry-points."semantik.plugins"]
my_embedding = "my_pkg.embedding:MyEmbeddingPlugin"
my_connector = "my_pkg.connector:MyConnector"
my_reranker = "my_pkg.reranker:MyReranker"
```

**Unified loader:**
```python
# packages/shared/plugins/loader.py
def load_all_plugins() -> dict[str, list[type[SemanticPlugin]]]:
    """Load all plugins, grouped by type"""
    plugins = {"embedding": [], "chunking": [], "connector": [], ...}

    for ep in importlib.metadata.entry_points(group="semantik.plugins"):
        plugin_cls = ep.load()
        plugin_type = plugin_cls.PLUGIN_TYPE
        plugins[plugin_type].append(plugin_cls)

    return plugins
```

---

### 2. Add New Plugin Types

Expand the plugin ecosystem beyond current types:

| Plugin Type | Interface | Use Cases |
|-------------|-----------|-----------|
| **Rerankers** | `rerank(query, docs) -> ranked_docs` | Cohere, BGE-reranker, cross-encoders |
| **Extractors** | `extract(text) -> metadata` | NER, topics, sentiment, language detection |
| **Preprocessors** | `preprocess(doc) -> doc` | OCR, PDF parsing, HTML cleaning, deduplication |
| **Postprocessors** | `postprocess(results) -> results` | Summarization, highlighting, grouping |
| **Auth Providers** | `authenticate(request) -> user` | LDAP, SAML, custom OIDC |
| **Storage Backends** | Vector store interface | Pinecone, Weaviate, Milvus, pgvector |
| **Notifiers** | `notify(event) -> None` | Slack, webhooks, email, PagerDuty |
| **Schedulers** | `schedule(job) -> None` | Custom sync schedules, cron expressions |

#### Example: Reranker Plugin Interface

```python
# packages/shared/plugins/types/reranker.py
from dataclasses import dataclass

@dataclass
class RerankResult:
    index: int
    score: float
    text: str

class BaseRerankerPlugin(SemanticPlugin):
    PLUGIN_TYPE = "reranker"

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query"""
        ...

    @abstractmethod
    def get_max_documents(self) -> int:
        """Maximum documents per rerank call"""
        ...

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Maximum tokens per document"""
        ...
```

#### Example: Extractor Plugin Interface

```python
# packages/shared/plugins/types/extractor.py
@dataclass
class ExtractionResult:
    entities: list[dict]  # [{"type": "PERSON", "text": "John", "start": 0, "end": 4}]
    topics: list[str]
    language: str | None
    sentiment: float | None  # -1.0 to 1.0
    custom: dict  # Plugin-specific extractions

class BaseExtractorPlugin(SemanticPlugin):
    PLUGIN_TYPE = "extractor"

    @abstractmethod
    async def extract(self, text: str, options: dict = None) -> ExtractionResult:
        """Extract metadata from text"""
        ...

    @classmethod
    def supported_extractions(cls) -> list[str]:
        """List of extraction types this plugin supports"""
        return ["entities", "topics", "language", "sentiment"]
```

---

### 3. Plugin CLI Tool

Create `semantik-plugin` for plugin development:

```bash
# Scaffold new plugin
semantik-plugin new my-embedder --type embedding
semantik-plugin new my-connector --type connector --template api

# Validate plugin contract
semantik-plugin validate ./my-plugin
semantik-plugin validate ./my-plugin --strict  # Check optional best practices

# Test plugin
semantik-plugin test ./my-plugin
semantik-plugin test ./my-plugin --model "sentence-transformers/all-MiniLM-L6-v2"
semantik-plugin benchmark ./my-plugin --dataset sample.jsonl

# Package and publish
semantik-plugin build ./my-plugin --output dist/
semantik-plugin publish ./my-plugin --registry https://plugins.semantik.io
```

#### Scaffold Structure

```
my-embedder/
├── pyproject.toml              # Pre-configured entry points
├── src/
│   └── my_embedder/
│       ├── __init__.py
│       ├── plugin.py           # Main plugin class
│       ├── config.py           # Configuration models
│       └── models.py           # Supported models
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_contract.py        # Auto-generated contract tests
│   └── test_plugin.py          # Custom tests
├── README.md                   # Documentation template
├── CHANGELOG.md
└── semantik-plugin.json        # Plugin manifest
```

#### Contract Tests (Auto-Generated)

```python
# tests/test_contract.py
import pytest
from semantik.plugins.testing import PluginContractTest

class TestMyEmbedderContract(PluginContractTest):
    plugin_class = MyEmbedderPlugin

    # These tests run automatically:
    # - test_has_required_attributes
    # - test_get_manifest_returns_valid_manifest
    # - test_initialize_and_cleanup
    # - test_embed_single_returns_correct_dimension
    # - test_embed_texts_batch_processing
    # - test_supports_model_returns_bool
```

---

### 4. Plugin Discovery API

```yaml
# API Endpoints
GET  /api/v2/plugins                    # List all plugins
GET  /api/v2/plugins?type=embedding     # Filter by type
GET  /api/v2/plugins/{id}               # Get plugin details
GET  /api/v2/plugins/{id}/manifest      # Get full manifest
GET  /api/v2/plugins/{id}/config-schema # Get configuration schema
POST /api/v2/plugins/{id}/enable        # Enable plugin
POST /api/v2/plugins/{id}/disable       # Disable plugin
POST /api/v2/plugins/{id}/configure     # Update plugin config
GET  /api/v2/plugins/{id}/health        # Health check
```

#### Response Examples

```json
// GET /api/v2/plugins
{
  "plugins": [
    {
      "id": "dense-local",
      "type": "embedding",
      "display_name": "Dense Local Embedder",
      "version": "2.0.0",
      "enabled": true,
      "builtin": true,
      "health": "healthy"
    },
    {
      "id": "cohere-reranker",
      "type": "reranker",
      "display_name": "Cohere Reranker",
      "version": "1.2.0",
      "enabled": false,
      "builtin": false,
      "health": "unknown"
    }
  ],
  "types": ["embedding", "chunking", "connector", "reranker"],
  "total": 12
}
```

```json
// GET /api/v2/plugins/cohere-reranker/manifest
{
  "id": "cohere-reranker",
  "version": "1.2.0",
  "type": "reranker",
  "display_name": "Cohere Reranker",
  "description": "Cross-encoder reranking using Cohere's rerank API",
  "author": "Semantik Community",
  "license": "MIT",
  "homepage": "https://github.com/semantik-plugins/cohere-reranker",
  "requires": ["cohere>=4.0"],
  "semantik_version": ">=2.0.0",
  "capabilities": {
    "max_documents": 1000,
    "max_tokens_per_doc": 4096,
    "models": ["rerank-english-v2.0", "rerank-multilingual-v2.0"]
  },
  "config_schema": {
    "type": "object",
    "properties": {
      "api_key": {"type": "string", "description": "Cohere API key"},
      "model": {"type": "string", "default": "rerank-english-v2.0"}
    },
    "required": ["api_key"]
  }
}
```

---

### 5. Plugin Configuration UI

Add Plugin Management page:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Settings > Plugins                                          [Refresh]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ Embedding Providers ────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  ✓ Dense Local               built-in    v2.0.0    [Configure]   │   │
│  │    sentence-transformers, BAAI/bge-*, Qwen/*                     │   │
│  │                                                                   │   │
│  │  ✓ OpenAI Embeddings         community   v1.3.0    [Configure]   │   │
│  │    text-embedding-3-small, text-embedding-3-large                │   │
│  │                                                                   │   │
│  │  ○ Voyage AI                 community   v0.9.0    [Enable]      │   │
│  │    voyage-large-2, voyage-code-2                                 │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Rerankers ──────────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  ○ Cohere Reranker           community   v1.2.0    [Enable]      │   │
│  │    rerank-english-v2.0, rerank-multilingual-v2.0   ⚠ Needs API key│   │
│  │                                                                   │   │
│  │  ○ BGE Reranker              community   v1.0.0    [Enable]      │   │
│  │    BAAI/bge-reranker-large                                       │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Connectors ─────────────────────────────────────────────────────┐   │
│  │                                                                   │   │
│  │  ✓ Local Directory           built-in    v2.0.0                  │   │
│  │  ✓ Git Repository            built-in    v2.0.0                  │   │
│  │  ✓ IMAP Email                built-in    v2.0.0                  │   │
│  │  ○ S3 Bucket                 community   v1.1.0    [Enable]      │   │
│  │  ○ Google Drive              community   v0.8.0    [Enable]      │   │
│  │  ○ Notion                    community   v0.5.0    [Enable]      │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [+ Install Plugin]  [Browse Registry]                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 6. Plugin Registry/Marketplace

#### Option A: Curated Registry (Recommended for v1)

```yaml
# Hosted at: https://plugins.semantik.io/registry.yaml
# Or self-hosted: SEMANTIK_PLUGIN_REGISTRY_URL

registry_version: "1.0"
last_updated: "2025-12-31T00:00:00Z"

plugins:
  - id: openai-embeddings
    type: embedding
    display_name: "OpenAI Embeddings"
    description: "text-embedding-3-small/large via OpenAI API"
    author: "Semantik Team"
    repository: "https://github.com/semantik-plugins/openai-embeddings"
    pypi: "semantik-plugin-openai-embeddings"
    versions:
      - version: "1.3.0"
        semantik_version: ">=2.0.0"
        changelog: "Added text-embedding-3-large support"
      - version: "1.2.0"
        semantik_version: ">=1.5.0"
    verified: true
    downloads: 1250

  - id: cohere-reranker
    type: reranker
    display_name: "Cohere Reranker"
    repository: "https://github.com/semantik-plugins/cohere-reranker"
    pypi: "semantik-plugin-cohere-reranker"
    versions:
      - version: "1.2.0"
        semantik_version: ">=2.0.0"
    verified: true
    downloads: 890
```

#### Option B: PyPI-Based Discovery

```toml
# Plugin's pyproject.toml
[project]
classifiers = [
    "Framework :: Semantik",
    "Framework :: Semantik :: Embedding",
]
```

```python
# Discovery via PyPI API
async def discover_plugins_from_pypi():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://pypi.org/simple/",
            params={"classifiers": "Framework :: Semantik"}
        )
        # Parse and return plugin metadata
```

#### Option C: GitHub Topics Discovery

```python
# Discovery via GitHub API
async def discover_plugins_from_github():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": "topic:semantik-plugin"},
            headers={"Accept": "application/vnd.github.v3+json"}
        )
        # Parse repos, fetch pyproject.toml, extract plugin metadata
```

---

### 7. Plugin Hooks/Events System

Allow plugins to hook into application lifecycle:

```python
# packages/shared/plugins/hooks.py
from enum import Enum
from typing import Callable, Any

class HookEvent(Enum):
    # Lifecycle
    APP_STARTUP = "app.startup"
    APP_SHUTDOWN = "app.shutdown"

    # Collections
    COLLECTION_CREATED = "collection.created"
    COLLECTION_DELETED = "collection.deleted"
    COLLECTION_UPDATED = "collection.updated"

    # Documents
    DOCUMENT_INDEXED = "document.indexed"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_UPDATED = "document.updated"

    # Search
    SEARCH_STARTED = "search.started"
    SEARCH_COMPLETED = "search.completed"

    # Operations
    OPERATION_STARTED = "operation.started"
    OPERATION_COMPLETED = "operation.completed"
    OPERATION_FAILED = "operation.failed"

class PluginHookRegistry:
    _hooks: dict[HookEvent, list[Callable]] = defaultdict(list)

    @classmethod
    def register(cls, event: HookEvent, handler: Callable):
        cls._hooks[event].append(handler)

    @classmethod
    async def emit(cls, event: HookEvent, data: dict[str, Any]):
        for handler in cls._hooks[event]:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Hook handler error for {event}: {e}")

# Plugin usage
class MyAnalyticsPlugin(SemanticPlugin):
    @classmethod
    def register_hooks(cls):
        PluginHookRegistry.register(
            HookEvent.SEARCH_COMPLETED,
            cls.log_search_analytics
        )

    @staticmethod
    async def log_search_analytics(data: dict):
        # data = {"query": "...", "results": [...], "duration_ms": 123}
        await analytics_service.log(data)
```

---

### 8. Plugin Versioning & Compatibility

```python
# packages/shared/plugins/compatibility.py
from packaging import version

def check_plugin_compatibility(
    plugin_manifest: PluginManifest,
    semantik_version: str
) -> tuple[bool, str | None]:
    """Check if plugin is compatible with current Semantik version"""

    if not plugin_manifest.semantik_version:
        return True, None  # No constraint specified

    # Parse version constraint (e.g., ">=2.0.0,<3.0.0")
    try:
        specifier = SpecifierSet(plugin_manifest.semantik_version)
        if version.parse(semantik_version) in specifier:
            return True, None
        else:
            return False, f"Requires Semantik {plugin_manifest.semantik_version}, got {semantik_version}"
    except Exception as e:
        return False, f"Invalid version constraint: {e}"

def resolve_plugin_conflicts(
    plugins: list[type[SemanticPlugin]]
) -> dict[str, type[SemanticPlugin]]:
    """Resolve conflicts when multiple plugins handle same capability"""

    resolved = {}
    conflicts = defaultdict(list)

    for plugin in plugins:
        manifest = plugin.get_manifest()
        key = f"{manifest.type}:{manifest.id}"

        if key in resolved:
            conflicts[key].append(plugin)
        else:
            resolved[key] = plugin

    if conflicts:
        for key, conflicting in conflicts.items():
            logger.warning(
                f"Plugin conflict for {key}: {[p.PLUGIN_ID for p in conflicting]}. "
                f"Using first registered: {resolved[key].PLUGIN_ID}"
            )

    return resolved
```

---

### 9. Plugin Sandboxing (Advanced)

For untrusted plugins, implement isolation:

```python
# packages/shared/plugins/sandbox.py
import subprocess
import json

class PluginPermissions:
    network: bool = False
    filesystem: list[str] = []  # Allowed paths
    env_vars: list[str] = []    # Allowed env vars
    max_memory_mb: int = 512
    max_cpu_seconds: int = 30

class SandboxedPluginRunner:
    """Run plugin in isolated subprocess with restricted permissions"""

    def __init__(self, plugin_path: str, permissions: PluginPermissions):
        self.plugin_path = plugin_path
        self.permissions = permissions
        self.process: subprocess.Popen | None = None

    async def call(self, method: str, *args, **kwargs) -> Any:
        """Call plugin method in sandboxed subprocess"""
        request = {
            "method": method,
            "args": args,
            "kwargs": kwargs
        }

        # Start isolated subprocess with resource limits
        self.process = await asyncio.create_subprocess_exec(
            "python", "-m", "semantik.plugins.sandbox_worker",
            "--plugin", self.plugin_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Apply cgroups/seccomp limits (Linux)
        # ...

        stdout, stderr = await asyncio.wait_for(
            self.process.communicate(json.dumps(request).encode()),
            timeout=self.permissions.max_cpu_seconds
        )

        return json.loads(stdout)
```

**Permission levels:**
- `TRUSTED` - Built-in and verified plugins, full access
- `RESTRICTED` - Community plugins, network + limited filesystem
- `SANDBOXED` - Untrusted plugins, subprocess with cgroups/seccomp

---

### 10. Plugin Templates Gallery

Official plugin templates repository:

```
github.com/semantik/plugin-templates/
├── embedding-local/          # Local model (sentence-transformers)
│   ├── src/plugin.py
│   ├── tests/
│   └── README.md
├── embedding-api/            # API-based (OpenAI, Cohere, Voyage)
├── chunking-custom/          # Custom chunking strategy
├── connector-rest-api/       # Generic REST API connector
├── connector-database/       # SQL/NoSQL database connector
├── connector-cloud-storage/  # S3, GCS, Azure Blob
├── reranker-api/             # API-based reranker
├── reranker-local/           # Local cross-encoder
├── extractor-spacy/          # spaCy NER extraction
├── extractor-llm/            # LLM-based extraction
├── notifier-webhook/         # Webhook notifications
└── notifier-slack/           # Slack notifications
```

Each template includes:
- Working implementation with best practices
- Comprehensive test suite
- GitHub Actions CI/CD
- Documentation with examples
- Performance benchmarks

---

## Quick Wins

**Achievable in 1-2 weeks:**

1. **Add connector entry points**
   - Align connectors with embedding/chunking patterns
   - Single entry point group: `semantik.plugins`
   - Update `ConnectorFactory` to use entry point discovery

2. **Create `/api/v2/plugins` endpoint**
   - List all installed plugins with metadata
   - Filter by type, enabled status
   - Return health status

3. **Write plugin scaffold command**
   - `semantik-plugin new <name> --type <type>`
   - Generate pyproject.toml, plugin class, tests
   - Include contract test template

4. **Create 2-3 example plugins as external packages**
   - `semantik-plugin-openai-embeddings`
   - `semantik-plugin-cohere-reranker`
   - Validate developer experience, document learnings

5. **Remove visual_example requirement for chunking plugins**
   - Make optional with fallback to generic icon
   - Lower barrier to entry

---

## Implementation Roadmap

### Revised Phase Summary (Post-Review)

| Phase | Features | Priority | Estimate | Status |
|-------|----------|----------|----------|--------|
| ~~**Phase 0: Bug Fixes & Prep**~~ | ~~Fix chunking idempotency, thread-safety, connector metadata~~ | ~~P0~~ | ~~1 week~~ | ✅ **COMPLETE** |
| ~~**Phase 1: Foundation**~~ | ~~Unified plugin base, connector entry points, `/api/v2/plugins`, CLI~~ | ~~P0~~ | ~~5-6 weeks~~ | ✅ **COMPLETE** |
| **Phase 1.6: Runtime Contract** | Shared state file, config at runtime, health check with config | **P0** | 1-2 weeks | **NEW - Do First** |
| ~~**Phase 2.1: Rerankers**~~ | ~~Reranker plugin wrapper~~ | ~~P1~~ | ~~3-4 days~~ | ✅ **COMPLETE** |
| ~~**Phase 2.2: Extractors**~~ | ~~Extractor plugins (NER, topics)~~ | ~~P1~~ | ~~1-2 weeks~~ | ✅ **COMPLETE** |
| ~~**Phase 2.3: Preprocessors**~~ | ~~PDF/HTML preprocessing~~ | - | - | **DEFERRED** |
| ~~**Phase 3: Experience**~~ | ~~Plugin UI, testing fixtures~~ | ~~P1~~ | ~~2-3 weeks~~ | ✅ **COMPLETE** |
| ~~**Phase 4: Registry**~~ | ~~Slim YAML registry + browse UI~~ | ~~P2~~ | ~~1 week~~ | ✅ **COMPLETE** |
| **Phase 5: Security** | Audit logging (env filtering is cooperative only) | P1 | 1 week | Planned (simplified) |
| **Phase 6: Example Plugins** | Real, published plugins for the registry | P2 | 2-3 weeks | Planned |
| **Phase 7: In-App Installation** | Install plugins from UI with Docker persistence | P3 | 2-3 weeks | Planned |

> **Scope Decisions (2026-01-01):**
> - **Phase 1.6 added**: Runtime contract fixes needed before UI/new types work correctly
> - **Phase 2.3 deferred**: Extraction happens inside connectors; no natural plugin hook without major refactor
> - **Phase 3 simplified**: Full Plugin UI kept, but hooks system deferred, testing reduced to pytest fixtures only, templates repo cut
> - **Phase 4 simplified**: Slim YAML-based registry instead of hosted marketplace service
> - **Phase 5 reframed**: Env filtering is cooperative (plugins run in-process), full sandboxing deferred
> - **Rationale**: Build for experimentation, not hypothetical ecosystem

---

### Phase 0: Bug Fixes & Preparation (P0) ✅ COMPLETE

**Goal:** Fix critical bugs and add missing infrastructure before building on top of the plugin system.

**Status:** Completed in branch `v0.7.3/groundwork-for-unified-plugins` (commit `dc8c665`)

#### 0.1 Fix Chunking Plugin Loader Idempotency

**Problem:** The embedding loader has `_plugins_loaded` flag, but chunking loader doesn't - can cause duplicate registration.

**Files to Change:**
- `packages/shared/chunking/plugin_loader.py`

**Implementation:**
```python
# Add to packages/shared/chunking/plugin_loader.py
_plugins_loaded = False
_registered_plugins: list[str] = []

def load_chunking_plugins() -> list[str]:
    global _plugins_loaded, _registered_plugins
    if _plugins_loaded:
        return list(_registered_plugins)

    # ... existing loading logic ...

    _plugins_loaded = True
    return _registered_plugins

def _reset_plugin_loader_state() -> None:
    """Reset for testing only."""
    global _plugins_loaded, _registered_plugins
    _plugins_loaded = False
    _registered_plugins = []
```

**Acceptance Criteria:**
- [x] Calling `load_chunking_plugins()` twice returns same result without re-registering
- [x] Test helper `_reset_plugin_loader_state()` exists
- [x] Tests pass in isolation and together

---

#### 0.2 Add Thread-Safety to Plugin Registries

**Problem:** All registry dictionaries lack locking, causing potential race conditions.

**Files to Change:**
- `packages/shared/embedding/factory.py`
- `packages/shared/embedding/provider_registry.py`
- `packages/webui/services/chunking/strategy_registry.py`

**Implementation:**
```python
# Add to each registry
from threading import Lock

_REGISTRY_LOCK = Lock()

def register_provider(internal_name: str, provider_class: type) -> None:
    with _REGISTRY_LOCK:
        if internal_name in _PROVIDER_CLASSES:
            logger.warning(f"Provider {internal_name} already registered, skipping")
            return
        _PROVIDER_CLASSES[internal_name] = provider_class
```

**Acceptance Criteria:**
- [x] All registry modifications use locks
- [x] Duplicate registration logs warning instead of silently overwriting
- [x] No race conditions under concurrent load

---

#### 0.3 Add Metadata to Existing Connectors

**Problem:** Connectors have no `PLUGIN_ID`, `METADATA`, or `get_config_fields()` unlike embedding/chunking plugins.

**Files to Change:**
- `packages/shared/connectors/local.py`
- `packages/shared/connectors/git.py`
- `packages/shared/connectors/imap.py`

**Implementation:**
```python
# Example: packages/shared/connectors/local.py
class LocalFileConnector(BaseConnector):
    # ADD these class attributes
    PLUGIN_ID: ClassVar[str] = "directory"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    METADATA: ClassVar[dict[str, Any]] = {
        "name": "Local Directory",
        "description": "Index files from a local directory on the server",
        "icon": "folder",
    }

    @classmethod
    def get_config_fields(cls) -> list[dict]:
        """Return config field definitions for UI."""
        return [
            {"name": "path", "type": "text", "label": "Directory Path", "required": True},
            {"name": "recursive", "type": "boolean", "label": "Include Subdirectories", "default": True},
            {"name": "include_patterns", "type": "glob_list", "label": "Include Patterns"},
            {"name": "exclude_patterns", "type": "glob_list", "label": "Exclude Patterns"},
        ]

    @classmethod
    def get_secret_fields(cls) -> list[dict]:
        """Return secret field definitions for UI."""
        return []  # Local connector has no secrets
```

**Acceptance Criteria:**
- [x] All 3 connectors have `PLUGIN_ID`, `PLUGIN_TYPE`, `METADATA`
- [x] All 3 connectors have `get_config_fields()` and `get_secret_fields()`
- [x] `connector_catalog.py` can now be removed (unblocked for Phase 1.2)
- [x] Existing connector usage unaffected

---

### Phase 1: Foundation (P0) ✅ COMPLETE

**Goal:** Establish a unified plugin architecture that simplifies both plugin development and internal maintenance.

#### 1.1 Unified Plugin Base Class

> **⚠️ ARCHITECTURE FINDING:** The three existing plugin systems have incompatible patterns:
> - **Embedding:** Two-phase async init (`__init__` then `await initialize(model_name)`), has `supports_model()` for auto-detection
> - **Chunking:** Sync-only, no class metadata, two different base classes (domain vs unified)
> - **Connector:** Dict config in `__init__`, no metadata attributes
>
> **Solution:** Use **protocol + adapter pattern** instead of forcing inheritance changes. Create adapters that wrap existing plugins to provide unified interface.

**Deliverables:**
- `packages/shared/plugins/base.py` - `SemanticPlugin` base class
- `packages/shared/plugins/protocols.py` - `PluginProtocol` for runtime checking
- `packages/shared/plugins/adapters.py` - Adapters for existing plugin types
- `packages/shared/plugins/manifest.py` - `PluginManifest` dataclass
- `packages/shared/plugins/exceptions.py` - Plugin-specific exceptions

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Create `PluginProtocol` | Runtime-checkable protocol for all plugins | `plugins/protocols.py` |
| Create `SemanticPlugin` ABC | Base class with `PLUGIN_TYPE`, `PLUGIN_ID`, `PLUGIN_VERSION` | `plugins/base.py` |
| Create adapters | `EmbeddingPluginAdapter`, `ChunkingPluginAdapter`, `ConnectorPluginAdapter` | `plugins/adapters.py` |
| Create `PluginManifest` | Dataclass with all plugin metadata fields | `plugins/manifest.py` |
| Define plugin exceptions | `PluginLoadError`, `PluginConfigError`, `PluginCompatibilityError` | `plugins/exceptions.py` |
| Create type-specific subclasses | `EmbeddingPlugin(SemanticPlugin)`, `ChunkingPlugin(SemanticPlugin)`, etc. | `plugins/types/*.py` |
| Add `get_config_schema()` | Optional JSON Schema for configuration | `plugins/base.py` |
| Add `health_check()` | Optional async health verification | `plugins/base.py` |

> **Note:** We use `PLUGIN_TYPE` (not `PLUGIN_CATEGORY`) as the standard attribute name. The embedding `PROVIDER_TYPE` attribute serves a different purpose (hosting model: "local"/"remote"/"hybrid") and does not conflict.

**Technical Details:**

```python
# packages/shared/plugins/protocols.py
from typing import Protocol, ClassVar, runtime_checkable

@runtime_checkable
class PluginProtocol(Protocol):
    """Minimal protocol all plugins should satisfy."""
    PLUGIN_TYPE: ClassVar[str]  # "embedding", "chunking", "connector"
    PLUGIN_ID: ClassVar[str]

    @classmethod
    def get_manifest(cls) -> "PluginManifest": ...

# packages/shared/plugins/base.py
from abc import ABC, abstractmethod
from typing import ClassVar, Any
from .manifest import PluginManifest

class SemanticPlugin(ABC):
    """Universal base for all Semantik plugins."""

    PLUGIN_TYPE: ClassVar[str]      # "embedding", "chunking", "connector", etc.
    PLUGIN_ID: ClassVar[str]        # Unique identifier, e.g., "dense-local"
    PLUGIN_VERSION: ClassVar[str] = "0.0.0"

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin metadata for discovery and UI."""
        ...

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        """Return JSON Schema for plugin configuration, or None if no config needed."""
        return None

    async def health_check(self) -> bool:
        """Check if plugin is healthy and operational."""
        return True

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize plugin with optional configuration."""
        pass

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass

# packages/shared/plugins/adapters.py
class EmbeddingPluginAdapter(SemanticPlugin):
    """Adapts existing BaseEmbeddingPlugin to unified interface."""

    PLUGIN_TYPE = "embedding"

    def __init__(self, legacy_plugin: "BaseEmbeddingPlugin"):
        self._plugin = legacy_plugin

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        # Build manifest from legacy plugin's get_definition()
        definition = cls._plugin_class.get_definition()
        return PluginManifest(
            id=definition.api_id,
            type="embedding",
            display_name=definition.display_name,
            description=definition.description or "",
            version=getattr(cls._plugin_class, 'PLUGIN_VERSION', '0.0.0'),
        )

    # Delegate to wrapped plugin
    async def initialize(self, config: dict | None = None) -> None:
        model_name = (config or {}).get("model_name", "")
        await self._plugin.initialize(model_name)
```

**Migration Strategy:**

1. Create new base class without breaking existing plugins
2. Add adapter layer: `class LegacyPluginAdapter(SemanticPlugin)`
3. Existing `BaseEmbeddingPlugin` inherits from both old interface and new `SemanticPlugin`
4. Deprecation warnings for direct `BaseEmbeddingPlugin` usage
5. Update documentation with migration guide

**Acceptance Criteria:**
- [x] All existing plugins continue working without modification
- [x] New plugins can use `SemanticPlugin` directly
- [x] `get_manifest()` returns valid `PluginManifest` for all plugins
- [x] Unit tests for base class and all type subclasses

---

#### 1.2 Connector Entry Points Migration

**Deliverables:**
- `semantik.plugins` entry point support for connectors
- Updated `ConnectorFactory` with entry point discovery
- Remove `connector_catalog.py` (replaced by `get_config_fields()`)
- Remove manual registration from `connector_factory.py`

**Tasks:**

| Task | Description | Files Changed | Status |
|------|-------------|---------------|--------|
| ~~Add metadata to connectors~~ | ~~`PLUGIN_ID`, `PLUGIN_TYPE`, `METADATA`, `get_config_fields()`, `get_secret_fields()`~~ | ~~`connectors/*.py`~~ | ✅ Phase 0 |
| Add `ConnectorPlugin` base | Extends `SemanticPlugin` with connector interface | `plugins/types/connector.py` | |
| Update `BaseConnector` | Inherit from `ConnectorPlugin`, add manifest methods | `connectors/base.py` | |
| Add entry point discovery | `load_connector_plugins()` function | `connectors/plugin_loader.py` (new) | |
| Update `ConnectorFactory` | Use entry point discovery only (remove manual) | `webui/services/connector_factory.py` | |
| Delete `connector_catalog.py` | No longer needed - connectors self-describe | `webui/services/connector_catalog.py` | |
| Add entry points to built-ins | Register directory, git, imap connectors | `pyproject.toml` | |
| Write migration docs | Guide for external connector authors | `docs/CONNECTOR_MIGRATION.md` | |

**Technical Details:**

```python
# packages/shared/plugins/types/connector.py
from abc import abstractmethod
from typing import AsyncIterator, Any
from ..base import SemanticPlugin
from semantik.shared.connectors.models import IngestedDocument

class ConnectorPlugin(SemanticPlugin):
    """Base class for data source connectors."""

    PLUGIN_TYPE = "connector"

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """Initialize connector with configuration."""
        ...

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with data source. Returns True if successful."""
        ...

    @abstractmethod
    async def load_documents(
        self,
        source_id: str | None = None
    ) -> AsyncIterator[IngestedDocument]:
        """Load documents from the data source."""
        ...

    def validate_config(self) -> None:
        """Validate configuration. Override to add custom validation."""
        pass

    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[dict]:
        """Return list of config field definitions for UI."""
        ...
```

```toml
# pyproject.toml (root)
[project.entry-points."semantik.plugins"]
connector-directory = "shared.connectors.local:LocalFileConnector"
connector-git = "shared.connectors.git:GitConnector"
connector-imap = "shared.connectors.imap:ImapConnector"
```

**Acceptance Criteria:**
- [x] All built-in connectors discoverable via entry points
- [x] External connector packages work with entry point registration
- [x] `ConnectorFactory.get_available_connectors()` includes all sources
- [x] No breaking changes for existing connector usage

---

#### 1.3 Plugin Discovery API

**Deliverables:**
- `/api/v2/plugins` endpoint with CRUD operations
- Plugin status tracking (enabled/disabled/error)
- Configuration storage in database

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Create plugin router | FastAPI router for `/api/v2/plugins` | `webui/api/routes/plugins.py` (new) |
| Create plugin schemas | Pydantic models for API responses | `webui/api/schemas/plugins.py` (new) |
| Create plugin service | Business logic for plugin management | `webui/services/plugin_service.py` (new) |
| Add plugin config table | Store enabled state and config per plugin | `shared/database/models/plugin.py` (new) |
| Create Alembic migration | `plugin_configs` table | `migrations/versions/xxxx_add_plugin_configs.py` |
| Add to router registry | Mount at `/api/v2/plugins` | `webui/api/routes/__init__.py` |

**API Endpoints:**

```python
# packages/webui/api/routes/plugins.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Literal

router = APIRouter(prefix="/api/v2/plugins", tags=["plugins"])

@router.get("")
async def list_plugins(
    type: str | None = None,
    enabled: bool | None = None,
    include_health: bool = False
) -> PluginListResponse:
    """
    List all installed plugins.

    Query params:
    - type: Filter by plugin type (embedding, chunking, connector, etc.)
    - enabled: Filter by enabled status
    - include_health: Run health checks and include status
    """
    ...

@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str) -> PluginDetailResponse:
    """Get detailed info for a specific plugin."""
    ...

@router.get("/{plugin_id}/manifest")
async def get_plugin_manifest(plugin_id: str) -> PluginManifest:
    """Get full manifest for a plugin."""
    ...

@router.get("/{plugin_id}/config-schema")
async def get_plugin_config_schema(plugin_id: str) -> dict | None:
    """Get JSON Schema for plugin configuration."""
    ...

@router.post("/{plugin_id}/enable")
async def enable_plugin(plugin_id: str) -> PluginStatusResponse:
    """Enable a plugin."""
    ...

@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str) -> PluginStatusResponse:
    """Disable a plugin."""
    ...

@router.put("/{plugin_id}/config")
async def update_plugin_config(
    plugin_id: str,
    config: dict
) -> PluginConfigResponse:
    """Update plugin configuration. Validates against config schema."""
    ...

@router.get("/{plugin_id}/health")
async def check_plugin_health(plugin_id: str) -> PluginHealthResponse:
    """Run health check for a plugin."""
    ...
```

**Database Schema:**

```python
# packages/shared/database/models/plugin.py
from sqlalchemy import Column, String, Boolean, JSON, DateTime
from sqlalchemy.sql import func

class PluginConfig(Base):
    __tablename__ = "plugin_configs"

    id = Column(String, primary_key=True)  # plugin_id
    type = Column(String, nullable=False)  # embedding, chunking, etc.
    enabled = Column(Boolean, default=True)
    config = Column(JSON, default={})
    last_health_check = Column(DateTime, nullable=True)
    health_status = Column(String, nullable=True)  # healthy, unhealthy, unknown
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
```

**Acceptance Criteria:**
- [x] All endpoints return correct data
- [x] Plugin enable/disable persists across restarts
- [x] Config validation against JSON Schema works
- [x] Health checks run without blocking API
- [x] OpenAPI docs generated correctly

---

#### 1.4 Plugin CLI Tool (Scaffold)

**Deliverables:**
- `semantik-plugin` CLI command
- `new` subcommand for scaffolding
- `validate` subcommand for contract checking
- Package installable via `pip install semantik[cli]`

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Create CLI package | `packages/cli/` directory structure | `packages/cli/*` (new) |
| Add Click application | Main CLI entry point | `cli/main.py` |
| Implement `new` command | Project scaffolding | `cli/commands/new.py` |
| Create templates | Jinja2 templates for each plugin type | `cli/templates/**/*.py.j2` |
| Implement `validate` | Check plugin contract | `cli/commands/validate.py` |
| Add to pyproject.toml | CLI entry point and deps | `packages/cli/pyproject.toml` |
| Add to workspace | Include in monorepo | `pyproject.toml` (root) |

**CLI Structure:**

```
packages/cli/
├── pyproject.toml
├── src/
│   └── semantik_cli/
│       ├── __init__.py
│       ├── main.py              # Click app entry point
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── new.py           # semantik-plugin new
│       │   ├── validate.py      # semantik-plugin validate
│       │   ├── test.py          # semantik-plugin test (Phase 2)
│       │   └── publish.py       # semantik-plugin publish (Phase 4)
│       ├── templates/
│       │   ├── embedding/
│       │   │   ├── plugin.py.j2
│       │   │   ├── pyproject.toml.j2
│       │   │   ├── tests/test_contract.py.j2
│       │   │   └── README.md.j2
│       │   ├── chunking/
│       │   ├── connector/
│       │   └── common/
│       │       ├── .gitignore.j2
│       │       └── CHANGELOG.md.j2
│       └── validators/
│           ├── __init__.py
│           ├── base.py
│           ├── embedding.py
│           └── connector.py
```

**Implementation:**

```python
# packages/cli/src/semantik_cli/main.py
import click
from .commands import new, validate

@click.group()
@click.version_option()
def cli():
    """Semantik Plugin Development CLI"""
    pass

cli.add_command(new.new)
cli.add_command(validate.validate)

# packages/cli/src/semantik_cli/commands/new.py
import click
from pathlib import Path
from jinja2 import Environment, PackageLoader

PLUGIN_TYPES = ["embedding", "chunking", "connector", "reranker", "extractor"]

@click.command()
@click.argument("name")
@click.option("--type", "-t", "plugin_type",
              type=click.Choice(PLUGIN_TYPES),
              required=True,
              help="Type of plugin to create")
@click.option("--output", "-o", "output_dir",
              type=click.Path(),
              default=".",
              help="Output directory")
@click.option("--template",
              type=click.Choice(["minimal", "full"]),
              default="full",
              help="Template variant")
def new(name: str, plugin_type: str, output_dir: str, template: str):
    """Create a new Semantik plugin project.

    Example:
        semantik-plugin new my-embedder --type embedding
        semantik-plugin new notion-connector --type connector
    """
    env = Environment(loader=PackageLoader("semantik_cli", "templates"))

    # Generate project structure
    project_dir = Path(output_dir) / name
    project_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "name": name,
        "plugin_type": plugin_type,
        "class_name": to_pascal_case(name),
        "module_name": to_snake_case(name),
    }

    # Render and write templates
    for template_name in get_templates_for_type(plugin_type, template):
        output_path = get_output_path(template_name, project_dir, context)
        content = env.get_template(template_name).render(**context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    click.echo(f"✓ Created plugin project at {project_dir}")
    click.echo(f"\nNext steps:")
    click.echo(f"  cd {name}")
    click.echo(f"  pip install -e .")
    click.echo(f"  semantik-plugin validate .")
```

**Acceptance Criteria:**
- [x] `semantik-plugin new my-plugin --type embedding` creates working project
- [x] Generated project passes `semantik-plugin validate`
- [x] Generated tests pass with `pytest`
- [x] Works for all plugin types: embedding, chunking, connector
- [x] CLI installable via pip

---

#### 1.5 Unified Plugin Loader

**Deliverables:**
- Single `load_all_plugins()` function
- Consistent error handling across all plugin types
- Plugin registry with lookup methods
- Remove separate `embedding/plugin_loader.py` and `chunking/plugin_loader.py`

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Create unified loader | `load_all_plugins()` function | `plugins/loader.py` (new) |
| Create plugin registry | In-memory registry with type indexing | `plugins/registry.py` (new) |
| Consolidate embedding loader | Move to unified loader, delete old file | `embedding/plugin_loader.py` → delete |
| Consolidate chunking loader | Move to unified loader, delete old file | `chunking/plugin_loader.py` → delete |
| Add connector loading | Include connectors in unified loader | `plugins/loader.py` |
| Update startup sequence | Single plugin loading call | `webui/main.py` |

**Technical Details:**

```python
# packages/shared/plugins/registry.py
from typing import TypeVar, Generic
from dataclasses import dataclass, field
from threading import Lock

T = TypeVar("T", bound="SemanticPlugin")

@dataclass
class PluginRegistry:
    """Thread-safe registry for all loaded plugins."""

    _plugins: dict[str, dict[str, type["SemanticPlugin"]]] = field(
        default_factory=lambda: {
            "embedding": {},
            "chunking": {},
            "connector": {},
            "reranker": {},
            "extractor": {},
        }
    )
    _lock: Lock = field(default_factory=Lock)
    _loaded: bool = False

    def register(self, plugin_cls: type["SemanticPlugin"]) -> None:
        """Register a plugin class."""
        with self._lock:
            plugin_type = plugin_cls.PLUGIN_TYPE
            plugin_id = plugin_cls.PLUGIN_ID

            if plugin_id in self._plugins[plugin_type]:
                existing = self._plugins[plugin_type][plugin_id]
                logger.warning(
                    f"Plugin conflict: {plugin_id} already registered "
                    f"({existing.__module__}), skipping {plugin_cls.__module__}"
                )
                return

            self._plugins[plugin_type][plugin_id] = plugin_cls
            logger.info(f"Registered plugin: {plugin_type}/{plugin_id}")

    def get(self, plugin_type: str, plugin_id: str) -> type["SemanticPlugin"] | None:
        """Get a specific plugin by type and ID."""
        return self._plugins.get(plugin_type, {}).get(plugin_id)

    def get_by_type(self, plugin_type: str) -> dict[str, type["SemanticPlugin"]]:
        """Get all plugins of a specific type."""
        return dict(self._plugins.get(plugin_type, {}))

    def get_all(self) -> dict[str, dict[str, type["SemanticPlugin"]]]:
        """Get all plugins grouped by type."""
        return {k: dict(v) for k, v in self._plugins.items()}

    def list_types(self) -> list[str]:
        """List all plugin types that have registered plugins."""
        return [k for k, v in self._plugins.items() if v]

# Global registry instance
plugin_registry = PluginRegistry()

# packages/shared/plugins/loader.py
import importlib.metadata
from .registry import plugin_registry
from .base import SemanticPlugin

def load_all_plugins(
    include_builtins: bool = True,
    entry_point_group: str = "semantik.plugins"
) -> PluginRegistry:
    """
    Load all plugins from entry points and built-in modules.

    Loading order:
    1. Built-in plugins (if include_builtins=True)
    2. External plugins from entry points

    Returns the populated PluginRegistry.
    """
    if plugin_registry._loaded:
        return plugin_registry

    with plugin_registry._lock:
        if plugin_registry._loaded:
            return plugin_registry

        # Load built-ins first
        if include_builtins:
            _load_builtin_plugins()

        # Load from entry points
        for ep in importlib.metadata.entry_points(group=entry_point_group):
            try:
                plugin_cls = ep.load()
                if not issubclass(plugin_cls, SemanticPlugin):
                    logger.warning(f"Entry point {ep.name} is not a SemanticPlugin")
                    continue
                plugin_registry.register(plugin_cls)
            except Exception as e:
                logger.error(f"Failed to load plugin {ep.name}: {e}")

        plugin_registry._loaded = True

    return plugin_registry

def _load_builtin_plugins():
    """Load built-in plugins from known modules."""
    # Import built-in modules to trigger registration
    from semantik.shared.embedding.providers import dense_local, litellm_provider
    from semantik.shared.chunking.domain.services.chunking_strategies import (
        semantic, sentence, fixed_token
    )
    # Connectors loaded via entry points in Phase 1.2
```

**Acceptance Criteria:**
- [x] Single call loads all plugin types
- [x] Built-ins always load before externals
- [x] Conflicts logged but don't crash
- [x] Registry queryable by type and ID
- [x] Thread-safe for concurrent access

---

### Phase 1.6: Runtime Contract (P0) — NEW

> **🔴 CRITICAL GAP IDENTIFIED:** The Phase 1 foundation has gaps that would make Plugin UI misleading:
> 1. VecPipe doesn't respect plugin enable/disable (calls `load_plugins()` without `disabled_plugin_ids`)
> 2. Plugin config is stored but never applied at runtime (nothing reads `PluginConfig.config`)
> 3. Health checks can't validate config-dependent state (called with no arguments)
>
> **This phase must be completed before Phase 2/3 to ensure the plugin system actually works end-to-end.**

**Goal:** Fix runtime contract gaps so that plugin enable/disable and configuration actually work across all services.

#### 1.6.1 Shared Plugin State File

**Problem:** WebUI stores plugin state in Postgres, but VecPipe has no database access.

**Solution:** Shared JSON file that WebUI writes and VecPipe reads.

**Secrets policy:** Plugin config **must not contain secrets**. Secrets (API keys, tokens) should be passed via environment variables. Config can reference env var names:

```python
# /data/plugin_state.json (or configurable path in shared volume)
{
    "version": 1,
    "updated_at": "2026-01-01T12:00:00Z",
    "disabled_ids": ["plugin-a", "plugin-b"],
    "configs": {
        "openai-embeddings": {
            "api_key_env": "OPENAI_API_KEY",  # Reference to env var, NOT the actual key
            "model": "text-embedding-3-small",
            "batch_size": 100
        },
        "cohere-reranker": {
            "api_key_env": "COHERE_API_KEY",
            "model": "rerank-english-v3"
        }
    }
}
```

**Why no secrets in config:**
- State file is plain JSON on shared volume (no encryption)
- Matches existing pattern: connectors use `get_secret_fields()` for env-var-based secrets
- Avoids complexity of secrets-at-rest encryption
- Plugin schema can declare `{"format": "env_var_reference"}` for key fields

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Define state file schema | JSON schema for plugin state | `shared/plugins/state.py` |
| Add env var resolution | Helper to resolve `*_env` fields to actual values | `shared/plugins/state.py` |
| Create state file writer | WebUI writes on config change | `webui/services/plugin_service.py` |
| Create state file reader | Load state at startup | `shared/plugins/loader.py` |
| Add file path config | `PLUGIN_STATE_FILE` env var | `shared/config/` |
| Update VecPipe startup | Read state file, pass to loader | `vecpipe/search/lifespan.py` |
| Validate no raw secrets | Reject config with actual secret values | `webui/services/plugin_service.py` |

#### 1.6.2 Config Passed at Runtime

**Problem:** `PluginConfig.config` is stored but never passed to plugins.

**Challenge:** The loader registers plugin *classes*, but plugins are *instantiated* later by type-specific factories. Config must reach each instantiation point.

**Integration points by plugin type:**

| Plugin Type | Registration | Instantiation | Config Integration Point |
|-------------|--------------|---------------|--------------------------|
| **Embedding** | `load_plugins()` registers in `provider_registry` | `EmbeddingProviderFactory.create_provider()` in `model_manager.py` | Factory must read config from state file and pass to constructor |
| **Chunking** | `load_plugins()` registers strategies | `ChunkingService` gets strategy by name | Service must pass config when instantiating strategy |
| **Connector** | `load_plugins()` registers in `connector_registry` | `ConnectorFactory.create()` | Factory must pass config to connector constructor |

**Solution:** Two-part approach:

1. **Base class accepts config in constructor** (not just `initialize()`):
```python
class SemanticPlugin:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    async def initialize(self) -> None:
        """Initialize using self._config."""
        pass
```

2. **Factories read config from state file:**
```python
# packages/shared/embedding/factory.py
from shared.plugins.state import load_plugin_state, resolve_env_vars

class EmbeddingProviderFactory:
    @classmethod
    def create_provider(cls, model_name: str, **kwargs) -> BaseEmbeddingPlugin:
        # Find provider class (existing logic)
        provider_cls = cls._find_provider(model_name)

        # NEW: Load config for this plugin
        state = load_plugin_state()
        plugin_id = provider_cls.PLUGIN_ID
        raw_config = state.get("configs", {}).get(plugin_id, {})
        config = resolve_env_vars(raw_config)  # Resolve api_key_env → actual value

        return provider_cls(config=config, **kwargs)
```

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Update base class | Accept `config` in `__init__` | `shared/plugins/base.py` |
| Add state loading to EmbeddingProviderFactory | Load and pass config at instantiation | `shared/embedding/factory.py` |
| Add state loading to ConnectorFactory | Load and pass config at instantiation | `webui/services/connector_factory.py` |
| Add state loading to ChunkingService | Pass config when getting strategy | `webui/services/chunking/` |
| Update all built-in plugins | Accept config param | Multiple files |
| Add `resolve_env_vars()` helper | Convert `*_env` references to values | `shared/plugins/state.py` |

#### 1.6.3 Health Check with Config

**Problem:** `health_check()` is called with no arguments, so it can't validate config-dependent state (e.g., "API key present").

**Solution:** Pass config to `health_check()`.

```python
# Before (current)
@classmethod
async def health_check(cls) -> bool:
    return True

# After
@classmethod
async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
    """Check plugin health with optional config context."""
    return True
```

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Update health_check signature | Add `config` param | `shared/plugins/base.py` |
| Update plugin_service | Pass config to health_check | `webui/services/plugin_service.py:379` |
| Update adapters | Pass config through adapter layer | `shared/plugins/adapters.py` |

#### 1.6.4 VecPipe Integration

**Problem:** VecPipe calls `load_plugins()` without `disabled_plugin_ids`.

**Files to update:**
- `packages/vecpipe/search/lifespan.py:59`
- `packages/vecpipe/model_manager.py:116`

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Read state file in VecPipe | Load disabled IDs at startup | `vecpipe/search/lifespan.py` |
| Pass disabled_ids to loader | Prevent disabled plugins from loading | `vecpipe/search/lifespan.py` |
| Update model_manager | Respect disabled state | `vecpipe/model_manager.py` |

**Acceptance Criteria:**
- [ ] Disabling a plugin in WebUI prevents it from loading in VecPipe (after restart)
- [ ] Plugin config set via API is available in `initialize()` method
- [ ] Health check can validate config-dependent state
- [ ] State file written atomically (no partial writes)
- [ ] Backward compatible: missing state file = all plugins enabled, no config

---

### Phase 2: New Plugin Types (P1)

**Goal:** Expand the plugin ecosystem with high-value plugin types that enable new use cases.

#### 2.1 Reranker Plugins (SCOPE REDUCED)

> **🟢 MAJOR DISCOVERY:** Reranking infrastructure already exists in VecPipe! This phase only needs to wrap existing code as a plugin interface.

**What Already Exists:**
- `use_reranker` and `rerank_model` parameters in SearchService ✅
- Qwen3 reranker configuration in `qwen3_search_config.py` ✅
- Cross-encoder support in `packages/vecpipe/reranker.py` ✅
- 5x candidate multiplier with bounds (20-200) ✅
- Hybrid search blending (70% vector / 30% rerank) ✅

**Revised Deliverables (3-4 days):**
- `BaseRerankerPlugin` interface (thin wrapper)
- Wrap existing Qwen3 reranker as built-in plugin
- Add `default_reranker` to Collection model
- Add `/api/v2/rerankers` discovery endpoint

**Revised Tasks:**

| Task | Description | Files Changed | Effort |
|------|-------------|---------------|--------|
| Define reranker interface | `BaseRerankerPlugin` with `rerank()` method | `plugins/types/reranker.py` | 0.5 day |
| Wrap Qwen3 reranker | Create plugin wrapper for existing code | `shared/plugins/builtins/qwen3_reranker.py` | 0.5 day |
| Add Collection field | `default_reranker` column + migration | `shared/database/models.py`, Alembic | 0.5 day |
| Add discovery endpoint | `GET /api/v2/rerankers` | `webui/api/v2/rerankers.py` | 0.5 day |
| Map plugin to existing params | `reranker_id` → `rerank_model` | `webui/services/search_service.py` | 0.5 day |
| Tests and docs | Contract tests, documentation | `tests/`, `docs/` | 1 day |

**Interface:**

```python
# packages/shared/plugins/types/reranker.py
from abc import abstractmethod
from dataclasses import dataclass
from ..base import SemanticPlugin

@dataclass
class RerankResult:
    """Result of reranking a document."""
    index: int          # Original index in input list
    score: float        # Relevance score (higher = more relevant)
    document: str       # The document text
    metadata: dict | None = None

@dataclass
class RerankerCapabilities:
    """Capabilities and limits of a reranker."""
    max_documents: int      # Max docs per request
    max_query_length: int   # Max query tokens/chars
    max_doc_length: int     # Max document tokens/chars
    supports_batching: bool # Can batch multiple queries
    models: list[str]       # Available model variants

class BaseRerankerPlugin(SemanticPlugin):
    """Base class for reranker plugins."""

    PLUGIN_TYPE = "reranker"

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict] | None = None
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of results to return (default: all)
            metadata: Optional metadata for each document

        Returns:
            List of RerankResult sorted by relevance (highest first)
        """
        ...

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> RerankerCapabilities:
        """Return reranker capabilities and limits."""
        ...

    async def rerank_batch(
        self,
        queries: list[str],
        documents_per_query: list[list[str]],
        top_k: int | None = None
    ) -> list[list[RerankResult]]:
        """
        Batch rerank multiple queries. Override for optimized batch processing.
        Default implementation calls rerank() for each query.
        """
        results = []
        for query, docs in zip(queries, documents_per_query):
            result = await self.rerank(query, docs, top_k)
            results.append(result)
        return results
```

**Search Integration:**

```python
# packages/webui/services/search_service.py (modified)

async def search(
    self,
    query: str,
    collection_id: str,
    top_k: int = 10,
    use_reranker: bool = False,
    reranker_id: str | None = None,
    rerank_top_k: int | None = None,  # How many to fetch before reranking
) -> SearchResponse:
    """
    Search with optional reranking.

    When use_reranker=True:
    1. Fetch rerank_top_k (default: top_k * 3) results from vector search
    2. Rerank using specified reranker (or collection default)
    3. Return top_k from reranked results
    """
    # Initial vector search
    fetch_count = rerank_top_k or (top_k * 3) if use_reranker else top_k
    vector_results = await self._vector_search(query, collection_id, fetch_count)

    if not use_reranker:
        return SearchResponse(results=vector_results[:top_k])

    # Get reranker
    reranker = await self._get_reranker(reranker_id, collection_id)
    if not reranker:
        logger.warning("Reranker requested but not available, returning vector results")
        return SearchResponse(results=vector_results[:top_k])

    # Rerank
    documents = [r.content for r in vector_results]
    reranked = await reranker.rerank(query, documents, top_k)

    # Map back to original results
    final_results = [
        SearchResult(
            **vector_results[rr.index].dict(),
            rerank_score=rr.score
        )
        for rr in reranked
    ]

    return SearchResponse(results=final_results, reranked=True)
```

**Acceptance Criteria:**
- [ ] Reranker plugin interface complete
- [ ] At least one working reranker (BGE or sentence-transformers)
- [ ] Search API accepts reranking parameters
- [ ] Reranking measurably improves search quality
- [ ] Performance acceptable (< 500ms for 100 docs)

---

#### 2.2 Extractor Plugins

**Deliverables:**
- `BaseExtractorPlugin` interface
- Integration with document ingestion pipeline
- Built-in spaCy NER extractor

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Define extractor interface | `BaseExtractorPlugin` with `extract()` method | `plugins/types/extractor.py` |
| Create extractor service | Service for running extractors | `webui/services/extractor_service.py` |
| Integrate with ingestion | Run extractors during document processing | `webui/services/ingestion_service.py` |
| Store extracted metadata | Add extraction results to document model | `shared/database/models/document.py` |
| Create spaCy extractor | NER, language detection built-in | `shared/extraction/spacy_extractor.py` |
| Add extraction config | Collection-level extraction settings | `shared/database/models/collection.py` |
| Expose in search filters | Filter by extracted entities | `webui/api/routes/search.py` |

**Interface:**

```python
# packages/shared/plugins/types/extractor.py
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from ..base import SemanticPlugin

class ExtractionType(Enum):
    ENTITIES = "entities"      # Named entities (PERSON, ORG, etc.)
    TOPICS = "topics"          # Topic classification
    LANGUAGE = "language"      # Language detection
    SENTIMENT = "sentiment"    # Sentiment analysis
    KEYWORDS = "keywords"      # Keyword extraction
    SUMMARY = "summary"        # Auto-summarization
    CUSTOM = "custom"          # Plugin-specific

@dataclass
class Entity:
    text: str
    type: str           # PERSON, ORG, LOC, DATE, etc.
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

@dataclass
class ExtractionResult:
    """Result of extracting metadata from text."""
    entities: list[Entity] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    language: str | None = None
    language_confidence: float | None = None
    sentiment: float | None = None  # -1.0 to 1.0
    keywords: list[str] = field(default_factory=list)
    summary: str | None = None
    custom: dict = field(default_factory=dict)

    def to_searchable_dict(self) -> dict:
        """Convert to dict suitable for search filtering."""
        return {
            "entities": {e.type: e.text for e in self.entities},
            "entity_types": list(set(e.type for e in self.entities)),
            "topics": self.topics,
            "language": self.language,
            "keywords": self.keywords,
        }

class BaseExtractorPlugin(SemanticPlugin):
    """Base class for metadata extraction plugins."""

    PLUGIN_TYPE = "extractor"

    @classmethod
    @abstractmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        """Return list of extraction types this plugin supports."""
        ...

    @abstractmethod
    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,
        options: dict | None = None
    ) -> ExtractionResult:
        """
        Extract metadata from text.

        Args:
            text: Text to extract from
            extraction_types: Which extractions to perform (default: all supported)
            options: Plugin-specific options

        Returns:
            ExtractionResult with extracted metadata
        """
        ...

    async def extract_batch(
        self,
        texts: list[str],
        extraction_types: list[ExtractionType] | None = None,
        options: dict | None = None
    ) -> list[ExtractionResult]:
        """Batch extraction. Override for optimized batch processing."""
        return [await self.extract(t, extraction_types, options) for t in texts]
```

**Ingestion Integration:**

```python
# packages/webui/services/ingestion_service.py (modified)

async def ingest_document(
    self,
    document: IngestedDocument,
    collection: Collection
) -> ProcessedDocument:
    """Ingest document with optional metadata extraction."""

    # ... chunking and embedding as before ...

    # Run extractors if configured
    if collection.extraction_config.enabled:
        extractors = await self._get_extractors(collection.extraction_config)
        extraction_results = []

        for extractor in extractors:
            try:
                result = await extractor.extract(
                    document.content,
                    extraction_types=collection.extraction_config.types
                )
                extraction_results.append(result)
            except Exception as e:
                logger.warning(f"Extractor {extractor.PLUGIN_ID} failed: {e}")

        # Merge extraction results
        merged = self._merge_extractions(extraction_results)
        document.extracted_metadata = merged.to_searchable_dict()

    # ... store document ...
```

**Acceptance Criteria:**
- [ ] Extractor plugin interface complete
- [ ] spaCy extractor works for NER and language detection
- [ ] Extracted metadata stored with documents
- [ ] Search can filter by extracted entities
- [ ] Extraction runs during ingestion without blocking

---

#### 2.3 Preprocessor Plugins — DEFERRED

> **Status:** Deferred - requires architectural changes to extraction pipeline.
>
> **Architectural Reality (discovered during review):**
>
> Content extraction currently happens **inside connectors** via `extract_and_serialize()`:
> - `packages/shared/connectors/local.py:236` - LocalFileConnector
> - `packages/shared/connectors/git.py:744` - GitConnector
> - `packages/shared/text_processing/extraction.py` - uses `unstructured.partition.auto.partition`
>
> By the time content reaches `IngestedDocument.content`, it is already **post-extraction plain text**.
> There is no natural hook point for preprocessor plugins without one of:
> - Modifying all connectors to call preprocessors (defeats generic plugin model)
> - Moving extraction to a centralized pipeline stage (significant refactor)
> - Changing contract so connectors hand off raw bytes + content_type
>
> **Recommendation:** Revisit when there's a concrete use case (e.g., specialized OCR, domain-specific parsing) that justifies the extraction pipeline refactor.

<details>
<summary>Original design (preserved for reference)</summary>

**Deliverables:**
- `BasePreprocessorPlugin` interface
- Per-collection preprocessing configuration
- Built-in HTML cleaner and PDF text extractor
- Content-type routing (auto-detect → configured preprocessor)

**UX Model:** Per-collection config where users select preprocessor per content type.

**Interface sketch:**
```python
class BasePreprocessorPlugin(SemanticPlugin):
    PLUGIN_TYPE = "preprocessor"

    @classmethod
    def supported_content_types(cls) -> list[str]: ...

    async def preprocess(self, content: str | bytes, content_type: str) -> PreprocessedDocument: ...
```

</details>

---

### Phase 3: Developer Experience (P1) — ✅ COMPLETE

**Goal:** Provide essential tooling for plugin management and testing.

**Status:** Completed in branch `0.7.5/phase3` (2026-01-01)

> **Scope reduction (2026-01-01):**
> - 3.1 Plugin UI: **Keep** (full implementation) ✅
> - 3.2 Testing: **Simplify** (pytest fixtures only, no CLI test command) ✅
> - 3.3 Hooks: **Deferred** (build when concrete use case arises)
> - 3.4 Templates repo: **Cut** (CLI scaffolding is sufficient)

#### 3.1 Plugin Management UI — ✅ COMPLETE

> **Implementation note:** Settings page restructured with tabs (Database | Plugins). Plugin Management tab displays all plugins grouped by type with enable/disable toggles, health status indicators, and configuration modals.
>
> **Pattern used:** Adapted `DynamicField` pattern for plugin config forms. JSON Schema from backend drives form rendering.

**Deliverables:**
- Settings > Plugins page in React UI
- Enable/disable toggles
- Configuration forms (adapt existing DynamicField pattern)
- Health status indicators

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Restructure Settings page | Add tabs/routing for multiple settings sections | `apps/webui-react/src/pages/SettingsPage.tsx` |
| Create Plugins tab/page | Main plugin management view | `apps/webui-react/src/pages/settings/Plugins.tsx` |
| Create PluginCard component | Card for each plugin with actions | `apps/webui-react/src/components/plugins/PluginCard.tsx` |
| Create PluginConfigModal | Modal for editing plugin config | `apps/webui-react/src/components/plugins/PluginConfigModal.tsx` |
| Adapt DynamicField for plugins | Reuse connector config pattern | `apps/webui-react/src/components/plugins/PluginConfigForm.tsx` |
| Create plugins API hooks | React Query hooks for plugin API | `apps/webui-react/src/hooks/usePlugins.ts` |

**Acceptance Criteria:**
- [x] Plugin list displays all installed plugins
- [x] Plugins can be enabled/disabled from UI
- [x] Configuration forms validate correctly
- [x] Health status updates in real-time
- [x] Mobile responsive design

---

#### 3.2 Plugin Testing Utilities — ✅ COMPLETE

**Deliverables:**
- `semantik.plugins.testing` module with pytest fixtures
- Contract test base classes for each plugin type
- Mock services for isolated testing

> **Simplified scope:** No CLI `test` command, no benchmark suite. Just pytest fixtures that plugin authors import.

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Create testing module | `packages/shared/plugins/testing/` | `plugins/testing/__init__.py` |
| Create contract tests | Base classes for each plugin type | `plugins/testing/contracts.py` |
| Create fixtures | Mock embedding service, mock documents | `plugins/testing/fixtures.py` |

**Usage:**

```python
# In plugin's test file
from semantik.plugins.testing import PluginContractTest, EmbeddingPluginContractTest

class TestMyEmbedder(EmbeddingPluginContractTest):
    plugin_class = MyEmbedderPlugin

    # Contract tests run automatically:
    # - test_has_required_class_attributes
    # - test_get_manifest_returns_valid_manifest
    # - test_embed_single_returns_list_of_floats
    # - test_embed_texts_batch_processing
```

**Acceptance Criteria:**
- [x] Contract tests catch missing methods/attributes
- [x] Fixtures work without requiring real services
- [x] Documentation explains testing best practices

---

#### 3.3 Plugin Hooks System — DEFERRED

> **Status:** Deferred until a concrete use case arises.
>
> **Rationale:** Hooks add complexity across the codebase (emit points in every service). Without a clear use case driving the design, we risk building the wrong abstraction. Will revisit when there's a specific need like analytics, audit logging, or external integrations that can't be solved another way.

<details>
<summary>Original design (preserved for reference)</summary>

**Deliverables:**
- Hook registry with typed events
- Async-safe hook execution
- Error isolation between hooks
- Built-in telemetry hooks

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Define hook events enum | All application lifecycle events | `plugins/hooks/events.py` |
| Create hook registry | Registration and dispatch system | `plugins/hooks/registry.py` |
| Create hook decorators | `@on_event(HookEvent.X)` decorator | `plugins/hooks/decorators.py` |
| Integrate with app lifecycle | Emit events at key points | `webui/main.py`, `services/*.py` |
| Create telemetry hook | Built-in hook for metrics | `plugins/hooks/telemetry.py` |
| Document hook system | Guide for using hooks | `docs/PLUGIN_HOOKS.md` |

</details>

---

#### 3.4 Plugin Templates Repository — CUT

> **Status:** Cut from scope.
>
> **Rationale:** The `semantik-plugin new` CLI command (Phase 1.4) already provides scaffolding via Jinja2 templates. A separate GitHub template repository adds maintenance burden without clear benefit. If more sophisticated templates are needed later, they can be added to the CLI.

---

### Phase 4: Plugin Registry (P2) — ✅ COMPLETE

**Goal:** Enable plugin discovery without hosted infrastructure.

**Status:** Completed in branch `0.7.5/phase4` (PR #273).

> **Scope reduction (2026-01-01):**
> - **No hosted service** - Registry is a YAML file in a GitHub repo
> - **No complex verification** - Just a "verified" boolean set by maintainer
> - **No community infrastructure** - Premature without community
> - **Basic versioning only** - Semver constraint checking, no compatibility matrix

#### 4.1 Slim Plugin Registry

**Approach:** A curated YAML file hosted on GitHub, fetched by the app.

```yaml
# https://raw.githubusercontent.com/semantik/plugin-registry/main/registry.yaml
# Or bundled in app, refreshed periodically

registry_version: "1.0"
last_updated: "2026-01-15T00:00:00Z"

plugins:
  - id: openai-embeddings
    type: embedding
    name: "OpenAI Embeddings"
    description: "text-embedding-3-small/large via OpenAI API"
    author: "semantik"
    repository: "https://github.com/semantik-plugins/openai-embeddings"
    pypi: "semantik-plugin-openai"
    verified: true
    min_semantik_version: "2.0.0"

  - id: cohere-reranker
    type: reranker
    name: "Cohere Reranker"
    description: "rerank-english-v3, rerank-multilingual-v3"
    author: "semantik"
    repository: "https://github.com/semantik-plugins/cohere-reranker"
    pypi: "semantik-plugin-cohere-reranker"
    verified: true
    min_semantik_version: "2.0.0"

  - id: tesseract-ocr
    type: preprocessor
    name: "Tesseract OCR"
    description: "OCR for scanned PDFs and images"
    author: "community"
    repository: "https://github.com/example/semantik-tesseract"
    pypi: "semantik-plugin-tesseract"
    verified: false
    min_semantik_version: "2.0.0"
```

**Deliverables:**
- Registry YAML schema
- GitHub repo `semantik/plugin-registry`
- Registry fetch/cache in backend
- Browse UI in Plugin Management page
- Basic install instructions (shows `pip install` command)

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Define registry schema | YAML structure for plugin entries | `docs/PLUGIN_REGISTRY.md` |
| Create registry repo | GitHub repo with initial plugins | `github.com/semantik/plugin-registry` |
| Add registry client | Fetch and cache registry YAML | `shared/plugins/registry_client.py` |
| Add browse UI | "Available Plugins" tab in Plugin UI | `apps/webui-react/` |
| Add version checking | Check `min_semantik_version` constraint | `shared/plugins/compatibility.py` |

**UI Design:**

```
Settings > Plugins > Available

┌─────────────────────────────────────────────────────────────┐
│ 🔍 Search plugins...                          [Refresh]     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  OpenAI Embeddings                    ✓ Verified            │
│  text-embedding-3-small/large via OpenAI API                │
│  [View on GitHub]  [Install: pip install semantik-plugin-openai] │
│                                                             │
│  Cohere Reranker                      ✓ Verified            │
│  rerank-english-v3, rerank-multilingual-v3                  │
│  [View on GitHub]  [Install: pip install semantik-plugin-cohere] │
│                                                             │
│  Tesseract OCR                        ⚠ Unverified          │
│  OCR for scanned PDFs and images                            │
│  [View on GitHub]  [Install: pip install semantik-plugin-tesseract] │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**To add a plugin to registry:** Submit PR to the YAML file.

**Acceptance Criteria:**
- [x] Registry YAML fetchable from GitHub (with env var override)
- [x] Plugins displayed in UI with install instructions
- [x] Version constraints checked before showing "compatible"
- [x] Verified vs unverified distinction visible
- [x] Search and filter functionality
- [x] Installed plugins marked in Available tab

---

#### 4.2 Plugin Versioning — SIMPLIFIED

**Scope:** Basic semver constraint checking only.

```python
# packages/shared/plugins/compatibility.py
from packaging.version import Version
from packaging.specifiers import SpecifierSet

def check_compatibility(plugin_constraint: str, semantik_version: str) -> bool:
    """Check if plugin is compatible with current Semantik version."""
    if not plugin_constraint:
        return True
    spec = SpecifierSet(f">={plugin_constraint}")
    return Version(semantik_version) in spec
```

No compatibility matrix, no upgrade notifications, no migration tooling.

---

#### 4.3 Community Infrastructure — CUT

> **Status:** Cut from scope. Premature without an actual community.

---

### Phase 5: Security (P1) — SIMPLIFIED

> **Context:** Current plugins run with full process privileges (access to env vars, network, filesystem). This is acceptable when the primary plugin author is trusted (yourself), but worth addressing for good hygiene.

**Goal:** Basic security hardening without complex sandboxing infrastructure.

> **Scope reduction (2026-01-01):**
> - 5.0 Security hardening: **Keep** (env filtering, audit logging)
> - 5.1-5.4 Sandboxing/permissions: **Deferred** (build when untrusted plugins are a real concern)
>
> **Rationale:** You're the primary plugin author and trust your own code. Full sandboxing is complex and premature. Document that users should only install trusted plugins.

#### 5.0 Security Hardening

> **⚠️ Important limitation:** Since plugins run **in-process**, they can always access `os.environ` directly by importing `os`. The env filtering below is "defense in depth" for plugins that cooperate with the API—it is **not a security boundary**.
>
> True isolation would require out-of-process execution (subprocess/WASM), which is deferred to 5.2. For now, the primary security measure is: **only install trusted plugins**.

**Deliverables:**
- Audit logging for plugin operations (primary value)
- Environment variable filtering (cooperative, not enforced)
- Security documentation

**Tasks:**

| Task | Description | Files Changed |
|------|-------------|---------------|
| Add audit logging | Log plugin load, health check, config changes | `shared/plugins/loader.py`, `webui/services/plugin_service.py` |
| Add env filtering helper | Utility for plugins that want sanitized env | `shared/plugins/security.py` |
| Document security model | "Plugins run with full privileges, only install trusted code" | `docs/PLUGIN_SECURITY.md` |

**Implementation:**

```python
# packages/shared/plugins/security.py
import os
import logging

logger = logging.getLogger(__name__)

SENSITIVE_ENV_PATTERNS = frozenset({
    "PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL", "API_KEY"
})

def get_sanitized_environment() -> dict[str, str]:
    """Return environment with sensitive values removed.

    NOTE: This is cooperative only. Plugins can still access os.environ directly.
    This utility is for plugins that want to be good citizens.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not any(pattern in key.upper() for pattern in SENSITIVE_ENV_PATTERNS)
    }

def audit_log(plugin_id: str, action: str, details: dict | None = None) -> None:
    """Log plugin action for security auditing."""
    logger.info(
        f"PLUGIN_AUDIT: {plugin_id} - {action}",
        extra={"plugin_id": plugin_id, "action": action, **(details or {})}
    )
```

**Acceptance Criteria:**
- [ ] Plugin operations logged with structured data
- [ ] Security documentation published
- [ ] Env filtering utility available (cooperative)

---

#### 5.1-5.4 Advanced Security — DEFERRED

> **Status:** Deferred until untrusted third-party plugins become a real concern.

<details>
<summary>Original designs (preserved for reference)</summary>

**5.1 Tiered Trust Model**
- Trusted (built-in), Verified (community reviewed), Untrusted (user-installed)
- UI indicators for trust level

**5.2 Plugin Sandboxing**
- Subprocess isolation for untrusted plugins
- Resource limits (CPU, memory, time)
- Options: cgroups, Docker containers, WASM, nsjail

**5.3 Permission System**
- Permission manifest in plugins (NETWORK, FILESYSTEM, GPU, etc.)
- User consent UI
- Runtime enforcement

**5.4 Plugin Verification**
- Code review process
- Automated security scanning
- Signing and attestation

</details>

---

### Phase 6: Example Plugins (P2)

**Goal:** Create real, published plugins that users can actually install, demonstrating the plugin system works end-to-end.

> **Rationale:** The registry currently contains placeholder entries. Publishing actual plugins:
> - Validates the plugin authoring experience
> - Provides working examples for community contributors
> - Makes the "Available Plugins" UI immediately useful
> - Proves the system works beyond built-in plugins

#### 6.1 Priority Plugins

| Plugin | Type | Priority | Notes |
|--------|------|----------|-------|
| **OpenAI Embeddings** | embedding | P0 | Most requested; `text-embedding-3-small/large` |
| **Cohere Reranker** | reranker | P1 | Popular reranking API |
| **Voyage AI Embeddings** | embedding | P2 | High-quality embeddings API |
| **S3 Connector** | connector | P2 | Common enterprise use case |

#### 6.2 Plugin Package Structure

Each plugin will be a separate repository following the standard structure:

```
semantik-plugin-openai/
├── pyproject.toml           # Package metadata, entry points
├── README.md                # Installation and usage docs
├── LICENSE                  # MIT
├── src/
│   └── semantik_plugin_openai/
│       ├── __init__.py
│       ├── provider.py      # Main plugin implementation
│       └── config.py        # Configuration schema
└── tests/
    └── test_provider.py     # Plugin tests
```

**Entry point registration:**
```toml
[project.entry-points."semantik.plugins"]
openai-embeddings = "semantik_plugin_openai:OpenAIEmbeddingPlugin"
```

#### 6.3 Tasks

| Task | Description | Deliverables |
|------|-------------|--------------|
| Create plugin template repo | GitHub template for new plugins | `github.com/semantik-plugins/plugin-template` |
| Implement OpenAI embeddings | Full embedding plugin with config | `semantik-plugin-openai` on PyPI |
| Implement Cohere reranker | Reranker plugin with API key config | `semantik-plugin-cohere-reranker` on PyPI |
| Implement Voyage embeddings | Embedding plugin | `semantik-plugin-voyage` on PyPI |
| Implement S3 connector | Connector with AWS credential config | `semantik-plugin-s3` on PyPI |
| Update registry | Replace placeholders with real entries | `packages/shared/plugins/data/registry.yaml` |
| Document plugin authoring | End-to-end guide with real examples | `docs/WRITING_PLUGINS.md` |

#### 6.4 OpenAI Embeddings Plugin (Reference Implementation)

```python
# src/semantik_plugin_openai/provider.py
from typing import ClassVar
import openai
from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest
from shared.plugins.types.embedding import EmbeddingPluginProtocol

class OpenAIEmbeddingPlugin(SemanticPlugin, EmbeddingPluginProtocol):
    PLUGIN_TYPE: ClassVar[str] = "embedding"
    PLUGIN_ID: ClassVar[str] = "openai-embeddings"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            version=cls.PLUGIN_VERSION,
            type=cls.PLUGIN_TYPE,
            display_name="OpenAI Embeddings",
            description="Generate embeddings using OpenAI's text-embedding models",
            author="semantik",
            homepage="https://github.com/semantik-plugins/openai-embeddings",
            config_schema={
                "api_key": {"type": "string", "secret": True, "required": True},
                "model": {"type": "string", "default": "text-embedding-3-small"},
                "organization": {"type": "string", "required": False},
            },
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name in cls.SUPPORTED_MODELS

    async def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        client = openai.AsyncOpenAI(
            api_key=self.config.get("api_key"),
            organization=self.config.get("organization"),
        )
        response = await client.embeddings.create(
            model=self.config.get("model", "text-embedding-3-small"),
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        model = self.config.get("model", "text-embedding-3-small")
        return self.SUPPORTED_MODELS.get(model, 1536)
```

#### 6.5 Publishing Workflow

1. Create GitHub repository under `semantik-plugins` org
2. Implement plugin following the template
3. Write tests with mocked API responses
4. Publish to PyPI: `uv build && uv publish`
5. Update bundled registry with real package info
6. Test installation: `pip install semantik-plugin-openai`
7. Verify plugin appears in UI and works

**Acceptance Criteria:**
- [ ] At least 2 plugins published to PyPI and installable
- [ ] Plugins appear in "Available" UI with working install commands
- [ ] Plugin template repository available for contributors
- [ ] End-to-end documentation for writing and publishing plugins

---

### Phase 7: In-App Plugin Installation (P3)

**Goal:** Allow users to install plugins directly from the UI without manual Docker/pip commands.

> **Context:** Semantik is Docker-first. Telling users to `pip install` doesn't work because:
> - Users don't have shell access to the container
> - Installs are lost on container restart
> - Manual Dockerfile customization is too technical for most users
>
> **Solution:** Persistent volume + in-app installation with hot-reload.

#### 7.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Host                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Semantik Container                      ││
│  │                                                          ││
│  │  ┌──────────────┐    pip install     ┌───────────────┐  ││
│  │  │   Web UI     │ ───────────────────▶│  /app/plugins │  ││
│  │  │  "Install"   │    --target         │   (volume)    │  ││
│  │  └──────────────┘                     └───────────────┘  ││
│  │         │                                    │           ││
│  │         │ WebSocket                          │ PYTHONPATH││
│  │         ▼                                    ▼           ││
│  │  ┌──────────────┐                    ┌───────────────┐  ││
│  │  │   Progress   │                    │ Plugin Loader │  ││
│  │  │   Updates    │                    │  (hot reload) │  ││
│  │  └──────────────┘                    └───────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│                    ┌─────────▼─────────┐                    │
│                    │  semantik_plugins │ ◀── Docker Volume  │
│                    │     (persistent)  │     (survives      │
│                    └───────────────────┘      restarts)     │
└─────────────────────────────────────────────────────────────┘
```

#### 7.2 Docker Configuration

```yaml
# docker-compose.yml
services:
  webui:
    image: semantik:latest
    volumes:
      - semantik_plugins:/app/plugins
    environment:
      SEMANTIK_PLUGINS_DIR: /app/plugins
      PYTHONPATH: /app/plugins

volumes:
  semantik_plugins:
```

#### 7.3 Backend Implementation

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/plugins/install` | POST | Install plugin from PyPI |
| `/api/v2/plugins/{id}/uninstall` | DELETE | Uninstall plugin |
| `/api/v2/plugins/install/status/{job_id}` | GET | Poll installation status |
| `/ws/plugins/install/{job_id}` | WebSocket | Stream installation progress |

**Install Request:**
```python
class PluginInstallRequest(BaseModel):
    package_name: str  # e.g., "semantik-plugin-openai"
    version: str | None = None  # e.g., "1.0.0" or None for latest

class PluginInstallResponse(BaseModel):
    job_id: str
    status: Literal["queued", "installing", "success", "failed"]
    message: str | None = None
```

**Installation Service:**
```python
# packages/webui/services/plugin_installer.py
import asyncio
import subprocess
from pathlib import Path

PLUGINS_DIR = Path(os.environ.get("SEMANTIK_PLUGINS_DIR", "/app/plugins"))

async def install_plugin(
    package_name: str,
    version: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Install a plugin package to the plugins directory.

    Returns (success, message).
    """
    target = f"{package_name}=={version}" if version else package_name

    cmd = [
        "pip", "install",
        "--target", str(PLUGINS_DIR),
        "--upgrade",
        "--no-deps",  # Deps handled separately to avoid conflicts
        target,
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_lines = []
    async for line in process.stdout:
        line_str = line.decode().strip()
        output_lines.append(line_str)
        if progress_callback:
            progress_callback(line_str)

    await process.wait()

    if process.returncode == 0:
        # Reload entry points to pick up new plugin
        await reload_plugins()
        return True, f"Successfully installed {package_name}"
    else:
        return False, "\n".join(output_lines)

async def uninstall_plugin(package_name: str) -> tuple[bool, str]:
    """Uninstall a plugin from the plugins directory."""
    # Find and remove package directory
    ...

async def reload_plugins() -> None:
    """Hot-reload plugin entry points without restart."""
    import importlib
    from importlib.metadata import distributions

    # Clear cached distributions
    importlib.invalidate_caches()

    # Re-scan entry points
    # ... trigger plugin loader refresh
```

#### 7.4 Frontend Implementation

**UI Flow:**
```
Available Plugins Tab
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  OpenAI Embeddings                    ✓ Verified            │
│  text-embedding-3-small/large via OpenAI API               │
│                                                             │
│  [Install]  ←── Click to install                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  OpenAI Embeddings                    ✓ Verified            │
│  text-embedding-3-small/large via OpenAI API               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Installing...                                        │   │
│  │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░  35%          │   │
│  │ Downloading semantik-plugin-openai...               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  OpenAI Embeddings                    ✓ Verified  ✓ Installed│
│  text-embedding-3-small/large via OpenAI API               │
│                                                             │
│  [Configure]  [Uninstall]                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- `PluginInstallButton` - Triggers install, shows progress
- `PluginInstallProgress` - WebSocket-connected progress display
- `usePluginInstall` hook - Manages install state and WebSocket

#### 7.5 Tasks

| Task | Description | Files Changed |
|------|-------------|---------------|
| Add plugins volume | Docker Compose config for persistent plugins | `docker-compose.yml`, `docker-compose.dev.yml` |
| Create installer service | pip subprocess with progress streaming | `webui/services/plugin_installer.py` |
| Add install endpoints | REST + WebSocket for installation | `webui/api/v2/plugins.py` |
| Add uninstall support | Remove plugin packages | `webui/services/plugin_installer.py` |
| Implement hot-reload | Reload entry points after install | `shared/plugins/loader.py` |
| Frontend install button | Replace "copy command" with "Install" | `AvailablePluginCard.tsx` |
| Frontend progress UI | WebSocket progress display | `PluginInstallProgress.tsx` |
| Handle errors gracefully | Dependency conflicts, network errors | Throughout |
| Add install permissions | Only admins can install plugins | `webui/api/v2/plugins.py` |
| Update documentation | Docker volume setup instructions | `docs/PLUGIN_INSTALLATION.md` |

#### 7.6 Security Considerations

| Risk | Mitigation |
|------|------------|
| Arbitrary code execution | Only install from registry (allowlist) |
| Dependency conflicts | Use `--no-deps`, install deps separately with conflict check |
| Admin-only | Require admin role for install/uninstall |
| Audit trail | Log all install/uninstall operations |
| Malicious packages | Only allow "verified" plugins by default, warn for unverified |

#### 7.7 Limitations

- **No automatic updates** - Users must manually update plugins
- **No rollback** - If install breaks something, manual intervention needed
- **Single version** - Can't have multiple versions of same plugin
- **Restart may still be needed** - Some plugins may require container restart for full initialization

**Acceptance Criteria:**
- [ ] Plugins can be installed from UI with one click
- [ ] Installation progress shown in real-time
- [ ] Installed plugins persist across container restarts
- [ ] Plugins can be uninstalled from UI
- [ ] Hot-reload works for most plugin types (or clear "restart required" messaging)
- [ ] Only admins can install/uninstall
- [ ] Docker Compose includes plugins volume by default

---

## Success Metrics

> **Adjusted for experimentation-first scope (2026-01-01)**

| Metric | Target | Notes |
|--------|--------|-------|
| Plugin types supported | 5 | embedding, chunking, connector, reranker, preprocessor |
| Plugins written by maintainer | 3-5 | OpenAI embeddings, Cohere reranker, Tesseract OCR, etc. |
| Time to write new plugin | < 2 hours | From idea to working plugin |
| Plugin UI functional | Yes | Browse, enable/disable, configure |
| Registry browsable | Yes | Slim YAML registry with UI |
| External contributions | Nice to have | Not a primary goal |

---

## Appendix: Current File Locations

| Component | Location |
|-----------|----------|
| **Unified Plugin System (Phase 1)** | |
| Unified plugin base | `packages/shared/plugins/base.py` |
| Plugin protocols | `packages/shared/plugins/protocols.py` |
| Plugin manifest | `packages/shared/plugins/manifest.py` |
| Plugin adapters | `packages/shared/plugins/adapters.py` |
| Plugin exceptions | `packages/shared/plugins/exceptions.py` |
| Unified plugin loader | `packages/shared/plugins/loader.py` |
| Plugin registry | `packages/shared/plugins/registry.py` |
| Plugin type bases | `packages/shared/plugins/types/` |
| Plugin service | `packages/webui/services/plugin_service.py` |
| Plugin API | `packages/webui/api/v2/plugins.py` |
| Plugin CLI | `packages/cli/src/semantik_cli/` |
| Connector registry | `packages/webui/services/connector_registry.py` |
| **Plugin Registry (Phase 4)** | |
| Compatibility checking | `packages/shared/plugins/compatibility.py` |
| Registry client | `packages/shared/plugins/registry_client.py` |
| Bundled registry | `packages/shared/plugins/data/registry.yaml` |
| Available plugins API | `packages/webui/api/v2/plugins.py` |
| Available plugins UI | `apps/webui-react/src/components/plugins/AvailablePluginsTab.tsx` |
| Available plugin card | `apps/webui-react/src/components/plugins/AvailablePluginCard.tsx` |
| Registry documentation | `docs/PLUGIN_REGISTRY.md` |
| **Legacy (still in use)** | |
| Embedding plugin base | `packages/shared/embedding/plugin_base.py` |
| Embedding provider registry | `packages/shared/embedding/provider_registry.py` |
| Embedding factory | `packages/shared/embedding/factory.py` |
| Chunking strategy base | `packages/shared/chunking/domain/services/chunking_strategies/base.py` |
| Chunking factory | `packages/shared/chunking/unified/factory.py` |
| Connector base | `packages/shared/connectors/base.py` |
| Connector factory | `packages/webui/services/connector_factory.py` |
| Plugin documentation | `docs/EMBEDDING_PLUGINS.md`, `docs/CHUNKING_PLUGINS.md`, `docs/CONNECTORS.md` |
|


