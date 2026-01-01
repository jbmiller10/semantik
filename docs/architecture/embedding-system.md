# Embedding System Architecture

> **Location:** `packages/shared/embedding/`

## Overview

Plugin-based embedding provider system supporting multiple models with automatic detection, asymmetric modes, and quantization options.

## Directory Structure

```
packages/shared/embedding/
├── plugin_base.py       # BaseEmbeddingPlugin abstract class
├── provider_registry.py # Provider metadata registry
├── factory.py           # EmbeddingProviderFactory
└── providers/
    ├── dense_local.py   # Local transformer models
    └── mock.py          # Testing mock provider

packages/shared/plugins/
├── loader.py            # Unified entry point discovery
├── registry.py          # Plugin registry
├── types.py             # EmbeddingMode enum
└── manifest.py          # PluginManifest metadata
```

## Core Components

### EmbeddingMode
```python
class EmbeddingMode(str, Enum):
    QUERY = "query"      # Search queries (applies prefixes/instructions)
    DOCUMENT = "document" # Document indexing (typically no prefix)
```

### EmbeddingProviderDefinition
```python
@dataclass
class EmbeddingProviderDefinition:
    id: str                    # Unique provider identifier
    name: str                  # Display name
    description: str           # Provider description
    supported_models: list[str] # Model name patterns
    supports_asymmetric: bool  # Supports query/document modes
    default_dimension: int     # Default embedding dimension
    supports_quantization: bool
    recommended_quantization: str | None
```

### BaseEmbeddingPlugin
```python
class BaseEmbeddingPlugin(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def definition(self) -> EmbeddingProviderDefinition:
        """Return provider definition."""
        pass

    @abstractmethod
    async def embed_texts(
        self,
        texts: list[str],
        mode: EmbeddingMode = EmbeddingMode.QUERY
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    async def get_dimension(self, model_name: str) -> int:
        """Get embedding dimension for model."""
        pass

    def supports_model(self, model_name: str) -> bool:
        """Check if provider supports the model."""
        for pattern in self.definition.supported_models:
            if fnmatch.fnmatch(model_name, pattern):
                return True
        return False
```

## Built-in Providers

### DenseLocalEmbeddingProvider
Local transformer models using sentence-transformers or direct HuggingFace.

```python
class DenseLocalEmbeddingProvider(BaseEmbeddingPlugin):
    def __init__(self, model_name: str, quantization: str = "float16"):
        self.model_name = model_name
        self.quantization = quantization
        self._model = None
        self._tokenizer = None

    @property
    def definition(self) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            id="dense-local",
            name="Local Dense Embeddings",
            description="Local transformer models (Qwen, BGE, etc.)",
            supported_models=[
                "Qwen/Qwen3-Embedding-*",
                "BAAI/bge-*",
                "sentence-transformers/*"
            ],
            supports_asymmetric=True,
            default_dimension=384,
            supports_quantization=True,
            recommended_quantization="float16"
        )

    async def embed_texts(
        self,
        texts: list[str],
        mode: EmbeddingMode = EmbeddingMode.QUERY
    ) -> list[list[float]]:
        model = await self._ensure_model_loaded()

        # Apply asymmetric processing
        processed_texts = self._apply_mode_processing(texts, mode)

        # Batch processing
        embeddings = []
        batch_size = self._get_batch_size()

        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())

        return embeddings

    def _apply_mode_processing(
        self,
        texts: list[str],
        mode: EmbeddingMode
    ) -> list[str]:
        """Apply query/document prefixes for asymmetric models."""
        config = self._get_model_config()

        if not config.is_asymmetric:
            return texts

        if mode == EmbeddingMode.QUERY:
            prefix = config.query_prefix or ""
            instruction = config.default_query_instruction or ""
            return [f"{prefix}{instruction}{text}" for text in texts]
        else:
            prefix = config.document_prefix or ""
            return [f"{prefix}{text}" for text in texts]

    async def _ensure_model_loaded(self):
        """Lazy load model on first use."""
        if self._model is None:
            self._model = await self._load_model()
        return self._model

    async def _load_model(self):
        """Load model with specified quantization."""
        if self.quantization == "int8":
            return self._load_int8_model()
        elif self.quantization == "float16":
            return self._load_fp16_model()
        else:
            return self._load_fp32_model()

    def _get_batch_size(self) -> int:
        """Get batch size based on model and quantization."""
        batch_sizes = {
            ("0.6B", "int8"): 256,
            ("0.6B", "float16"): 128,
            ("4B", "float32"): 16,
            ("8B", "float16"): 8,
        }
        # Determine model size from name
        size = self._detect_model_size()
        return batch_sizes.get((size, self.quantization), 64)
```

### MockEmbeddingProvider
Deterministic embeddings for testing.

```python
class MockEmbeddingProvider(BaseEmbeddingPlugin):
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    @property
    def definition(self) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            id="mock",
            name="Mock Embeddings",
            description="Deterministic embeddings for testing",
            supported_models=["mock", "test"],
            supports_asymmetric=False,
            default_dimension=self.dimension,
            supports_quantization=False,
            recommended_quantization=None
        )

    async def embed_texts(
        self,
        texts: list[str],
        mode: EmbeddingMode = EmbeddingMode.QUERY
    ) -> list[list[float]]:
        embeddings = []
        for text in texts:
            # Deterministic based on text content
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.randn(self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        return embeddings

    async def get_dimension(self, model_name: str) -> int:
        return self.dimension
```

## Provider Registry

```python
class ProviderRegistry:
    """Registry for embedding provider metadata."""

    _providers: dict[str, EmbeddingProviderDefinition] = {}
    _model_to_provider: dict[str, str] = {}

    @classmethod
    def register(cls, definition: EmbeddingProviderDefinition) -> None:
        """Register a provider definition."""
        cls._providers[definition.id] = definition

        # Build model-to-provider mapping
        for pattern in definition.supported_models:
            cls._model_to_provider[pattern] = definition.id

    @classmethod
    @lru_cache(maxsize=100)
    def get_provider_for_model(cls, model_name: str) -> str | None:
        """Find provider ID that supports the model."""
        for pattern, provider_id in cls._model_to_provider.items():
            if fnmatch.fnmatch(model_name, pattern):
                return provider_id
        return None

    @classmethod
    def list_providers(cls) -> list[EmbeddingProviderDefinition]:
        """List all registered providers."""
        return list(cls._providers.values())
```

## Provider Factory

```python
class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    _plugins: dict[str, type[BaseEmbeddingPlugin]] = {}

    @classmethod
    def register_plugin(cls, plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Register a plugin class."""
        definition = plugin_class().definition
        cls._plugins[definition.id] = plugin_class
        ProviderRegistry.register(definition)

    @classmethod
    def create_provider(
        cls,
        model_name: str,
        quantization: str = "float16"
    ) -> BaseEmbeddingPlugin:
        """Create provider instance for model."""
        # Auto-detect provider
        provider_id = ProviderRegistry.get_provider_for_model(model_name)

        if not provider_id:
            raise ValueError(f"No provider found for model: {model_name}")

        plugin_class = cls._plugins.get(provider_id)
        if not plugin_class:
            raise ValueError(f"Plugin not loaded: {provider_id}")

        # Create instance with model configuration
        if provider_id == "dense-local":
            return plugin_class(model_name=model_name, quantization=quantization)
        else:
            return plugin_class()

    @classmethod
    def list_models(cls) -> dict[str, dict]:
        """List all available models with metadata."""
        models = {}
        for provider_id, plugin_class in cls._plugins.items():
            plugin = plugin_class()
            for pattern in plugin.definition.supported_models:
                models[pattern] = {
                    "provider": provider_id,
                    "dimension": plugin.definition.default_dimension,
                    "supports_quantization": plugin.definition.supports_quantization
                }
        return models
```

## Plugin Loading

```python
from shared.plugins.loader import load_plugins

def load_embedding_plugins() -> None:
    """Discover and load embedding plugins via the unified loader."""
    load_plugins(plugin_types={"embedding"})
```

## Model Configuration

```python
@dataclass
class ModelConfig:
    """Configuration for a specific embedding model."""
    model_name: str
    dimension: int
    is_asymmetric: bool = False
    query_prefix: str | None = None
    document_prefix: str | None = None
    default_query_instruction: str | None = None
    max_sequence_length: int = 8192

# Known model configurations
MODEL_CONFIGS = {
    "Qwen/Qwen3-Embedding-0.6B": ModelConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        is_asymmetric=True,
        query_prefix="query: ",
        document_prefix="",
        default_query_instruction="Represent this query for retrieval: "
    ),
    "BAAI/bge-large-en-v1.5": ModelConfig(
        model_name="BAAI/bge-large-en-v1.5",
        dimension=1024,
        is_asymmetric=True,
        query_prefix="",
        document_prefix="",
        default_query_instruction="Represent this sentence for searching: "
    ),
}
```

## API Endpoints

```python
@router.get("/embedding/providers")
async def list_providers() -> list[EmbeddingProviderDefinition]:
    """List all available embedding providers."""
    return ProviderRegistry.list_providers()

@router.get("/embedding/providers/{provider_id}")
async def get_provider(provider_id: str) -> EmbeddingProviderDefinition:
    """Get details for a specific provider."""
    providers = ProviderRegistry.list_providers()
    for p in providers:
        if p.id == provider_id:
            return p
    raise HTTPException(404, "Provider not found")

@router.get("/embedding/models")
async def list_models() -> dict[str, dict]:
    """List all available models."""
    return EmbeddingProviderFactory.list_models()

@router.get("/embedding/models/{model_name}/supported")
async def check_model_support(model_name: str) -> dict:
    """Check if model is supported."""
    provider_id = ProviderRegistry.get_provider_for_model(model_name)
    return {
        "model": model_name,
        "supported": provider_id is not None,
        "provider": provider_id
    }
```

## Usage Examples

```python
# Create provider for specific model
provider = EmbeddingProviderFactory.create_provider(
    "Qwen/Qwen3-Embedding-0.6B",
    quantization="float16"
)

# Generate query embeddings
query_embeddings = await provider.embed_texts(
    ["What is authentication?"],
    mode=EmbeddingMode.QUERY
)

# Generate document embeddings
doc_embeddings = await provider.embed_texts(
    ["Authentication is the process of..."],
    mode=EmbeddingMode.DOCUMENT
)

# Get model dimension
dimension = await provider.get_dimension("Qwen/Qwen3-Embedding-0.6B")
```

## Extension Points

### Adding a New Provider
1. Create provider class extending `BaseEmbeddingPlugin`
2. Implement required methods
3. Register via entry point or direct registration
4. Add model configurations
5. Write tests

### Adding External Plugin
```python
# In your package's pyproject.toml
[project.entry-points."semantik.plugins"]
my_provider = "my_package.embedding:MyEmbeddingProvider"

# Your provider class
class MyEmbeddingProvider(BaseEmbeddingPlugin):
    @property
    def definition(self):
        return EmbeddingProviderDefinition(
            id="my-provider",
            name="My Custom Provider",
            ...
        )

    async def embed_texts(self, texts, mode):
        # Your implementation
        ...
```
