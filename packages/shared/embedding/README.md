# Embedding Service

Embedding generation for Semantik. Supports sentence-transformers and Qwen models with quantization.

## Structure

```
embedding/
├── base.py          # Abstract base class
├── dense.py         # Dense embeddings (sentence-transformers, Qwen)
├── service.py       # Singleton management
├── models.py        # Model configurations
└── __init__.py
```

## Usage

```python
from shared.embedding import embedding_service

# Load and generate
embedding_service.load_model("BAAI/bge-large-en-v1.5", quantization="float16")
embeddings = embedding_service.generate_embeddings(texts, model_name="BAAI/bge-large-en-v1.5")
```

### Async

```python
from shared.embedding import get_embedding_service, initialize_embedding_service

service = await initialize_embedding_service("Qwen/Qwen3-Embedding-0.6B", quantization="float16")
embeddings = await service.embed_texts(["Query text"], instruction="Represent this text for similarity search")
```

## Adding Models

```python
from shared.embedding.models import ModelConfig, add_model_config

add_model_config(ModelConfig(
    name="your-org/your-model",
    dimension=768,
    max_sequence_length=512,
    supports_quantization=True,
    recommended_quantization="float16"
))
```

## Testing

```bash
pytest tests/test_embedding_*.py -v
```
