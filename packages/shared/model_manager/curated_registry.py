"""Curated model aggregation for the model manager.

Aggregates model configurations from multiple sources into a unified registry
for use by the model manager UI and APIs.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache


class ModelType(str, Enum):
    """Type of model in the curated registry."""

    EMBEDDING = "embedding"
    LLM = "llm"
    RERANKER = "reranker"
    SPLADE = "splade"


@dataclass(frozen=True)
class CuratedModel:
    """A curated model in the model manager registry.

    Frozen dataclass for hashability (enables LRU caching).
    """

    id: str
    name: str
    description: str
    model_type: ModelType
    memory_mb: dict[str, int] = field(default_factory=dict)

    # Embedding-specific fields
    dimension: int | None = None
    max_sequence_length: int | None = None
    pooling_method: str | None = None
    is_asymmetric: bool = False
    query_prefix: str = ""
    document_prefix: str = ""
    default_query_instruction: str = ""

    # LLM-specific fields
    context_window: int | None = None

    def __hash__(self) -> int:
        """Custom hash for frozen dataclass with dict field."""
        return hash(
            (
                self.id,
                self.model_type,
                tuple(sorted(self.memory_mb.items())) if self.memory_mb else (),
            )
        )


def _aggregate_embedding_models() -> list[CuratedModel]:
    """Aggregate embedding models from MODEL_CONFIGS."""
    from shared.embedding.models import MODEL_CONFIGS

    models = []
    for model_id, config in MODEL_CONFIGS.items():
        models.append(
            CuratedModel(
                id=model_id,
                name=config.name,
                description=config.description,
                model_type=ModelType.EMBEDDING,
                memory_mb=dict(config.memory_estimate) if config.memory_estimate else {},
                dimension=config.dimension,
                max_sequence_length=config.max_sequence_length,
                pooling_method=config.pooling_method,
                is_asymmetric=config.is_asymmetric,
                query_prefix=config.query_prefix,
                document_prefix=config.document_prefix,
                default_query_instruction=config.default_query_instruction,
            )
        )
    return models


def _aggregate_llm_models() -> list[CuratedModel]:
    """Aggregate local LLM models from model_registry."""
    from shared.llm.model_registry import get_models_by_provider

    models = []
    for model_info in get_models_by_provider("local"):
        models.append(
            CuratedModel(
                id=model_info.id,
                name=model_info.display_name,
                description=model_info.description,
                model_type=ModelType.LLM,
                memory_mb=dict(model_info.memory_mb) if model_info.memory_mb else {},
                context_window=model_info.context_window,
            )
        )
    return models


def _aggregate_reranker_models() -> list[CuratedModel]:
    """Aggregate reranker models from qwen3_reranker plugin."""
    from shared.plugins.builtins.qwen3_reranker import SUPPORTED_MODELS

    # Try to get memory estimates from vecpipe
    memory_estimates: dict[str, dict[str, int]] = {}
    try:
        from vecpipe.memory_utils import MODEL_MEMORY_REQUIREMENTS

        for (model_id, quant), memory_mb in MODEL_MEMORY_REQUIREMENTS.items():
            if model_id not in memory_estimates:
                memory_estimates[model_id] = {}
            memory_estimates[model_id][quant] = memory_mb
    except ImportError:
        pass

    models = []
    for model_id in SUPPORTED_MODELS:
        # Determine model size from name for description
        if "0.6B" in model_id:
            description = "Cross-encoder reranker, 0.6B parameters"
        elif "4B" in model_id:
            description = "Cross-encoder reranker, 4B parameters"
        elif "8B" in model_id:
            description = "Cross-encoder reranker, 8B parameters"
        else:
            description = "Cross-encoder reranker"

        models.append(
            CuratedModel(
                id=model_id,
                name=model_id.split("/")[-1],
                description=description,
                model_type=ModelType.RERANKER,
                memory_mb=memory_estimates.get(model_id, {}),
            )
        )
    return models


def _aggregate_splade_models() -> list[CuratedModel]:
    """Aggregate SPLADE models from splade_indexer plugin."""
    from shared.plugins.builtins.splade_indexer import DEFAULT_MODEL

    # Curated SPLADE models
    splade_model_ids = [
        DEFAULT_MODEL,  # naver/splade-cocondenser-ensembledistil
        "naver/splade-v3",
    ]

    # Try to get memory estimates from vecpipe
    memory_estimates: dict[str, dict[str, int]] = {}
    try:
        from vecpipe.memory_utils import MODEL_MEMORY_REQUIREMENTS

        for (model_id, quant), memory_mb in MODEL_MEMORY_REQUIREMENTS.items():
            if model_id not in memory_estimates:
                memory_estimates[model_id] = {}
            memory_estimates[model_id][quant] = memory_mb
    except ImportError:
        pass

    models = []
    for model_id in splade_model_ids:
        if model_id == DEFAULT_MODEL:
            description = "SPLADE learned sparse model (default)"
        else:
            description = "SPLADE v3 learned sparse model"

        models.append(
            CuratedModel(
                id=model_id,
                name=model_id.split("/")[-1],
                description=description,
                model_type=ModelType.SPLADE,
                memory_mb=memory_estimates.get(model_id, {}),
            )
        )
    return models


def _merge_models(existing: CuratedModel, new: CuratedModel) -> CuratedModel:
    """Merge two CuratedModel instances, preferring existing (first-seen) values.

    For memory_mb dicts, combines keys from both (existing values take precedence).

    Args:
        existing: The model seen first (higher precedence).
        new: The model seen later (lower precedence).

    Returns:
        Merged CuratedModel with combined memory estimates.
    """
    # Merge memory_mb dicts: existing values take precedence
    merged_memory = dict(new.memory_mb)
    merged_memory.update(existing.memory_mb)

    # Use all fields from existing, but with merged memory_mb
    return CuratedModel(
        id=existing.id,
        name=existing.name,
        description=existing.description,
        model_type=existing.model_type,
        memory_mb=merged_memory,
        dimension=existing.dimension,
        max_sequence_length=existing.max_sequence_length,
        pooling_method=existing.pooling_method,
        is_asymmetric=existing.is_asymmetric,
        query_prefix=existing.query_prefix,
        document_prefix=existing.document_prefix,
        default_query_instruction=existing.default_query_instruction,
        context_window=existing.context_window,
    )


@lru_cache(maxsize=1)
def get_curated_models() -> tuple[CuratedModel, ...]:
    """Get all curated models, aggregated from multiple sources.

    Returns an immutable tuple sorted by (model_type, name) for deterministic ordering.
    Uses LRU cache to avoid repeated aggregation.

    De-duplicates by (id, model_type) with first-seen precedence. If the same model
    appears in multiple sources, fields are merged (first source values take precedence,
    memory_mb dicts are combined with first source values winning on conflicts).

    Returns:
        Tuple of CuratedModel instances.
    """
    # Track seen models by (id, model_type) for de-duplication
    seen: dict[tuple[str, ModelType], CuratedModel] = {}

    # Aggregate from all sources in precedence order
    for source_models in [
        _aggregate_embedding_models(),
        _aggregate_llm_models(),
        _aggregate_reranker_models(),
        _aggregate_splade_models(),
    ]:
        for model in source_models:
            key = (model.id, model.model_type)
            if key not in seen:
                seen[key] = model
            else:
                # Merge with existing, preserving first-seen precedence
                seen[key] = _merge_models(seen[key], model)

    # Convert to list and sort by (model_type.value, name) for deterministic ordering
    models = list(seen.values())
    models.sort(key=lambda m: (m.model_type.value, m.name))

    return tuple(models)


def get_curated_model_ids() -> set[str]:
    """Get the set of all curated model IDs.

    Returns:
        Set of model ID strings.
    """
    return {model.id for model in get_curated_models()}


def get_models_by_type(model_type: ModelType) -> tuple[CuratedModel, ...]:
    """Get curated models filtered by type.

    Args:
        model_type: The type of models to return.

    Returns:
        Tuple of CuratedModel instances of the specified type.
    """
    return tuple(m for m in get_curated_models() if m.model_type == model_type)


def get_model_by_id(model_id: str) -> CuratedModel | None:
    """Get a specific curated model by ID.

    Args:
        model_id: The model ID to look up.

    Returns:
        CuratedModel if found, None otherwise.
    """
    for model in get_curated_models():
        if model.id == model_id:
            return model
    return None


__all__ = [
    "CuratedModel",
    "ModelType",
    "get_curated_models",
    "get_curated_model_ids",
    "get_models_by_type",
    "get_model_by_id",
]
