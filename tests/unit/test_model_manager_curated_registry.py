"""Unit tests for shared.model_manager.curated_registry."""

import pytest

from shared.model_manager.curated_registry import (
    CuratedModel,
    ModelType,
    get_curated_model_ids,
    get_curated_models,
    get_model_by_id,
    get_models_by_type,
)


class TestGetCuratedModels:
    """Tests for get_curated_models function."""

    def test_returns_tuple(self) -> None:
        """get_curated_models should return an immutable tuple."""
        models = get_curated_models()
        assert isinstance(models, tuple)

    def test_returns_curated_model_instances(self) -> None:
        """All items should be CuratedModel instances."""
        models = get_curated_models()
        assert all(isinstance(m, CuratedModel) for m in models)

    def test_returns_at_least_one_embedding_model(self) -> None:
        """Should return at least one embedding model."""
        models = get_curated_models()
        embedding_models = [m for m in models if m.model_type == ModelType.EMBEDDING]
        assert len(embedding_models) >= 1

    def test_qwen3_embedding_model_exists(self) -> None:
        """Qwen/Qwen3-Embedding-0.6B should be in the curated list."""
        models = get_curated_models()
        model_ids = [m.id for m in models]
        assert "Qwen/Qwen3-Embedding-0.6B" in model_ids

    def test_returns_at_least_one_llm_model(self) -> None:
        """Should return at least one local LLM model."""
        models = get_curated_models()
        llm_models = [m for m in models if m.model_type == ModelType.LLM]
        assert len(llm_models) >= 1

    def test_returns_at_least_one_reranker_model(self) -> None:
        """Should return at least one reranker model."""
        models = get_curated_models()
        reranker_models = [m for m in models if m.model_type == ModelType.RERANKER]
        assert len(reranker_models) >= 1

    def test_returns_at_least_one_splade_model(self) -> None:
        """Should return at least one SPLADE model."""
        models = get_curated_models()
        splade_models = [m for m in models if m.model_type == ModelType.SPLADE]
        assert len(splade_models) >= 1

    def test_no_duplicate_id_type_pairs(self) -> None:
        """No duplicate (id, model_type) pairs should exist."""
        models = get_curated_models()
        seen: set[tuple[str, ModelType]] = set()
        for model in models:
            key = (model.id, model.model_type)
            assert key not in seen, f"Duplicate found: {key}"
            seen.add(key)

    def test_deterministic_ordering(self) -> None:
        """Repeated calls should return the same order."""
        # Clear the cache to test fresh aggregation
        get_curated_models.cache_clear()

        first_call = get_curated_models()
        second_call = get_curated_models()

        assert first_call == second_call
        assert [m.id for m in first_call] == [m.id for m in second_call]

    def test_sorted_by_type_then_name(self) -> None:
        """Models should be sorted by (model_type.value, name)."""
        models = get_curated_models()

        for i in range(len(models) - 1):
            curr = models[i]
            next_model = models[i + 1]

            # Should be sorted by type first, then name
            curr_key = (curr.model_type.value, curr.name)
            next_key = (next_model.model_type.value, next_model.name)
            assert curr_key <= next_key, f"Not sorted: {curr_key} > {next_key}"

    def test_embedding_models_have_positive_dimension(self) -> None:
        """Embedding models should have dimension > 0."""
        models = get_curated_models()
        embedding_models = [m for m in models if m.model_type == ModelType.EMBEDDING]

        for model in embedding_models:
            assert model.dimension is not None, f"{model.id} has no dimension"
            assert model.dimension > 0, f"{model.id} has invalid dimension: {model.dimension}"

    def test_llm_models_have_positive_context_window(self) -> None:
        """LLM models should have context_window > 0."""
        models = get_curated_models()
        llm_models = [m for m in models if m.model_type == ModelType.LLM]

        for model in llm_models:
            assert model.context_window is not None, f"{model.id} has no context_window"
            assert model.context_window > 0, f"{model.id} has invalid context_window: {model.context_window}"


class TestGetCuratedModelIds:
    """Tests for get_curated_model_ids function."""

    def test_returns_set(self) -> None:
        """Should return a set."""
        ids = get_curated_model_ids()
        assert isinstance(ids, set)

    def test_returns_strings(self) -> None:
        """All IDs should be strings."""
        ids = get_curated_model_ids()
        assert all(isinstance(id, str) for id in ids)

    def test_contains_known_model(self) -> None:
        """Should contain known model IDs."""
        ids = get_curated_model_ids()
        assert "Qwen/Qwen3-Embedding-0.6B" in ids


class TestGetModelsByType:
    """Tests for get_models_by_type function."""

    def test_returns_tuple(self) -> None:
        """Should return an immutable tuple."""
        models = get_models_by_type(ModelType.EMBEDDING)
        assert isinstance(models, tuple)

    def test_filters_by_type(self) -> None:
        """Should only return models of the specified type."""
        for model_type in ModelType:
            models = get_models_by_type(model_type)
            for model in models:
                assert model.model_type == model_type

    def test_embedding_type_has_models(self) -> None:
        """EMBEDDING type should have at least one model."""
        models = get_models_by_type(ModelType.EMBEDDING)
        assert len(models) >= 1


class TestGetModelById:
    """Tests for get_model_by_id function."""

    def test_returns_model_for_known_id(self) -> None:
        """Should return CuratedModel for known ID."""
        model = get_model_by_id("Qwen/Qwen3-Embedding-0.6B")
        assert model is not None
        assert model.id == "Qwen/Qwen3-Embedding-0.6B"

    def test_returns_none_for_unknown_id(self) -> None:
        """Should return None for unknown ID."""
        model = get_model_by_id("nonexistent/model")
        assert model is None


class TestCuratedModelDataclass:
    """Tests for CuratedModel dataclass."""

    def test_frozen(self) -> None:
        """CuratedModel should be frozen (immutable)."""
        model = CuratedModel(
            id="test/model",
            name="Test Model",
            description="Test description",
            model_type=ModelType.EMBEDDING,
        )
        with pytest.raises(AttributeError):
            model.id = "changed"  # type: ignore[misc]

    def test_hashable(self) -> None:
        """CuratedModel should be hashable."""
        model = CuratedModel(
            id="test/model",
            name="Test Model",
            description="Test description",
            model_type=ModelType.EMBEDDING,
            memory_mb={"float16": 1000},
        )
        # Should not raise
        hash(model)
        # Should be usable in sets
        _ = {model}


class TestModelTypeEnum:
    """Tests for ModelType enum."""

    def test_has_expected_values(self) -> None:
        """Should have all expected values."""
        assert ModelType.EMBEDDING.value == "embedding"
        assert ModelType.LLM.value == "llm"
        assert ModelType.RERANKER.value == "reranker"
        assert ModelType.SPLADE.value == "splade"
