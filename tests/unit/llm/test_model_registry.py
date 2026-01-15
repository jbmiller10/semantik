"""Tests for LLM model registry."""

import pytest

from shared.llm.model_registry import (
    ModelInfo,
    get_all_models,
    get_default_model,
    get_model_by_id,
    get_models_by_provider,
    load_model_registry,
)


class TestLoadModelRegistry:
    """Tests for load_model_registry function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        registry = load_model_registry()
        assert isinstance(registry, dict)

    def test_has_anthropic_provider(self):
        """Registry contains anthropic provider."""
        registry = load_model_registry()
        assert "anthropic" in registry

    def test_has_openai_provider(self):
        """Registry contains openai provider."""
        registry = load_model_registry()
        assert "openai" in registry

    def test_anthropic_models_are_model_info(self):
        """Anthropic models are ModelInfo instances."""
        registry = load_model_registry()
        for model in registry["anthropic"]:
            assert isinstance(model, ModelInfo)

    def test_openai_models_are_model_info(self):
        """OpenAI models are ModelInfo instances."""
        registry = load_model_registry()
        for model in registry["openai"]:
            assert isinstance(model, ModelInfo)

    def test_model_has_required_fields(self):
        """Models have all required fields."""
        registry = load_model_registry()
        for provider_models in registry.values():
            for model in provider_models:
                assert model.id
                assert model.name
                assert model.display_name
                assert model.provider
                assert model.tier_recommendation in ("high", "low")
                assert model.context_window > 0
                assert model.description

    def test_is_cached(self):
        """Registry is cached (returns same instance)."""
        registry1 = load_model_registry()
        registry2 = load_model_registry()
        assert registry1 is registry2

    def test_has_local_provider(self):
        """Registry contains local provider."""
        registry = load_model_registry()
        assert "local" in registry

    def test_local_models_are_model_info(self):
        """Local models are ModelInfo instances."""
        registry = load_model_registry()
        for model in registry["local"]:
            assert isinstance(model, ModelInfo)

    def test_local_models_have_memory_mb(self):
        """Local models have memory_mb field populated."""
        registry = load_model_registry()
        for model in registry["local"]:
            assert model.memory_mb is not None
            assert "int8" in model.memory_mb
            assert "int4" in model.memory_mb


class TestGetDefaultModel:
    """Tests for get_default_model function."""

    def test_anthropic_high_tier(self):
        """Returns correct model for anthropic high tier."""
        model_id = get_default_model("anthropic", "high")
        assert model_id  # Not empty
        assert isinstance(model_id, str)

    def test_anthropic_low_tier(self):
        """Returns correct model for anthropic low tier."""
        model_id = get_default_model("anthropic", "low")
        assert model_id
        assert isinstance(model_id, str)

    def test_openai_high_tier(self):
        """Returns correct model for openai high tier."""
        model_id = get_default_model("openai", "high")
        assert model_id
        assert isinstance(model_id, str)

    def test_openai_low_tier(self):
        """Returns correct model for openai low tier."""
        model_id = get_default_model("openai", "low")
        assert model_id
        assert isinstance(model_id, str)

    def test_unknown_provider_raises(self):
        """Raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="unknown"):
            get_default_model("unknown", "high")

    def test_unknown_tier_raises(self):
        """Raises ValueError for unknown tier."""
        with pytest.raises(ValueError, match="unknown"):
            get_default_model("anthropic", "unknown")

    def test_different_tiers_return_different_models(self):
        """High and low tiers return different models."""
        high = get_default_model("anthropic", "high")
        low = get_default_model("anthropic", "low")
        assert high != low


class TestGetAllModels:
    """Tests for get_all_models function."""

    def test_returns_list(self):
        """Returns a list."""
        models = get_all_models()
        assert isinstance(models, list)

    def test_returns_model_info_instances(self):
        """All items are ModelInfo instances."""
        models = get_all_models()
        for model in models:
            assert isinstance(model, ModelInfo)

    def test_includes_anthropic_models(self):
        """Includes models from anthropic provider."""
        models = get_all_models()
        anthropic_models = [m for m in models if m.provider == "anthropic"]
        assert len(anthropic_models) > 0

    def test_includes_openai_models(self):
        """Includes models from openai provider."""
        models = get_all_models()
        openai_models = [m for m in models if m.provider == "openai"]
        assert len(openai_models) > 0

    def test_flat_list(self):
        """Returns flat list (not nested)."""
        models = get_all_models()
        for model in models:
            assert not isinstance(model, list)

    def test_includes_local_models(self):
        """Includes models from local provider."""
        models = get_all_models()
        local_models = [m for m in models if m.provider == "local"]
        assert len(local_models) > 0


class TestGetModelsByProvider:
    """Tests for get_models_by_provider function."""

    def test_returns_list(self):
        """Returns a list for valid provider."""
        models = get_models_by_provider("anthropic")
        assert isinstance(models, list)

    def test_returns_model_info_instances(self):
        """All items are ModelInfo instances."""
        models = get_models_by_provider("anthropic")
        for model in models:
            assert isinstance(model, ModelInfo)

    def test_returns_empty_for_unknown_provider(self):
        """Returns empty list for unknown provider."""
        models = get_models_by_provider("unknown_provider")
        assert models == []

    def test_returns_local_models(self):
        """Returns local models with memory_mb."""
        models = get_models_by_provider("local")
        assert len(models) > 0
        for model in models:
            assert model.provider == "local"
            assert model.memory_mb is not None

    def test_only_returns_specified_provider(self):
        """Returns only models from specified provider."""
        models = get_models_by_provider("openai")
        for model in models:
            assert model.provider == "openai"

    def test_anthropic_provider_has_models(self):
        """Anthropic provider returns models."""
        models = get_models_by_provider("anthropic")
        assert len(models) > 0
        for model in models:
            assert model.provider == "anthropic"


class TestGetModelById:
    """Tests for get_model_by_id function."""

    def test_finds_existing_model(self):
        """Can find a model by ID."""
        # Get a known model ID from the registry
        all_models = get_all_models()
        assert len(all_models) > 0

        first_model = all_models[0]
        found = get_model_by_id(first_model.id)

        assert found is not None
        assert found.id == first_model.id

    def test_returns_none_for_unknown_model(self):
        """Returns None for unknown model ID."""
        result = get_model_by_id("unknown-model-that-does-not-exist")
        assert result is None

    def test_returns_correct_model_info(self):
        """Returns complete ModelInfo for found model."""
        all_models = get_all_models()
        for model in all_models:
            found = get_model_by_id(model.id)
            assert found is not None
            assert found.id == model.id
            assert found.provider == model.provider
            assert found.context_window == model.context_window

    def test_finds_model_with_provider_filter(self):
        """Can find model when filtering by provider."""
        local_models = get_models_by_provider("local")
        assert len(local_models) > 0
        first_local = local_models[0]
        found = get_model_by_id(first_local.id, provider="local")
        assert found is not None
        assert found.id == first_local.id
        assert found.provider == "local"

    def test_returns_none_for_wrong_provider(self):
        """Returns None when model exists but provider doesn't match."""
        anthropic_models = get_models_by_provider("anthropic")
        assert len(anthropic_models) > 0
        first_anthropic = anthropic_models[0]
        # Try to find it with wrong provider
        found = get_model_by_id(first_anthropic.id, provider="openai")
        assert found is None

    def test_backward_compatible_without_provider(self):
        """Works without provider parameter (backward compatible)."""
        all_models = get_all_models()
        first_model = all_models[0]
        # Should find without provider
        found = get_model_by_id(first_model.id)
        assert found is not None
        assert found.id == first_model.id


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_is_frozen(self):
        """ModelInfo is immutable."""
        model = ModelInfo(
            id="test-model",
            name="Test",
            display_name="Test Model",
            provider="test",
            tier_recommendation="high",
            context_window=4096,
            description="Test description",
        )

        with pytest.raises(AttributeError):
            model.id = "modified"  # type: ignore[misc]

    def test_equality(self):
        """Two ModelInfo with same values are equal."""
        model1 = ModelInfo(
            id="test",
            name="Test",
            display_name="Test",
            provider="test",
            tier_recommendation="high",
            context_window=4096,
            description="Test",
        )
        model2 = ModelInfo(
            id="test",
            name="Test",
            display_name="Test",
            provider="test",
            tier_recommendation="high",
            context_window=4096,
            description="Test",
        )

        assert model1 == model2

    def test_hashable(self):
        """ModelInfo is hashable (can be used in sets)."""
        model = ModelInfo(
            id="test",
            name="Test",
            display_name="Test",
            provider="test",
            tier_recommendation="high",
            context_window=4096,
            description="Test",
        )

        # Should not raise - can be hashed and added to set
        hash(model)
        model_set = {model}
        assert len(model_set) == 1

    def test_to_dict_returns_expected_keys(self):
        """to_dict() returns all required keys."""
        model = ModelInfo(
            id="test-model",
            name="Test",
            display_name="Test Model",
            provider="test",
            tier_recommendation="high",
            context_window=4096,
            description="Test description",
        )
        result = model.to_dict()

        assert "id" in result
        assert "name" in result
        assert "display_name" in result
        assert "provider" in result
        assert "tier_recommendation" in result
        assert "context_window" in result
        assert "description" in result
        # memory_mb should be excluded when None
        assert "memory_mb" not in result

    def test_to_dict_includes_memory_mb_when_present(self):
        """to_dict() includes memory_mb when populated."""
        model = ModelInfo(
            id="test-model",
            name="Test",
            display_name="Test Model",
            provider="local",
            tier_recommendation="high",
            context_window=4096,
            description="Test description",
            memory_mb={"int8": 2000, "int4": 1200},
        )
        result = model.to_dict()

        assert "memory_mb" in result
        assert result["memory_mb"]["int8"] == 2000
        assert result["memory_mb"]["int4"] == 1200

    def test_memory_mb_default_is_none(self):
        """memory_mb defaults to None."""
        model = ModelInfo(
            id="test",
            name="Test",
            display_name="Test",
            provider="test",
            tier_recommendation="high",
            context_window=4096,
            description="Test",
        )
        assert model.memory_mb is None

    def test_to_dict_values_match_original(self):
        """to_dict() values match original ModelInfo values."""
        model = ModelInfo(
            id="test-id",
            name="Test Name",
            display_name="Test Display Name",
            provider="local",
            tier_recommendation="low",
            context_window=8192,
            description="Test description text",
            memory_mb={"int8": 3000, "int4": 1500, "float16": 5000},
        )
        result = model.to_dict()

        assert result["id"] == "test-id"
        assert result["name"] == "Test Name"
        assert result["display_name"] == "Test Display Name"
        assert result["provider"] == "local"
        assert result["tier_recommendation"] == "low"
        assert result["context_window"] == 8192
        assert result["description"] == "Test description text"
        assert result["memory_mb"]["float16"] == 5000
