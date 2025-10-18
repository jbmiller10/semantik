"""Additional coverage for chunking strategy registry and helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.webui.api.v2.chunking_schemas import ChunkingStrategy
from packages.webui.services import chunking_strategy_factory as factory_module
from packages.webui.services.chunking.strategy_registry import (
    get_strategy_defaults,
    recommend_strategy,
    resolve_api_identifier,
    resolve_internal_strategy_name,
)
from packages.webui.services.chunking_config_builder import ChunkingConfigBuilder
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.chunking_strategy_factory import ChunkingStrategyError, ChunkingStrategyFactory


def test_strategy_registry_alias_resolution() -> None:
    """Aliases should resolve to canonical identifiers and internal names."""

    assert resolve_api_identifier("mixed") == ChunkingStrategy.HYBRID.value
    assert resolve_internal_strategy_name("document_structure") == "markdown"


def test_strategy_registry_default_contexts() -> None:
    """Each context should surface its specific defaults."""

    manager_defaults = get_strategy_defaults(ChunkingStrategy.SEMANTIC, context="manager")
    builder_defaults = get_strategy_defaults(ChunkingStrategy.SEMANTIC, context="builder")
    factory_defaults = get_strategy_defaults(ChunkingStrategy.SEMANTIC, context="factory")

    assert manager_defaults["embedding_model"] == "sentence-transformers"
    assert builder_defaults["embedding_model"] == "default"
    assert factory_defaults["buffer_size"] == 1


def test_recommend_strategy_fallback_behaviour() -> None:
    """Recommendation helper falls back to recursive when no matches exist."""

    assert recommend_strategy(["unknown-ext"]) == ChunkingStrategy.RECURSIVE
    assert recommend_strategy([]) == ChunkingStrategy.RECURSIVE


def test_config_builder_resolves_alias_and_defaults() -> None:
    """Builder should map common aliases and return canonical defaults."""

    builder = ChunkingConfigBuilder()
    result = builder.build_config("mixed")

    assert result.strategy == ChunkingStrategy.HYBRID
    assert result.config["primary_strategy"] == "semantic"
    assert result.validation_errors is None


def test_config_builder_unknown_strategy_reports_validation_error() -> None:
    """Unknown strategy names produce a validation error result."""

    builder = ChunkingConfigBuilder()
    result = builder.build_config("not_a_strategy")

    assert result.validation_errors is not None
    assert "Unknown strategy" in result.validation_errors[0]


def test_strategy_factory_alias_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory resolves aliases before delegating to the shared registry."""

    sentinel = object()

    def fake_get_strategy(name: str) -> object:
        assert name == "character"
        return sentinel

    monkeypatch.setitem(factory_module.STRATEGY_REGISTRY, "character", object())
    monkeypatch.setattr(factory_module, "get_strategy", fake_get_strategy)

    strategy = ChunkingStrategyFactory.create_strategy("fixed", {})
    assert strategy is sentinel


def test_strategy_factory_unknown_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory should raise a clear error for unknown identifiers."""

    monkeypatch.setattr(factory_module, "get_strategy", lambda _: object())

    with pytest.raises(ChunkingStrategyError):
        ChunkingStrategyFactory.create_strategy("does_not_exist", {})


def test_strategy_factory_metadata_uses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata lookup should utilise registry helpers even for aliases."""

    monkeypatch.setattr(
        factory_module,
        "get_strategy_metadata",
        lambda _: {"description": "Test description"},
    )

    info = ChunkingStrategyFactory.get_strategy_info("fixed")

    assert info["internal_name"] == "character"
    assert info["description"] == "Test description"


@pytest.mark.asyncio()
async def test_service_get_strategy_details_resolves_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """ChunkingService should resolve API aliases before returning details."""

    service = ChunkingService(AsyncMock(), MagicMock(), MagicMock())
    strategy_payload = {
        "id": "markdown",
        "name": "Markdown",
        "description": "Handles markdown structure",
        "best_for": ["md"],
        "pros": ["Preserves headings"],
        "cons": ["Markdown only"],
        "default_config": {"strategy": "markdown"},
        "performance_characteristics": {"speed": "moderate"},
    }

    monkeypatch.setattr(
        service,
        "get_available_strategies",
        AsyncMock(return_value=[strategy_payload]),
    )

    sentinel = object()

    def fake_build_strategy_info(data: dict[str, object]) -> object:
        assert data["id"] == "markdown"
        return sentinel

    monkeypatch.setattr(service, "_build_strategy_info", fake_build_strategy_info)

    result = await service.get_strategy_details("document_structure")
    assert result is sentinel
