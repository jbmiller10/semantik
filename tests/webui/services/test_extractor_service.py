"""Tests for ExtractorService."""

from __future__ import annotations

from typing import Any

import pytest

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from shared.plugins.types.extractor import ExtractionResult, ExtractionType, ExtractorPlugin
from webui.services import extractor_service as extractor_service_module


class DummyExtractor(ExtractorPlugin):
    """Extractor that returns a single keyword."""

    PLUGIN_ID = "dummy-extractor"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.init_calls: list[dict[str, Any] | None] = []

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type="extractor",
            version="1.0.0",
            display_name="Dummy Extractor",
            description="Dummy extractor for tests.",
            capabilities={"supported_extractions": ["keywords"]},
        )

    @classmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        return [ExtractionType.KEYWORDS]

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        await super().initialize(config)
        self.init_calls.append(config)

    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        if extraction_types is not None and ExtractionType.KEYWORDS not in extraction_types:
            return ExtractionResult()
        return ExtractionResult(keywords=[options.get("keyword", "default") if options else "default"])


class ErrorExtractor(DummyExtractor):
    """Extractor that raises when extracting."""

    PLUGIN_ID = "error-extractor"

    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    plugin_registry.reset()
    monkeypatch.setattr(extractor_service_module, "load_plugins", lambda **_: plugin_registry)
    yield
    plugin_registry.reset()


def _register_extractor(plugin_cls: type[ExtractorPlugin]) -> None:
    manifest = plugin_cls.get_manifest()
    record = PluginRecord(
        plugin_type="extractor",
        plugin_id=manifest.id,
        plugin_version=manifest.version,
        manifest=manifest,
        plugin_class=plugin_cls,
        source=PluginSource.BUILTIN,
    )
    plugin_registry.register(record)


@pytest.mark.asyncio()
async def test_run_extractors_empty_inputs() -> None:
    service = extractor_service_module.ExtractorService()

    result = await service.run_extractors(text="", extractor_ids=["dummy-extractor"])
    assert result == ExtractionResult()

    result = await service.run_extractors(text="hello", extractor_ids=[])
    assert result == ExtractionResult()


@pytest.mark.asyncio()
async def test_run_extractors_ignores_unknown_types() -> None:
    _register_extractor(DummyExtractor)
    service = extractor_service_module.ExtractorService()

    result = await service.run_extractors(
        text="hello world",
        extractor_ids=["dummy-extractor"],
        extraction_types=["keywords", "unknown-type"],
        options={"keyword": "value"},
    )

    assert result.keywords == ["value"]


@pytest.mark.asyncio()
async def test_get_extractor_instance_caches_by_config() -> None:
    _register_extractor(DummyExtractor)
    service = extractor_service_module.ExtractorService()

    first = await service._get_extractor_instance("dummy-extractor", {"a": 1})
    second = await service._get_extractor_instance("dummy-extractor", {"a": 1})
    third = await service._get_extractor_instance("dummy-extractor", {"a": 2})

    assert first is second
    assert first is not third


@pytest.mark.asyncio()
async def test_run_extractors_continues_on_error() -> None:
    _register_extractor(ErrorExtractor)
    _register_extractor(DummyExtractor)
    service = extractor_service_module.ExtractorService()

    result = await service.run_extractors(
        text="hello world",
        extractor_ids=["error-extractor", "dummy-extractor"],
        options={"keyword": "ok"},
    )

    assert result.keywords == ["ok"]


@pytest.mark.asyncio()
async def test_extract_for_collection_returns_searchable_dict() -> None:
    _register_extractor(DummyExtractor)
    service = extractor_service_module.ExtractorService()

    extraction_config = {
        "enabled": True,
        "extractor_ids": ["dummy-extractor"],
        "types": ["keywords"],
        "options": {"keyword": "value"},
    }

    result = await service.extract_for_collection("hello world", extraction_config)
    assert result == {"keywords": ["value"]}


@pytest.mark.asyncio()
async def test_cleanup_handles_errors(monkeypatch) -> None:
    _register_extractor(DummyExtractor)
    service = extractor_service_module.ExtractorService()

    instance = await service._get_extractor_instance("dummy-extractor", None)
    assert instance is not None

    async def boom() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(instance, "cleanup", boom)

    await service.cleanup()
    assert service._extractor_instances == {}
