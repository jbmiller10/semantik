"""Ensure the chunking plugin registry integrates with the orchestrator."""

import pytest

from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.services.chunking_strategies import STRATEGY_REGISTRY
import packages.webui.services.chunking.strategy_registry as strategy_registry
from packages.webui.services.chunking import ChunkingCache, ChunkingConfigManager, ChunkingMetrics, ChunkingProcessor, ChunkingValidator
from packages.webui.services.chunking.orchestrator import ChunkingOrchestrator
from packages.webui.services.chunking.strategy_registry import register_strategy_definition
from packages.webui.services.chunking_strategy_factory import ChunkingStrategyFactory


class DemoPluginStrategy(ChunkingStrategy):
    INTERNAL_NAME = "demo_plugin"
    API_ID = "demo_plugin"

    def __init__(self) -> None:
        super().__init__(self.INTERNAL_NAME)

    def chunk(self, text: str, _config):
        return [text.upper()]

    def estimate_chunks(self, content_length: int, _config) -> int:
        return max(1, content_length // 50)

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        if not content:
            return False, "empty"
        return True, None


@pytest.mark.asyncio()
async def test_plugin_strategy_runs_through_orchestrator():
    # Register plugin definition and implementation
    register_strategy_definition(
        api_id="demo_plugin",
        internal_id="demo_plugin",
        display_name="Demo Plugin",
        description="Test plugin strategy",
        manager_defaults={"chunk_size": 128, "chunk_overlap": 0},
        builder_defaults={"chunk_size": 128, "chunk_overlap": 0},
        visual_example={"url": "https://example.com/demo.png"},
        is_plugin=True,
    )
    ChunkingStrategyFactory.register_strategy("demo_plugin", DemoPluginStrategy)

    orchestrator = ChunkingOrchestrator(
        processor=ChunkingProcessor(),
        cache=ChunkingCache(redis_client=None),
        metrics=ChunkingMetrics(),
        validator=ChunkingValidator(),
        config_manager=ChunkingConfigManager(),
    )

    try:
        strategies = await orchestrator.get_available_strategies()
        assert any(s.id == "demo_plugin" for s in strategies)

        chunks = await orchestrator.execute_ingestion_chunking(
            content="plugin demo",
            strategy="demo_plugin",
            config={"chunk_size": 128, "chunk_overlap": 0},
        )
        assert chunks[0]["content"] == "PLUGIN DEMO"
    finally:
        STRATEGY_REGISTRY.pop("demo_plugin", None)
        strategy_registry._STRATEGIES.pop("demo_plugin", None)
        strategy_registry._clear_caches()
