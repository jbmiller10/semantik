"""Cover dev-mode config paths in the orchestrator."""

import pytest

from packages.webui.services.chunking import (
    ChunkingCache,
    ChunkingConfigManager,
    ChunkingMetrics,
    ChunkingProcessor,
    ChunkingValidator,
)
from packages.webui.services.chunking.orchestrator import ChunkingOrchestrator


@pytest.mark.asyncio()
async def test_save_configuration_dev_mode_returns_ephemeral():
    orchestrator = ChunkingOrchestrator(
        processor=ChunkingProcessor(),
        cache=ChunkingCache(redis_client=None),
        metrics=ChunkingMetrics(),
        validator=ChunkingValidator(),
        config_manager=ChunkingConfigManager(profile_repo=None),
    )

    dto = await orchestrator.save_configuration(
        name="dev",
        description=None,
        strategy="recursive",
        config={"chunk_size": 256},
        is_default=False,
        tags=[],
        user_id=0,
    )

    assert dto.id
    assert dto.strategy == "recursive"
    assert dto.created_by == 0


@pytest.mark.asyncio()
async def test_list_configurations_dev_mode_returns_empty():
    orchestrator = ChunkingOrchestrator(
        processor=ChunkingProcessor(),
        cache=ChunkingCache(redis_client=None),
        metrics=ChunkingMetrics(),
        validator=ChunkingValidator(),
        config_manager=ChunkingConfigManager(profile_repo=None),
    )

    result = await orchestrator.list_configurations(user_id=0)
    assert result == []
