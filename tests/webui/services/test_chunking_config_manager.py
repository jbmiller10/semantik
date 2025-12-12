"""Tests for the DB-backed chunking config manager."""

from typing import Any

import pytest

from shared.database.models import ChunkingConfigProfile
from shared.database.repositories.chunking_config_profile_repository import ChunkingConfigProfileRepository
from webui.services.chunking.config_manager import ChunkingConfigManager


class _FakeResult:
    def __init__(self, value: Any = None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Any]:
        return self._value or []


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.executed: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:  # pragma: no cover - trivial
        self.executed.append(stmt)
        return _FakeResult()

    async def flush(self) -> None:  # pragma: no cover - no-op
        return None

    def add(self, obj: Any) -> None:
        self.added.append(obj)


@pytest.mark.asyncio()
async def test_config_manager_persists_and_lists_profiles():
    session = _FakeSession()
    repo = ChunkingConfigProfileRepository(session)
    manager = ChunkingConfigManager(profile_repo=repo)

    dto = await manager.save_user_config(
        user_id=7,
        name="demo",
        strategy="recursive",
        config={"chunk_size": 256},
        description=None,
        is_default=True,
        tags=["team"],
    )

    assert session.added, "Profile should be added to session"
    assert dto.name == "demo"
    assert dto.strategy == "recursive"
    assert dto.config["chunk_size"] == 256

    # fake list result
    _profile = ChunkingConfigProfile(
        name="demo",
        description=None,
        strategy="recursive",
        config={"chunk_size": 256},
        created_by=7,
        is_default=False,
    )
    listed = await manager.list_user_configs(user_id=7, strategy=None, is_default=None)
    assert listed == [], "Fake session returns empty list by design"


@pytest.mark.asyncio()
async def test_config_manager_merge_defaults():
    manager = ChunkingConfigManager(profile_repo=None)

    merged = manager.merge_configs("recursive", {"chunk_size": 123})
    assert merged["chunk_size"] == 123
    assert "strategy" not in merged


def test_recommend_strategy_for_markdown_and_short_content():
    manager = ChunkingConfigManager(profile_repo=None)

    rec = manager.recommend_strategy(file_type=".md", content_length=400, document_type="technical")

    assert rec["strategy"]
    assert rec["suggested_config"]["chunk_size"] == 500
