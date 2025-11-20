"""Repository-level tests for chunking config profiles (lightweight, no DB)."""

import pytest

from packages.shared.database.models import ChunkingConfigProfile
from packages.shared.database.repositories.chunking_config_profile_repository import (
    ChunkingConfigProfileRepository,
)


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._value


class _Session:
    def __init__(self):
        self.added = []
        self.executed = []
        self.flushed = False

    async def execute(self, stmt):
        self.executed.append(stmt)
        # Return empty result by default
        return _Result(None)

    async def flush(self):  # pragma: no cover - trivial
        self.flushed = True

    def add(self, obj):
        self.added.append(obj)


@pytest.mark.asyncio()
async def test_upsert_profile_inserts_when_missing():
    session = _Session()
    repo = ChunkingConfigProfileRepository(session)

    profile = await repo.upsert_profile(
        user_id=1,
        name="demo",
        strategy="recursive",
        config={"chunk_size": 1000},
        description=None,
        is_default=True,
        tags=["a"],
    )

    assert isinstance(profile, ChunkingConfigProfile)
    assert profile.name == "demo"
    assert session.added, "Profile should be added"


@pytest.mark.asyncio()
async def test_list_profiles_filters_and_orders():
    session = _Session()
    repo = ChunkingConfigProfileRepository(session)

    # Override execute to return two profiles
    p1 = ChunkingConfigProfile(name="demo", strategy="recursive", config={}, created_by=2)
    p2 = ChunkingConfigProfile(name="other", strategy="semantic", config={}, created_by=2)

    async def execute_override(_stmt):  # pragma: no cover - minimal stub
        return _Result([p1, p2])

    session.execute = execute_override  # type: ignore[assignment]

    items = await repo.list_profiles(user_id=2, strategy=None, is_default=None)
    assert len(items) == 2
    assert items[0].name in {"demo", "other"}
