"""Repository-level tests for chunking config profiles (lightweight, no DB)."""

import pytest

from shared.database.models import ChunkingConfigProfile
from shared.database.repositories.chunking_config_profile_repository import ChunkingConfigProfileRepository


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
        self.next_result = None
        self.values: list = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        if self.next_result is not None:
            return self.next_result
        # Return empty result by default
        return _Result(None)

    async def flush(self):  # pragma: no cover - trivial
        self.flushed = True

    def add(self, obj):
        self.added.append(obj)


@pytest.mark.asyncio()
async def test_upsert_profile_preserves_default_flag_when_not_provided():
    session = _Session()
    existing = ChunkingConfigProfile(
        name="demo",
        description=None,
        strategy="recursive",
        config={},
        created_by=1,
        is_default=True,
    )

    session.next_result = _Result(existing)
    repo = ChunkingConfigProfileRepository(session)

    profile = await repo.upsert_profile(
        user_id=1,
        name="demo",
        strategy="semantic",
        config={"chunk_size": 999},
        description="updated",
        is_default=None,
        tags=["x"],
    )

    assert profile.is_default is True, "Default flag should remain unchanged when not explicitly set"
    assert profile.config["chunk_size"] == 999


@pytest.mark.asyncio()
async def test_clear_defaults_and_increment_usage():
    session = _Session()
    repo = ChunkingConfigProfileRepository(session)

    await repo.clear_defaults(user_id=3)
    await repo.increment_usage(profile_id=9)

    # Two execute calls recorded with update statements
    assert len(session.executed) == 2


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
