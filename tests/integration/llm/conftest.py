"""Fixtures for LLM integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from shared.database.models import LLMProviderConfig, LLMUsageEvent
from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.database.repositories.llm_usage_repository import LLMUsageRepository
from shared.utils.encryption import SecretEncryption, generate_fernet_key

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


@pytest.fixture(autouse=True)
def _initialize_encryption():
    """Initialize encryption for all LLM tests.

    Uses a fresh Fernet key for each test to ensure isolation.
    Resets encryption state after each test.
    """
    test_key = generate_fernet_key()
    SecretEncryption.initialize(test_key)
    yield
    SecretEncryption.reset()


@pytest.fixture()
def llm_config_repo(db_session: AsyncSession) -> LLMProviderConfigRepository:
    """Create LLMProviderConfigRepository with real session."""
    return LLMProviderConfigRepository(db_session)


@pytest.fixture()
def llm_usage_repo(db_session: AsyncSession) -> LLMUsageRepository:
    """Create LLMUsageRepository with real session."""
    return LLMUsageRepository(db_session)


@pytest_asyncio.fixture()
async def llm_config(
    db_session: AsyncSession,
    test_user_db: User,
) -> LLMProviderConfig:
    """Create a basic LLMProviderConfig for testing.

    Creates config with anthropic as provider for both tiers.
    """
    config = LLMProviderConfig(
        user_id=test_user_db.id,
        high_quality_provider="anthropic",
        high_quality_model="claude-opus-4-5-20251101",
        low_quality_provider="anthropic",
        low_quality_model="claude-sonnet-4-5-20250929",
    )
    db_session.add(config)
    await db_session.flush()
    return config


@pytest_asyncio.fixture()
async def llm_config_with_key(
    db_session: AsyncSession,
    llm_config: LLMProviderConfig,
    llm_config_repo: LLMProviderConfigRepository,
) -> LLMProviderConfig:
    """LLMProviderConfig with an encrypted API key stored.

    Adds an anthropic API key to the config.
    """
    await llm_config_repo.set_api_key(
        cast(int, llm_config.id),
        "anthropic",
        "sk-ant-test-key-for-integration-tests",
    )
    await db_session.flush()
    return llm_config


@pytest_asyncio.fixture()
async def llm_config_multi_provider(
    db_session: AsyncSession,
    test_user_db: User,
    llm_config_repo: LLMProviderConfigRepository,
) -> LLMProviderConfig:
    """LLMProviderConfig with different providers per tier.

    HIGH tier: OpenAI (gpt-4o)
    LOW tier: Anthropic (claude-sonnet)
    """
    config = LLMProviderConfig(
        user_id=test_user_db.id,
        high_quality_provider="openai",
        high_quality_model="gpt-4o",
        low_quality_provider="anthropic",
        low_quality_model="claude-sonnet-4-5-20250929",
    )
    db_session.add(config)
    await db_session.flush()

    # Add keys for both providers
    config_id = cast(int, config.id)
    await llm_config_repo.set_api_key(config_id, "anthropic", "sk-ant-test-key")
    await llm_config_repo.set_api_key(config_id, "openai", "sk-openai-test-key")
    await db_session.flush()

    return config


@pytest_asyncio.fixture()
async def llm_usage_event(
    db_session: AsyncSession,
    test_user_db: User,
    llm_usage_repo: LLMUsageRepository,
) -> LLMUsageEvent:
    """Create a single LLM usage event for testing."""
    event = await llm_usage_repo.record_usage(
        user_id=cast(int, test_user_db.id),
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        quality_tier="low",
        feature="hyde",
        input_tokens=100,
        output_tokens=200,
    )
    await db_session.flush()
    return event


@pytest.fixture()
def mock_anthropic_response():
    """Mock response from Anthropic SDK."""
    response = MagicMock()
    response.content = [MagicMock(text="Generated response text")]
    response.model = "claude-sonnet-4-5-20250929"
    response.usage = MagicMock(input_tokens=50, output_tokens=100)
    response.stop_reason = "end_turn"
    return response


@pytest.fixture()
def mock_openai_response():
    """Mock response from OpenAI SDK."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Generated response text"))]
    response.model = "gpt-4o"
    response.usage = MagicMock(prompt_tokens=50, completion_tokens=100)
    response.choices[0].finish_reason = "stop"
    return response


@pytest.fixture()
def mock_anthropic_client(mock_anthropic_response):
    """Mock AsyncAnthropic client."""
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=mock_anthropic_response)
    return client


@pytest.fixture()
def mock_openai_client(mock_openai_response):
    """Mock AsyncOpenAI client."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    return client
