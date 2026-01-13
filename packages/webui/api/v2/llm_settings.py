"""LLM settings API endpoints."""

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Query
from starlette.requests import Request  # noqa: TCH002 - Required at runtime for rate limiter

from shared.database import get_db
from shared.database.exceptions import EntityNotFoundError
from shared.database.repositories.llm_provider_config_repository import (
    LLMProviderConfigRepository,
)
from shared.database.repositories.llm_usage_repository import LLMUsageRepository
from shared.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMTimeoutError,
)
from shared.llm.model_registry import get_all_models, get_default_model
from shared.llm.providers.anthropic_provider import AnthropicLLMProvider
from shared.llm.providers.openai_provider import OpenAILLMProvider
from webui.api.schemas import ErrorResponse
from webui.api.v2.llm_schemas import (
    AvailableModel,
    AvailableModelsResponse,
    LLMSettingsResponse,
    LLMSettingsUpdate,
    LLMTestRequest,
    LLMTestResponse,
    TokenUsageResponse,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig
from webui.rate_limiter import limiter

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/llm", tags=["llm-settings-v2"])


async def _get_config_repo(
    db: "AsyncSession" = Depends(get_db),
) -> LLMProviderConfigRepository:
    """Get LLM config repository instance."""
    return LLMProviderConfigRepository(db)


async def _get_usage_repo(
    db: "AsyncSession" = Depends(get_db),
) -> LLMUsageRepository:
    """Get LLM usage repository instance."""
    return LLMUsageRepository(db)


@router.get(
    "/settings",
    response_model=LLMSettingsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "LLM settings not configured"},
    },
)
async def get_llm_settings(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: LLMProviderConfigRepository = Depends(_get_config_repo),
) -> LLMSettingsResponse:
    """Get the current user's LLM configuration.

    Returns 404 if the user has not configured LLM settings.
    """
    user_id = int(current_user["id"])
    config = await repo.get_by_user_id(user_id)

    if not config:
        raise EntityNotFoundError("LLMProviderConfig", f"user:{user_id}")

    # Check which providers have keys configured
    anthropic_has_key = await repo.has_api_key(config.id, "anthropic")
    openai_has_key = await repo.has_api_key(config.id, "openai")

    return LLMSettingsResponse(
        high_quality_provider=config.high_quality_provider,
        high_quality_model=config.high_quality_model,
        low_quality_provider=config.low_quality_provider,
        low_quality_model=config.low_quality_model,
        anthropic_has_key=anthropic_has_key,
        openai_has_key=openai_has_key,
        default_temperature=config.default_temperature,
        default_max_tokens=config.default_max_tokens,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.put(
    "/settings",
    response_model=LLMSettingsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or encryption not configured"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def update_llm_settings(
    update: LLMSettingsUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: LLMProviderConfigRepository = Depends(_get_config_repo),
    db: "AsyncSession" = Depends(get_db),
) -> LLMSettingsResponse:
    """Create or update the current user's LLM settings.

    API keys are write-only and will never be returned in responses.
    To update a key, simply provide the new value. To check if a key
    is configured, use the `anthropic_has_key` and `openai_has_key`
    fields in the response.
    """
    user_id = int(current_user["id"])

    # Get or create config
    config = await repo.get_or_create(user_id)

    # Update tier configuration if provided
    update_fields: dict[str, Any] = {}
    if update.high_quality_provider is not None:
        update_fields["high_quality_provider"] = update.high_quality_provider
    if update.high_quality_model is not None:
        update_fields["high_quality_model"] = update.high_quality_model
    if update.low_quality_provider is not None:
        update_fields["low_quality_provider"] = update.low_quality_provider
    if update.low_quality_model is not None:
        update_fields["low_quality_model"] = update.low_quality_model
    if update.default_temperature is not None:
        update_fields["default_temperature"] = update.default_temperature
    if update.default_max_tokens is not None:
        update_fields["default_max_tokens"] = update.default_max_tokens

    if update_fields:
        config = await repo.update(user_id, **update_fields)

    # Store API keys if provided (encrypted)
    if update.anthropic_api_key:
        await repo.set_api_key(config.id, "anthropic", update.anthropic_api_key)
        logger.info("Updated Anthropic API key for user %s", user_id)

    if update.openai_api_key:
        await repo.set_api_key(config.id, "openai", update.openai_api_key)
        logger.info("Updated OpenAI API key for user %s", user_id)

    await db.commit()

    # Refresh config to get updated timestamps
    config = await repo.get_by_user_id(user_id)
    if not config:
        raise EntityNotFoundError("LLMProviderConfig", f"user:{user_id}")

    # Check which providers have keys configured
    anthropic_has_key = await repo.has_api_key(config.id, "anthropic")
    openai_has_key = await repo.has_api_key(config.id, "openai")

    return LLMSettingsResponse(
        high_quality_provider=config.high_quality_provider,
        high_quality_model=config.high_quality_model,
        low_quality_provider=config.low_quality_provider,
        low_quality_model=config.low_quality_model,
        anthropic_has_key=anthropic_has_key,
        openai_has_key=openai_has_key,
        default_temperature=config.default_temperature,
        default_max_tokens=config.default_max_tokens,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
)
async def list_available_models() -> AvailableModelsResponse:
    """List available LLM models from the curated registry.

    Returns a static list of recommended models. For custom model IDs,
    use the test endpoint to validate before saving.
    """
    models = get_all_models()

    return AvailableModelsResponse(
        models=[
            AvailableModel(
                id=m.id,
                name=m.name,
                display_name=m.display_name,
                provider=m.provider,
                tier_recommendation=m.tier_recommendation,
                context_window=m.context_window,
                description=m.description,
            )
            for m in models
        ]
    )


@router.post(
    "/test",
    response_model=LLMTestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid API key or provider error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit(RateLimitConfig.LLM_TEST_RATE)
async def test_api_key(
    request: Request,  # noqa: ARG001 - Required for rate limiter
    test_request: LLMTestRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> LLMTestResponse:
    """Test an API key's validity with a minimal request.

    This endpoint is rate limited to 5 requests per minute to prevent abuse.
    The API key is NOT saved - use PUT /settings to store keys.
    """
    provider_type = test_request.provider
    api_key = test_request.api_key

    # Get a default model for testing
    try:
        test_model = get_default_model(provider_type, "low")
    except ValueError:
        return LLMTestResponse(
            success=False,
            message=f"Unknown provider: {provider_type}",
        )

    # Create provider instance
    provider = AnthropicLLMProvider() if provider_type == "anthropic" else OpenAILLMProvider()

    try:
        # Initialize and make minimal request
        await provider.initialize(api_key, test_model)

        async with provider:
            # Make a minimal request to verify the key works
            await provider.generate(
                prompt="Say 'OK' and nothing else.",
                max_tokens=5,
                timeout=10.0,
            )

        return LLMTestResponse(
            success=True,
            message="API key is valid",
            model_tested=test_model,
        )

    except LLMAuthenticationError as e:
        logger.info("API key test failed for %s: authentication error", provider_type)
        return LLMTestResponse(
            success=False,
            message=f"Authentication failed: {e}",
        )
    except LLMTimeoutError:
        logger.warning("API key test timed out for %s", provider_type)
        return LLMTestResponse(
            success=False,
            message="Request timed out. The key may be valid but the API is slow.",
        )
    except LLMProviderError as e:
        logger.warning("API key test failed for %s: %s", provider_type, e)
        return LLMTestResponse(
            success=False,
            message=f"Provider error: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error testing API key for %s: %s", provider_type, e)
        return LLMTestResponse(
            success=False,
            message="An unexpected error occurred while testing the key.",
        )


@router.get(
    "/usage",
    response_model=TokenUsageResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_usage_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to include in statistics"),
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: LLMUsageRepository = Depends(_get_usage_repo),
) -> TokenUsageResponse:
    """Get token usage statistics for the current user.

    Returns aggregated token usage over the specified period, broken down
    by feature (HyDE, summary, etc.) and provider (Anthropic, OpenAI).
    """
    user_id = int(current_user["id"])

    summary = await repo.get_user_usage_summary(user_id, days)

    return TokenUsageResponse(
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        total_tokens=summary.total_tokens,
        by_feature=summary.by_feature,
        by_provider=summary.by_provider,
        event_count=summary.event_count,
        period_days=summary.period_days,
    )
