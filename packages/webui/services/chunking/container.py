"""Composition helpers for chunking services."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from packages.shared.config import settings
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.chunking_config_profile_repository import (
    ChunkingConfigProfileRepository,
)
from packages.webui.services.chunking.cache import ChunkingCache
from packages.webui.services.chunking.config_manager import ChunkingConfigManager
from packages.webui.services.chunking.metrics import ChunkingMetrics
from packages.webui.services.chunking.operation_manager import (
    DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    DEFAULT_DEAD_LETTER_TTL_SECONDS,
    ChunkingOperationManager,
)
from packages.webui.services.chunking.orchestrator import ChunkingOrchestrator
from packages.webui.services.chunking.processor import ChunkingProcessor
from packages.webui.services.chunking.validator import ChunkingValidator
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.redis_manager import RedisConfig, RedisManager
from packages.webui.services.type_guards import ensure_async_redis, ensure_sync_redis
from packages.webui.utils.error_classifier import get_default_chunking_error_classifier

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from redis import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_redis_manager: RedisManager | None = None
_chunking_metrics: ChunkingMetrics | None = None
_chunking_processor: ChunkingProcessor | None = None


def get_redis_manager() -> RedisManager:
    """Return process-wide Redis manager."""

    global _redis_manager
    if _redis_manager is None:
        config = RedisConfig(
            url=settings.REDIS_URL,
            max_connections=50,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
            socket_keepalive=True,
        )
        _redis_manager = RedisManager(config)
        logger.info("Initialized Redis manager for chunking container (url=%s)", settings.REDIS_URL)
    return _redis_manager


async def get_async_redis_client() -> aioredis.Redis | None:
    """Return async Redis client or None when unavailable."""

    try:
        manager = get_redis_manager()
        client = await manager.async_client()
        return ensure_async_redis(client)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Async Redis unavailable for chunking cache: %s", exc)
        return None


def get_sync_redis_client() -> Redis | None:
    """Return sync Redis client or None when unavailable."""

    try:
        manager = get_redis_manager()
        client = manager.sync_client
        return ensure_sync_redis(client)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Sync Redis unavailable for chunking tasks: %s", exc)
        return None


def build_chunking_operation_manager(
    *,
    redis_client: Redis | None = None,
    error_handler: ChunkingErrorHandler | None = None,
    error_classifier: Any | None = None,
    logger_: logging.Logger | None = None,
    expected_circuit_breaker_exceptions: tuple[type[Exception], ...] | None = None,
    failure_threshold: int | None = None,
    recovery_timeout: int | None = None,
    dead_letter_ttl_seconds: int | None = None,
    memory_usage_gauge: Any | None = None,
) -> ChunkingOperationManager:
    """Construct a ChunkingOperationManager with dependency defaults."""

    resolved_logger = logger_ or logger.getChild("operation_manager")

    resolved_redis = ensure_sync_redis(redis_client) if redis_client else get_sync_redis_client()
    resolved_error_handler = error_handler or ChunkingErrorHandler(redis_client=None)
    resolved_classifier = error_classifier or get_default_chunking_error_classifier()

    return ChunkingOperationManager(
        redis_client=resolved_redis,
        error_handler=resolved_error_handler,
        error_classifier=resolved_classifier,
        logger=resolved_logger,
        expected_circuit_breaker_exceptions=expected_circuit_breaker_exceptions,
        failure_threshold=failure_threshold or DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=recovery_timeout or DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        dead_letter_ttl_seconds=dead_letter_ttl_seconds or DEFAULT_DEAD_LETTER_TTL_SECONDS,
        memory_usage_gauge=memory_usage_gauge,
    )


def build_chunking_processor() -> ChunkingProcessor:
    global _chunking_processor
    if _chunking_processor is None:
        _chunking_processor = ChunkingProcessor()
    return _chunking_processor


def build_chunking_cache(redis_client: aioredis.Redis | None) -> ChunkingCache:
    return ChunkingCache(redis_client)


def build_chunking_metrics() -> ChunkingMetrics:
    global _chunking_metrics
    if _chunking_metrics is None:
        _chunking_metrics = ChunkingMetrics()
    return _chunking_metrics


def build_chunking_config_manager(
    profile_repo: ChunkingConfigProfileRepository | None = None,
) -> ChunkingConfigManager:
    """Construct a config manager bound to the provided repository."""

    return ChunkingConfigManager(profile_repo)


def build_chunking_validator(
    db_session: AsyncSession,
    collection_repo: CollectionRepository,
    document_repo: DocumentRepository,
) -> ChunkingValidator:
    return ChunkingValidator(db_session=db_session, collection_repo=collection_repo, document_repo=document_repo)


async def build_chunking_orchestrator(
    db_session: AsyncSession,
    *,
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
    enable_cache: bool = True,
) -> ChunkingOrchestrator:
    """Assemble orchestrator with collaborators bound to session."""

    collection_repo = collection_repo or CollectionRepository(db_session)
    document_repo = document_repo or DocumentRepository(db_session)

    redis_client = await get_async_redis_client() if enable_cache else None
    processor = build_chunking_processor()
    cache = build_chunking_cache(redis_client)
    metrics = build_chunking_metrics()
    validator = build_chunking_validator(db_session, collection_repo, document_repo)
    config_repo = ChunkingConfigProfileRepository(db_session)
    config_manager = build_chunking_config_manager(config_repo)

    return ChunkingOrchestrator(
        processor=processor,
        cache=cache,
        metrics=metrics,
        validator=validator,
        config_manager=config_manager,
        db_session=db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
    )


async def get_chunking_orchestrator(
    db_session: AsyncSession,
    *,
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
) -> ChunkingOrchestrator:
    """Return orchestrator honoring feature toggle."""
    return await build_chunking_orchestrator(
        db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
    )


async def resolve_celery_chunking_orchestrator(
    db_session: AsyncSession,
    *,
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
) -> ChunkingOrchestrator:
    """Provide orchestrator instance tailored for Celery workers (no cache)."""

    return await build_chunking_orchestrator(
        db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
        enable_cache=False,
    )
