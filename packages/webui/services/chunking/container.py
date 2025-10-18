"""Composition helpers for chunking services."""

from __future__ import annotations

import logging
import redis.asyncio as aioredis
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.config import settings
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository

from ..redis_manager import RedisConfig, RedisManager
from ..type_guards import ensure_async_redis, ensure_sync_redis
from ..chunking_service import ChunkingService
from .adapter import ChunkingServiceAdapter
from .cache import ChunkingCache
from .config_manager import ChunkingConfigManager
from .metrics import ChunkingMetrics
from .orchestrator import ChunkingOrchestrator
from .processor import ChunkingProcessor
from .validator import ChunkingValidator

logger = logging.getLogger(__name__)

_redis_manager: RedisManager | None = None


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


def build_chunking_processor() -> ChunkingProcessor:
    return ChunkingProcessor()


def build_chunking_cache(redis_client: aioredis.Redis | None) -> ChunkingCache:
    return ChunkingCache(redis_client)


def build_chunking_metrics() -> ChunkingMetrics:
    return ChunkingMetrics()


def build_chunking_config_manager() -> ChunkingConfigManager:
    return ChunkingConfigManager()


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
    config_manager = build_chunking_config_manager()

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

    if not settings.USE_CHUNKING_ORCHESTRATOR:
        raise RuntimeError(
            "Chunking orchestrator disabled via USE_CHUNKING_ORCHESTRATOR. "
            "Request adapter or legacy service instead."
        )
    return await build_chunking_orchestrator(
        db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
    )


async def get_chunking_service_adapter(
    db_session: AsyncSession,
    *,
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
) -> ChunkingServiceAdapter:
    """Return adapter that emulates ChunkingService API."""

    orchestrator = await build_chunking_orchestrator(
        db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
    )
    return ChunkingServiceAdapter(
        orchestrator=orchestrator,
        db_session=db_session,
        collection_repo=orchestrator.collection_repo,
        document_repo=orchestrator.document_repo,
    )


async def get_legacy_chunking_service(
    db_session: AsyncSession,
    *,
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
    with_cache: bool = True,
) -> ChunkingService:
    """Return legacy ChunkingService for fallback scenarios."""

    collection_repo = collection_repo or CollectionRepository(db_session)
    document_repo = document_repo or DocumentRepository(db_session)

    redis_client = await get_async_redis_client() if with_cache else None
    return ChunkingService(
        db_session=db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
        redis_client=redis_client,
    )


async def resolve_api_chunking_dependency(
    db_session: AsyncSession,
    *,
    prefer_adapter: bool = False,
) -> ChunkingOrchestrator | ChunkingServiceAdapter | ChunkingService:
    """Select orchestrator or legacy service for API use."""

    if settings.USE_CHUNKING_ORCHESTRATOR:
        if prefer_adapter:
            return await get_chunking_service_adapter(db_session)
        return await get_chunking_orchestrator(db_session)
    return await get_legacy_chunking_service(db_session)


async def resolve_celery_chunking_service(
    db_session: AsyncSession,
    *,
    collection_repo: CollectionRepository | None = None,
    document_repo: DocumentRepository | None = None,
) -> ChunkingService | ChunkingServiceAdapter:
    """Provide Celery-friendly chunking dependency."""

    if settings.USE_CHUNKING_ORCHESTRATOR:
        # Celery tasks run sync contexts, so bypass Redis cache inside adapter.
        orchestrator = await build_chunking_orchestrator(
            db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
            enable_cache=False,
        )
        return ChunkingServiceAdapter(
            orchestrator=orchestrator,
            db_session=db_session,
            collection_repo=orchestrator.collection_repo,
            document_repo=orchestrator.document_repo,
        )

    return await get_legacy_chunking_service(
        db_session,
        collection_repo=collection_repo,
        document_repo=document_repo,
        with_cache=False,
    )
