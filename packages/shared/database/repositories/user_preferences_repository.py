"""Repository implementation for user preferences."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from sqlalchemy import select

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError, ValidationError
from shared.database.models import UserPreferences

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final[_UnsetType] = _UnsetType()


class UserPreferencesRepository:
    """Repository for user preferences management.

    This repository manages per-user preferences for search behavior
    and collection creation defaults.

    Example:
        ```python
        repo = UserPreferencesRepository(session)

        # Get or create user's preferences (returns defaults if not configured)
        prefs = await repo.get_or_create(user_id=123)

        # Update search preferences
        await repo.update(
            user_id=123,
            search_top_k=20,
            search_mode="hybrid",
        )

        # Reset search preferences to defaults
        await repo.reset_search(user_id=123)

        # Reset collection defaults
        await repo.reset_collection_defaults(user_id=123)
        ```
    """

    # Valid values for constrained fields
    VALID_SEARCH_MODES = frozenset({"dense", "sparse", "hybrid"})
    VALID_QUANTIZATION = frozenset({"none", "scalar", "binary"})
    VALID_CHUNKING_STRATEGIES = frozenset({"character", "recursive", "markdown", "semantic"})
    VALID_SPARSE_TYPES = frozenset({"bm25", "splade"})

    # Default values for reset operations
    SEARCH_DEFAULTS: dict[str, int | str | bool | None] = {
        "search_top_k": 10,
        "search_mode": "dense",
        "search_use_reranker": False,
        "search_rrf_k": 60,
        "search_similarity_threshold": None,
    }

    COLLECTION_DEFAULTS: dict[str, int | str | bool | None] = {
        "default_embedding_model": None,
        "default_quantization": "none",
        "default_chunking_strategy": "recursive",
        "default_chunk_size": 1024,
        "default_chunk_overlap": 200,
        "default_enable_sparse": False,
        "default_sparse_type": "bm25",
        "default_enable_hybrid": False,
    }

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    def _validate_search_mode(self, mode: str) -> None:
        """Validate search mode value."""
        if mode not in self.VALID_SEARCH_MODES:
            raise ValidationError(
                f"Invalid search_mode '{mode}'. Must be one of: {', '.join(sorted(self.VALID_SEARCH_MODES))}",
                field="search_mode",
            )

    def _validate_quantization(self, value: str) -> None:
        """Validate quantization value."""
        if value not in self.VALID_QUANTIZATION:
            raise ValidationError(
                f"Invalid quantization '{value}'. Must be one of: {', '.join(sorted(self.VALID_QUANTIZATION))}",
                field="default_quantization",
            )

    def _validate_chunking_strategy(self, value: str) -> None:
        """Validate chunking strategy value."""
        if value not in self.VALID_CHUNKING_STRATEGIES:
            raise ValidationError(
                f"Invalid chunking_strategy '{value}'. Must be one of: {', '.join(sorted(self.VALID_CHUNKING_STRATEGIES))}",
                field="default_chunking_strategy",
            )

    def _validate_sparse_type(self, value: str) -> None:
        """Validate sparse type value."""
        if value not in self.VALID_SPARSE_TYPES:
            raise ValidationError(
                f"Invalid sparse_type '{value}'. Must be one of: {', '.join(sorted(self.VALID_SPARSE_TYPES))}",
                field="default_sparse_type",
            )

    def _validate_range(
        self,
        value: int | float,
        min_val: int | float,
        max_val: int | float,
        field: str,
    ) -> None:
        """Validate value is within range."""
        if not (min_val <= value <= max_val):
            raise ValidationError(
                f"{field} must be between {min_val} and {max_val}, got {value}",
                field=field,
            )

    def _validate_hybrid_sparse_consistency(
        self,
        enable_hybrid: bool | _UnsetType,
        enable_sparse: bool | _UnsetType,
        current_sparse: bool,
    ) -> None:
        """Validate hybrid requires sparse constraint.

        Args:
            enable_hybrid: New hybrid value (or UNSET)
            enable_sparse: New sparse value (or UNSET)
            current_sparse: Current sparse value in database
        """
        # Determine effective sparse value
        effective_sparse = enable_sparse if enable_sparse is not UNSET else current_sparse

        # If enabling hybrid, sparse must also be enabled
        if enable_hybrid is not UNSET and enable_hybrid and not effective_sparse:
            raise ValidationError(
                "Cannot enable hybrid search without enabling sparse indexing first",
                field="default_enable_hybrid",
            )

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_user_id(self, user_id: int) -> UserPreferences | None:
        """Get preferences for a user.

        Args:
            user_id: The user's ID

        Returns:
            UserPreferences instance or None if not configured

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error("Failed to get preferences for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get", "UserPreferences", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_or_create(self, user_id: int) -> UserPreferences:
        """Get existing preferences or create with defaults.

        Args:
            user_id: The user's ID

        Returns:
            UserPreferences instance (existing or newly created with defaults)

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            prefs = await self.get_by_user_id(user_id)
            if prefs is not None:
                return prefs

            # Create new preferences with all defaults
            prefs = UserPreferences(user_id=user_id)
            self.session.add(prefs)
            await self.session.flush()

            logger.info(f"Created user preferences for user_id={user_id}")
            return prefs

        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to get/create preferences for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get_or_create", "UserPreferences", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def update(
        self,
        user_id: int,
        *,
        # Search preferences
        search_top_k: int | _UnsetType = UNSET,
        search_mode: str | _UnsetType = UNSET,
        search_use_reranker: bool | _UnsetType = UNSET,
        search_rrf_k: int | _UnsetType = UNSET,
        search_similarity_threshold: float | None | _UnsetType = UNSET,
        # Collection defaults
        default_embedding_model: str | None | _UnsetType = UNSET,
        default_quantization: str | _UnsetType = UNSET,
        default_chunking_strategy: str | _UnsetType = UNSET,
        default_chunk_size: int | _UnsetType = UNSET,
        default_chunk_overlap: int | _UnsetType = UNSET,
        default_enable_sparse: bool | _UnsetType = UNSET,
        default_sparse_type: str | _UnsetType = UNSET,
        default_enable_hybrid: bool | _UnsetType = UNSET,
    ) -> UserPreferences:
        """Update user preferences.

        Only updates fields that are explicitly provided.
        Creates preferences if they don't exist.

        Args:
            user_id: The user's ID
            search_top_k: Number of search results (5-50)
            search_mode: Search mode ('dense', 'sparse', 'hybrid')
            search_use_reranker: Enable reranking
            search_rrf_k: RRF constant (1-100)
            search_similarity_threshold: Minimum similarity (0.0-1.0 or None)
            default_embedding_model: Default embedding model or None
            default_quantization: Quantization type ('none', 'scalar', 'binary')
            default_chunking_strategy: Chunking strategy
            default_chunk_size: Chunk size (256-4096)
            default_chunk_overlap: Chunk overlap (0-512)
            default_enable_sparse: Enable sparse indexing
            default_sparse_type: Sparse type ('bm25', 'splade')
            default_enable_hybrid: Enable hybrid search (requires sparse)

        Returns:
            Updated UserPreferences instance

        Raises:
            ValidationError: If values are invalid or violate constraints
            DatabaseOperationError: For database errors
        """
        # Validate constrained fields
        if not isinstance(search_top_k, _UnsetType):
            self._validate_range(search_top_k, 5, 50, "search_top_k")
        if not isinstance(search_mode, _UnsetType):
            self._validate_search_mode(search_mode)
        if not isinstance(search_rrf_k, _UnsetType):
            self._validate_range(search_rrf_k, 1, 100, "search_rrf_k")
        if not isinstance(search_similarity_threshold, _UnsetType) and search_similarity_threshold is not None:
            self._validate_range(search_similarity_threshold, 0.0, 1.0, "search_similarity_threshold")
        if not isinstance(default_quantization, _UnsetType):
            self._validate_quantization(default_quantization)
        if not isinstance(default_chunking_strategy, _UnsetType):
            self._validate_chunking_strategy(default_chunking_strategy)
        if not isinstance(default_chunk_size, _UnsetType):
            self._validate_range(default_chunk_size, 256, 4096, "default_chunk_size")
        if not isinstance(default_chunk_overlap, _UnsetType):
            self._validate_range(default_chunk_overlap, 0, 512, "default_chunk_overlap")
        if not isinstance(default_sparse_type, _UnsetType):
            self._validate_sparse_type(default_sparse_type)

        try:
            prefs = await self.get_or_create(user_id)

            # Validate hybrid/sparse consistency before updating
            self._validate_hybrid_sparse_consistency(
                default_enable_hybrid,
                default_enable_sparse,
                prefs.default_enable_sparse,
            )

            # Update only provided fields - Search preferences
            if search_top_k is not UNSET:
                prefs.search_top_k = search_top_k
            if search_mode is not UNSET:
                prefs.search_mode = search_mode
            if search_use_reranker is not UNSET:
                prefs.search_use_reranker = search_use_reranker
            if search_rrf_k is not UNSET:
                prefs.search_rrf_k = search_rrf_k
            if search_similarity_threshold is not UNSET:
                prefs.search_similarity_threshold = search_similarity_threshold

            # Update only provided fields - Collection defaults
            if default_embedding_model is not UNSET:
                prefs.default_embedding_model = default_embedding_model
            if default_quantization is not UNSET:
                prefs.default_quantization = default_quantization
            if default_chunking_strategy is not UNSET:
                prefs.default_chunking_strategy = default_chunking_strategy
            if default_chunk_size is not UNSET:
                prefs.default_chunk_size = default_chunk_size
            if default_chunk_overlap is not UNSET:
                prefs.default_chunk_overlap = default_chunk_overlap
            if default_enable_sparse is not UNSET:
                prefs.default_enable_sparse = default_enable_sparse
            if default_sparse_type is not UNSET:
                prefs.default_sparse_type = default_sparse_type
            if default_enable_hybrid is not UNSET:
                prefs.default_enable_hybrid = default_enable_hybrid

            prefs.updated_at = datetime.now(UTC)

            await self.session.flush()
            logger.debug(f"Updated preferences for user_id={user_id}")

            return prefs

        except ValidationError:
            raise
        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to update preferences for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("update", "UserPreferences", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def reset_search(self, user_id: int) -> UserPreferences:
        """Reset search preferences to defaults.

        Resets: search_top_k, search_mode, search_use_reranker,
                search_rrf_k, search_similarity_threshold

        Args:
            user_id: The user's ID

        Returns:
            Updated UserPreferences with default search settings

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            prefs = await self.get_or_create(user_id)

            # Reset all search fields to defaults
            prefs.search_top_k = self.SEARCH_DEFAULTS["search_top_k"]
            prefs.search_mode = self.SEARCH_DEFAULTS["search_mode"]
            prefs.search_use_reranker = self.SEARCH_DEFAULTS["search_use_reranker"]
            prefs.search_rrf_k = self.SEARCH_DEFAULTS["search_rrf_k"]
            prefs.search_similarity_threshold = self.SEARCH_DEFAULTS["search_similarity_threshold"]

            prefs.updated_at = datetime.now(UTC)

            await self.session.flush()
            logger.info(f"Reset search preferences for user_id={user_id}")

            return prefs

        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to reset search preferences for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("reset_search", "UserPreferences", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def reset_collection_defaults(self, user_id: int) -> UserPreferences:
        """Reset collection defaults to system defaults.

        Resets: default_embedding_model, default_quantization,
                default_chunking_strategy, default_chunk_size,
                default_chunk_overlap, default_enable_sparse,
                default_sparse_type, default_enable_hybrid

        Args:
            user_id: The user's ID

        Returns:
            Updated UserPreferences with default collection settings

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            prefs = await self.get_or_create(user_id)

            # Reset all collection default fields
            prefs.default_embedding_model = self.COLLECTION_DEFAULTS["default_embedding_model"]
            prefs.default_quantization = self.COLLECTION_DEFAULTS["default_quantization"]
            prefs.default_chunking_strategy = self.COLLECTION_DEFAULTS["default_chunking_strategy"]
            prefs.default_chunk_size = self.COLLECTION_DEFAULTS["default_chunk_size"]
            prefs.default_chunk_overlap = self.COLLECTION_DEFAULTS["default_chunk_overlap"]
            prefs.default_enable_sparse = self.COLLECTION_DEFAULTS["default_enable_sparse"]
            prefs.default_sparse_type = self.COLLECTION_DEFAULTS["default_sparse_type"]
            prefs.default_enable_hybrid = self.COLLECTION_DEFAULTS["default_enable_hybrid"]

            prefs.updated_at = datetime.now(UTC)

            await self.session.flush()
            logger.info(f"Reset collection defaults for user_id={user_id}")

            return prefs

        except DatabaseOperationError:
            raise
        except Exception as e:
            logger.error("Failed to reset collection defaults for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("reset_collection_defaults", "UserPreferences", str(e)) from e
