"""
Chunking validator service.

Handles validation of inputs, configurations, and permissions for chunking operations.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.infrastructure.exceptions import (
    PermissionDeniedError,
    ValidationError,
)
from packages.shared.database.repositories.collection_repository import (
    CollectionRepository,
)
from packages.shared.database.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class ChunkingValidator:
    """Service responsible for validation of chunking operations."""

    # Configuration constraints
    MIN_CHUNK_SIZE = 50
    MAX_CHUNK_SIZE = 10000
    MIN_OVERLAP = 0
    MAX_OVERLAP_RATIO = 0.5  # 50% max overlap
    MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB

    # Valid strategies
    VALID_STRATEGIES = [
        "fixed_size",
        "sliding_window",
        "semantic",
        "recursive",
        "document_structure",
        "markdown",
        "hierarchical",
        "hybrid",
    ]

    def __init__(
        self,
        db_session: AsyncSession | None = None,
        collection_repo: CollectionRepository | None = None,
        document_repo: DocumentRepository | None = None,
    ):
        """
        Initialize the validator service.

        Args:
            db_session: Database session for permission checks
            collection_repo: Repository for collection access checks
            document_repo: Repository for document access checks
        """
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo

    async def validate_preview_request(
        self,
        content: str | None,
        document_id: str | None,
        strategy: str,
        config: dict[str, Any] | None,
    ) -> None:
        """
        Validate a preview request.

        Args:
            content: Direct content to preview
            document_id: Document ID to preview
            strategy: Chunking strategy
            config: Strategy configuration

        Raises:
            ValidationError: If validation fails
        """
        # Must have either content or document_id
        if not content and not document_id:
            raise ValidationError(
                field="content/document_id",
                value=None,
                reason="Either content or document_id must be provided"
            )

        if content and document_id:
            raise ValidationError(
                field="content/document_id",
                value="both provided",
                reason="Cannot provide both content and document_id"
            )

        # Validate content size
        if content:
            self.validate_content(content)

        # Validate strategy
        self.validate_strategy(strategy)

        # Validate configuration
        if config:
            self.validate_config(strategy, config)

    def validate_content(self, content: str) -> None:
        """
        Validate content for chunking.

        Args:
            content: Content to validate

        Raises:
            ValidationError: If content is invalid
        """
        if not content:
            raise ValidationError(
                field="content",
                value=None,
                reason="Content cannot be empty"
            )

        if len(content) > self.MAX_CONTENT_SIZE:
            raise ValidationError(
                field="content",
                value=f"{len(content)} bytes",
                reason=f"Content size exceeds maximum {self.MAX_CONTENT_SIZE}"
            )

        # Check for suspicious patterns (basic security check)
        if "<script" in content.lower() or "javascript:" in content.lower():
            logger.warning("Potentially malicious content detected")
            # Don't reject but log for monitoring

    def validate_strategy(self, strategy: str) -> None:
        """
        Validate chunking strategy.

        Args:
            strategy: Strategy name

        Raises:
            ValidationError: If strategy is invalid
        """
        if not strategy:
            raise ValidationError(
                field="strategy",
                value=None,
                reason="Strategy is required"
            )

        if strategy not in self.VALID_STRATEGIES:
            raise ValidationError(
                field="strategy",
                value=strategy,
                reason=f"Invalid strategy. Valid strategies: {', '.join(self.VALID_STRATEGIES)}"
            )

    def validate_config(self, strategy: str, config: dict[str, Any]) -> None:
        """
        Validate strategy configuration.

        Args:
            strategy: Strategy name
            config: Configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError(
                field="config",
                value=type(config).__name__,
                reason="Configuration must be a dictionary"
            )

        # Validate common parameters
        if "chunk_size" in config:
            chunk_size = config["chunk_size"]
            if not isinstance(chunk_size, int):
                try:
                    chunk_size = int(chunk_size)
                except (ValueError, TypeError) as e:
                    raise ValidationError(
                        field="chunk_size",
                        value=chunk_size,
                        reason=f"Must be an integer: {e}"
                    ) from e

            if chunk_size < self.MIN_CHUNK_SIZE:
                raise ValidationError(
                    field="chunk_size",
                    value=chunk_size,
                    reason=f"Below minimum {self.MIN_CHUNK_SIZE}"
                )

            if chunk_size > self.MAX_CHUNK_SIZE:
                raise ValidationError(
                    field="chunk_size",
                    value=chunk_size,
                    reason=f"Exceeds maximum {self.MAX_CHUNK_SIZE}"
                )

        if "chunk_overlap" in config:
            overlap = config["chunk_overlap"]
            if not isinstance(overlap, int):
                try:
                    overlap = int(overlap)
                except (ValueError, TypeError) as e:
                    raise ValidationError(
                        field="chunk_overlap",
                        value=overlap,
                        reason=f"Must be an integer: {e}"
                    ) from e

            if overlap < self.MIN_OVERLAP:
                raise ValidationError(
                    field="chunk_overlap",
                    value=overlap,
                    reason="Cannot be negative"
                )

            # Check overlap ratio if chunk_size is present
            if "chunk_size" in config:
                chunk_size = int(config["chunk_size"])
                if overlap > chunk_size * self.MAX_OVERLAP_RATIO:
                    raise ValidationError(
                        field="chunk_overlap",
                        value=overlap,
                        reason=f"Exceeds {self.MAX_OVERLAP_RATIO * 100}% of chunk_size"
                    )

        # Strategy-specific validation
        self._validate_strategy_specific_config(strategy, config)

    def _validate_strategy_specific_config(
        self,
        strategy: str,
        config: dict[str, Any],
    ) -> None:
        """Validate strategy-specific configuration parameters."""
        if strategy == "semantic":
            if "embedding_model" in config:
                valid_models = ["sentence-transformers", "openai", "cohere"]
                if config["embedding_model"] not in valid_models:
                    raise ValidationError(f"Invalid embedding_model. Valid options: {', '.join(valid_models)}")

        elif strategy == "hierarchical" and "max_level" in config:
            max_level = config["max_level"]
            if not isinstance(max_level, int) or max_level < 1 or max_level > 5:
                raise ValidationError("max_level must be an integer between 1 and 5")

    async def validate_document_access(
        self,
        document_id: str,
        user_id: int,
    ) -> None:
        """
        Validate user has access to document.

        Args:
            document_id: Document ID
            user_id: User ID

        Raises:
            PermissionDeniedError: If user lacks access
            ValidationError: If document not found
        """
        if not self.document_repo:
            logger.warning("Document repository not available for access check")
            return

        document = await self.document_repo.get(document_id)
        if not document:
            raise ValidationError(f"Document {document_id} not found")

        # Check if user owns the collection containing the document
        if document.collection and document.collection.owner_id != user_id:
            raise PermissionDeniedError(f"User {user_id} does not have access to document {document_id}")

    async def validate_collection_access(
        self,
        collection_id: str,
        user_id: int,
    ) -> None:
        """
        Validate user has access to collection.

        Args:
            collection_id: Collection ID
            user_id: User ID

        Raises:
            PermissionDeniedError: If user lacks access
            ValidationError: If collection not found
        """
        if not self.collection_repo:
            logger.warning("Collection repository not available for access check")
            return

        collection = await self.collection_repo.get(collection_id)
        if not collection:
            raise ValidationError(f"Collection {collection_id} not found")

        if collection.owner_id != user_id:
            raise PermissionDeniedError(f"User {user_id} does not have access to collection {collection_id}")

    def validate_collection_config(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate and normalize collection-level chunking configuration.

        Args:
            config: Configuration to validate

        Returns:
            Normalized configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        if not config:
            return {}

        if not isinstance(config, dict):
            raise ValidationError("Collection config must be a dictionary")

        # Validate strategy if present
        if "strategy" in config:
            self.validate_strategy(config["strategy"])

        # Validate strategy config if present
        if "strategy_config" in config:
            if "strategy" not in config:
                raise ValidationError("strategy_config requires strategy to be specified")

            self.validate_config(config["strategy"], config["strategy_config"])

        return config

    def validate_operation_params(
        self,
        operation_type: str,
        params: dict[str, Any],
    ) -> None:
        """
        Validate parameters for a chunking operation.

        Args:
            operation_type: Type of operation
            params: Operation parameters

        Raises:
            ValidationError: If parameters are invalid
        """
        valid_operations = ["preview", "process", "reprocess"]
        if operation_type not in valid_operations:
            raise ValidationError(
                f"Invalid operation type '{operation_type}'. Valid types: {', '.join(valid_operations)}"
            )

        if operation_type == "process" and "collection_id" not in params:
            raise ValidationError(
                field="collection_id",
                value=None,
                reason="Required for process operation"
            )

        if operation_type == "reprocess" and "collection_id" not in params:
            raise ValidationError("collection_id is required for reprocess operation")
        if operation_type == "reprocess" and "strategy" not in params:
            raise ValidationError("strategy is required for reprocess operation")
