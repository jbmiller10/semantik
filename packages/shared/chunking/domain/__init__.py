#!/usr/bin/env python3
"""
Pure domain layer for chunking operations.

This module provides the core business logic for text chunking,
completely independent of any infrastructure concerns.
"""

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.entities.chunk_collection import ChunkCollection
from packages.shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from packages.shared.chunking.domain.exceptions import (
    ChunkingDomainError,
    ChunkSizeViolationError,
    DocumentTooLargeError,
    InvalidChunkError,
    InvalidConfigurationError,
    InvalidStateError,
    OverlapConfigurationError,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus

__all__ = [
    # Entities
    "Chunk",
    "ChunkCollection",
    "ChunkingOperation",
    # Value Objects
    "ChunkConfig",
    "ChunkMetadata",
    "OperationStatus",
    # Exceptions
    "ChunkingDomainError",
    "InvalidStateError",
    "InvalidChunkError",
    "InvalidConfigurationError",
    "DocumentTooLargeError",
    "ChunkSizeViolationError",
    "OverlapConfigurationError",
]
