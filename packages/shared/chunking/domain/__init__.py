#!/usr/bin/env python3
"""
Pure domain layer for chunking operations.

This module provides the core business logic for text chunking,
completely independent of any infrastructure concerns.
"""

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.entities.chunk_collection import ChunkCollection
from shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from shared.chunking.domain.exceptions import (
    ChunkingDomainError,
    ChunkSizeViolationError,
    DocumentTooLargeError,
    InvalidChunkError,
    InvalidConfigurationError,
    InvalidStateError,
    OverlapConfigurationError,
)
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.domain.value_objects.operation_status import OperationStatus

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
