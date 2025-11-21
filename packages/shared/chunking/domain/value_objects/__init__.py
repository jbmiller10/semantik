#!/usr/bin/env python3
"""
Value objects for chunking domain.

Value objects are immutable objects that represent concepts in the domain
without identity.
"""

from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.domain.value_objects.operation_status import OperationStatus

__all__ = ["ChunkConfig", "ChunkMetadata", "OperationStatus"]
