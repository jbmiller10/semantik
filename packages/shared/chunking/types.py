"""
Shared chunking types used across services.

This module provides core type definitions for chunking operations that need
to be shared between the API layer and service layer without creating circular
imports.
"""

from enum import Enum


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"
    # Legacy/aliases retained for backward compatibility (not used in tests)
    SLIDING_WINDOW = "sliding_window"
    DOCUMENT_STRUCTURE = "document_structure"


class ChunkingStatus(str, Enum):
    """Status of chunking operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class QualityLevel(str, Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
