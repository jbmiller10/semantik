"""
Compatibility module for ChunkingService.

This module provides backward compatibility by re-exporting the ChunkingServiceAdapter
as ChunkingService. This allows existing code to continue working while we transition
to the new domain-driven architecture.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .chunking_service_adapter import ChunkingServiceAdapter as ChunkingService


@dataclass
class ChunkingStatistics:
    """Statistics for chunking operations."""
    
    total_chunks: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    total_documents: int = 0
    failed_documents: int = 0
    processing_time_ms: float = 0.0
    strategy_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.strategy_distribution is None:
            self.strategy_distribution = {}


__all__ = ["ChunkingService", "ChunkingStatistics"]