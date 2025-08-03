#!/usr/bin/env python3
"""
Semantic chunking strategy using embedding similarity.

This module implements semantic chunking using LlamaIndex's SemanticSplitterNodeParser,
which determines natural breakpoints in text based on embedding similarity.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult
from packages.shared.utils.gpu_memory_monitor import GPUMemoryMonitor

# Conditional imports for CI compatibility
try:
    from llama_index.core import Document
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.openai import OpenAIEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    # Fallback for CI environments
    Document = None
    MockEmbedding = None  
    SemanticSplitterNodeParser = None
    OpenAIEmbedding = None
    LLAMA_INDEX_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """Semantic chunking using embeddings to find natural topic boundaries."""

    def __init__(
        self,
        embed_model: Optional[Any] = None,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1,
        max_chunk_size: int = 3000,
        max_retries: int = 3,
        embed_batch_size: Optional[int] = None,
    ):
        """Initialize semantic chunker with embedding model and parameters.

        Args:
            embed_model: Embedding model (defaults to OpenAI or Mock for testing)
            breakpoint_percentile_threshold: Sensitivity for topic boundaries (0-100)
            buffer_size: Context sentences to include around breakpoints
            max_chunk_size: Maximum tokens per chunk for safety
            max_retries: Maximum retry attempts for embedding API failures
        """
        # Check if LlamaIndex is available
        if not LLAMA_INDEX_AVAILABLE:
            logger.warning("LlamaIndex not available, semantic chunking will use fallback strategy")
            
        # Smart model selection prioritizing local embeddings (data privacy)
        if embed_model is None:
            if not LLAMA_INDEX_AVAILABLE:
                # Fallback when LlamaIndex is not available (CI environments)
                embed_model = None
                logger.info("LlamaIndex unavailable - using fallback mode")
            elif os.getenv("TESTING", "false").lower() == "true":
                embed_model = MockEmbedding(embed_dim=384)
                logger.info("Using MockEmbedding for testing")
            else:
                # ALWAYS use local embeddings by default to respect data privacy
                try:
                    from sentence_transformers import SentenceTransformer
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    
                    # Use high-quality local model that balances speed and accuracy
                    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Fast, good quality
                    self.model_name = model_name
                    
                    # Calculate optimal batch size if not provided
                    if embed_batch_size is None:
                        embed_batch_size = self._calculate_optimal_batch_size()
                    
                    # Initialize with GPU if available, CPU otherwise
                    embed_model = HuggingFaceEmbedding(
                        model_name=model_name,
                        device="cuda" if self._has_gpu() else "cpu",
                        embed_batch_size=embed_batch_size,
                        cache_folder=os.getenv("HF_CACHE_DIR", "./models"),
                    )
                    logger.info(f"Using local embedding model: {model_name} with batch_size={embed_batch_size} (preserving data privacy)")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize local embeddings: {e}")
                    
                    # Only fall back to OpenAI if explicitly enabled (DISCOURAGED)
                    if os.getenv("ENABLE_OPENAI_EMBEDDINGS", "false").lower() == "true":
                        try:
                            from llama_index.embeddings.openai import OpenAIEmbedding
                            embed_model = OpenAIEmbedding(
                                api_key=os.getenv("OPENAI_API_KEY"),
                                embed_batch_size=100,
                            )
                            logger.warning(
                                "Using OpenAI embeddings - DATA WILL LEAVE YOUR SYSTEM! "
                                "This violates Semantik's privacy principles. "
                                "Consider using local models instead."
                            )
                        except Exception as openai_error:
                            logger.error(f"OpenAI embeddings also failed: {openai_error}")
                            # Final fallback to mock - but NEVER in production
                            if os.getenv("TESTING", "false").lower() == "true":
                                embed_model = MockEmbedding(embed_dim=384)
                                logger.info("Using MockEmbedding for testing")
                            else:
                                # CRITICAL: Never silently degrade in production
                                logger.critical(
                                    "PRODUCTION FAILURE: Cannot initialize semantic chunking. "
                                    "Both local and OpenAI embeddings failed. "
                                    "Semantic chunking will not function."
                                )
                                self._alert_degraded_mode()
                                raise RuntimeError(
                                    "Cannot initialize semantic chunking: All embedding models failed. "
                                    "Please check your configuration and dependencies."
                                )
                    else:
                        # Final fallback to mock - but NEVER in production
                        if os.getenv("TESTING", "false").lower() == "true":
                            embed_model = MockEmbedding(embed_dim=384)
                            logger.info("Using MockEmbedding for testing")
                        else:
                            # CRITICAL: Never silently degrade in production
                            logger.critical(
                                "PRODUCTION FAILURE: Cannot initialize semantic chunking. "
                                "Local embeddings failed and OpenAI embeddings disabled. "
                                "Semantic chunking will not function."
                            )
                            self._alert_degraded_mode()
                            raise RuntimeError(
                                "Cannot initialize semantic chunking: No embedding model available. "
                                "Please install sentence-transformers, ensure GPU/CUDA support, "
                                "or enable external embeddings with ENABLE_OPENAI_EMBEDDINGS=true"
                            )

        try:
            if LLAMA_INDEX_AVAILABLE:
                self.splitter = SemanticSplitterNodeParser(
                    embed_model=embed_model,
                    breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                    buffer_size=buffer_size,
                )
            else:
                # Fallback for CI environments
                self.splitter = None
                logger.warning("SemanticSplitterNodeParser unavailable - using fallback mode")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticSplitterNodeParser: {e}")
            raise ValueError(f"Invalid semantic chunker configuration: {e}")

        self.max_chunk_size = max_chunk_size
        self.max_retries = max_retries
        self.strategy_name = "semantic"
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.buffer_size = buffer_size
        self.embed_batch_size = embed_batch_size if embed_batch_size is not None else self._calculate_optimal_batch_size()
        self._gpu_memory_monitor = GPUMemoryMonitor() if self._has_gpu() else None
        
        # Store model name if not set
        if not hasattr(self, 'model_name'):
            self.model_name = "unknown"

        logger.info(
            f"Initialized SemanticChunker with threshold={breakpoint_percentile_threshold}, "
            f"buffer_size={buffer_size}, max_chunk_size={max_chunk_size}, "
            f"embed_batch_size={self.embed_batch_size}"
        )

    def _has_gpu(self) -> bool:
        """Check if GPU is available for local embeddings.
        
        Returns:
            True if CUDA GPU is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.debug("PyTorch not available, GPU detection failed")
            return False
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory.
        
        Returns:
            Optimal batch size for current GPU memory state
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 8  # Conservative CPU batch size
            
            # Get GPU memory info
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            gpu_free_mb = (torch.cuda.get_device_properties(0).total_memory - 
                          torch.cuda.memory_allocated(0)) // (1024 * 1024)
            
            # Conservative estimation based on model size
            model_memory_per_batch = 50  # MB for all-MiniLM-L6-v2
            if hasattr(self, 'model_name'):
                if "large" in self.model_name.lower():
                    model_memory_per_batch = 150
                elif "base" in self.model_name.lower():
                    model_memory_per_batch = 100
                    
            # Use 70% of free memory, minimum 4, maximum 128
            safe_batch_size = max(4, min(128, int((gpu_free_mb * 0.7) // model_memory_per_batch)))
            
            logger.info(f"GPU memory: {gpu_memory_mb}MB total, {gpu_free_mb}MB free, "
                       f"using batch_size={safe_batch_size}")
            return safe_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}, using default")
            return 16  # Safe default

    def _alert_degraded_mode(self):
        """Alert monitoring systems about degraded functionality."""
        # Log for monitoring systems to pick up
        logger.critical(
            "SEMANTIC_CHUNKING_DEGRADED",
            extra={
                "alert_type": "service_degradation",
                "service": "semantic_chunking",
                "severity": "critical",
                "action_required": "install_local_embeddings"
            }
        )
        
        # TODO: Add integration with monitoring system (Prometheus, etc.)

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """Asynchronously chunk text using semantic boundaries.

        Args:
            text: Text to chunk
            doc_id: Document identifier for chunk IDs
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects representing semantic chunks
        """
        # Validate inputs
        self._validate_input(text, doc_id, metadata)
        
        if not text.strip():
            return []
            
        # Fallback when LlamaIndex is not available (CI environments)
        if not LLAMA_INDEX_AVAILABLE or self.splitter is None:
            logger.warning("Semantic chunking unavailable, falling back to basic chunking")
            # Simple fallback: split by sentences
            sentences = text.split('. ')
            chunks = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunks.append(ChunkResult(
                        text=sentence.strip() + ('.' if not sentence.endswith('.') else ''),
                        chunk_id=f"{doc_id}_fallback_{i}",
                        metadata={**(metadata or {}), "strategy": "semantic_fallback"}
                    ))
            return chunks
            
        # Start GPU memory monitoring if available
        if self._gpu_memory_monitor:
            self._gpu_memory_monitor.start_monitoring()

        try:
            # Create document for LlamaIndex
            doc = Document(text=text, metadata=metadata or {})

            # Check memory usage and adjust batch size if needed
            if self._gpu_memory_monitor and self._gpu_memory_monitor.memory_usage() > 0.8:
                logger.warning("High GPU memory usage detected, reducing batch size")
                original_batch = self.embed_batch_size
                new_batch = max(2, self.embed_batch_size // 2)
                self.embed_batch_size = new_batch
                self._update_embed_batch_size(new_batch)
                try:
                    nodes = await self._process_with_memory_management_async(doc)
                finally:
                    self.embed_batch_size = original_batch
                    self._update_embed_batch_size(original_batch)
            else:
                # Run CPU-bound operation in executor to avoid blocking
                loop = asyncio.get_event_loop()
                nodes = await loop.run_in_executor(
                    None,
                    self._chunk_with_retry,
                    doc,
                )

            # Convert nodes to ChunkResult format
            results = []
            for idx, node in enumerate(nodes):
                # Ensure chunk size safety
                chunk_text = node.text
                if len(chunk_text) > self.max_chunk_size:
                    chunk_text = chunk_text[: self.max_chunk_size]
                    logger.warning(
                        f"Truncated oversized semantic chunk: "
                        f"{len(node.text)} -> {self.max_chunk_size} characters"
                    )

                # Calculate character offsets
                start_offset = node.start_char_idx if node.start_char_idx is not None else 0
                end_offset = node.end_char_idx if node.end_char_idx is not None else start_offset + len(chunk_text)

                # Build comprehensive metadata
                chunk_metadata = {
                    **(metadata or {}),
                    "strategy": self.strategy_name,
                    "breakpoint_threshold": self.breakpoint_percentile_threshold,
                    "buffer_size": self.buffer_size,
                    "chunk_index": idx,
                    "total_chunks": len(nodes),
                }

                # Add semantic-specific metadata if available
                if hasattr(node, "metadata") and node.metadata:
                    chunk_metadata.update(node.metadata)

                results.append(
                    ChunkResult(
                        chunk_id=f"{doc_id}_{idx:04d}",
                        text=chunk_text,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        metadata=chunk_metadata,
                    )
                )

            logger.info(
                f"Semantic chunking created {len(results)} chunks for document {doc_id} "
                f"({len(text)} characters)"
            )
            return results

        except Exception as e:
            logger.error(f"Semantic chunking failed for document {doc_id}: {e}")
            # Fallback to recursive chunking
            return await self._fallback_chunking(text, doc_id, metadata)
        finally:
            # Clean up GPU memory
            if self._gpu_memory_monitor:
                self._gpu_memory_monitor.cleanup()

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """Synchronously chunk text using semantic boundaries.

        Args:
            text: Text to chunk
            doc_id: Document identifier for chunk IDs
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects representing semantic chunks
        """
        # Validate inputs
        self._validate_input(text, doc_id, metadata)
        
        if not text.strip():
            return []
            
        # Fallback when LlamaIndex is not available (CI environments)
        if not LLAMA_INDEX_AVAILABLE or self.splitter is None:
            logger.warning("Semantic chunking unavailable, falling back to basic chunking")
            # Simple fallback: split by sentences
            sentences = text.split('. ')
            chunks = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunks.append(ChunkResult(
                        text=sentence.strip() + ('.' if not sentence.endswith('.') else ''),
                        chunk_id=f"{doc_id}_fallback_{i}",
                        metadata={**(metadata or {}), "strategy": "semantic_fallback"}
                    ))
            return chunks
            
        # Start GPU memory monitoring if available  
        if self._gpu_memory_monitor:
            self._gpu_memory_monitor.start_monitoring()

        try:
            # Create document for LlamaIndex
            doc = Document(text=text, metadata=metadata or {})

            # Check memory usage and adjust batch size if needed
            if self._gpu_memory_monitor and self._gpu_memory_monitor.memory_usage() > 0.8:
                logger.warning("High GPU memory usage detected, reducing batch size")
                original_batch = self.embed_batch_size
                new_batch = max(2, self.embed_batch_size // 2)
                self.embed_batch_size = new_batch
                self._update_embed_batch_size(new_batch)
                try:
                    nodes = self._chunk_with_retry(doc)
                finally:
                    self.embed_batch_size = original_batch
                    self._update_embed_batch_size(original_batch)
            else:
                # Process with retry logic
                nodes = self._chunk_with_retry(doc)

            # Convert nodes to ChunkResult format (same logic as async)
            results = []
            for idx, node in enumerate(nodes):
                chunk_text = node.text
                if len(chunk_text) > self.max_chunk_size:
                    chunk_text = chunk_text[: self.max_chunk_size]
                    logger.warning(
                        f"Truncated oversized semantic chunk: "
                        f"{len(node.text)} -> {self.max_chunk_size} characters"
                    )

                start_offset = node.start_char_idx if node.start_char_idx is not None else 0
                end_offset = node.end_char_idx if node.end_char_idx is not None else start_offset + len(chunk_text)

                chunk_metadata = {
                    **(metadata or {}),
                    "strategy": self.strategy_name,
                    "breakpoint_threshold": self.breakpoint_percentile_threshold,
                    "buffer_size": self.buffer_size,
                    "chunk_index": idx,
                    "total_chunks": len(nodes),
                }

                if hasattr(node, "metadata") and node.metadata:
                    chunk_metadata.update(node.metadata)

                results.append(
                    ChunkResult(
                        chunk_id=f"{doc_id}_{idx:04d}",
                        text=chunk_text,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        metadata=chunk_metadata,
                    )
                )

            logger.info(
                f"Semantic chunking created {len(results)} chunks for document {doc_id} "
                f"({len(text)} characters)"
            )
            return results

        except Exception as e:
            logger.error(f"Semantic chunking failed for document {doc_id}: {e}")
            # Fallback to recursive chunking (sync version)
            return self._fallback_chunking_sync(text, doc_id, metadata)
        finally:
            # Clean up GPU memory
            if self._gpu_memory_monitor:
                self._gpu_memory_monitor.cleanup()

    def _chunk_with_retry(self, doc: Document, max_retries: Optional[int] = None) -> List[Any]:
        """Chunk document with retry logic for embedding API failures.

        Args:
            doc: LlamaIndex document to chunk
            max_retries: Override default max_retries

        Returns:
            List of LlamaIndex nodes

        Raises:
            Exception: If all retry attempts fail
        """
        retries = max_retries if max_retries is not None else self.max_retries

        for attempt in range(retries):
            try:
                nodes = self.splitter.get_nodes_from_documents([doc])
                if nodes:  # Success!
                    if attempt > 0:
                        logger.info(f"Semantic chunking succeeded on attempt {attempt + 1}")
                    return nodes
                else:
                    logger.warning("Semantic chunking returned empty nodes")
                    if attempt < retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    continue

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Semantic chunking attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Semantic chunking failed after {retries} attempts: {e}")
                    raise

        # Should not reach here, but safety fallback
        raise RuntimeError(f"Semantic chunking failed after {retries} attempts")

    async def _fallback_chunking(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        """Fallback to recursive chunking on semantic failure.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            metadata: Metadata to preserve

        Returns:
            List of ChunkResult from recursive chunking
        """
        logger.info(f"Falling back to recursive chunking for document {doc_id}")

        try:
            # Import here to avoid circular import
            from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker

            fallback = RecursiveChunker(chunk_size=600, chunk_overlap=100)
            chunks = await fallback.chunk_text_async(text, doc_id, metadata)

            # Update metadata to indicate fallback was used
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "original_strategy": self.strategy_name,
                        "fallback_strategy": "recursive",
                        "fallback_reason": "semantic_chunking_failed",
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Fallback chunking also failed for document {doc_id}: {e}")
            # Last resort: return single chunk
            return [
                ChunkResult(
                    chunk_id=f"{doc_id}_0000",
                    text=text[: self.max_chunk_size],
                    start_offset=0,
                    end_offset=min(len(text), self.max_chunk_size),
                    metadata={
                        **(metadata or {}),
                        "strategy": "emergency_fallback",
                        "original_strategy": self.strategy_name,
                        "fallback_reason": "all_chunking_failed",
                    },
                )
            ]

    def _fallback_chunking_sync(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkResult]:
        """Synchronous fallback to recursive chunking.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            metadata: Metadata to preserve

        Returns:
            List of ChunkResult from recursive chunking
        """
        logger.info(f"Falling back to recursive chunking for document {doc_id}")

        try:
            from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker

            fallback = RecursiveChunker(chunk_size=600, chunk_overlap=100)
            chunks = fallback.chunk_text(text, doc_id, metadata)

            # Update metadata to indicate fallback was used
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "original_strategy": self.strategy_name,
                        "fallback_strategy": "recursive",
                        "fallback_reason": "semantic_chunking_failed",
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Fallback chunking also failed for document {doc_id}: {e}")
            # Last resort: return single chunk
            return [
                ChunkResult(
                    chunk_id=f"{doc_id}_0000",
                    text=text[: self.max_chunk_size],
                    start_offset=0,
                    end_offset=min(len(text), self.max_chunk_size),
                    metadata={
                        **(metadata or {}),
                        "strategy": "emergency_fallback",
                        "original_strategy": self.strategy_name,
                        "fallback_reason": "all_chunking_failed",
                    },
                )
            ]

    def validate_config(self, params: Dict[str, Any]) -> bool:
        """Validate semantic chunking configuration parameters.

        Args:
            params: Configuration parameters to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate breakpoint_percentile_threshold
            threshold = params.get("breakpoint_percentile_threshold", self.breakpoint_percentile_threshold)
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 100):
                logger.error(f"Invalid breakpoint_percentile_threshold: {threshold}")
                return False

            # Validate buffer_size
            buffer_size = params.get("buffer_size", self.buffer_size)
            if not isinstance(buffer_size, int) or buffer_size < 0:
                logger.error(f"Invalid buffer_size: {buffer_size}")
                return False

            # Validate max_chunk_size
            max_chunk_size = params.get("max_chunk_size", self.max_chunk_size)
            if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
                logger.error(f"Invalid max_chunk_size: {max_chunk_size}")
                return False

            return True

        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    def estimate_chunks(self, text_length: int, params: Dict[str, Any]) -> int:
        """Estimate number of chunks for given text length.

        Args:
            text_length: Length of text in characters
            params: Chunking parameters

        Returns:
            Estimated number of chunks
        """
        # Semantic chunking is content-dependent, so this is a rough estimate
        # Based on typical performance: ~600-800 characters per chunk
        avg_chunk_size = params.get("avg_chunk_size", 700)
        base_estimate = max(1, text_length // avg_chunk_size)

        # Adjust based on breakpoint threshold (higher threshold = fewer chunks)
        threshold = params.get("breakpoint_percentile_threshold", self.breakpoint_percentile_threshold)
        threshold_factor = 0.5 + (threshold / 200)  # Range: 0.5 to 1.0

        estimated = max(1, int(base_estimate * threshold_factor))

        logger.debug(
            f"Estimated {estimated} chunks for {text_length} characters "
            f"with threshold {threshold}"
        )

        return estimated
    
    async def _process_with_memory_management_async(self, doc: Document) -> List[Any]:
        """Process document with active memory management in async context.
        
        Args:
            doc: Document to process
            
        Returns:
            List of nodes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._chunk_with_retry,
            doc,
        )
    
    def _update_embed_batch_size(self, new_batch_size: int):
        """Update the embedding model's batch size dynamically.
        
        Args:
            new_batch_size: New batch size to use
        """
        try:
            # Update the splitter's embed model batch size if possible
            if hasattr(self.splitter, '_embed_model') and hasattr(self.splitter._embed_model, 'embed_batch_size'):
                self.splitter._embed_model.embed_batch_size = new_batch_size
                logger.info(f"Updated embedding model batch size to {new_batch_size}")
            elif hasattr(self.splitter, 'embed_model') and hasattr(self.splitter.embed_model, 'embed_batch_size'):
                self.splitter.embed_model.embed_batch_size = new_batch_size
                logger.info(f"Updated embedding model batch size to {new_batch_size}")
            else:
                logger.debug("Could not update embedding model batch size - attribute not found")
        except Exception as e:
            logger.warning(f"Failed to update embedding model batch size: {e}")