#!/usr/bin/env python3
"""
Text chunking module - handles splitting text into token-based chunks.
Uses tiktoken for accurate token counting.
"""

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class TokenChunker:
    """Chunk text by token count using tiktoken"""

    def __init__(self, model_name: str = "cl100k_base", chunk_size: int = 600, chunk_overlap: int = 200):
        """Initialize tokenizer for chunking"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Validate parameters to prevent infinite loops
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(
                f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}), setting overlap to chunk_size/2"
            )
            self.chunk_overlap = self.chunk_size // 2

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        # Use tiktoken for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception:
            # Fallback to default encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"Initialized tokenizer: {model_name}, chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}"
        )

    def chunk_text(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into overlapping chunks by token count"""
        if not text.strip():
            return []

        logger.info(f"Starting tokenization for doc_id: {doc_id}, text length: {len(text)} chars")

        # Tokenize entire text
        import time

        start_time = time.time()
        tokens = self.tokenizer.encode(text)
        tokenize_time = time.time() - start_time
        total_tokens = len(tokens)

        logger.info(f"Tokenization complete in {tokenize_time:.2f}s: {total_tokens} tokens from {len(text)} chars")

        if total_tokens <= self.chunk_size:
            # Text fits in single chunk
            chunk_data = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_0000",
                "text": text.strip(),
                "token_count": total_tokens,
                "start_token": 0,
                "end_token": total_tokens,
            }
            if metadata:
                chunk_data["metadata"] = metadata
            return [chunk_data]

        chunks = []
        chunk_id = 0
        start = 0

        logger.info(
            f"Starting chunking: {total_tokens} tokens, chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}"
        )

        while start < total_tokens:
            # Determine chunk boundaries
            end = min(start + self.chunk_size, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Try to break at sentence boundary if not at end
            if end < total_tokens:
                # Look for sentence end in last 10% of chunk
                search_start = int(len(chunk_tokens) * 0.9)
                best_break = len(chunk_tokens)

                for i in range(search_start, len(chunk_tokens)):
                    decoded = self.tokenizer.decode(chunk_tokens[:i])
                    if decoded.rstrip().endswith((".", "!", "?", "\n\n")):
                        best_break = i
                        break

                # Adjust chunk if we found a better break point
                if best_break < len(chunk_tokens):
                    chunk_tokens = chunk_tokens[:best_break]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    end = start + best_break

            chunk_data = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_id:04d}",
                "text": chunk_text.strip(),
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
            }
            if metadata:
                chunk_data["metadata"] = metadata

            chunks.append(chunk_data)

            # Free chunk_tokens reference
            del chunk_tokens

            chunk_id += 1

            # Move start position with overlap
            next_start = end - self.chunk_overlap

            # Ensure progress - this is normal at the end of the document
            if next_start <= start:
                # We've reached the end of the document or overlap is too large
                # Just move to the end position
                next_start = end

            start = next_start

            # Safety check for actual infinite loops
            if chunk_id > 10000:
                logger.error(f"Too many chunks created ({chunk_id}), stopping to prevent infinite loop")
                break

        # Free the tokens list after processing
        del tokens

        logger.debug(f"Created {len(chunks)} chunks from {total_tokens} tokens")
        return chunks
