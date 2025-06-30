#!/usr/bin/env python3
"""
Document extraction and chunking module V2
Uses tiktoken for accurate token counting
Implements file change tracking with SHA256
Now uses unstructured library for unified document parsing
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Unstructured for document parsing
from unstructured.partition.auto import partition

from vecpipe.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 600  # tokens
DEFAULT_CHUNK_OVERLAP = 200  # tokens


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
        except:
            # Fallback to default encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"Initialized tokenizer: {model_name}, chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}"
        )

    def chunk_text(self, text: str, doc_id: str, metadata: dict | None = None) -> list[dict]:
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


class FileChangeTracker:
    """Track file changes using SHA256 and SCD-like approach"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(settings.FILE_TRACKING_DB)
        self.db_path = db_path
        self.tracking_data = self._load_tracking_data()

    def _load_tracking_data(self) -> dict:
        """Load tracking data from JSON file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path) as f:
                    return json.load(f)
            except:
                logger.warning(f"Failed to load tracking data from {self.db_path}")
        return {"files": {}}

    def _save_tracking_data(self):
        """Save tracking data to JSON file"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.tracking_data, f, indent=2)

    def get_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def should_process_file(self, filepath: str) -> tuple[bool, str | None]:
        """Check if file should be processed based on hash"""
        current_hash = self.get_file_hash(filepath)
        file_key = str(Path(filepath).absolute())

        if file_key in self.tracking_data["files"]:
            file_info = self.tracking_data["files"][file_key]
            if file_info["hash"] == current_hash:
                # File unchanged
                return False, current_hash
            else:
                # File changed
                logger.info(f"File changed: {filepath}")
                return True, current_hash
        else:
            # New file
            logger.info(f"New file: {filepath}")
            return True, current_hash

    def update_file_tracking(self, filepath: str, file_hash: str, doc_id: str, chunks_created: int):
        """Update tracking information for a file"""
        file_key = str(Path(filepath).absolute())
        now = datetime.now().isoformat()

        if file_key in self.tracking_data["files"]:
            # Update existing entry
            file_info = self.tracking_data["files"][file_key]
            file_info["hash"] = file_hash
            file_info["last_seen"] = now
            file_info["last_processed"] = now
            file_info["doc_id"] = doc_id
            file_info["chunks_created"] = chunks_created
            file_info["process_count"] = file_info.get("process_count", 0) + 1
        else:
            # Create new entry
            self.tracking_data["files"][file_key] = {
                "hash": file_hash,
                "first_seen": now,
                "last_seen": now,
                "last_processed": now,
                "doc_id": doc_id,
                "chunks_created": chunks_created,
                "process_count": 1,
            }

        self._save_tracking_data()

    def get_removed_files(self, current_files: list[str]) -> list[dict]:
        """Find files that were tracked but no longer exist"""
        current_file_keys = {str(Path(f).absolute()) for f in current_files}
        removed_files = []

        for file_key, file_info in self.tracking_data["files"].items():
            if file_key not in current_file_keys:
                removed_files.append(
                    {"path": file_key, "doc_id": file_info.get("doc_id"), "last_seen": file_info.get("last_seen")}
                )

        return removed_files

    def remove_file(self, filepath: str):
        """Remove a file from tracking"""
        file_key = str(Path(filepath).absolute()) if not os.path.isabs(filepath) else filepath
        if file_key in self.tracking_data["files"]:
            del self.tracking_data["files"][file_key]
            logger.info(f"Removed file from tracking: {file_key}")

    def save(self):
        """Save tracking data to disk"""
        self._save_tracking_data()


def extract_and_serialize(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Uses unstructured to partition a file and serializes structured data.
    Returns list of (text, metadata) tuples."""
    ext = Path(filepath).suffix.lower()

    # Use unstructured for all file types
    try:
        elements = partition(
            filename=filepath,
            strategy="auto",  # Let unstructured determine the best strategy
            include_page_breaks=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
        )

        results = []
        current_page = 1

        for element in elements:
            # Extract text content
            text = str(element)
            if not text.strip():
                continue

            # Build metadata
            metadata = {"filename": os.path.basename(filepath), "file_type": ext[1:] if ext else "unknown"}

            # Add element-specific metadata
            if hasattr(element, "metadata"):
                elem_meta = element.metadata
                if hasattr(elem_meta, "page_number") and elem_meta.page_number:
                    metadata["page_number"] = elem_meta.page_number
                    current_page = elem_meta.page_number
                else:
                    metadata["page_number"] = current_page

                if hasattr(elem_meta, "category"):
                    metadata["element_type"] = elem_meta.category

                # Add any coordinates if available
                if hasattr(elem_meta, "coordinates"):
                    metadata["has_coordinates"] = True

            results.append((text, metadata))

        return results

    except Exception as e:
        logger.error(f"Failed to extract from {filepath} using unstructured: {e}")
        raise


def extract_text(filepath: str, timeout: int = 300) -> str:
    """Legacy function for backward compatibility - extracts text without metadata
    Note: timeout parameter is kept for backward compatibility but not used"""
    try:
        results = extract_and_serialize(filepath)
        # Concatenate all text parts
        text_parts = [text for text, _ in results]
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from {filepath}: {e}")
        raise


def process_file_v2(filepath: str, output_dir: str, chunker: TokenChunker, tracker: FileChangeTracker) -> str | None:
    """Process a single file with change tracking and metadata preservation"""
    try:
        # Check if file needs processing
        should_process, file_hash = tracker.should_process_file(filepath)

        if not should_process:
            logger.info(f"Skipping unchanged file: {filepath}")
            return None

        # Generate document ID from file path
        doc_id = hashlib.md5(filepath.encode()).hexdigest()[:16]

        # Check if output already exists (for resume capability)
        output_path = os.path.join(output_dir, f"{doc_id}.parquet")
        if os.path.exists(output_path):
            # Verify the existing output is valid
            try:
                existing_table = pq.read_table(output_path)
                if len(existing_table) > 0:
                    logger.info(f"Valid output exists for: {filepath}")
                    tracker.update_file_tracking(filepath, file_hash, doc_id, len(existing_table))
                    return output_path
            except:
                logger.warning(f"Invalid existing output, reprocessing: {filepath}")

        # Extract text and metadata
        logger.info(f"Extracting: {filepath}")
        text_blocks = extract_and_serialize(filepath)

        if not text_blocks:
            logger.warning(f"No text extracted from: {filepath}")
            tracker.update_file_tracking(filepath, file_hash, doc_id, 0)
            return None

        # Process each text block
        all_chunks = []
        for text, metadata in text_blocks:
            if not text.strip():
                continue

            # Chunk text using token-based chunking
            chunks = chunker.chunk_text(text, doc_id, metadata)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from: {filepath}")

        # Convert to parquet with metadata
        if all_chunks:
            # Prepare data for parquet
            df_data = {
                "doc_id": [],
                "chunk_id": [],
                "path": [],
                "text": [],
                "token_count": [],
                "file_hash": [],
                "metadata": [],
            }

            for chunk in all_chunks:
                df_data["doc_id"].append(chunk["doc_id"])
                df_data["chunk_id"].append(chunk["chunk_id"])
                df_data["path"].append(filepath)
                df_data["text"].append(chunk["text"])
                df_data["token_count"].append(chunk["token_count"])
                df_data["file_hash"].append(file_hash)
                # Store metadata as JSON string for parquet compatibility
                df_data["metadata"].append(json.dumps(chunk.get("metadata", {})))

            table = pa.table(df_data)
            pq.write_table(table, output_path)

            # Update tracking
            tracker.update_file_tracking(filepath, file_hash, doc_id, len(all_chunks))

            return output_path

        return None

    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        # Log error to file
        with open(ERROR_LOG, "a") as f:
            f.write(f"{filepath}\t{str(e)}\n")
        return None


# Compatibility wrappers for old API
_default_chunker = None


def chunk_text(text: str, doc_id: str) -> list[dict]:
    """Compatibility wrapper for old chunk_text function"""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = TokenChunker()
    return _default_chunker.chunk_text(text, doc_id)


def process_file(filepath: str, output_dir: str) -> str | None:
    """Compatibility wrapper for old process_file function"""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = TokenChunker()
    tracker = FileChangeTracker()
    return process_file_v2(filepath, output_dir, _default_chunker, tracker)


def main():
    parser = argparse.ArgumentParser(description="Extract and chunk documents (V2)")
    parser.add_argument("--input", "-i", required=True, help="Input file list (null-delimited)")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="Output directory for parquet files")
    parser.add_argument("--chunk-size", "-cs", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in tokens")
    parser.add_argument(
        "--chunk-overlap", "-co", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in tokens"
    )
    parser.add_argument("--model-encoding", "-m", default="cl100k_base", help="Tiktoken model encoding")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Files to process per batch")

    args = parser.parse_args()

    # Initialize components
    chunker = TokenChunker(model_name=args.model_encoding, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    tracker = FileChangeTracker()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Read file list
    with open(args.input, "rb") as f:
        file_list = f.read().decode("utf-8").split("\0")
        file_list = [f for f in file_list if f]  # Remove empty strings

    logger.info(f"Found {len(file_list)} files to process")

    # Check for removed files
    removed_files = tracker.get_removed_files(file_list)
    if removed_files:
        logger.info(f"Found {len(removed_files)} removed files")
        for removed in removed_files:
            logger.info(f"File removed: {removed['path']} (doc_id: {removed['doc_id']})")

    # Process files
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for filepath in tqdm(file_list, desc="Processing files"):
        result = process_file_v2(filepath, args.output, chunker, tracker)
        if result:
            processed_count += 1
        elif result is None:
            failed_count += 1
        else:
            skipped_count += 1

    logger.info(f"Processing complete: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")


if __name__ == "__main__":
    main()
