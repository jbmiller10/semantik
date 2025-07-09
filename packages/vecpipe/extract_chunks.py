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
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import from shared package
from shared.config import settings
from shared.text_processing.chunking import TokenChunker
from shared.text_processing.extraction import extract_and_serialize

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 600  # tokens
DEFAULT_CHUNK_OVERLAP = 200  # tokens


class FileChangeTracker:
    """Track file changes using SHA256 and SCD-like approach"""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(settings.file_tracking_db)
        self.db_path = db_path
        self.tracking_data = self._load_tracking_data()

    def _load_tracking_data(self) -> dict[str, Any]:
        """Load tracking data from JSON file"""
        if Path(self.db_path).exists():
            try:
                with Path(self.db_path).open() as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception:
                logger.warning(f"Failed to load tracking data from {self.db_path}")
        return {"files": {}}

    def _save_tracking_data(self) -> None:
        """Save tracking data to JSON file"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(self.db_path).open("w") as f:
            json.dump(self.tracking_data, f, indent=2)

    def get_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with Path(filepath).open("rb") as f:
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
            # File changed
            logger.info(f"File changed: {filepath}")
            return True, current_hash
        # New file
        logger.info(f"New file: {filepath}")
        return True, current_hash

    def update_file_tracking(self, filepath: str, file_hash: str, doc_id: str, chunks_created: int) -> None:
        """Update tracking information for a file"""
        file_key = str(Path(filepath).absolute())
        now = datetime.now(UTC).isoformat()

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

    def remove_file(self, filepath: str) -> None:
        """Remove a file from tracking"""
        file_key = str(Path(filepath).absolute()) if not Path(filepath).is_absolute() else filepath
        if file_key in self.tracking_data["files"]:
            del self.tracking_data["files"][file_key]
            logger.info(f"Removed file from tracking: {file_key}")

    def save(self) -> None:
        """Save tracking data to disk"""
        self._save_tracking_data()


def process_file_v2(filepath: str, output_dir: str, chunker: TokenChunker, tracker: FileChangeTracker) -> str | None:
    """Process a single file with change tracking and metadata preservation"""
    try:
        # Check if file needs processing
        should_process, file_hash = tracker.should_process_file(filepath)
        assert file_hash is not None  # file_hash is always returned

        if not should_process:
            logger.info(f"Skipping unchanged file: {filepath}")
            return None

        # Generate document ID from file path
        doc_id = hashlib.md5(filepath.encode()).hexdigest()[:16]

        # Check if output already exists (for resume capability)
        output_path = Path(output_dir) / f"{doc_id}.parquet"
        if output_path.exists():
            # Verify the existing output is valid
            try:
                existing_table = pq.read_table(output_path)
                if len(existing_table) > 0:
                    logger.info(f"Valid output exists for: {filepath}")
                    tracker.update_file_tracking(filepath, file_hash, doc_id, len(existing_table))
                    return str(output_path)
            except Exception:
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
            df_data: dict[str, list[Any]] = {
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
            pq.write_table(table, str(output_path))

            # Update tracking
            tracker.update_file_tracking(filepath, file_hash, doc_id, len(all_chunks))

            return str(output_path)

        return None

    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        # Log error to file
        with Path(settings.error_log).open("a") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk documents (V2)")
    parser.add_argument("--input", "-i", required=True, help="Input file list (null-delimited)")
    parser.add_argument("--output", "-o", default=str(settings.output_dir), help="Output directory for parquet files")
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
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Read file list
    with Path(args.input).open("rb") as f:
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
