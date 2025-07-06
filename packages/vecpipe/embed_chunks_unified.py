#!/usr/bin/env python3
"""
Unified CLI entry point for embedding generation
Uses the webui.embedding_service.EmbeddingService for all embedding operations
"""

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm

# Add parent directory to path to import webui module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from webui.embedding_service import EmbeddingService

from .config import settings
from .metrics import (
    TimingContext,
    embedding_batch_duration,
    extraction_duration,
    ingestion_duration,
    metrics_collector,
    record_chunks_created,
    record_embeddings_generated,
    record_file_failed,
    record_file_processed,
    start_metrics_server,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants - matching the old embed_chunks.py
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 96
INPUT_DIR = str(settings.EXTRACT_DIR)
OUTPUT_DIR = str(settings.INGEST_DIR)
MAX_CONCURRENT_IO = 4


async def read_parquet_async(file_path: str) -> dict[str, Any]:
    """Async read parquet file"""
    loop = asyncio.get_event_loop()

    def _read() -> dict[str, Any]:
        with TimingContext(extraction_duration):
            table = pq.read_table(file_path)
            result = {
                "doc_ids": table.column("doc_id").to_pylist(),
                "chunk_ids": table.column("chunk_id").to_pylist(),
                "paths": table.column("path").to_pylist(),
                "texts": table.column("text").to_pylist(),
                "file_path": file_path,
            }

            # Check if metadata column exists
            if "metadata" in table.column_names:
                import json

                # Parse JSON metadata strings
                metadata_strs = table.column("metadata").to_pylist()
                result["metadata"] = [json.loads(m) if m else {} for m in metadata_strs]
            else:
                result["metadata"] = [{}] * len(result["doc_ids"])

            return result

    return await loop.run_in_executor(None, _read)


async def write_parquet_async(output_path: str, data: dict[str, Any]) -> None:
    """Async write parquet file"""
    loop = asyncio.get_event_loop()

    def _write() -> None:
        with TimingContext(ingestion_duration):
            output_table = pa.table(data)
            pq.write_table(output_table, output_path)

    await loop.run_in_executor(None, _write)


async def process_file_async(
    file_path: str, output_dir: str, embedding_service: EmbeddingService, args: argparse.Namespace
) -> str | None:
    """Process a single file asynchronously"""
    try:
        # Generate output filename
        filename = Path(file_path).stem
        output_path = Path(output_dir) / f"{filename}_embedded.parquet"

        # Skip if already processed
        if output_path.exists():
            logger.info(f"Skipping already processed: {file_path}")
            return str(output_path)

        logger.info(f"Processing: {file_path}")

        # Read input asynchronously
        data = await read_parquet_async(file_path)

        texts = data["texts"]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        # Record chunks created
        record_chunks_created(len(texts))

        # Generate embeddings using the unified service with timing
        with TimingContext(embedding_batch_duration):
            embeddings = embedding_service.generate_embeddings(
                texts=texts,
                model_name=args.model,
                quantization=args.quantization,
                batch_size=args.batch_size,
                show_progress=False,  # Disable per-file progress since we have overall progress
            )

        if embeddings is None:
            logger.error(f"Failed to generate embeddings for {file_path}")
            record_file_failed("embedding", "generation_error")
            return None

        # Record embeddings generated
        record_embeddings_generated(len(embeddings))

        # Generate unique IDs for each point
        point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        # Prepare output data
        output_data = {
            "id": point_ids,
            "vector": embeddings.tolist(),
            "payload": [
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "path": path,
                    "content": text,  # Add full text content for hybrid search
                    "metadata": metadata,  # Include metadata from extraction
                }
                for doc_id, chunk_id, path, text, metadata in zip(
                    data["doc_ids"], data["chunk_ids"], data["paths"], texts, data["metadata"], strict=False
                )
            ],
        }

        # Write output asynchronously
        await write_parquet_async(str(output_path), output_data)

        logger.info(f"Saved embeddings to: {output_path}")
        record_file_processed("embedding")
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        record_file_failed("embedding", type(e).__name__)
        return None


async def process_files_parallel(
    file_paths: list[str], output_dir: str, embedding_service: EmbeddingService, args: argparse.Namespace
) -> list[str | None]:
    """Process multiple files in parallel"""
    # Create semaphore to limit concurrent I/O operations
    io_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IO)

    async def process_with_limit(file_path: str) -> str | None:
        async with io_semaphore:
            return await process_file_async(file_path, output_dir, embedding_service, args)

    # Process all files concurrently
    tasks = [process_with_limit(fp) for fp in file_paths]

    # Use tqdm for progress tracking
    results = []
    for coro in tqdm.as_completed(tasks, desc="Processing files"):
        result = await coro
        results.append(result)

    return results


async def main_async(args: argparse.Namespace) -> None:
    """Main async function"""
    # Start metrics server if requested
    if args.metrics_port:
        try:
            start_metrics_server(args.metrics_port)
            logger.info(f"Metrics server started on port {args.metrics_port}")
        except OSError as e:
            logger.warning(f"Failed to start metrics server on port {args.metrics_port}: {e}")
            logger.info("Continuing without metrics server")

    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Initialize unified embedding service
    logger.info("Initializing unified EmbeddingService...")
    embedding_service = EmbeddingService(mock_mode=args.mock)

    # Load the model
    if not args.mock:
        logger.info(f"Loading model: {args.model} with quantization: {args.quantization}")
        if not embedding_service.load_model(args.model, args.quantization):
            logger.error("Failed to load model")
            return

    try:
        # Find input files
        input_path = Path(args.input)
        input_files = list(input_path.glob(args.pattern))

        if not input_files:
            logger.warning(f"No files found matching pattern: {input_path / args.pattern}")
            return

        logger.info(f"Found {len(input_files)} files to process")

        # Process files in parallel
        results = await process_files_parallel([str(f) for f in input_files], args.output, embedding_service, args)

        # Count successes
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Successfully processed {successful}/{len(input_files)} files")

        # Update metrics periodically
        metrics_collector.update_resource_metrics()

        # Log GPU usage if available
        if args.device == "cuda" and not args.mock:
            import torch

            memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak GPU memory usage: {memory_used:.2f} GB")

    finally:
        # Cleanup
        logger.info("Shutting down embedding service")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings using unified service")
    parser.add_argument("--input", "-i", default=INPUT_DIR, help="Input directory with parquet files")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="Output directory for embedded parquet files")
    parser.add_argument("--pattern", "-p", default="*.parquet", help="File pattern to match")
    parser.add_argument("--device", "-d", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--model", "-m", default=MODEL_NAME, help="Model name")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--quantization",
        "-q",
        default="float32",
        choices=["float32", "float16", "int8"],
        help="Model quantization (float32, float16, int8)",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings for testing (no GPU required)")
    parser.add_argument("--metrics-port", type=int, default=None, help="Port for Prometheus metrics server")

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
