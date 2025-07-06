#!/usr/bin/env python3
"""
Qdrant ingestion script (VS-030)
Bulk loads vectors from parquet files into Qdrant collection
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 4000
# PARALLEL_WORKERS = 4  # Not currently used
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


def process_parquet_file(file_path: str, client: QdrantClient, batch_size: int = BATCH_SIZE) -> bool:
    """Process a single parquet file and upload to Qdrant"""
    try:
        logger.info(f"Processing: {file_path}")

        # Read parquet file
        table = pq.read_table(file_path)

        # Extract data
        ids = table.column("id").to_pylist()
        vectors = table.column("vector").to_pylist()
        payloads = table.column("payload").to_pylist()

        total_points = len(ids)
        logger.info(f"Uploading {total_points} points from {Path(file_path).name}")

        # Upload in batches
        successful_batches = 0
        for i in range(0, total_points, batch_size):
            batch_end = min(i + batch_size, total_points)

            # Prepare batch
            batch_points = []
            for j in range(i, batch_end):
                point = PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j])
                batch_points.append(point)

            # Upload with retry
            success = False
            for retry in range(MAX_RETRIES):
                try:
                    client.upsert(collection_name=settings.DEFAULT_COLLECTION, points=batch_points)
                    successful_batches += 1
                    success = True
                    break
                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        logger.warning(f"Retry {retry + 1}/{MAX_RETRIES} for batch {i//batch_size + 1}: {e}")
                        time.sleep(RETRY_DELAY * (2**retry))  # Exponential backoff
                    else:
                        logger.error(f"Failed to upload batch after {MAX_RETRIES} retries: {e}")

            if not success:
                logger.error(f"Failed to upload batch {i//batch_size + 1} after {MAX_RETRIES} retries")
                return False

        logger.info(f"Successfully uploaded {successful_batches} batches ({total_points} points)")
        return True

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        import traceback

        traceback.print_exc()
        return False


def move_file(src: str, dst_dir: str):
    """Move file to destination directory"""
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    dst = dst_path / src_path.name
    src_path.rename(dst)
    logger.info(f"Moved {src_path.name} to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ingest embeddings into Qdrant")
    parser.add_argument(
        "--input", "-i", default=str(settings.INGEST_DIR), help="Input directory with embedded parquet files"
    )
    parser.add_argument(
        "--loaded", "-l", default=str(settings.LOADED_DIR), help="Directory for successfully loaded files"
    )
    parser.add_argument("--rejects", "-r", default=str(settings.REJECT_DIR), help="Directory for rejected files")
    parser.add_argument("--pattern", "-p", default="*_embedded.parquet", help="File pattern to match")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size for uploads")
    parser.add_argument("--host", default=settings.QDRANT_HOST, help="Qdrant host")
    parser.add_argument("--port", type=int, default=settings.QDRANT_PORT, help="Qdrant port")

    args = parser.parse_args()

    # Create directories
    Path(args.loaded).mkdir(parents=True, exist_ok=True)
    Path(args.rejects).mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = QdrantClient(url=f"http://{args.host}:{args.port}")

    # Get collection info
    try:
        info = client.get_collection(settings.DEFAULT_COLLECTION)
        logger.info(f"Collection '{settings.DEFAULT_COLLECTION}' has {info.points_count} points")
    except Exception as e:
        logger.error(f"Collection '{settings.DEFAULT_COLLECTION}' not found: {e}")
        sys.exit(1)

    # Find input files
    input_path = Path(args.input)
    input_files = list(input_path.glob(args.pattern))

    if not input_files:
        logger.info(f"No files found matching pattern: {input_path / args.pattern}")
        return

    logger.info(f"Found {len(input_files)} files to ingest")

    # Process files
    successful = 0
    failed = 0

    for file_path in tqdm(input_files, desc="Ingesting files"):
        if process_parquet_file(str(file_path), client, args.batch_size):
            move_file(str(file_path), args.loaded)
            successful += 1
        else:
            move_file(str(file_path), args.rejects)
            failed += 1

    # Get final collection info
    try:
        final_info = client.get_collection(settings.DEFAULT_COLLECTION)
        logger.info(f"Collection now has {final_info.points_count} points")
    except Exception as e:
        logger.error(f"Failed to get final collection info: {e}")

    logger.info(f"Ingestion complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
