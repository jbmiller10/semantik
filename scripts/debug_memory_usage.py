#!/usr/bin/env python3
"""
Test script to debug memory issues with embedding jobs
"""
import gc
import logging
import sys
import time
import traceback
from pathlib import Path

import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.text_processing.chunking import TokenChunker
from shared.text_processing.extraction import extract_text

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("/tmp/test_memory_issue.log")],
)
logger = logging.getLogger(__name__)


def monitor_memory():
    """Print current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=0.1),
    }


def test_file(filepath: str):
    """Test processing a single file"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing file: {filepath}")
    logger.info(f"File size: {Path(filepath).stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"Initial memory: {monitor_memory()}")

    try:
        # Test text extraction
        logger.info("Starting text extraction...")
        start_time = time.time()
        text = extract_text(filepath, timeout=60)  # 60 second timeout
        extract_time = time.time() - start_time

        logger.info(f"Extraction complete in {extract_time:.2f}s")
        logger.info(f"Extracted text length: {len(text)} characters")
        logger.info(f"Memory after extraction: {monitor_memory()}")

        # Test chunking
        logger.info("Starting chunking...")
        chunker = TokenChunker(chunk_size=600, chunk_overlap=200)
        start_time = time.time()
        chunks = chunker.chunk_text(text, "test_doc")
        chunk_time = time.time() - start_time

        logger.info(f"Chunking complete in {chunk_time:.2f}s")
        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"Memory after chunking: {monitor_memory()}")

        # Clean up
        del text
        del chunks

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_memory_issue.py <file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]

    if Path(path).is_file():
        # Test single file
        test_file(path)
    elif Path(path).is_dir():
        # Test all supported files in directory
        extensions = {".pdf", ".docx", ".doc", ".txt", ".text"}
        files = []

        for ext in extensions:
            files.extend(Path(path).glob(f"**/*{ext}"))

        logger.info(f"Found {len(files)} files to test")

        for i, filepath in enumerate(files[:5]):  # Test first 5 files
            logger.info(f"\nTesting file {i+1}/{min(5, len(files))}")
            if not test_file(str(filepath)):
                logger.error(f"Stopping after failure on file: {filepath}")
                break

            # Force garbage collection between files
            gc.collect()
            logger.info(f"Memory after GC: {monitor_memory()}")

            # Small delay to let system stabilize
            time.sleep(1)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
